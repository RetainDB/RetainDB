import assert from "node:assert/strict";
import { randomBytes, randomUUID } from "node:crypto";
import { after, test } from "node:test";

const databaseUrl = process.env.RETAINDB_TEST_DATABASE_URL;
const run = databaseUrl ? test : test.skip;
if (databaseUrl) process.env.DATABASE_URL = databaseUrl;
const fixtureId = randomUUID();
const projectIds: string[] = [];

run("provisions the default organization and retains tenant references", async () => {
  const { db } = await import("../src/db/index.js");
  const organizations = await db.$queryRaw<Array<{ id: string; ownerId: string | null }>>`
    SELECT id, "ownerId" FROM organizations WHERE id = 'default'
  `;
  assert.equal(organizations.length, 1);
  assert.equal(organizations[0].id, "default");
  await assert.rejects(db.project.create({
    data: { orgId: `missing-${fixtureId}`, name: fixtureId, slug: fixtureId },
  }), (error: unknown) => error instanceof Error && "code" in error && error.code === "P2003");
});

run("creates and resolves the default OSS project after production migrations", async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const name = `bootstrap-${fixtureId}`;
  const created = await ensureProject("default", name);
  projectIds.push(created.id);
  const resolved = await ensureProject("default", name);

  assert.equal(created.id, resolved.id);
  assert.equal(created.orgId, "default");
  assert.equal(created.name, name);
});

run("creates a source in the default OSS organization without cloud identity records", async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const project = await ensureProject("default", `source-bootstrap-${fixtureId}`);
  projectIds.push(project.id);
  const source = await db.source.create({
    data: {
      orgId: "default",
      projectId: project.id,
      name: "bootstrap-source",
      type: "text",
      status: "READY",
    },
  });

  assert.equal(source.projectId, project.id);
  assert.equal(source.orgId, "default");
  assert.equal(source.status, "READY");
});

run("keeps project references enforced without cloud organization records", async () => {
  const { db } = await import("../src/db/index.js");

  await assert.rejects(db.source.create({
    data: {
      orgId: "default",
      projectId: randomUUID(),
      name: `missing-project-${fixtureId}`,
      type: "text",
    },
  }), (error: unknown) => error instanceof Error && "code" in error && error.code === "P2003");
});

run("persists source versions, chunks and retrieval vectors through document ingestion", {
  skip: process.env.RETAINDB_TEST_LOCAL_EMBEDDINGS !== "true",
  timeout: 180_000,
}, async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const modulePath = "../src/engine/ingest.js";
  const { ingestDocument } = await import(modulePath);
  const project = await ensureProject("default", `document-bootstrap-${fixtureId}`);
  projectIds.push(project.id);
  const source = await db.source.create({
    data: { orgId: "default", projectId: project.id, name: "documents", type: "text", status: "READY" },
  });
  const input = {
    projectId: project.id,
    sourceId: source.id,
    externalId: "operating-notes",
    title: "Database recovery procedure",
    content: "To restore the PostgreSQL service, stop the application, recover the database from its verified backup, and restart the application after checking database health. Preserve the original backup until retrieval checks pass.",
    ingestionProfile: "plain_text",
    strategyOverride: "fixed",
    skipEntityExtraction: true,
    metadata: { provenance: fixtureId },
  };
  const document = await ingestDocument(input);
  const repeated = await ingestDocument(input);
  const chunks = await db.$queryRaw<Array<{ id: string; dimensions: number; content: string }>>`
    SELECT id, vector_dims(embedding) AS dimensions, content FROM chunks WHERE "documentId" = ${document.id}
  `;

  assert.equal(document.id, repeated.id);
  assert.equal(document.status, "READY");
  assert.ok(document.sourceVersionId);
  assert.ok(chunks.length > 0);
  assert.ok(chunks.every(chunk => chunk.dimensions === 1024));
  assert.ok(chunks.some(chunk => chunk.content.includes("verified backup")));

  const retrievalPath = "../src/engine/retriever.js";
  const { retrieve } = await import(retrievalPath);
  const result = await retrieve({
    projectId: project.id,
    query: "How should I recover the database safely?",
    useCache: false,
    hybridSearch: false,
    rerank: false,
    compress: false,
    topK: 3,
  });
  assert.ok(chunks.some(chunk => chunk.id === result.results[0]?.id));
  assert.equal(result.results[0]?.source, "vector");
  assert.equal(result.results[0]?.metadata.source_id, source.id);
  assert.ok(result.context.includes("verified backup"));
});

run("accepts and completes asynchronous ingestion for the default organization", { timeout: 10_000 }, async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const queuePath = "../src/engine/ingestion-queue.js";
  const { ingestionQueue: queue } = await import(queuePath);
  const project = await ensureProject("default", `async-bootstrap-${fixtureId}`);
  projectIds.push(project.id);

  const jobId = await queue.createJob({ orgId: "default", projectId: project.id, userId: "local", documents: [] });
  try {
    const status = await queue.getJobStatus(jobId);
    assert.equal(status.orgId, "default");
    let completed = status;
    for (let attempt = 0; attempt < 100 && (completed.status === "PENDING" || completed.status === "PROCESSING"); attempt++) {
      await new Promise(resolve => setTimeout(resolve, 20));
      completed = await queue.getJobStatus(jobId);
    }
    assert.equal(completed.status, "COMPLETED");
    assert.equal(completed.totalDocuments, 0);
  } finally {
    await db.$executeRaw`DELETE FROM ingestion_jobs WHERE id = ${jobId}`;
  }
});

run("reads persisted connector job health after production migrations", async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const project = await ensureProject("default", `connector-health-${fixtureId}`);
  projectIds.push(project.id);
  const source = await db.source.create({
    data: { projectId: project.id, orgId: "default", name: "connector", type: fixtureId, status: "READY" },
  });
  const job = await db.syncJob.create({
    data: { sourceId: source.id, status: "COMPLETED", documentsTotal: 2, documentsIndexed: 1, documentsFailed: 1, progress: 50.5, partialFailure: true, warningCodes: ["PARTIAL_INDEX"] },
  });
  assert.equal(job.progress, 50.5);
  const healthPath = "../src/engine/ops-observability.js";
  const { getConnectorHealthSummary } = await import(healthPath);
  const health = await getConnectorHealthSummary();
  const connector = health.connectors.find((entry: { connector_type: string }) => entry.connector_type === fixtureId);
  assert.equal(connector?.partial_failures, 1);
  assert.equal(connector?.documents_indexed, 1);
  assert.equal(connector?.documents_failed, 1);
});

run("loads the self-hosted API without an OpenAI credential", async () => {
  assert.equal(process.env.OPENAI_API_KEY, undefined);
  process.env.ENCRYPTION_KEY ||= randomBytes(32).toString("hex");
  const appPath = "../src/api/routes.js";
  const { api } = await import(appPath);
  assert.equal(typeof api.fetch, "function");
  const response = await api.request("/v1/projects");
  assert.equal(response.status, 200);
  assert.ok(Array.isArray((await response.json()).projects));
});

run("reports chunk persistence failures instead of successful ingestion", async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const modulePath = "../src/engine/ingest.js";
  const { ingestDocument } = await import(modulePath);
  const project = await ensureProject("default", `failed-ingest-${fixtureId}`);
  projectIds.push(project.id);
  const source = await db.source.create({
    data: { projectId: project.id, orgId: "default", name: "failing-store", type: "text", status: "READY" },
  });
  const constraint = `test_chunk_${fixtureId.replaceAll("-", "")}`;
  await db.$executeRawUnsafe(`ALTER TABLE chunks ADD CONSTRAINT "${constraint}" CHECK ((metadata->>'failure_fixture') IS DISTINCT FROM '${fixtureId}') NOT VALID`);
  try {
    await assert.rejects(ingestDocument({
      projectId: project.id,
      sourceId: source.id,
      externalId: "rejected-document",
      title: "Rejected storage",
      content: "This document must not be reported as indexed when its chunks cannot be persisted.",
      metadata: { failure_fixture: fixtureId },
      ingestionProfile: "plain_text",
      strategyOverride: "fixed",
      skipEntityExtraction: true,
    }));
    const document = await db.document.findFirst({ where: { sourceId: source.id } });
    assert.equal(document?.status, "ERROR");
    const queuePath = "../src/engine/ingestion-queue.js";
    const { ingestionQueue: queue } = await import(queuePath);
    const jobId = await queue.createJob({
      orgId: "default", projectId: project.id, userId: "local",
      documents: [{ title: "Rejected queued storage", content: "The queued document must report failed persistence, not completed ingestion.", metadata: { failure_fixture: fixtureId }, ingestion_profile: "plain_text", strategy_override: "fixed" }],
    });
    try {
      let status = await queue.getJobStatus(jobId);
      for (let attempt = 0; attempt < 200 && (status.status === "PENDING" || status.status === "PROCESSING"); attempt++) {
        await new Promise(resolve => setTimeout(resolve, 20));
        status = await queue.getJobStatus(jobId);
      }
      assert.equal(status.status, "FAILED");
      assert.equal(status.processedChunks, 0);
    } finally {
      await db.$executeRaw`DELETE FROM ingestion_jobs WHERE id = ${jobId}`;
    }
  } finally {
    await db.$executeRawUnsafe(`ALTER TABLE chunks DROP CONSTRAINT "${constraint}"`);
  }
});

run("processes a queued document and persists the actual chunk count", {
  skip: process.env.RETAINDB_TEST_LOCAL_EMBEDDINGS !== "true",
  timeout: 20_000,
}, async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const queuePath = "../src/engine/ingestion-queue.js";
  const { ingestionQueue: queue } = await import(queuePath);
  const project = await ensureProject("default", `async-document-${fixtureId}`);
  projectIds.push(project.id);
  const jobId = await queue.createJob({
    orgId: "default", projectId: project.id, userId: "local",
    documents: [{ title: "Queued recovery", content: "Restore the PostgreSQL backup only after verifying its checksum and stopping writes.", ingestion_profile: "plain_text", strategy_override: "fixed" }],
  });
  try {
    let status = await queue.getJobStatus(jobId);
    for (let attempt = 0; attempt < 400 && (status.status === "PENDING" || status.status === "PROCESSING"); attempt++) {
      await new Promise(resolve => setTimeout(resolve, 20));
      status = await queue.getJobStatus(jobId);
    }
    assert.equal(status.status, "COMPLETED");
    const chunks = await db.chunk.count({ where: { projectId: project.id } });
    assert.ok(chunks > 0);
    assert.equal(status.totalChunks, chunks);
    assert.equal(status.processedChunks, chunks);
    const rows = await db.$queryRaw<Array<{ document_id: string; status: string }>>`SELECT document_id, status FROM ingestion_documents WHERE job_id = ${jobId}`;
    assert.ok(rows[0]?.document_id);
    assert.equal(rows[0]?.status, "COMPLETED");
  } finally {
    await db.$executeRaw`DELETE FROM ingestion_jobs WHERE id = ${jobId}`;
  }
});

after(async () => {
  if (!databaseUrl) return;
  const { db } = await import("../src/db/index.js");
  try {
    await db.project.deleteMany({ where: { id: { in: projectIds } } });
  } finally {
    await db.$disconnect();
  }
});
