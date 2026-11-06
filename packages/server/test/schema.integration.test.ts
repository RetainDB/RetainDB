import assert from "node:assert/strict";
import { randomBytes, randomUUID } from "node:crypto";
import { after, test } from "node:test";

const databaseUrl = process.env.RETAINDB_TEST_DATABASE_URL;
const run = databaseUrl ? test : test.skip;
if (databaseUrl) process.env.DATABASE_URL = databaseUrl;
const fixtureId = randomUUID();
const projectIds: string[] = [];

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
  const { db } = await import("../src/db/index.js");
  const jobs = await db.syncJob.findMany({ take: 1 });
  assert.ok(Array.isArray(jobs));
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

after(async () => {
  if (!databaseUrl) return;
  const { db } = await import("../src/db/index.js");
  try {
    await db.project.deleteMany({ where: { id: { in: projectIds } } });
  } finally {
    await db.$disconnect();
  }
});
