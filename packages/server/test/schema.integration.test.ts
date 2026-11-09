import assert from "node:assert/strict";
import { randomBytes, randomUUID } from "node:crypto";
import { after, test } from "node:test";

const databaseUrl = process.env.RETAINDB_TEST_DATABASE_URL;
const run = databaseUrl ? test : test.skip;
const environmentKeys = ["DATABASE_URL", "OPENAI_API_KEY", "RETAINDB_API_KEY", "ENCRYPTION_KEY", "EMBEDDING_MODE", "DISABLE_AUTO_EXTRACTION", "LARGE_BATCH_THRESHOLD"] as const;
const originalEnvironment = Object.fromEntries(environmentKeys.map(key => [key, process.env[key]]));
if (databaseUrl) {
  process.env.DATABASE_URL = databaseUrl;
  process.env.ENCRYPTION_KEY = randomBytes(32).toString("hex");
  process.env.EMBEDDING_MODE = "local";
  process.env.DISABLE_AUTO_EXTRACTION = "true";
  delete process.env.OPENAI_API_KEY;
  delete process.env.RETAINDB_API_KEY;
  delete process.env.LARGE_BATCH_THRESHOLD;
}
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

run("uses vector indexes that support initially empty databases without training", async () => {
  const { db } = await import("../src/db/index.js");
  const indexes = await db.$queryRaw<Array<{ tablename: string; indexdef: string }>>`
    SELECT tablename, indexdef FROM pg_indexes
    WHERE schemaname = 'public' AND tablename IN ('chunks', 'memories', 'entities')
      AND indexdef LIKE '%embedding%'
  `;
  for (const table of ["chunks", "memories", "entities"]) {
    assert.ok(indexes.some(index => index.tablename === table && index.indexdef.includes("USING hnsw")), table);
    assert.ok(!indexes.some(index => index.tablename === table && index.indexdef.includes("USING ivfflat")), table);
  }
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

run("keeps bulk document embeddings local without credentials", {
  skip: process.env.RETAINDB_TEST_LOCAL_EMBEDDINGS !== "true",
  timeout: 180_000,
}, async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const modulePath = "../src/engine/ingest.js";
  const { ingestDocument } = await import(modulePath);
  const project = await ensureProject("default", `bulk-local-${fixtureId}`);
  projectIds.push(project.id);
  const source = await db.source.create({
    data: { orgId: "default", projectId: project.id, name: "bulk-documents", type: "text", status: "READY" },
  });
  const content = Array.from({ length: 24 }, (_, index) =>
    `# Recovery section ${index}\nBefore restoring PostgreSQL partition ${index}, verify its backup checksum and stop writes to the database. Keep the verified backup until record counts and semantic retrieval have been checked.`
  ).join("\n\n");
  const document = await ingestDocument({
    sourceId: source.id, projectId: project.id, externalId: "bulk-notes", title: "Partition recovery", content,
    ingestionProfile: "plain_text", strategyOverride: "fixed", skipEntityExtraction: true,
  });
  const chunks = await db.$queryRaw<Array<{ dimensions: number | null }>>`
    SELECT vector_dims(embedding) AS dimensions FROM chunks WHERE "documentId" = ${document.id}
  `;
  assert.ok(document.searchableChunksCreated > 20);
  assert.equal(chunks.length, document.searchableChunksCreated);
  assert.ok(chunks.every(chunk => chunk.dimensions === 1024));
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
  const appPath = "../src/api/routes.js";
  const { api } = await import(appPath);
  assert.equal(typeof api.fetch, "function");
  const response = await api.request("/v1/projects");
  assert.equal(response.status, 200);
  assert.ok(Array.isArray((await response.json()).projects));
});

run("text learns persist Unicode across fixed chunk boundaries without changing source content", {
  skip: process.env.RETAINDB_TEST_LOCAL_EMBEDDINGS !== "true",
  timeout: 180_000,
}, async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const appPath = "../src/api/routes.js";
  const { api } = await import(appPath);
  const project = await ensureProject("default", `unicode-learn-${fixtureId}`);
  projectIds.push(project.id);
  const content = "a".repeat(499) + "🧬" + "b".repeat(198) + "𐐀" + "c".repeat(900);
  const response = await api.request("/v1/learn", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode: "text",
      project: project.id,
      title: "Unicode boundary persistence",
      content,
      metadata: { unicode_fixture: fixtureId },
      options: { async: false, ingestion_profile: "plain_text", strategy_override: "fixed" },
    }),
  });
  assert.equal(response.status, 200, await response.text());
  const document = await db.document.findFirstOrThrow({ where: { projectId: project.id } });
  assert.equal(document.content, content);
  assert.equal(document.status, "READY");
  const chunks = await db.$queryRaw<Array<{ content: string; dimensions: number | null }>>`
    SELECT content, vector_dims(embedding) AS dimensions FROM chunks
    WHERE "documentId" = ${document.id} ORDER BY "chunkIndex"
  `;
  assert.ok(chunks.length > 1);
  for (const chunk of chunks) {
    assert.doesNotMatch(chunk.content, /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u);
    assert.ok(content.includes(chunk.content));
    assert.equal(chunk.dimensions, 1024);
  }
  assert.ok(chunks.some(chunk => chunk.content.includes("🧬")));
  assert.ok(chunks.some(chunk => chunk.content.includes("𐐀")));
});

run("concurrent first text learns share one project source and persist both documents", {
  skip: process.env.RETAINDB_TEST_LOCAL_EMBEDDINGS !== "true",
  timeout: 180_000,
}, async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const appPath = "../src/api/routes.js";
  const { api } = await import(appPath);
  const embeddingPath = "../src/engine/embeddings.js";
  const { embedSingle } = await import(embeddingPath);
  await embedSingle("Initialize genuine local embeddings before concurrent document ingestion.");
  const project = await ensureProject("default", `concurrent-learn-${fixtureId}`);
  projectIds.push(project.id);
  const barrier = `source_barrier_${fixtureId.replaceAll("-", "")}`;
  await db.$executeRawUnsafe(`CREATE FUNCTION "${barrier}"() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NEW."projectId" = '${project.id}' THEN
        PERFORM pg_advisory_xact_lock(hashtext(NEW."projectId"));
      END IF;
      RETURN NEW;
    END $$`);
  await db.$executeRawUnsafe(`CREATE TRIGGER "${barrier}" BEFORE INSERT ON sources FOR EACH ROW EXECUTE FUNCTION "${barrier}"()`);
  try {
    const learn = (label: string) => api.request("/v1/learn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: "text", project: project.id, title: `Concurrent ${label} document`,
        content: `The ${label} document preserves independent knowledge while sharing its project's ingestion source.`,
        metadata: { provenance: fixtureId, label },
        options: { async: false, ingestion_profile: "plain_text", strategy_override: "fixed" },
      }),
    });
    let pending!: Promise<Response[]>;
    await db.$transaction(async transaction => {
      await transaction.$executeRaw`SELECT pg_advisory_xact_lock(hashtext(${project.id}))`;
      pending = Promise.all([learn("first"), learn("second")]);
      const deadline = Date.now() + 10_000;
      while (true) {
        const locks = await db.$queryRaw<Array<{ waiting: number }>>`
          SELECT COUNT(*)::int AS waiting FROM pg_locks WHERE locktype = 'advisory' AND NOT granted
        `;
        if (locks[0].waiting === 2) break;
        assert.ok(Date.now() < deadline, "both first writes must reach source allocation before release");
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    }, { timeout: 15_000 });
    const responses = await pending;
    const bodies = await Promise.all(responses.map(response => response.json()));
    assert.deepEqual(responses.map(response => response.status), [200, 200], JSON.stringify(bodies));
    for (const body of bodies) {
      assert.equal(body.success, true);
      assert.equal(body.status, "completed");
    }
    assert.equal(bodies[0].source_id, bodies[1].source_id);
    assert.equal(await db.source.count({ where: { projectId: project.id } }), 1);
    const versions = await db.sourceVersion.findMany({ where: { sourceId: bodies[0].source_id } });
    assert.equal(versions.length, 1);
    assert.equal(versions[0].status, "ACTIVE");
    const source = await db.source.findUniqueOrThrow({ where: { id: bodies[0].source_id } });
    assert.equal(source.activeVersionId, versions[0].id);
    const documents = await db.document.findMany({ where: { projectId: project.id } });
    assert.equal(documents.length, 2);
    assert.ok(documents.every(document => document.status === "READY" && document.sourceVersionId === source.activeVersionId));
    const vectors = await db.$queryRaw<Array<{ count: bigint }>>`
      SELECT COUNT(*) AS count FROM chunks c
      JOIN documents d ON d.id = c."documentId"
      JOIN sources s ON s.id = d."sourceId"
      WHERE c."projectId" = ${project.id} AND d."sourceVersionId" = s."activeVersionId"
        AND vector_dims(c.embedding) = 1024
    `;
    assert.equal(Number(vectors[0].count), 2);
  } finally {
    await db.$executeRawUnsafe(`DROP TRIGGER "${barrier}" ON sources`);
    await db.$executeRawUnsafe(`DROP FUNCTION "${barrier}"()`);
  }
});

run("runs scheduled operational health against the migrated webhook delivery schema", async () => {
  const { db } = await import("../src/db/index.js");
  const hook = await db.webhook.create({
    data: { orgId: "default", url: "http://127.0.0.1:1/not-called", events: ["document.indexed"], isActive: false },
  });
  const delivery = await db.webhookDelivery.create({
    data: { webhookId: hook.id, event: "document.indexed", payload: {}, errorCode: "DELIVERY_FAILED", errorMessage: "Fixture failure", responseBody: "Fixture response", parentTraceId: fixtureId },
  });
  const modulePath = "../src/engine/ops-observability.js";
  const { evaluateOperationalAlerts } = await import(modulePath);
  try {
    const result = await evaluateOperationalAlerts();
    assert.ok(Array.isArray(result.alerts));
    assert.ok(result.webhooks.recent_failures.some(item => item.id === delivery.id && item.errorCode === "DELIVERY_FAILED"));
    assert.ok(result.webhooks.deliveries.some(item => item.event === "document.indexed"));
  } finally {
    await db.webhookDelivery.delete({ where: { id: delivery.id } });
    await db.webhook.delete({ where: { id: hook.id } });
  }
});

run("allocates unique source versions after retention gaps and concurrent staging", async () => {
  const { ensureProject } = await import("../src/api/helpers.js");
  const { db } = await import("../src/db/index.js");
  const modulePath = "../src/engine/source-versions.js";
  const { ensureActiveSourceVersion, createStagedSourceVersion } = await import(modulePath);
  const project = await ensureProject("default", `version-allocation-${fixtureId}`);
  projectIds.push(project.id);
  const source = await db.source.create({
    data: { projectId: project.id, orgId: "default", name: "version-allocation", type: "text", status: "READY" },
  });
  await db.sourceVersion.createMany({
    data: [1, 3].map(versionNumber => ({
      sourceId: source.id, orgId: "default", projectId: project.id, versionNumber, status: "SUPERSEDED",
    })),
  });
  const active = await ensureActiveSourceVersion(source.id);
  assert.equal(active?.versionNumber, 4);
  const staged = await Promise.all(Array.from({ length: 4 }, () => createStagedSourceVersion(source.id)));
  assert.deepEqual(staged.map(version => version.versionNumber).sort((a, b) => a - b), [5, 6, 7, 8]);
  const current = await db.source.findUniqueOrThrow({ where: { id: source.id } });
  assert.equal(current.activeVersionId, active?.id);
  assert.equal(await db.sourceVersion.count({ where: { sourceId: source.id, status: "ACTIVE" } }), 1);
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
  try {
    const { db } = await import("../src/db/index.js");
    try {
      await db.project.deleteMany({ where: { id: { in: projectIds } } });
    } finally {
      await db.$disconnect();
    }
  } finally {
    for (const key of environmentKeys) {
      if (originalEnvironment[key] === undefined) delete process.env[key];
      else process.env[key] = originalEnvironment[key];
    }
  }
});
