import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { after, test } from "node:test";

const databaseUrl = process.env.RETAINDB_TEST_DATABASE_URL;
const run = databaseUrl ? test : test.skip;
const savedEnvironment = { ...process.env };
if (databaseUrl) {
  process.env.DATABASE_URL = databaseUrl;
  process.env.DISABLE_AUTO_EXTRACTION = "true";
  process.env.EMBEDDING_MODE = "local";
}
const ingestPath = "../src/engine/ingest.js";
const projectIds: string[] = [];

function vectors(texts: string[]): number[][] {
  return texts.map((text) => {
    const vector = Array<number>(1024).fill(0);
    vector[0] = 1;
    vector[1] = text.length / 1000;
    return vector;
  });
}

async function fixture() {
  const { db } = await import("../src/db/index.js");
  const { ensureActiveSourceVersion } = await import("../src/engine/source-versions.js");
  const name = `ingest-${randomUUID()}`;
  const project = await db.project.create({ data: { name, slug: name, orgId: "default" } });
  projectIds.push(project.id);
  const source = await db.source.create({
    data: { projectId: project.id, orgId: "default", name, type: "text", status: "READY" },
  });
  const version = await ensureActiveSourceVersion(source.id);
  assert.ok(version);
  return { db, project, source, version };
}

function input(sourceId: string, projectId: string, versionId: string, embedTexts?: (texts: string[]) => Promise<number[][]>) {
  return {
    sourceId,
    projectId,
    sourceVersionId: versionId,
    externalId: `document-${randomUUID()}.txt`,
    title: "Recovery",
    content: "Verify the PostgreSQL backup checksum before restoring partitions.",
    ingestionProfile: "plain_text" as const,
    strategyOverride: "fixed" as const,
    skipEntityExtraction: true,
    skipSourceCountUpdate: true,
    embedTexts,
  };
}

run("a required embedding failure records ERROR and leaves no chunks", { timeout: 30_000 }, async () => {
  const { ingestDocument } = await import(ingestPath);
  const { db, project, source, version } = await fixture();
  const document = input(source.id, project.id, version.id, async () => {
    throw new Error("embedding unavailable");
  });

  await assert.rejects(ingestDocument(document), /embedding unavailable/);

  const stored = await db.document.findFirstOrThrow({ where: { externalId: document.externalId } });
  assert.equal(stored.status, "ERROR");
  assert.equal(stored.indexedAt, null);
  assert.match(stored.parseError ?? "", /embedding unavailable/);
  assert.equal(await db.chunk.count({ where: { documentId: stored.id } }), 0);
  await assert.rejects(
    ingestDocument({ ...document, externalId: `second-${randomUUID()}.txt` }),
    /embedding unavailable/,
  );
  await db.source.update({ where: { id: source.id }, data: { status: "READY" } });
  await assert.rejects(ingestDocument({ ...document, externalId: `third-${randomUUID()}.txt`, skipSourceCountUpdate: false }), /embedding unavailable/);
  const failedSource = await db.source.findUniqueOrThrow({ where: { id: source.id } });
  assert.equal(failedSource.status, "ERROR");
});

run("a successful ingest publishes READY with vectors for every chunk", { timeout: 30_000 }, async () => {
  const { ingestDocument } = await import(ingestPath);
  const { db, project, source, version } = await fixture();
  const document = input(source.id, project.id, version.id, async (texts) => vectors(texts));

  const result = await ingestDocument({ ...document, skipSourceCountUpdate: false });
  assert.equal(result.status, "READY");

  const chunks = await db.$queryRaw<Array<{ dimensions: number }>>`
    SELECT vector_dims(embedding) AS dimensions FROM chunks WHERE "documentId" = ${result.documentId}
  `;
  assert.ok(chunks.length > 0);
  assert.ok(chunks.every((chunk) => chunk.dimensions === 1024));
  const healthySource = await db.source.findUniqueOrThrow({ where: { id: source.id } });
  assert.equal(healthySource.status, "READY");
});

run("a failed batch settles owned work before rejecting", { timeout: 30_000 }, async () => {
  const { ingestDocuments } = await import(ingestPath);
  const { db, project, source, version } = await fixture();
  const documents = Array.from({ length: 5 }, () => input(source.id, project.id, version.id, async () => {
    throw new Error("batch embedding unavailable");
  }));

  await assert.rejects(ingestDocuments(documents), /batch embedding unavailable/);
  await new Promise((resolve) => setTimeout(resolve, 200));

  const stored = await db.document.findMany({ where: { sourceId: source.id } });
  assert.equal(stored.length, documents.length);
  assert.ok(stored.every((document) => document.status === "ERROR"));
  assert.equal(await db.chunk.count({ where: { document: { sourceId: source.id } } }), 0);
});

after(async () => {
  if (!databaseUrl) return;
  const { db } = await import("../src/db/index.js");
  if (projectIds.length > 0) await db.project.deleteMany({ where: { id: { in: projectIds } } });
  await db.$disconnect();
  process.env = savedEnvironment;
});
