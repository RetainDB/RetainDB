import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { test } from "node:test";

const databaseUrl = process.env.RETAINDB_TEST_DATABASE_URL;
const run = databaseUrl ? test : test.skip;
const savedEnvironment = { ...process.env };
if (databaseUrl) process.env.DATABASE_URL = databaseUrl;

run("concurrent package syncs share one source row", { timeout: 30_000 }, async () => {
  const { db } = await import("../src/db/index.js");
  const { ensurePackageSource } = await import("../src/engine/autosubscribe.js");
  const name = `autosubscribe-${randomUUID()}`;
  const project = await db.project.create({ data: { name, slug: name, orgId: "default" } });
  try {
    const pkg = { name: "left-pad", ecosystem: "npm" };
    const ids = await Promise.all(Array.from({ length: 5 }, () => ensurePackageSource(project.id, "default", pkg)));
    assert.equal(new Set(ids).size, 1);
    assert.equal(await db.source.count({ where: { projectId: project.id, deletedAt: null } }), 1);
    assert.equal(await ensurePackageSource(project.id, "default", pkg), ids[0]);
  } finally {
    await db.project.delete({ where: { id: project.id } });
    await db.$disconnect();
    process.env = savedEnvironment;
  }
});
