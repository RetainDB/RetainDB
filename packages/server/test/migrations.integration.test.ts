import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import postgres from "postgres";

const databaseUrl = process.env.RETAINDB_TEST_DATABASE_URL;
const run = databaseUrl ? test : test.skip;
const migrationRoot = new URL("../prisma/migrations/", import.meta.url);
const schemaPath = fileURLToPath(new URL("../prisma/schema.prisma", import.meta.url));
const prismaCli = fileURLToPath(new URL("../node_modules/prisma/build/index.js", import.meta.url));
const preparation = "20261106001500_preserve_bootstrap_slug";
const bootstrap = "20261106002000_bootstrap_oss_organization";

async function withDatabase(fn: (url: URL, client: ReturnType<typeof postgres>) => Promise<void>) {
  const name = `migrations_${randomUUID().replaceAll("-", "")}`;
  const admin = postgres(databaseUrl!, { max: 1, onnotice: () => {} });
  const isolatedUrl = new URL(databaseUrl!);
  isolatedUrl.pathname = `/${name}`;
  let client: ReturnType<typeof postgres> | undefined;
  try {
    await admin.unsafe(`CREATE DATABASE "${name}"`);
    client = postgres(isolatedUrl.toString(), { max: 1, onnotice: () => {} });
    await fn(isolatedUrl, client);
  } finally {
    await client?.end({ timeout: 5 });
    await admin.unsafe(`DROP DATABASE IF EXISTS "${name}" WITH (FORCE)`);
    await admin.end({ timeout: 5 });
  }
}

async function migrateDeploy(url: URL) {
  return promisify(execFile)(process.execPath, [prismaCli, "migrate", "deploy", "--schema", schemaPath], {
    env: { ...process.env, DATABASE_URL: url.toString() },
    timeout: 30_000,
  });
}

run("migrate deploy creates a complete schema and provisions the default organization", { timeout: 60_000 }, async () => {
  await withDatabase(async (url, client) => {
    await migrateDeploy(url);
    const expected = (await readdir(migrationRoot, { withFileTypes: true })).filter((entry) => entry.isDirectory()).length;
    const applied = await client`SELECT count(*)::int AS count FROM _prisma_migrations WHERE finished_at IS NOT NULL`;
    assert.equal(applied[0].count, expected);
    const organization = await client`SELECT id, slug, "ownerId" FROM organizations WHERE id = 'default'`;
    assert.equal(organization.length, 1);
    assert.equal(organization[0].ownerId, null);
  });
});

run("the package registry migration is idempotent over an existing table", { timeout: 60_000 }, async () => {
  await withDatabase(async (_url, client) => {
    await client.unsafe(`
      CREATE TABLE "packages" (
        "id" text PRIMARY KEY,
        "orgId" text NOT NULL,
        "name" text NOT NULL,
        "ecosystem" text NOT NULL,
        "version" text NOT NULL,
        "autoSync" boolean NOT NULL DEFAULT true,
        "docsIndexed" integer NOT NULL DEFAULT 0,
        "createdAt" timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" timestamp(3) NOT NULL
      );
      CREATE UNIQUE INDEX "packages_orgId_ecosystem_name_key" ON "packages" ("orgId", "ecosystem", "name");
    `);
    const sql = await readFile(new URL("20261110000000_package_registry/migration.sql", migrationRoot), "utf8");
    await client.unsafe(sql).simple();
    await client.unsafe(sql).simple();
    const columns = await client`SELECT column_name FROM information_schema.columns WHERE table_name = 'packages'`;
    assert.ok(columns.some((column) => column.column_name === "projectId"));
    const indexes = await client`SELECT indexname FROM pg_indexes WHERE tablename = 'packages'`;
    assert.ok(indexes.some((index) => index.indexname === "packages_orgId_ecosystem_name_key"));
  });
});

run("bootstrap preserves an organization that already occupies the default slug", { timeout: 60_000 }, async () => {
  await withDatabase(async (_url, client) => {
    const migrations = (await readdir(migrationRoot, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory() && entry.name < preparation)
      .map((entry) => entry.name)
      .sort();
    for (const migration of migrations) {
      await client.unsafe(await readFile(new URL(`${migration}/migration.sql`, migrationRoot), "utf8")).simple();
    }
    await client`INSERT INTO users (id, email, "updatedAt") VALUES ('owner', 'owner@example.test', NOW())`;
    await client`INSERT INTO organizations (id, name, slug, "ownerId", "updatedAt")
      VALUES ('existing', 'Existing', 'default', 'owner', NOW())`;

    for (const migration of [preparation, bootstrap]) {
      await client.unsafe(await readFile(new URL(`${migration}/migration.sql`, migrationRoot), "utf8")).simple();
    }

    const rows = await client`SELECT id, slug, "ownerId" FROM organizations ORDER BY id`;
    assert.deepEqual(rows.map((row) => row.id), ["default", "existing"]);
    assert.equal(rows[0].slug, "default-oss-1");
    assert.equal(rows[0].ownerId, null);
    assert.equal(rows[1].slug, "default");
    await assert.rejects(
      client`INSERT INTO organizations (id, name, slug, "ownerId", "updatedAt")
        VALUES ('invalid', 'Invalid', 'invalid', NULL, NOW())`,
      (error: unknown) => error instanceof Error && "code" in error && error.code === "23514",
    );
  });
});
