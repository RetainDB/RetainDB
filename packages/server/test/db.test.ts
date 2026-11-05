import assert from "node:assert/strict";
import { after, test } from "node:test";
import { Prisma } from "@prisma/client";

process.env.NODE_ENV = "production";
process.env.DATABASE_URL = "postgresql://localhost:1/retaindb_unused";

const clientModule = new URL("../src/db/index.ts", import.meta.url);

async function clientReturning(result: unknown) {
  const { db } = await import("../src/db/index.js");
  return db.$extends({
    query: {
      async $allOperations() {
        return result;
      },
    },
  });
}

after(async () => {
  const { db } = await import("../src/db/index.js");
  await db.$disconnect();
});

test("initializes the database client with the locked Prisma version", async () => {
  const { db, prisma } = await import(clientModule.href);

  assert.equal(db, prisma);
  assert.equal(typeof db.$connect, "function");
  assert.equal(typeof db.$transaction, "function");
});

test("reuses the extended client across production module reloads", async () => {
  const first = await import(clientModule.href);
  const second = await import(`${clientModule.href}?reload`);

  assert.equal(first.db, second.db);
  assert.equal(first.prisma, second.prisma);
});

test("converts nested BigInt values without mutating query results", async () => {
  const original = [{ count: 9007199254740993n, nested: { values: [0n, -2n] } }];
  const db = await clientReturning(original);
  const result = await db.$queryRaw`SELECT 1`;

  assert.deepEqual(result, [{ count: "9007199254740993", nested: { values: ["0", "-2"] } }]);
  assert.equal(original[0].count, 9007199254740993n);
  assert.deepEqual(original[0].nested.values, [0n, -2n]);
  assert.doesNotThrow(() => JSON.stringify(result));
});

test("preserves Date, bytes and Decimal values beside BigInt results", async () => {
  const createdAt = new Date("2026-01-02T03:04:05.000Z");
  const bytes = new Uint8Array([1, 2, 3]);
  const buffer = Buffer.from([4, 5]);
  const amount = new Prisma.Decimal("123.456");
  const db = await clientReturning([{ count: 1n, createdAt, bytes, buffer, amount }]);
  const [result] = await db.$queryRaw<Array<Record<string, unknown>>>`SELECT 1`;

  assert.equal(result.count, "1");
  assert.equal(result.createdAt, createdAt);
  assert.equal(result.bytes, bytes);
  assert.equal(result.buffer, buffer);
  assert.equal(result.amount, amount);
});

test("leaves query results without BigInt unchanged", async () => {
  const original = [{ name: "project", count: 2, settings: { enabled: true }, missing: null }];
  const db = await clientReturning(original);

  assert.equal(await db.$queryRaw`SELECT 1`, original);
});

test("preserves null and undefined in results containing BigInt", async () => {
  const db = await clientReturning([{ count: 1n, missing: null, absent: undefined }]);

  assert.deepEqual(await db.$queryRaw`SELECT 1`, [{
    count: "1",
    missing: null,
    absent: undefined,
  }]);
});

test("preserves primitive query results", async () => {
  for (const value of [null, undefined, false, true, 0, "value"]) {
    const db = await clientReturning(value);
    assert.equal(await db.$queryRaw`SELECT 1`, value);
  }
});

test("normalizes BigInt in objects with a null prototype", async () => {
  const original = Object.assign(Object.create(null), { count: 7n });
  const db = await clientReturning([original]);

  assert.deepEqual(await db.$queryRaw`SELECT 1`, [{ count: "7" }]);
  assert.equal(original.count, 7n);
});

test("preserves model query results", async () => {
  const db = await clientReturning({ id: "project-id", settings: { enabled: true } });
  const project = await db.project.findUnique({ where: { id: "project-id" } });

  assert.deepEqual(project, { id: "project-id", settings: { enabled: true } });
});

test("normalizes unsafe raw-query results", async () => {
  const db = await clientReturning([{ count: 4n }]);

  assert.deepEqual(await db.$queryRawUnsafe("SELECT 1"), [{ count: "4" }]);
});

test("preserves raw mutation row counts", async () => {
  const db = await clientReturning(2);

  assert.equal(await db.$executeRaw`UPDATE projects SET name = name`, 2);
  assert.equal(await db.$executeRawUnsafe("UPDATE projects SET name = name"), 2);
});

test("keeps Prisma query promises lazy", async () => {
  const { db } = await import("../src/db/index.js");
  let calls = 0;
  const client = db.$extends({
    query: {
      async $allOperations() {
        calls += 1;
        return [{ count: 5n }];
      },
    },
  });
  const result = client.$queryRaw`SELECT 1`;

  assert.equal(calls, 0);
  assert.equal(result[Symbol.toStringTag], "PrismaPromise");
  assert.deepEqual(await result, [{ count: "5" }]);
  assert.equal(calls, 1);
});

test("keeps query promises compatible with batch transactions", async () => {
  const db = await clientReturning([{ count: 6n }]);
  const results = await db.$transaction([
    db.$queryRaw`SELECT 1`,
    db.$queryRawUnsafe("SELECT 2"),
  ]);

  assert.deepEqual(results, [[{ count: "6" }], [{ count: "6" }]]);
});

test("passes query arguments through unchanged", async () => {
  const { db } = await import("../src/db/index.js");
  const args = { where: { id: "project-id" }, select: { name: true as const } };
  const client = db.$extends({
    query: {
      async $allOperations(request) {
        assert.equal(request.model, "Project");
        assert.equal(request.operation, "findUnique");
        assert.deepEqual(request.args, args);
        return { name: "project" };
      },
    },
  });

  assert.deepEqual(await client.project.findUnique(args), { name: "project" });
});

test("propagates query failures unchanged", async () => {
  const { db } = await import("../src/db/index.js");
  const failure = new Error("query failed");
  const client = db.$extends({
    query: {
      async $allOperations() {
        throw failure;
      },
    },
  });

  await assert.rejects(client.$queryRaw`SELECT 1`, (error) => error === failure);
});
