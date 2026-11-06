import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { randomBytes, randomUUID } from "node:crypto";
import { createServer } from "node:http";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { PrismaClient } from "@prisma/client";

const databaseUrl = process.env.RETAINDB_TEST_DATABASE_URL;
const run = databaseUrl ? test : test.skip;

run("a stalled native embedding attempt cannot block later receipts or write after timeout", { timeout: 30_000 }, async () => {
  const admin = new PrismaClient({ datasources: { db: { url: databaseUrl } } });
  const databaseName = `native_timeout_${randomUUID().replaceAll("-", "")}`;
  const isolatedUrl = new URL(databaseUrl!);
  isolatedUrl.pathname = `/${databaseName}`;
  await admin.$executeRawUnsafe(`CREATE DATABASE "${databaseName}"`);
  try {
    await promisify(execFile)(process.execPath, [
      fileURLToPath(new URL("../node_modules/prisma/build/index.js", import.meta.url)),
      "migrate", "deploy", "--schema", fileURLToPath(new URL("../prisma/schema.prisma", import.meta.url)),
    ], { env: { ...process.env, DATABASE_URL: isolatedUrl.toString() }, timeout: 15_000 });
    await exerciseStalledAttempt(isolatedUrl.toString());
  } finally {
    await admin.$executeRawUnsafe(`DROP DATABASE "${databaseName}" WITH (FORCE)`);
    await admin.$disconnect();
  }
});

async function exerciseStalledAttempt(isolatedDatabaseUrl: string): Promise<void> {
  const keys = ["DATABASE_URL", "ENCRYPTION_KEY", "EMBEDDING_MODE", "REMOTE_INFERENCE_REQUIRED", "INFERENCE_TIMEOUT_MS", "INFERENCE_BASE_URL", "EMBEDDING_INFERENCE_BASE_URL", "NATIVE_CAPTURE_EMBEDDING_TIMEOUT_MS"];
  const original = Object.fromEntries(keys.map(key => [key, process.env[key]]));
  let calls = 0;
  let stalledConnectionClosed = false;
  const inference = createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    calls++;
    if (body.includes("First fixture blocks")) {
      response.on("close", () => { stalledConnectionClosed = true; });
      return;
    }
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ embeddings: [Array.from({ length: 1024 }, (_, index) => index === 0 ? 1 : 0)] }));
  });
  await new Promise<void>(resolve => inference.listen(0, "127.0.0.1", resolve));
  const address = inference.address();
  assert.ok(address && typeof address !== "string");
  Object.assign(process.env, {
    DATABASE_URL: isolatedDatabaseUrl, ENCRYPTION_KEY: randomBytes(32).toString("hex"),
    EMBEDDING_MODE: "remote", REMOTE_INFERENCE_REQUIRED: "true", INFERENCE_TIMEOUT_MS: "15000",
    INFERENCE_BASE_URL: `http://127.0.0.1:${address.port}`, EMBEDDING_INFERENCE_BASE_URL: `http://127.0.0.1:${address.port}`,
    NATIVE_CAPTURE_EMBEDDING_TIMEOUT_MS: "2000",
  });
  const { db } = await import("../src/db/index.js");
  const capturePath = "../src/engine/native-capture.js";
  const capture = await import(capturePath);
  const project = await db.project.create({ data: { orgId: "default", name: randomUUID(), slug: randomUUID() } });
  try {
    const first = await capture.acceptNativeCapture({ orgId: "default", projectId: project.id, sessionId: "stalled", agentId: "test", hookType: "prompt", data: { prompt: "First fixture blocks in the embedding provider." } });
    for (let attempt = 0; attempt < 200 && calls === 0; attempt++) await delay(20);
    assert.equal(calls, 1);
    const second = await capture.acceptNativeCapture({ orgId: "default", projectId: project.id, sessionId: "later", agentId: "test", hookType: "prompt", data: { prompt: "Second fixture must be indexed after the first attempt is terminated." } });
    let indexed = false;
    for (let attempt = 0; attempt < 300; attempt++) {
      const message = await db.message.findUniqueOrThrow({ where: { id: second.id } });
      if ((message.metadata as Record<string, unknown>).native_capture_state === "indexed") { indexed = true; break; }
      await delay(20);
    }
    assert.ok(indexed, "Later receipt must index before the provider's own timeout");
    assert.ok(stalledConnectionClosed, "Timed-out model process must close its request before work is retried");
    const failed = await db.message.findUniqueOrThrow({ where: { id: first.id } });
    assert.equal((failed.metadata as Record<string, unknown>).native_capture_state, "pending");
    assert.match(String((failed.metadata as Record<string, unknown>).last_error), /timed out/i);
  } finally {
    const embeddingsPath = "../src/engine/native-embeddings.js";
    const embeddings = await import(embeddingsPath);
    await capture.drainNativeCaptures();
    await embeddings.stopNativeEmbeddingProcess();
    await db.project.delete({ where: { id: project.id } });
    await db.$disconnect();
    inference.closeAllConnections();
    await new Promise<void>(resolve => inference.close(() => resolve()));
    for (const key of keys) {
      if (original[key] === undefined) delete process.env[key];
      else process.env[key] = original[key];
    }
  }
}
