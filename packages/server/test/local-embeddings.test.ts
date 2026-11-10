import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

test("local embeddings retain real normalized vectors when native inference falls back to WASM", {
  skip: process.env.RETAINDB_TEST_LOCAL_EMBEDDINGS !== "true",
  timeout: 180_000,
}, async (context) => {
  const { env } = await import("@xenova/transformers");
  const require = createRequire(import.meta.url);
  const runtimePath = require.resolve("onnxruntime-node", {
    paths: [dirname(fileURLToPath(import.meta.resolve("@xenova/transformers")))],
  });
  const { onnxruntimeBackend } = require(join(dirname(runtimePath), "backend.js"));
  const nativeFailure = new Error("Injected native model construction failure");
  let nativeAttempts = 0;
  context.mock.method(onnxruntimeBackend, "createSessionHandler", async () => {
    nativeAttempts++;
    throw nativeFailure;
  });
  const warn = console.warn;
  context.mock.method(console, "warn", (...messages: unknown[]) => {
    if (messages[0] !== nativeFailure && !String(messages[0]).startsWith("Something went wrong during model construction")) {
      warn(...messages);
    }
  });
  const originalThreads = env.backends.onnx.wasm.numThreads;
  context.after(() => { env.backends.onnx.wasm.numThreads = originalThreads; });
  env.backends.onnx.wasm.numThreads = 2;
  const { embedLocal } = await import("../src/engine/embeddings-local.js");
  const vectors = await embedLocal([
    "Verify PostgreSQL database backup checksums before restoring a database.",
    "Check the database backup checksum before PostgreSQL recovery.",
    "Fresh strawberries and whipped cream make a summer dessert.",
  ]);

  assert.equal(nativeAttempts, 1);
  assert.equal(vectors.length, 3);
  for (const vector of vectors) {
    assert.equal(vector.length, 1024);
    assert.ok(vector.every(Number.isFinite));
    assert.ok(Math.abs(Math.hypot(...vector) - 1) < 0.001);
  }
  const similarity = (left: number[], right: number[]) => left.reduce((sum, value, index) => sum + value * right[index], 0);
  assert.ok(similarity(vectors[0], vectors[1]) > similarity(vectors[0], vectors[2]));
});
