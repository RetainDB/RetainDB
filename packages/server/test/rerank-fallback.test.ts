import assert from "node:assert/strict";
import { test } from "node:test";

process.env.DATABASE_URL ??= "postgresql://localhost:1/retaindb_unused";
const { applyRerank } = await import("../src/engine/retriever.js");

test("rerank replaces ordering when the reranker answers", async () => {
  const outcome = await applyRerank([{ id: "a" }, { id: "b" }], async (results) => [...results].reverse());
  assert.deepEqual(outcome, { results: [{ id: "b" }, { id: "a" }], degraded: false });
});

test("a failing reranker keeps the fused order and reports degradation", async () => {
  const input = [{ id: "a" }, { id: "b" }];
  const outcome = await applyRerank(input, async () => {
    throw new Error("model unavailable");
  });
  assert.equal(outcome.degraded, true);
  assert.equal(outcome.results, input);
});
