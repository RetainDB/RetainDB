import assert from "node:assert/strict";
import { test } from "node:test";

test("semantic cache never reuses another project or scope result", async () => {
  const cachePath = "../src/engine/cache.js";
  const { getFromSemanticCache, setInSemanticCache } = await import(cachePath);
  const embedding = [0.71, 0.29, 0.31, 0.73, 0.14, 0.27, 0.19, 0.21];
  const results = [{ memory: { id: "project-a-memory", content: "Project A knowledge" }, similarity: 1 }];
  await setInSemanticCache(embedding, results, 60, "project-a:scope-project");
  assert.deepEqual((await getFromSemanticCache(embedding, "project-a:scope-project"))?.results, results);
  assert.equal(await getFromSemanticCache(embedding, "project-b:scope-project"), null);
  assert.equal(await getFromSemanticCache(embedding, "project-a:scope-user"), null);
});
