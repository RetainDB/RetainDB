import assert from "node:assert/strict";
import { test } from "node:test";
import { chunkText } from "../src/engine/chunker.js";

test("fixed chunks keep whole characters at the boundary", () => {
  const content = `${"a".repeat(18)}🌱🌱`;
  const { chunks } = chunkText(content, { chunkSize: 20, chunkOverlap: 0 });
  assert.ok(chunks.length > 0);
  for (const chunk of chunks) {
    assert.ok(!/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/.test(chunk.content), "chunk ends on a lone high surrogate");
    assert.ok(!/(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/.test(chunk.content), "chunk starts on a lone low surrogate");
  }
  assert.equal(chunks.map((chunk) => chunk.content).join(""), content);
});

test("fixed chunks make progress when one character exceeds the chunk size", () => {
  const content = "🌱🌱🌱";
  const { chunks } = chunkText(content, { chunkSize: 1, chunkOverlap: 0 });
  assert.ok(chunks.length >= 1);
  assert.equal(chunks.map((chunk) => chunk.content).join(""), content);
});
