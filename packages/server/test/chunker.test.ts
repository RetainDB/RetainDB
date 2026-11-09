import assert from "node:assert/strict";
import { test } from "node:test";
import { chunkText } from "../src/engine/chunker.js";

const loneSurrogate = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u;

function fixedChunks(content: string, chunkOverlap: number, chunkSize = 1000) {
  return chunkText(content, {
    ingestionProfile: "plain_text",
    strategyOverride: "fixed",
    chunkOverlap,
    chunkSize,
  }).chunks;
}

test("fixed chunks preserve complete Unicode characters at the end boundary", () => {
  const content = "a".repeat(699) + "🧬" + "b".repeat(900);
  const chunks = fixedChunks(content, 0);
  for (const chunk of chunks) assert.doesNotMatch(chunk.content, loneSurrogate);
  assert.equal(chunks.filter(chunk => chunk.searchable).map(chunk => chunk.content).join(""), content);
});

test("fixed chunks preserve complete Unicode characters at the overlap boundary", () => {
  const content = "a".repeat(499) + "🧬" + "b".repeat(900);
  const chunks = fixedChunks(content, 200);
  const searchable = chunks.filter(chunk => chunk.searchable);
  assert.ok(searchable.length > 1);
  for (const chunk of chunks) {
    assert.doesNotMatch(chunk.content, loneSurrogate);
    assert.ok(content.includes(chunk.content));
  }
  assert.ok(searchable.some(chunk => chunk.content.includes("🧬")));
});

test("fixed chunks make progress when a Unicode character exceeds the chunk size", () => {
  const content = "A🧬B𐐀C";
  const chunks = fixedChunks(content, 0, 1);
  for (const chunk of chunks) assert.doesNotMatch(chunk.content, loneSurrogate);
  assert.deepEqual(chunks.filter(chunk => chunk.searchable).map(chunk => chunk.content), ["A", "🧬", "B", "𐐀", "C"]);
});

test("fixed chunks retain existing ASCII sizes and overlaps", () => {
  const chunks = fixedChunks("abcdefghijklmnopqrst", 2, 8);
  assert.deepEqual(chunks.filter(chunk => chunk.searchable).map(chunk => chunk.content), [
    "abcdefgh",
    "ghijklmn",
    "mnopqrst",
  ]);
});
