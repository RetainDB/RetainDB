import assert from "node:assert/strict";
import { test } from "node:test";
import { verifyApiKey } from "../src/middleware/auth.js";

test("api keys compare by value, not prefix", () => {
  assert.equal(verifyApiKey("correct-horse", "correct-horse"), true);
  assert.equal(verifyApiKey("correct-hors", "correct-horse"), false);
  assert.equal(verifyApiKey("correct-horse ", "correct-horse"), false);
  assert.equal(verifyApiKey("", "correct-horse"), false);
  assert.equal(verifyApiKey("", ""), false);
});
