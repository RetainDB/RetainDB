import assert from "node:assert/strict";
import { test } from "node:test";

process.env.DATABASE_URL ??= "postgresql://localhost:1/retaindb_unused";
const { captureRetryPlan } = await import("../src/engine/native-capture.js");

test("capture retries back off and stop at the attempt limit", () => {
  assert.deepEqual(captureRetryPlan(1), { state: "pending", retryAfterSeconds: 2 });
  assert.deepEqual(captureRetryPlan(4), { state: "pending", retryAfterSeconds: 16 });
  assert.deepEqual(captureRetryPlan(8), { state: "failed", retryAfterSeconds: null });
  assert.deepEqual(captureRetryPlan(9), { state: "failed", retryAfterSeconds: null });
});

test("capture backoff never exceeds five minutes", () => {
  for (let attempts = 1; attempts < 8; attempts += 1) {
    const plan = captureRetryPlan(attempts);
    assert.ok(plan.retryAfterSeconds !== null && plan.retryAfterSeconds <= 300, String(attempts));
  }
});
