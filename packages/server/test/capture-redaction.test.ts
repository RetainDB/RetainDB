import assert from "node:assert/strict";
import { test } from "node:test";

process.env.DATABASE_URL ??= "postgresql://localhost:1/retaindb_unused";
const { redactCapture } = await import("../src/engine/native-capture.js");

test("capture removes credential fields and their echoes", () => {
  const value = {
    command: "export API_TOKEN=fixture-token-value",
    nested: { password: "fixture-token-value" },
    echoed: "token fixture-token-value present",
    keep: "unrelated text",
  };
  const redacted = JSON.stringify(redactCapture(value));
  assert.ok(!redacted.includes("fixture-token-value"), redacted);
  assert.ok(redacted.includes("unrelated text"));
});

test("capture removes typed echoes of credential values", () => {
  const value = { password: 867530912, typed: 867530912, keep: 42 };
  const redacted = redactCapture(value) as Record<string, unknown>;
  assert.equal(redacted.typed, "[REDACTED]");
  assert.equal(redacted.keep, 42);
});

test("capture removes private key blocks and connection credentials", () => {
  const value = {
    key: "-----BEGIN PRIVATE KEY-----\nfixture-material\n-----END PRIVATE KEY-----",
    url: "postgres://fixture-user:fixture-password@localhost:5432/memory",
  };
  const redacted = JSON.stringify(redactCapture(value));
  assert.ok(!redacted.includes("fixture-material"), redacted);
  assert.ok(!redacted.includes("fixture-password"), redacted);
});

test("capture does not redact prose that merely contains a short secret", () => {
  const value = { password: "1", note: "Run 1 migration before 1 restart", count: 1 };
  const redacted = redactCapture(value) as Record<string, unknown>;
  assert.equal(redacted.password, "[REDACTED]");
  assert.equal(redacted.note, "Run 1 migration before 1 restart");
  assert.equal(redacted.count, "[REDACTED]");
});
