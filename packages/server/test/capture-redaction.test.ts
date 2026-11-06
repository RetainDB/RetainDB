import assert from "node:assert/strict";
import { test } from "node:test";

process.env.DATABASE_URL = "postgresql://localhost:1/retaindb_unused";

for (const sample of [
  { env: { RETAINDB_API_KEY: "fixture-private-one", OPENAI_API_KEY: "fixture-private-two", DATABASE_PASSWORD: "fixture-private-three", AWS_SECRET_ACCESS_KEY: "fixture-private-four" } },
  { details: 'refresh_token="fixture private refresh" password="fixture private phrase" RETAINDB_API_KEY=fixture-private-key' },
  { details: JSON.stringify({ env: { RETAINDB_API_KEY: "fixture-private-json", OPENAI_API_KEY: "fixture private json phrase" } }) },
]) {
  test("capture redacts prefixed credentials and complete quoted values", async () => {
    const modulePath = "../src/engine/native-capture.js";
    const { redactCapture } = await import(modulePath);
    const result = JSON.stringify(redactCapture(sample));
    assert.ok(!result.includes("fixture"), "Credential values must be absent from capture");
    assert.ok(!result.includes("private phrase") && !result.includes("private refresh"), "Quoted credentials must be completely redacted");
    assert.ok(result.includes("[REDACTED]"));
    assert.equal(redactCapture("normal words and token_budget=1200"), "normal words and token_budget=1200");
  });
}
