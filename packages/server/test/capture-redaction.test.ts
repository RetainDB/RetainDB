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

test("capture removes connection credentials and complete private key blocks", async () => {
  const modulePath = "../src/engine/native-capture.js";
  const { redactCapture } = await import(modulePath);
  const samples = [
    "postgresql://fixture-user:fixture-private-database@localhost:5432/memory?sslmode=require",
    "redis://fixture-user:fixture-private-cache@localhost:6379/0",
    "https://fixture-user:fixture-private-http@example.test/evidence",
    "-----BEGIN PRIVATE KEY-----\nfixture-private-line-one\nfixture-private-line-two\n-----END PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY-----\nfixture-private-rsa\n-----END RSA PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----\nfixture-private-ssh\n-----END OPENSSH PRIVATE KEY-----",
  ];
  for (const sample of samples) {
    const result = JSON.stringify(redactCapture({ tool_response: sample }));
    assert.ok(!result.includes("fixture-private"), "Credential material must not reach captured tool evidence");
    assert.ok(result.includes("[REDACTED]"));
  }
  assert.equal(redactCapture("https://example.test/public/path postgres://localhost:5432/memory"),
    "https://example.test/public/path postgres://localhost:5432/memory");
});

test("capture removes recognized credential echoes across the whole event", async () => {
  const modulePath = "../src/engine/native-capture.js";
  const { redactCapture } = await import(modulePath);
  const samples = [
    { tool_response: "result fixture-echo-password remains private", tool_input: JSON.stringify({ env: { DATABASE_PASSWORD: "fixture-echo-password" } }) },
    { tool_response: "fixture quoted echo", tool_input: 'APP_TOKEN="fixture quoted echo"' },
    { tool_response: "fixture-private-db", connection: "postgres://fixture-user:fixture-private-db@localhost/memory" },
    { tool_response: "fixture-private-body", key: "-----BEGIN PRIVATE KEY-----\nfixture-private-body\n-----END PRIVATE KEY-----" },
    { nested: [{ output: "fixture-echo-array" }], env: { API_KEY: "fixture-echo-array" } },
    { tool_response: JSON.stringify({ output: "fixture quoted echo" }), env: { API_KEY: "fixture quoted echo" } },
  ];
  for (const sample of samples) {
    const before = JSON.stringify(sample);
    const result = JSON.stringify(redactCapture(sample));
    assert.ok(!result.includes("fixture-echo") && !result.includes("fixture-private") && !result.includes("fixture quoted"),
      "A credential found anywhere in the event must not survive an unlabeled echo");
    assert.equal(JSON.stringify(sample), before, "Redaction must not mutate the source event");
    assert.ok(result.includes("[REDACTED]"));
  }
  assert.deepEqual(redactCapture({ prompt: "Keep useful evidence", token_budget: 1200 }),
    { prompt: "Keep useful evidence", token_budget: 1200 });
});

test("connection usernames do not become secret aliases outside the URL", async () => {
  const modulePath = "../src/engine/native-capture.js";
  const { redactCapture } = await import(modulePath);
  const input = { project: "fixture-user", connection: "postgres://fixture-user:fixture-password@localhost/db",
    tool_response: "fixture-user checked the database using fixture-password" };
  const result = redactCapture(input) as typeof input;
  assert.equal(result.project, input.project);
  assert.ok(result.tool_response.includes("fixture-user"));
  assert.ok(!result.tool_response.includes("fixture-password"));
  assert.equal(result.connection, "postgres://[REDACTED]@localhost/db");
});

test("short credential echoes do not corrupt unrelated words or project identifiers", async () => {
  const modulePath = "../src/engine/native-capture.js";
  const { redactCapture } = await import(modulePath);
  const input = { project: "fixture-project", details: "password=f", tool_response: "f fixture-project useful evidence" };
  const result = redactCapture(input) as typeof input;
  assert.equal(result.project, input.project);
  assert.equal(result.tool_response, "[REDACTED] fixture-project useful evidence");
  assert.equal(result.details, "password=[REDACTED]");
});

test("capture discovers embedded credentials before discarding sensitive fields", async () => {
  const modulePath = "../src/engine/native-capture.js";
  const { redactCapture } = await import(modulePath);
  for (const authorization of [
    "Bearer fixture-embedded-token",
    { header: "Bearer fixture-embedded-token" },
    JSON.stringify({ header: "Bearer fixture-embedded-token" }),
  ]) {
    const input = { tool_response: "result fixture-embedded-token", authorization, prompt: "Keep useful evidence" };
    const before = JSON.stringify(input);
    assert.deepEqual(redactCapture(input), {
      tool_response: "result [REDACTED]", authorization: "[REDACTED]", prompt: input.prompt,
    });
    assert.equal(JSON.stringify(input), before);
  }
});

test("capture preserves primitive credential text when removing sibling echoes", async () => {
  const modulePath = "../src/engine/native-capture.js";
  const { redactCapture } = await import(modulePath);
  for (const credential of ["12345678", "12345678.00", "1.2345678e7", "true", "null"]) {
    const input = { tool_response: `result ${credential}`, tool_input: `PASSWORD=${credential}`, project: "fixture-project" };
    assert.deepEqual(redactCapture(input), {
      tool_response: "result [REDACTED]", tool_input: "PASSWORD=[REDACTED]", project: input.project,
    });
    assert.equal(input.tool_response, `result ${credential}`);
  }
});

test("capture removes string echoes of typed credential primitives", async () => {
  const modulePath = "../src/engine/native-capture.js";
  const { redactCapture } = await import(modulePath);
  for (const password of [12345678, true]) {
    const input = { tool_response: `result ${password}`, password, project: "fixture-project" };
    assert.deepEqual(redactCapture(input), {
      tool_response: "result [REDACTED]", password: "[REDACTED]", project: input.project,
    });
  }
});
