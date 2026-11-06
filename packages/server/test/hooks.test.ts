import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { after, test } from "node:test";

const routesPath = "../src/api/routes.js";
const environmentKeys = ["DATABASE_URL", "ENCRYPTION_KEY", "OPENAI_API_KEY", "RETAINDB_API_KEY"] as const;
const savedEnvironment = Object.fromEntries(environmentKeys.map(key => [key, process.env[key]]));
process.env.DATABASE_URL = "postgresql://localhost:1/retaindb_unused";
process.env.ENCRYPTION_KEY = randomBytes(32).toString("hex");
delete process.env.OPENAI_API_KEY;
delete process.env.RETAINDB_API_KEY;

after(async () => {
  const { db } = await import("../src/db/index.js");
  await db.$disconnect();
  for (const key of environmentKeys) {
    if (savedEnvironment[key] === undefined) delete process.env[key];
    else process.env[key] = savedEnvironment[key];
  }
});

test("native hooks share API authentication and reject invalid capture before database access", async () => {
  const originalKey = process.env.RETAINDB_API_KEY;
  const originalOpenAI = process.env.OPENAI_API_KEY;
  process.env.RETAINDB_API_KEY = randomBytes(32).toString("hex");
  delete process.env.OPENAI_API_KEY;
  try {
    const { api } = await import(routesPath);
    for (const path of ["/retaindb/observe", "/v1/agent-events", "/v1/context/pack", "/v1/context/delta", "/v1/context/compress-output"]) {
      const unauthorized = await api.request(path, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
      });
      assert.equal(unauthorized.status, 401, path);
      const invalid = await api.request(path, {
        method: "POST", headers: { Authorization: `Bearer ${process.env.RETAINDB_API_KEY}`, "Content-Type": "application/json" },
        body: "{}",
      });
      assert.equal(invalid.status, 400, path);
    }
  } finally {
    if (originalKey === undefined) delete process.env.RETAINDB_API_KEY;
    else process.env.RETAINDB_API_KEY = originalKey;
    if (originalOpenAI === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = originalOpenAI;
  }
});

test("native output compression is local, bounded and redacts credentials", async () => {
  const { api } = await import(routesPath);
  const secret = `ghp_${"s".repeat(32)}`;
  const response = await api.request("/v1/context/compress-output", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output: `${"ordinary output\n".repeat(300)}ERROR: recovery failed\nAuthorization: Bearer ${secret}`, token_budget: 100 }),
  });
  assert.equal(response.status, 200, await response.clone().text());
  const result = await response.json();
  assert.ok(result.compressed.includes("recovery failed"));
  assert.ok(!result.compressed.includes(secret));
  assert.ok(result.estimated_tokens <= 100);
  assert.ok(result.compression_ratio < 1);
});
