import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { randomBytes, randomUUID } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";

const project = `retaindb-compose-${randomUUID()}`;
const composeArgs = ["compose", "--project-name", project, "--file", "docker-compose.yml", "--file", "docker-compose.ci.yml"];
const baseUrl = "http://127.0.0.1:3000";
const apiKey = randomBytes(32).toString("hex");
const environment = {
  ...process.env,
  RETAINDB_API_KEY: apiKey,
  ENCRYPTION_KEY: randomBytes(32).toString("hex"),
  OPENAI_API_KEY: "",
  EMBEDDING_MODE: "local",
  PORT: "3000",
};

function compose(args, timeout = 120_000) {
  const result = spawnSync("docker", [...composeArgs, ...args], {
    encoding: "utf8", env: environment, timeout, killSignal: "SIGKILL", maxBuffer: 8 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
  return result.stdout;
}

async function request(path, body, timeout = 120_000) {
  const response = await fetch(`${baseUrl}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeout),
  });
  const text = await response.text();
  assert.ok(response.ok, `${path}: HTTP ${response.status} ${text}`);
  return JSON.parse(text);
}

function query(sql) {
  return compose(["exec", "--no-TTY", "postgres", "psql", "--username", "retaindb", "--dbname", "retaindb", "--tuples-only", "--no-align", "--set", "ON_ERROR_STOP=1", "--command", sql]).trim();
}

async function waitForReady() {
  for (let attempt = 0; attempt < 120; attempt++) {
    const response = await fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(2000) }).catch(() => null);
    if (response?.ok) {
      const logs = compose(["logs", "--no-color", "server"]);
      if (logs.includes("Local embedding model ready")) return;
      assert.ok(!logs.includes("Local model pre-load failed"), "Local embedding model failed to initialize");
    }
    await delay(2000);
  }
  throw new Error("Compose API and local embedding model did not become ready");
}

try {
  compose(["up", "--detach", "--build", "--wait", "--wait-timeout", "180"], 900_000);
  await waitForReady();
  assert.equal((await fetch(`${baseUrl}/v1/projects`)).status, 401);
  const migrations = Number(query('SELECT COUNT(*) FROM "_prisma_migrations" WHERE finished_at IS NOT NULL'));
  assert.ok(migrations >= 13);
  assert.equal(query("SELECT COUNT(*) FROM users"), "0");
  assert.equal(query("SELECT COUNT(*) FROM organizations WHERE id='default'"), "1");

  const name = `compose-${randomUUID()}`;
  const content = Array.from({ length: 24 }, (_, index) =>
    `# PostgreSQL recovery partition ${index}\nBefore restoring database partition ${index}, verify the backup checksum and stop incoming writes. Preserve the verified backup until row counts and semantic retrieval pass.`
  ).join("\n\n");
  const learned = await request("/v1/learn", {
    mode: "text", project: name, title: "Verified database recovery", content,
    metadata: { provenance: project },
    options: { async: false, ingestion_profile: "plain_text", strategy_override: "fixed" },
  }, 180_000);
  assert.equal(learned.success, true);
  assert.equal(learned.status, "completed");
  assert.ok(learned.chunks_indexed > 20);
  assert.match(learned.project, /^[a-f0-9-]+$/i);
  const scope = learned.project;
  const chunkCounts = query(`SELECT COUNT(*) || ':' || COUNT(embedding) || ':' || MIN(vector_dims(embedding)) FROM chunks WHERE "projectId"='${scope}'`);
  const [chunks, vectors, dimensions] = chunkCounts.split(":").map(Number);
  assert.ok(chunks > 20);
  assert.equal(vectors, chunks);
  assert.equal(dimensions, 1024);

  const retrieval = await request("/v1/context/query", {
    project: name, query: "How should database recovery verify a backup before restoring partitions?",
    top_k: 3, hybrid: false, rerank: false, compress: false, use_cache: false,
    include_memories: false,
  });
  assert.ok(retrieval.results.length > 0);
  assert.equal(retrieval.results[0].retrieval_source, "vector");
  assert.ok(retrieval.context.includes("backup checksum"));

  const queued = await request("/v1/learn", {
    mode: "text", project: name, title: "Queued database recovery",
    content: "PostgreSQL recovery requires a verified backup before replaying committed writes.",
    options: { ingestion_profile: "plain_text", strategy_override: "fixed" },
  });
  assert.equal(queued.status, "processing");
  assert.match(queued.job_id, /^[a-f0-9-]+$/i);
  let job;
  for (let attempt = 0; attempt < 120; attempt++) {
    job = (await request(`/v1/jobs/${queued.job_id}`)).job;
    if (!["PENDING", "PROCESSING"].includes(job.status)) break;
    await delay(500);
  }
  assert.equal(job.status, "COMPLETED");
  assert.ok(job.processedChunks > 0);

  const memory = await request("/v1/memory", {
    project: name, content: "PostgreSQL recovery requires verified backup checksums before restoring partitions.",
    memory_type: "factual", auto_detect: false, write_mode: "sync", scope_target: "PROJECT",
    confidence: 0.95, importance: 0.9, metadata: { provenance: project },
  });
  assert.equal(memory.success, true);
  assert.match(memory.memory_id, /^[a-f0-9-]+$/i);
  const storedMemory = query(`SELECT content || ':' || vector_dims(embedding) || ':' || (metadata->>'semantic_status') FROM memories WHERE id='${memory.memory_id}'`);
  assert.ok(storedMemory.endsWith(":1024:ready"));
  const memories = await request("/v1/memory/search", {
    project: name, query: "What must PostgreSQL recovery verify before restoring partitions?",
    include_chunks: false, include_relations: false, include_pending: false, profile: "fast",
  });
  assert.ok(memories.results.some(result => result.memory.id === memory.memory_id));

  compose(["restart", "server"]);
  await waitForReady();
  assert.equal(query(`SELECT COUNT(*) FROM memories WHERE id='${memory.memory_id}'`), "1");
  assert.ok(Number(query(`SELECT COUNT(*) FROM chunks WHERE "projectId"='${scope}' AND embedding IS NOT NULL`)) > 20);
  const afterRestart = await request("/v1/context/query", {
    project: name, query: "Verify database recovery backup checksums", hybrid: false, rerank: false,
    compress: false, use_cache: false, top_k: 3,
  });
  assert.ok(afterRestart.results.length > 0);

  const observe = await fetch(`${baseUrl}/retaindb/observe`, {
    method: "POST", headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({ project: name, sessionId: "compose-compatibility", hookType: "session_start", data: {} }),
    signal: AbortSignal.timeout(5000),
  });
  console.log(JSON.stringify({
    project: name, migrations, chunks, vectors, dimensions,
    vector_retrieval: "passed", queued_ingestion: "passed", memory_write_and_recall: "passed",
    restart_persistence: "passed", native_observe_status: observe.status,
  }));
} catch (error) {
  try { process.stderr.write(compose(["logs", "--no-color", "--tail", "200"])); } catch {}
  throw error;
} finally {
  compose(["down", "--volumes", "--remove-orphans", "--rmi", "local"], 120_000);
}
