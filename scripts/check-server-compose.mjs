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

const keepAlive = setInterval(() => {}, 1000);
try {
  console.log("Building and starting isolated Compose runtime");
  compose(["up", "--detach", "--build", "--wait", "--wait-timeout", "180"], 900_000);
  await waitForReady();
  console.log("Compose API, migrations and local model ready");
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
  console.log("Vector ingestion and retrieval passed");

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
  console.log("Queued ingestion and canonical memory recall passed");

  compose(["restart", "server"]);
  await waitForReady();
  assert.equal(query(`SELECT COUNT(*) FROM memories WHERE id='${memory.memory_id}'`), "1");
  assert.ok(Number(query(`SELECT COUNT(*) FROM chunks WHERE "projectId"='${scope}' AND embedding IS NOT NULL`)) > 20);
  const afterRestart = await request("/v1/context/query", {
    project: name, query: "Verify database recovery backup checksums", hybrid: false, rerank: false,
    compress: false, use_cache: false, top_k: 3,
  });
  assert.ok(afterRestart.results.length > 0);
  console.log("Initial restart persistence passed");

  const observation = {
    project: name, sessionId: "compose-compatibility", hookType: "session_start", agent_id: "compose-agent",
    timestamp: new Date().toISOString(), data: { prompt: "Restore PostgreSQL only after verifying backup checksums.", token: "private-fixture-value" },
  };
  const observe = await request("/retaindb/observe", observation, 3000);
  assert.equal(observe.accepted, true);
  const replay = await request("/retaindb/observe", observation, 3000);
  assert.equal(replay.event_id, observe.event_id);
  console.log("Native event accepted durably; restarting during indexing");
  compose(["restart", "server"]);
  await waitForReady();
  let captured;
  for (let attempt = 0; attempt < 120; attempt++) {
    captured = await request(`/v1/agent-events/${observe.event_id}`);
    if (captured.status === "indexed") break;
    await delay(500);
  }
  assert.equal(captured.status, "indexed", JSON.stringify(captured));
  console.log("Native event recovered after restart");
  assert.equal(query(`SELECT COUNT(*) FROM messages WHERE id='${observe.event_id}'`), "1");
  assert.equal(query(`SELECT COUNT(*) FROM chunks WHERE "documentId"='${captured.document_id}' AND vector_dims(embedding)=1024`), "1");
  assert.equal(query(`SELECT COUNT(*) FROM messages WHERE id='${observe.event_id}' AND content LIKE '%private-fixture-value%'`), "0");
  const event = await request("/v1/agent-events", {
    project: name, session_id: "compose-cli-hook", agent_id: "compose-agent",
    event: { kind: "tool_result", summary: "Backup verification succeeded before PostgreSQL recovery.", timestamp: new Date().toISOString() },
  }, 3000);
  assert.equal(event.accepted, true);
  const plugin = spawnSync(process.execPath, ["--input-type=module", "--eval", `
    import { capture } from './packages/local/plugin/scripts/_capture.mjs';
    await capture('session_start', { session_id: 'packaged-native-plugin', prompt: 'Retain the PostgreSQL backup checksum before recovery.' });
  `], {
    encoding: "utf8", timeout: 10_000,
    env: { ...environment, RETAINDB_BASE_URL: baseUrl, RETAINDB_PROJECT: name, RETAINDB_SDK_CHILD: "0" },
  });
  assert.equal(plugin.status, 0, plugin.stderr);
  assert.equal(query("SELECT COUNT(*) FROM messages m JOIN sessions s ON s.id=m.\"sessionId\" WHERE s.\"sessionId\"='packaged-native-plugin'"), "1");
  for (let attempt = 0; attempt < 120; attempt++) {
    if (query("SELECT COUNT(*) FROM messages WHERE metadata->>'native_capture_state' != 'indexed'") === "0") break;
    await delay(500);
  }
  assert.equal(query("SELECT COUNT(*) FROM messages WHERE metadata->>'native_capture_state' != 'indexed'"), "0");
  const pack = await request("/v1/context/pack", { project: name, query: "PostgreSQL backup checksum recovery", token_budget: 400 });
  assert.ok(pack.context.includes("backup"));
  assert.ok(pack.estimated_tokens <= 400);
  const delta = await request("/v1/context/delta", { project: name, query: "PostgreSQL backup checksum recovery", token_budget: 400, previous_context_hash: pack.context_hash });
  assert.equal(delta.unchanged, true);
  assert.equal(delta.delta_context, "");
  const compressed = await request("/v1/context/compress-output", { output: `${"ordinary output\n".repeat(100)}ERROR: restore failed`, token_budget: 50 });
  assert.ok(compressed.compressed.includes("restore failed"));
  assert.ok(compressed.estimated_tokens <= 50);
  console.log(JSON.stringify({
    project: name, migrations, chunks, vectors, dimensions,
    vector_retrieval: "passed", queued_ingestion: "passed", memory_write_and_recall: "passed",
    restart_persistence: "passed", native_capture_and_replay: "passed", native_context_and_compression: "passed",
  }));
} catch (error) {
  try { process.stderr.write(compose(["logs", "--no-color", "--tail", "200"])); } catch {}
  throw error;
} finally {
  try {
    compose(["down", "--volumes", "--remove-orphans", "--rmi", "local"], 120_000);
  } finally {
    clearInterval(keepAlive);
  }
}
