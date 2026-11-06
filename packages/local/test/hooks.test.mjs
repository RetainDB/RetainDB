import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { createServer } from "node:http";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const packageRoot = fileURLToPath(new URL("../", import.meta.url));
const cliPath = fileURLToPath(new URL("../src/cli.ts", import.meta.url));
const compactContext = "Recorder project context";
const compressedOutput = "Recorder compact tool output";

async function startRecorder(t, requests, apiKey) {
  const server = createServer(async (request, response) => {
    let input = "";
    for await (const chunk of request) input += chunk;
    const authorization = request.headers.authorization;
    const authenticated = !apiKey || authorization === `Bearer ${apiKey}`;
    const status = authenticated ? 200 : 401;
    requests.push({
      path: request.url,
      method: request.method,
      contentType: request.headers["content-type"],
      authorization: !authorization ? "absent" : authenticated ? "accepted" : "rejected",
      status,
      body: JSON.parse(input),
    });
    let body;
    if (!authenticated) body = { error: "Unauthorized" };
    else if (request.url === "/v1/context/pack") body = { context: compactContext };
    else if (request.url === "/v1/context/compress-output") body = { compressed: compressedOutput };
    else if (request.url === "/v1/agent-events") body = { captured: true };
    else body = { error: "Unexpected request" };
    response.writeHead(status, { "Content-Type": "application/json" });
    response.end(JSON.stringify(body));
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  t.after(async () => {
    server.closeAllConnections();
    await new Promise((resolve, reject) => {
      server.close((error) => error ? reject(error) : resolve());
    });
  });
  return `http://127.0.0.1:${server.address().port}`;
}

async function invokeHook(baseUrl, kind, data, apiKey) {
  const env = Object.fromEntries(Object.entries(process.env).filter(([name]) => !name.startsWith("RETAINDB_")));
  Object.assign(env, {
    RETAINDB_BASE_URL: baseUrl,
    RETAINDB_PROJECT: "native-hook-regression",
    RETAINDB_SESSION_ID: "hook-session",
    RETAINDB_AUTO_CONTEXT: "true",
    RETAINDB_COMPRESS_TOOL_OUTPUT: "true",
    RETAINDB_TOKEN_BUDGET: "1200",
    TSX_DISABLE_CACHE: "1",
    NODE_NO_WARNINGS: "1",
  });
  if (apiKey !== undefined) env.RETAINDB_API_KEY = apiKey;
  return new Promise((resolve, reject) => {
    const child = execFile(process.execPath, ["--import", "tsx/esm", cliPath, "hook", `--kind=${kind}`, "--agent=native-test"], {
      cwd: packageRoot,
      env,
      timeout: 10000,
    }, (error, stdout, stderr) => {
      if (error) reject(error);
      else resolve({ stdout, stderr });
    });
    child.stdin.end(JSON.stringify(data));
  });
}

function assertRequests(requests, paths, protectedAccess) {
  assert.deepEqual(requests.map(({ path, status, authorization }) => ({ path, status, authorization })),
    paths.map((path) => ({ path, status: 200, authorization: protectedAccess ? "accepted" : "absent" })));
  for (const request of requests) {
    assert.equal(request.method, "POST");
    assert.equal(request.contentType, "application/json");
  }
}

for (const mode of ["protected", "open-unset", "open-empty"]) {
  for (const kind of ["prompt", "pre_tool_use", "pre_compact"]) {
    test(`${kind} authenticates context and event requests (${mode})`, { timeout: 20000 }, async (t) => {
      const requests = [];
      const key = mode === "protected" ? randomUUID() : undefined;
      const baseUrl = await startRecorder(t, requests, key);
      const input = { prompt: "Recall project context", cwd: packageRoot, files: ["src/cli.ts"], previous_context_hash: "prior-context" };
      const result = await invokeHook(baseUrl, kind, input, mode === "open-empty" ? "" : key);

      assertRequests(requests, ["/v1/context/pack", "/v1/agent-events"], mode === "protected");
      assert.deepEqual(requests[0].body, {
        project: "native-hook-regression",
        query: input.prompt,
        cwd: packageRoot,
        files: input.files,
        previous_context_hash: input.previous_context_hash,
        token_budget: 1200,
      });
      assert.equal(requests[1].body.event.kind, kind);
      assert.equal(requests[1].body.session_id, "hook-session");
      assert.equal(requests[1].body.agent_id, "native-test");
      assert.deepEqual(JSON.parse(requests[1].body.event.details), input);
      assert.equal(result.stdout, `\nRetainDB compact context:\n${compactContext}\n`);
      assert.equal(result.stderr, "");
    });
  }

  test(`tool_result authenticates compression and event requests (${mode})`, { timeout: 20000 }, async (t) => {
    const requests = [];
    const key = mode === "protected" ? randomUUID() : undefined;
    const baseUrl = await startRecorder(t, requests, key);
    const input = { tool_name: "Read", tool_output: "Tool output diagnostic\n".repeat(100) };
    const result = await invokeHook(baseUrl, "tool_result", input, mode === "open-empty" ? "" : key);

    assertRequests(requests, ["/v1/context/compress-output", "/v1/agent-events"], mode === "protected");
    assert.deepEqual(requests[0].body, {
      output: input.tool_output,
      token_budget: 1200,
    });
    assert.equal(requests[1].body.event.kind, "tool_result");
    assert.equal(requests[1].body.event.summary, JSON.stringify(input));
    const details = JSON.parse(requests[1].body.event.details);
    assert.equal(details.tool_output, compressedOutput);
    assert.equal(details.tool_response, compressedOutput);
    assert.equal(details.output_compressed, true);
    assert.match(details.raw_output_hash, /^[a-f0-9]{64}$/);
    assert.equal(result.stdout, "");
    assert.equal(result.stderr, "");
  });

  test(`session_end authenticates its event without extra requests (${mode})`, { timeout: 20000 }, async (t) => {
    const requests = [];
    const key = mode === "protected" ? randomUUID() : undefined;
    const baseUrl = await startRecorder(t, requests, key);
    const input = { summary: "Finished native hook regression" };
    const result = await invokeHook(baseUrl, "session_end", input, mode === "open-empty" ? "" : key);

    assertRequests(requests, ["/v1/agent-events"], mode === "protected");
    assert.equal(requests[0].body.event.kind, "session_end");
    assert.equal(requests[0].body.event.summary, JSON.stringify(input));
    assert.equal(requests[0].body.project, "native-hook-regression");
    assert.equal(result.stdout, "");
    assert.equal(result.stderr, "");
  });
}

for (const credentials of ["missing", "incorrect"]) {
  test(`protected recorder rejects ${credentials} credentials`, { timeout: 20000 }, async (t) => {
    const requests = [];
    const baseUrl = await startRecorder(t, requests, randomUUID());
    const result = await invokeHook(baseUrl, "session_end", { summary: "Rejected event" }, credentials === "missing" ? undefined : randomUUID());

    assert.deepEqual(requests.map(({ path, status, authorization }) => ({ path, status, authorization })), [{
      path: "/v1/agent-events",
      status: 401,
      authorization: credentials === "missing" ? "absent" : "rejected",
    }]);
    assert.equal(result.stderr, "");
    assert.equal(result.stdout, "");
  });
}
