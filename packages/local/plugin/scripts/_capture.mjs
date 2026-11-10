#!/usr/bin/env node

const BASE_URL = (process.env.RETAINDB_BASE_URL || "http://localhost:3111").replace(/\/+$/, "");
const API_KEY = process.env.RETAINDB_API_KEY || "";
const AUTO_CONTEXT = process.env.RETAINDB_AUTO_CONTEXT !== "false";
const COMPRESS_TOOL_OUTPUT = process.env.RETAINDB_COMPRESS_TOOL_OUTPUT !== "false";
const TOKEN_BUDGET = Number(process.env.RETAINDB_TOKEN_BUDGET || 1200);

function headers() {
  const value = { "Content-Type": "application/json" };
  if (API_KEY) value.Authorization = `Bearer ${API_KEY}`;
  return value;
}

function truncate(value, max = 8000) {
  if (typeof value === "string") return value.length > max ? `${value.slice(0, max)}\n[...truncated]` : value;
  try {
    const text = JSON.stringify(value);
    return text.length > max ? `${text.slice(0, max)}...[truncated]` : value;
  } catch {
    return value;
  }
}

export async function readHookInput() {
  let input = "";
  for await (const chunk of process.stdin) input += chunk;
  let data = {};
  try {
    data = input.trim() ? JSON.parse(input) : {};
  } catch {
    data = { raw: input.slice(0, 8000) };
  }
  return { input, data };
}

export function hookMeta(data = {}) {
  return {
    sessionId: data.session_id || data.sessionId || process.env.RETAINDB_SESSION_ID || "unknown",
    project: process.env.RETAINDB_PROJECT || data.cwd || data.project || "default",
    agentId: process.env.RETAINDB_AGENT_ID || data.agent_id || data.agentId || "agent",
    cwd: data.cwd || process.cwd(),
  };
}

export async function post(path, body, timeout = 2500) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeout),
  });
  return res.ok ? res.json() : null;
}

export function shouldInjectContext() {
  return AUTO_CONTEXT && process.env.RETAINDB_INJECT_CONTEXT !== "false";
}

export async function buildContextPack(data, hookType, query) {
  if (!shouldInjectContext()) return null;
  const meta = hookMeta(data);
  const previous = process.env[`RETAINDB_CONTEXT_HASH_${meta.sessionId}`] || data.previous_context_hash || data.context_hash;
  return post("/v1/context/pack", {
    project: meta.project,
    query: query || data.prompt || data.message || data.tool_input || data.tool_name || hookType,
    cwd: meta.cwd,
    files: data.files || data.filePaths || data.paths || [],
    tool_output: COMPRESS_TOOL_OUTPUT ? data.tool_output || data.tool_response || "" : "",
    previous_context_hash: previous,
    token_budget: TOKEN_BUDGET,
  });
}

export async function printContextPack(data, hookType, query) {
  try {
    const pack = await buildContextPack(data, hookType, query);
    if (pack?.context_hash) process.env[`RETAINDB_CONTEXT_HASH_${hookMeta(data).sessionId}`] = pack.context_hash;
    const text = pack?.delta_context ?? pack?.context;
    if (text && String(text).trim()) process.stdout.write(`\nRetainDB compact context:\n${text}\n`);
  } catch {
    // Hooks must never block the agent.
  }
}

async function compressedData(data) {
  if (!COMPRESS_TOOL_OUTPUT) return data;
  const raw = data.tool_output ?? data.tool_response;
  if (typeof raw !== "string" || raw.length < 1200) return data;
  try {
    const body = await post("/v1/context/compress-output", { output: raw, token_budget: Math.min(TOKEN_BUDGET, 600) }, 2500);
    if (body?.compressed) return { ...data, tool_output: body.compressed, tool_response: body.compressed, raw_output_hash: await hash(raw), output_compressed: true };
  } catch {}
  return data;
}

async function hash(value) {
  const bytes = new TextEncoder().encode(String(value));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function capture(hookType, providedData) {
  const { data } = providedData ? { data: providedData } : await readHookInput();
  if (process.env.RETAINDB_SDK_CHILD === "1" || data.entrypoint === "sdk-ts") return;

  const meta = hookMeta(data);
  const safeData = await compressedData(data);

  const payload = {
    hookType,
    sessionId: meta.sessionId,
    project: meta.project,
    cwd: meta.cwd,
    timestamp: new Date().toISOString(),
    data: {
      ...safeData,
      tool_output: truncate(safeData.tool_output ?? safeData.tool_response),
      prompt: truncate(safeData.prompt),
    },
  };

  try {
    await post("/retaindb/observe", payload, 3000);
  } catch {
    // Hooks must never block the agent.
  }
}
