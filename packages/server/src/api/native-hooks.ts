import { createHash } from "node:crypto";
import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import type { AuthContext } from "../middleware/auth.js";
import { rateLimitMiddleware, RateLimits } from "../middleware/rate-limit.js";
import { acceptNativeCapture, redactCapture } from "../engine/native-capture.js";
import { retrieve } from "../engine/retriever.js";
import { searchMemories } from "../engine/memory/search.js";
import { ensureProject } from "./helpers.js";
import { prisma } from "../db/index.js";

export const nativeHookRoutes = new Hono<{ Variables: { auth: AuthContext } }>();
const identity = {
  project: z.string().trim().min(1).max(4096).optional(),
  session_id: z.string().trim().min(1).max(512),
  agent_id: z.string().trim().min(1).max(512).optional().default("agent"),
};
const observationSchema = z.object({
  project: identity.project,
  session_id: identity.session_id.optional(), sessionId: identity.session_id.optional(),
  agent_id: z.string().trim().min(1).max(512).optional(),
  agentId: z.string().trim().min(1).max(512).optional(),
  hook_type: z.string().trim().min(1).max(128).optional(),
  hookType: z.string().trim().min(1).max(128).optional(),
  cwd: z.string().max(4096).optional(),
  data: z.record(z.unknown()),
  timestamp: z.string().datetime({ offset: true }).optional(),
}).refine(body => Boolean(body.session_id || body.sessionId) && Boolean(body.hook_type || body.hookType));

nativeHookRoutes.post("/retaindb/observe", rateLimitMiddleware(RateLimits.ingest), zValidator("json", observationSchema), async c => {
  const auth = c.get("auth");
  const body = c.req.valid("json");
  const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);
  const event = await acceptNativeCapture({
    orgId: auth.orgId, projectId: project.id, sessionId: (body.session_id || body.sessionId)!,
    agentId: body.agent_id || body.agentId || String(body.data.agent_id || body.data.agentId || "agent"),
    hookType: (body.hook_type || body.hookType)!,
    data: { ...body.data, ...(body.cwd ? { cwd: body.cwd } : {}) }, timestamp: body.timestamp,
  });
  return c.json({ success: true, accepted: true, event_id: event.id, status: event.status, memories_created: 0 }, 202);
});

nativeHookRoutes.post("/v1/agent-events", rateLimitMiddleware(RateLimits.ingest), zValidator("json", z.object({
  ...identity,
  event: z.object({
    kind: z.string().trim().min(1).max(128), summary: z.string().max(64000).optional(),
    details: z.string().max(256000).optional(), timestamp: z.string().datetime({ offset: true }).optional(),
  }).passthrough(),
})), async c => {
  const auth = c.get("auth");
  const body = c.req.valid("json");
  const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);
  const event = await acceptNativeCapture({
    orgId: auth.orgId, projectId: project.id, sessionId: body.session_id, agentId: body.agent_id,
    hookType: body.event.kind, data: body.event, timestamp: body.event.timestamp,
  });
  return c.json({ success: true, accepted: true, event_id: event.id, status: event.status, memories_created: 0 }, 202);
});

nativeHookRoutes.get("/v1/agent-events/:eventId", rateLimitMiddleware(RateLimits.query), async c => {
  const auth = c.get("auth");
  const message = await prisma.message.findFirst({
    where: { id: c.req.param("eventId"), session: { orgId: auth.orgId }, metadata: { path: ["source"], equals: "native_agent_hook" } },
    select: { id: true, metadata: true },
  });
  if (!message) return c.json({ error: "Agent event not found" }, 404);
  const metadata = message.metadata as Record<string, unknown>;
  return c.json({
    event_id: message.id, status: metadata.native_capture_state,
    attempts: metadata.capture_attempts, document_id: metadata.document_id || null,
    retry_at: metadata.retry_at || null, last_error: metadata.last_error || null,
  });
});

function estimatedTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function truncate(text: string, budget: number): string {
  return text.slice(0, Math.max(0, budget) * 4);
}

function compressOutput(output: string, budget: number): string {
  const lines = String(redactCapture(output)).split(/\r?\n/);
  const important = lines.filter(line => /\b(error|failed|failure|exception|traceback|assert|timeout|denied|not found|expected|received)\b/i.test(line));
  return truncate([
    `Original output: ${lines.length} lines, approx ${estimatedTokens(output)} tokens.`,
    ...(important.length ? important : lines.slice(-40)),
  ].join("\n"), budget);
}

nativeHookRoutes.post("/v1/context/compress-output", rateLimitMiddleware(RateLimits.query), zValidator("json", z.object({
  output: z.string().max(1024 * 1024).optional(), tool_output: z.string().max(1024 * 1024).optional(),
  token_budget: z.number().int().min(1).max(8000).optional().default(500),
}).refine(body => body.output !== undefined || body.tool_output !== undefined)), c => {
  const body = c.req.valid("json");
  const output = body.output ?? body.tool_output ?? "";
  const compressed = compressOutput(output, body.token_budget);
  return c.json({
    compressed, estimated_tokens: estimatedTokens(compressed), original_estimated_tokens: estimatedTokens(output),
    compression_ratio: estimatedTokens(compressed) / Math.max(1, estimatedTokens(output)),
  });
});

const packSchema = z.object({
  project: identity.project,
  query: z.preprocess(value => typeof value === "object" && value !== null ? JSON.stringify(redactCapture(value)) : value, z.string().trim().min(1).max(64000)),
  token_budget: z.number().int().min(1).max(8000).optional().default(1200),
  previous_context_hash: z.string().max(128).optional(), context_hash: z.string().max(128).optional(),
  include_memory: z.boolean().optional().default(true), include_company_brain: z.boolean().optional().default(true),
  files: z.array(z.string().max(4096)).max(100).optional().default([]),
  cwd: z.string().max(4096).optional(), tool_output: z.string().max(1024 * 1024).optional(),
});

for (const path of ["/v1/context/pack", "/v1/context/delta"]) {
  nativeHookRoutes.post(path, rateLimitMiddleware(RateLimits.query), zValidator("json", packSchema), async c => {
    const auth = c.get("auth");
    const body = c.req.valid("json");
    const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);
    const [documents, memories] = await Promise.all([
      body.include_company_brain ? retrieve({
        projectId: project.id, query: body.query, hybridSearch: false, rerank: false, compress: false,
        includeMemories: false, includeGraph: false, useCache: false, topK: 6, maxTokens: body.token_budget,
      }) : null,
      body.include_memory ? searchMemories({
        projectId: project.id, orgId: auth.orgId, query: body.query, questionDate: new Date(), topK: 6, fastMode: true,
      }) : [],
    ]);
    const sections: string[] = [];
    if (body.tool_output) sections.push(`Compressed tool output:\n${compressOutput(body.tool_output, Math.max(1, Math.floor(body.token_budget / 4)))}`);
    if (memories.length) sections.push(`Relevant memory:\n${memories.map(item => `- ${item.memory.content}`).join("\n")}`);
    if (documents?.context) sections.push(`Indexed context:\n${documents.context}`);
    const context = truncate(String(redactCapture(sections.join("\n\n"))), body.token_budget);
    const contextHash = createHash("sha256").update(JSON.stringify([auth.orgId, project.id, context])).digest("hex");
    const previousHash = body.previous_context_hash || body.context_hash || null;
    const delta = previousHash === contextHash ? "" : context;
    return c.json({
      context, delta_context: delta, context_hash: contextHash, previous_context_hash: previousHash,
      token_budget: body.token_budget, estimated_tokens: estimatedTokens(context), estimated_delta_tokens: estimatedTokens(delta),
      unchanged: previousHash === contextHash, files_packed: [],
      diagnostics: body.files.length || body.cwd ? ["Server context uses indexed sources; client filesystem paths are not read."] : [],
      results: documents?.results || [], memories: memories.map(item => item.memory),
    });
  });
}
