import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import { Prisma } from "@prisma/client";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve as resolvePath } from "node:path";
import { prisma } from "../db/index.js";
import { authMiddleware, type AuthContext } from "../middleware/auth.js";
import { rateLimitMiddleware, RateLimits } from "../middleware/rate-limit.js";
import { retrieve } from "../engine/retriever.js";
import { syncGitHub } from "../connectors/github.js";
import { syncGitLab } from "../connectors/gitlab.js";
import { syncUrl } from "../connectors/url.js";
import { syncSitemap } from "../connectors/sitemap.js";
import { syncText } from "../connectors/text.js";
import { syncPdf } from "../connectors/pdf.js";
import { syncApiSpec } from "../connectors/api_spec.js";
import { syncNotion } from "../connectors/notion.js";
import { syncConfluence } from "../connectors/confluence.js";
import { syncSlack } from "../connectors/slack.js";
import { syncDiscord } from "../connectors/discord.js";
import { syncArxiv } from "../connectors/arxiv.js";
import { syncNpmPackage } from "../connectors/npm_package.js";
import { syncPyPIPackage } from "../connectors/pypi_package.js";
import { syncDatabase } from "../connectors/database.js";
import { syncHuggingFace } from "../connectors/huggingface.js";
import { syncWeb } from "../connectors/web.js";
import { ingestDocument } from "../engine/ingest.js";
import { fireWebhookEvent } from "../engine/webhooks.js";
import { embedSingle } from "../engine/embeddings.js";
import { ingestSession as ingestMemorySession, searchMemories } from "../engine/memory/index.js";
import { writeMemoryCanonical } from "../engine/memory/write.js";
import { nanoid } from "nanoid";
import { createHash } from "crypto";
import { memoryRoutes } from "./memory.js";
import { contextRoutes } from "./context.js";
import { nativeHookRoutes } from "./native-hooks.js";
import { optimizationRoutes } from "./optimization.js";
import { searchRoutes } from "./search.js";
import { fileRoutes } from "./files.js";
import { agentTaskRoutes } from "./agent-tasks.js";
import { researchAgentRoutes } from "./research-agent.js";
import { resolveProjectReference, ensureProject, getEffectiveOrgId } from "./helpers.js";
import { getContractHeaders, getPublicContractMetadata } from "../contracts/runtime.mjs";
import { getLatencySummary, getLatencyTraceConfig, getLatencyGateStatus, resetLatencySummary } from "../engine/latency-tracing.js";
import {
  getExtractionAlerts,
  getExtractionGateStatus,
  getExtractionPhase0Config,
  getExtractionStats,
  resetExtractionObservability,
} from "../engine/extraction-observability.js";
import { getTraceIdFromRequest } from "../lib/trace.js";
import {
  getIdempotencyKey,
  hashIdempotencyPayload,
  loadIdempotentResponse,
  storeIdempotentResponse,
} from "./idempotency.js";
import {
  buildRouteControlMatrix,
  assertRouteControlCoverage,
  getRouteControl,
} from "../security/route-controls.js";
import {
  getSourceVersion,
  listSourceVersions,
  serializeSourceVersion,
  restoreSource,
  softDeleteSource,
  markStaleSourceVersionsFailed,
} from "../engine/source-versions.js";
import { redeliverWebhookDelivery } from "../engine/webhooks.js";
import { exportIndexBundle } from "../engine/index-bundle.js";
import {
  evaluateOperationalAlerts,
  getConnectorHealthSummary,
  getOperationalCounters,
  getQueueHealthSummary,
  getRetrievalHealthSummary,
  getWebhookFailureSummary,
} from "../engine/ops-observability.js";

// Type augmentation for Hono context variables
type Variables = {
  auth: AuthContext;
  traceId: string;
};

export const api = new Hono<{ Variables: Variables }>();
const DEPLOY_REGION = process.env.DEPLOY_REGION || process.env.AWS_REGION || "us-east-1";
const STACK_NAME = process.env.RETAINDB_STACK || "ec2";
const ORGANIZATION_PLAN = {
  FREE: "FREE",
  OSS: "OSS",
  PAY_AS_YOU_GO: "PAY_AS_YOU_GO",
  PRO: "PRO",
  SCALE: "SCALE",
  ENTERPRISE: "ENTERPRISE",
} as const;
  ORGANIZATION_PLAN
type DeviceAuthRecord = {
  userCode: string;
  expiresAt: number;
  apiKey?: string;
};
const LEGACY_MEMORY_SUNSET = "Wed, 01 Jul 2026 00:00:00 GMT";

api.use("/*", async (c, next) => {
  const headers = getContractHeaders();
  for (const [name, value] of Object.entries(headers)) {
    c.header(name as any, value as any);
  }
  const traceId = getTraceIdFromRequest(c);
  c.set("traceId", traceId);
  c.header("x-trace-id", traceId);
  c.header("x-request-id", traceId);
  await next();
});


function getLegacyMemoryReplacement(method: string, path: string): string {
  const upperMethod = method.toUpperCase();
  if (path === "/v1/memories" && upperMethod === "POST") return "/v1/memory";
  if (path === "/v1/memories/search") return "/v1/memory/search";
  if (path === "/v1/memories" && upperMethod === "GET") {
    return "/v1/memory/profile/:userId or /v1/memory/session/:sessionId";
  }
  if (/^\/v1\/memories\/[^/]+$/.test(path)) return "/v1/memory/:memoryId";
  return "/v1/memory";
}

function markLegacyMemoryRoute(c: {
  req: { path: string; method: string };
  header: (name: string, value: string) => void;
}) {
  c.header("Deprecation", "true");
  c.header("Sunset", LEGACY_MEMORY_SUNSET);
  c.header("Link", `</docs/api-reference#legacy-memory-routes>; rel="deprecation"; type="text/markdown"`);
  c.header("X-RetainDB-Replacement-Route", getLegacyMemoryReplacement(c.req.method, c.req.path));
}



function bigIntJson(value: any): any {
  if (value === null || value === undefined) return value;
  if (typeof value === "bigint") return value.toString();
  if (typeof value === "number" && !Number.isFinite(value)) return value.toString();
  if (Array.isArray(value)) return value.map(bigIntJson);
  if (value instanceof Date) return value.toISOString();
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, bigIntJson(v)])
    );
  }
  return value;
}

function extractToken(raw: string | undefined): string {
  if (!raw) return "";
  const trimmed = raw.trim();
  const bearer = trimmed.match(/^Bearer\s+(.+)$/i)?.[1];
  return (bearer || trimmed).trim();
}

function sanitizeWizardMetadata(input: unknown): Record<string, any> {
  if (!input || typeof input !== "object" || Array.isArray(input)) return {};
  const blocked = new Set(["query", "content", "prompt", "file", "path", "token", "apiKey"]);
  const out: Record<string, any> = {};
  for (const [key, value] of Object.entries(input as Record<string, any>)) {
    if (blocked.has(key)) continue;
    if (typeof value === "string") {
      out[key] = value.length > 180 ? `${value.slice(0, 180)}...` : value;
      continue;
    }
    if (typeof value === "number" || typeof value === "boolean" || value === null) {
      out[key] = value;
      continue;
    }
  }
  return out;
}







api.use("/*", authMiddleware);
api.use("/*", async (c, next) => {
  const auth = c.get("auth") as AuthContext;
  const traceId = c.get("traceId");
  const routeMatrix = getNodeRouteMatrix();
  const control = getRouteControl(routeMatrix, c.req.method, c.req.path);
  if (!control) {
    return c.json({ error: "Route control missing", trace_id: traceId }, 500);
  }
  if (control.authMode === "admin_only" && !auth.isAdmin) {
    return c.json({ error: "Admin access required", trace_id: traceId }, 403);
  }
  await next();
  if (control.auditRequired) {
  }
});

// ─── SOTA Routes ─────────────────────────────────────────────
// Mount SOTA memory, context, and optimization APIs
api.route("/", memoryRoutes);
api.route("/", contextRoutes);
api.route("/", nativeHookRoutes);
api.route("/", optimizationRoutes);
api.route("/", searchRoutes);
api.route("/", fileRoutes);
api.route("/", agentTaskRoutes);
api.route("/", researchAgentRoutes);

// ─── Helper ──────────────────────────────────────────────────

function slugify(s: string) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function getNodeRouteMatrix() {
  const routes = ((api as any).routes || []) as Array<{ method: string; path: string }>;
  const matrix = buildRouteControlMatrix(routes, "node");
  assertRouteControlCoverage(matrix, routes, "node");
  return matrix;
}

function serializeSource(source: any, activeJob?: any) {
  const activeVersion = source.activeVersion || null;
  const latestVersion = source.sourceVersions?.[0] || activeVersion;
  const jobResult = (activeJob?.result || {}) as Record<string, any>;
  return {
    id: String(source.id),
    orgId: String(source.orgId),
    projectId: String(source.projectId),
    name: source.name,
    type: source.type,
    connectorType: source.connectorType,
    config: source.config,
    status: source.status,
    syncSchedule: source.syncSchedule,
    lastSyncAt: source.lastSyncAt,
    lastError: source.syncError || source.lastSyncError || null,
    createdAt: source.createdAt,
    updatedAt: source.updatedAt,
    partial_failure: latestVersion?.partialFailure || jobResult.partialFailure || false,
    documents_total: latestVersion?.documentCount ?? source.documentCount ?? 0,
    documents_indexed: latestVersion?.documentCount ?? source.documentCount ?? 0,
    documents_failed: jobResult.documentsFailed ?? 0,
    warning_codes: latestVersion?.warningCodes || jobResult.warningCodes || [],
    error_code: latestVersion?.errorCode || jobResult.errorCode || null,
    outcome: latestVersion?.partialFailure
      ? "partial_failure"
      : latestVersion?.errorCode && (latestVersion?.documentCount || 0) <= 0
        ? "failed"
        : "success",
    active_version: source.activeVersionId || null,
    restore_until: null,
  };
}

const sourceSummarySelect = {
  id: true,
  orgId: true,
  projectId: true,
  name: true,
  type: true,
  connectorType: true,
  config: true,
  status: true,
  syncSchedule: true,
  lastSyncAt: true,
  syncError: true,
  lastSyncError: true,
  documentCount: true,
  chunkCount: true,
  createdAt: true,
  updatedAt: true,
  activeVersionId: true,
  sourceVersions: {
    take: 1,
    orderBy: { createdAt: "desc" as const },
    select: {
      id: true,
      partialFailure: true,
      warningCodes: true,
      errorCode: true,
      documentCount: true,
      status: true,
      versionNumber: true,
      chunkCount: true,
      errorMessage: true,
      syncJobId: true,
      promotedAt: true,
      supersededAt: true,
      failedAt: true,
      createdAt: true,
      sourceId: true,
      orgId: true,
    },
  },
};

function serializeSyncJob(job: any) {
  const result = (job?.result || {}) as Record<string, any>;
  return {
    id: job.id,
    source_id: job.sourceId,
    source_version_id: job.sourceVersionId || null,
    status: job.status,
    progress: job.progress,
    result: job.result,
    error: job.error,
    started_at: job.startedAt,
    completed_at: job.completedAt,
    partial_failure: result.partialFailure || false,
    documents_total: result.documentsTotal ?? null,
    documents_indexed: result.documentsIndexed ?? null,
    documents_failed: result.documentsFailed ?? null,
    warning_codes: result.warningCodes || [],
    error_code: result.errorCode || null,
    outcome: result.outcome || (result.partialFailure ? "partial_failure" : job.error ? "failed" : "success"),
    mode: result.mode || job.mode || null,
    effective_mode: result.effectiveMode || job.effectiveMode || null,
    active_version: result.activeVersion || null,
  };
}

async function loadMutationReplay(
  c: any,
  auth: AuthContext,
  endpoint: string,
  payload: Record<string, any>
) {
  const idempotencyKey = getIdempotencyKey({
    "idempotency-key": c.req.header("idempotency-key"),
    "Idempotency-Key": c.req.header("Idempotency-Key"),
  });
  if (!idempotencyKey) return { idempotencyKey: null, requestHash: null, replay: null as any };

  const requestHash = hashIdempotencyPayload(payload);
  const replay = await loadIdempotentResponse({
    orgId: auth.orgId,
    endpoint,
    idempotencyKey,
    requestHash,
  });
  return { idempotencyKey, requestHash, replay };
}

async function storeMutationReplay(params: {
  auth: AuthContext;
  endpoint: string;
  idempotencyKey?: string | null;
  requestHash?: string | null;
  statusCode: number;
  body: Record<string, any>;
  ttlSeconds?: number;
}) {
  if (!params.idempotencyKey || !params.requestHash) return;
  await storeIdempotentResponse({
    orgId: params.auth.orgId,
    endpoint: params.endpoint,
    idempotencyKey: params.idempotencyKey,
    requestHash: params.requestHash,
    statusCode: params.statusCode,
    body: params.body,
    ttlSeconds: params.ttlSeconds,
  });
}

const DEFAULT_LEARN_PROJECT = (process.env.RETAINDB_PROJECT || "").trim();
const LEARN_LOCAL_ALLOWLIST = (process.env.RETAINDB_LOCAL_ALLOWLIST || "")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);
const LEARN_AUTO_DEBOUNCE_MS = Math.max(50, parseInt(process.env.LEARN_AUTO_DEBOUNCE_MS || "250", 10));
const learnAutoTimers = new Map<string, NodeJS.Timeout>();

const learnTextOptionsSchema = z.object({
  async: z.boolean().optional(),
  ingestion_profile: z.enum(["auto", "repo", "web_docs", "pdf_layout", "video_transcript", "plain_text"]).optional(),
  strategy_override: z.enum(["fixed", "recursive", "semantic", "hierarchical", "adaptive"]).optional(),
  profile_config: z.record(z.any()).optional(),
}).optional();

const learnSourceOptionsSchema = z.object({
  async: z.boolean().optional(),
  auto_index: z.boolean().optional(),
  ingestion_profile: z.enum(["auto", "repo", "web_docs", "pdf_layout", "video_transcript", "plain_text"]).optional(),
  strategy_override: z.enum(["fixed", "recursive", "semantic", "hierarchical", "adaptive"]).optional(),
  profile_config: z.record(z.any()).optional(),
  crawl_depth: z.number().int().min(0).optional(),
  include_paths: z.array(z.string()).optional(),
  exclude_paths: z.array(z.string()).optional(),
  glob: z.string().optional(),
  max_files: z.number().int().min(1).optional(),
  max_pages: z.number().int().min(1).optional(),
  extract_mode: z.enum(["text", "structured", "markdown"]).optional(),
  workspace_id: z.string().optional(),
  allow_stt_fallback: z.boolean().optional(),
  max_duration_minutes: z.number().int().min(1).optional(),
  max_chunks: z.number().int().min(1).optional(),
}).optional();

const learnRequestSchema = z.discriminatedUnion("mode", [
  z.object({
    mode: z.literal("conversation"),
    project: z.string().optional(),
    user_id: z.string().optional(),
    agent_id: z.string().optional(),
    task_id: z.string().trim().min(1).max(128).optional(),
    session_id: z.string().min(1).max(128),
    messages: z.array(z.object({
      role: z.string().min(1),
      content: z.string().min(1),
      timestamp: z.string().optional(),
    })).min(1),
    events: z.array(z.object({
      kind: z.enum(["decision", "constraint", "outcome", "failure", "task_update", "file_edit", "tool_result"]),
      summary: z.string().min(1).max(1000),
      details: z.string().max(5000).optional(),
      salience: z.enum(["low", "medium", "high"]).optional(),
      timestamp: z.string().optional(),
      filePaths: z.array(z.string()).optional(),
      toolName: z.string().optional(),
      success: z.boolean().optional(),
    })).optional(),
    promotion_mode: z.enum(["session_state_v1", "user_specific_legacy"]).optional(),
  }),
  z.object({
    mode: z.literal("text"),
    project: z.string().optional(),
    title: z.string().min(1).max(500),
    content: z.string().min(1),
    metadata: z.record(z.any()).optional(),
    tags: z.array(z.string()).optional(),
    namespace: z.string().optional(),
    options: learnTextOptionsSchema,
  }),
  z.object({
    mode: z.literal("source"),
    project: z.string().optional(),
    type: z.enum(["github", "web", "url", "playwright", "pdf", "local", "slack", "video"]),
    name: z.string().optional(),
    metadata: z.record(z.string()).optional(),
    owner: z.string().optional(),
    repo: z.string().optional(),
    branch: z.string().optional(),
    paths: z.array(z.string()).optional(),
    url: z.string().url().optional(),
    file_path: z.string().optional(),
    path: z.string().optional(),
    channel_ids: z.array(z.string()).optional(),
    since: z.string().optional(),
    token: z.string().optional(),
    auth_ref: z.string().optional(),
    platform: z.enum(["youtube", "loom", "generic"]).optional(),
    language: z.string().optional(),
    options: learnSourceOptionsSchema,
  }),
]);

const indexApiConnectorTypeSchema = z.enum([
  "github",
  "gitlab",
  "github-tarball",
  "url",
  "sitemap",
  "web",
  "playwright",
  "pdf",
  "text",
  "api_spec",
  "dataset",
  "database",
  "confluence",
  "notion",
  "slack",
  "discord",
  "arxiv",
  "huggingface",
  "npm_package",
  "pypi_package",
  "video",
  "custom",
  "local",
]);

const indexApiSourceSchema = z.object({
  type: indexApiConnectorTypeSchema,
  name: z.string().optional(),
  config: z.record(z.any()).optional(),
  sync_schedule: z.string().optional(),
  auto_index: z.boolean().optional(),

  // Convenience fields for common connectors (optional; merged into config if set).
  owner: z.string().optional(),
  repo: z.string().optional(),
  branch: z.string().optional(),
  paths: z.array(z.string()).optional(),
  url: z.string().optional(),
  channel_ids: z.array(z.string()).optional(),
  since: z.string().optional(),
  token: z.string().optional(),
  auth_ref: z.string().optional(),
  path: z.string().optional(),
  file_path: z.string().optional(),
  platform: z.enum(["youtube", "loom", "generic"]).optional(),
  language: z.string().optional(),

  options: learnSourceOptionsSchema,
  metadata: z.record(z.any()).optional(),
});

const indexBundleRequestSchema = z.object({
  project: z.string().optional(),
  include: z.object({
    sources: z.boolean().optional().default(true),
    documents: z.boolean().optional().default(true),
    chunks: z.boolean().optional().default(false),
    memories: z.boolean().optional().default(true),
    entities: z.boolean().optional().default(false),
    relations: z.boolean().optional().default(false),
  }).optional().default({}),
  limits: z.object({
    maxSources: z.number().int().min(1).max(2000).optional(),
    maxDocuments: z.number().int().min(1).max(5000).optional(),
    maxChunks: z.number().int().min(1).max(200000).optional(),
    maxChunkChars: z.number().int().min(200).max(200000).optional(),
    maxMemories: z.number().int().min(1).max(20000).optional(),
    maxEntities: z.number().int().min(1).max(200000).optional(),
    maxRelations: z.number().int().min(1).max(200000).optional(),
  }).optional(),
  redact_secrets: z.boolean().optional().default(true),
});

const indexApiRequestSchema = z.object({
  project: z.string().optional(),
  sources: z.array(indexApiSourceSchema).min(1).max(25),
  auto_index: z.boolean().optional().default(true),
  return_bundle: z.boolean().optional().default(false),
  bundle: indexBundleRequestSchema.omit({ project: true }).optional(),
});

function resolveLearnProjectRef(project?: string): string | undefined {
  const explicit = project?.trim();
  if (explicit) return explicit;
  return DEFAULT_LEARN_PROJECT || undefined;
}

function getLearnLocalAllowlistRoots(): string[] {
  return LEARN_LOCAL_ALLOWLIST.length > 0 ? LEARN_LOCAL_ALLOWLIST : [process.cwd()];
}

function isLearnLocalPathAllowed(targetPath: string): { allowed: boolean; allowlist: string[] } {
  const normalized = targetPath.replace(/\\/g, "/").toLowerCase();
  const allowlist = getLearnLocalAllowlistRoots();
  const allowed = allowlist.some((root) => normalized.startsWith(root.replace(/\\/g, "/").toLowerCase()));
  return { allowed, allowlist };
}

function shouldSkipLearnSensitivePath(filePath: string): boolean {
  const normalized = filePath.replace(/\\/g, "/").toLowerCase();
  const denySnippets = [
    "/node_modules/",
    "/.git/",
    "/dist/",
    "/build/",
    "/.next/",
    "/.aws/",
    "/.ssh/",
    ".pem",
    ".key",
    ".env",
    "credentials",
  ];
  return denySnippets.some((snippet) => normalized.includes(snippet));
}

function redactLearnSecrets(content: string): string {
  return content
    .replace(/(api[_-]?key\s*[=:]\s*)[^\s"'`]+/gi, "$1[REDACTED]")
    .replace(/(token\s*[=:]\s*)[^\s"'`]+/gi, "$1[REDACTED]")
    .replace(/(secret\s*[=:]\s*)[^\s"'`]+/gi, "$1[REDACTED]");
}

async function ensureLearnProject(auth: AuthContext, project?: string) {
  const projectRef = resolveLearnProjectRef(project);
  return ensureProject(auth.orgId, projectRef, auth.isAdmin);
}

function mergeIndexSourceConfig(input: z.infer<typeof indexApiSourceSchema>): Record<string, any> {
  const base =
    input.config && typeof input.config === "object" && !Array.isArray(input.config)
      ? { ...input.config }
      : {};

  const merged: Record<string, any> = { ...base };

  if (input.owner) merged.owner = input.owner;
  if (input.repo) merged.repo = input.repo;
  if (input.branch) merged.branch = input.branch;
  if (input.paths) merged.paths = input.paths;
  if (input.url) merged.url = input.url;
  if (input.channel_ids) merged.channel_ids = input.channel_ids;
  if (input.since) merged.since = input.since;
  if (input.token) merged.token = input.token;
  if (input.auth_ref) merged.auth_ref = input.auth_ref;
  if (input.path) merged.path = input.path;
  if (input.file_path) merged.file_path = input.file_path;
  if (input.platform) merged.platform = input.platform;
  if (input.language) merged.language = input.language;
  if (input.metadata) merged.metadata = input.metadata;
  if (input.options && typeof input.options === "object") merged.options = input.options;

  return merged;
}

async function ensureDirectIngestSource(projectId: string, orgId: string) {
  const directIngestName = `direct-ingest-${projectId}`;
  const findSource = () => prisma.source.findFirst({
    where: {
      orgId,
      projectId,
      connectorType: "custom",
      OR: [
        { name: "direct-ingest" },
        { name: directIngestName },
      ],
    },
  });
  const directSource = await findSource();
  if (directSource) return directSource;

  try {
    return await prisma.source.create({
      data: {
        orgId,
        projectId,
        name: directIngestName,
        type: "custom",
        connectorType: "custom",
        config: {},
        status: "READY",
      },
    });
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "P2002") {
      const concurrent = await findSource();
      if (concurrent) return concurrent;
    }
    throw error;
  }
}

async function learnTextContent(params: {
  auth: AuthContext;
  project: { id: string; orgId: string; name: string; slug: string | null };
  traceId: string;
  input: z.infer<typeof learnRequestSchema> & { mode: "text" };
}) {
  const { auth, project, traceId, input } = params;
  const asyncMode = input.options?.async !== false;
  if (asyncMode) {
    const jobId = await ingestionQueue.createJob({
      orgId: auth.orgId,
      projectId: project.id,
      userId: auth.userId || "system",
      documents: [{
        title: input.title,
        content: input.content,
        metadata: {
          ...(input.metadata || {}),
          tags: input.tags || [],
        },
        namespace: input.namespace,
        tags: input.tags || [],
        ingestion_profile: input.options?.ingestion_profile,
        strategy_override: input.options?.strategy_override,
        profile_config: input.options?.profile_config as any,
      }],
    });
    return {
      success: true as const,
      mode: "text" as const,
      project: project.id,
      status: "processing" as const,
      job_id: jobId,
      source_id: null,
    };
  }

  const directSource = await ensureDirectIngestSource(project.id, auth.orgId);
  const result = await ingestDocument({
    sourceId: directSource.id,
    projectId: project.id,
    externalId: `learn-${input.title}`,
    title: input.title,
    content: input.content,
    metadata: {
      ...(input.metadata || {}),
      tags: input.tags || [],
    },
    ingestionProfile: input.options?.ingestion_profile,
    strategyOverride: input.options?.strategy_override,
    profileConfig: input.options?.profile_config as any,
  });
  return {
    success: true as const,
    mode: "text" as const,
    project: project.id,
    status: "completed" as const,
    chunks_indexed: Number(result?.chunksCreated || 1),
    source_id: directSource.id,
  };
}

async function learnSourceContent(params: {
  auth: AuthContext;
  project: { id: string; orgId: string; name: string; slug: string | null };
  traceId: string;
  input: z.infer<typeof learnRequestSchema> & { mode: "source" };
}) {
  const { auth, project, traceId, input } = params;
  if (input.options?.async === false) {
    throw new Error("source learn is background-only; remove options.async=false");
  }

  const autoIndex = input.options?.auto_index ?? true;
  const baseConfig: Record<string, any> = {
    ...(input.metadata ? { metadata: input.metadata } : {}),
    ...(input.options?.ingestion_profile ? { ingestion_profile: input.options.ingestion_profile } : {}),
    ...(input.options?.strategy_override ? { strategy_override: input.options.strategy_override } : {}),
    ...(input.options?.profile_config ? { profile_config: input.options.profile_config } : {}),
  };

  let connectorType = input.type;
  const config: Record<string, any> = { ...baseConfig };

  if (input.type === "github") {
    if (!input.owner || !input.repo) throw new Error("github learn requires owner and repo");
    config.owner = input.owner;
    config.repo = input.repo;
    if (input.branch) config.branch = input.branch;
    if (input.paths) config.paths = input.paths;
  } else if (input.type === "web" || input.type === "url" || input.type === "playwright") {
    if (!input.url) throw new Error("web learn requires url");
    connectorType = "web"; // all three map to the same connector
    config.url = input.url;
    if (input.options?.crawl_depth !== undefined) config.maxDepth = input.options.crawl_depth;
    if (input.options?.max_pages !== undefined) config.maxPages = input.options.max_pages;
    if (input.options?.include_paths) config.includePaths = input.options.include_paths;
    if (input.options?.exclude_paths) config.excludePaths = input.options.exclude_paths;
    // playwright mode uses browser rendering — keep as separate connector
    if (input.type === "playwright") {
      connectorType = "playwright";
      if (input.options?.extract_mode) config.extractMode = input.options.extract_mode;
    }
  } else if (input.type === "pdf") {
    if (!input.url && !input.file_path) throw new Error("pdf learn requires url or file_path");
    if (input.url) config.url = input.url;
    if (input.file_path) config.file_path = input.file_path;
  } else if (input.type === "slack") {
    config.channel_ids = input.channel_ids || [];
    if (input.since) config.since = input.since;
    if (input.options?.workspace_id) config.workspace_id = input.options.workspace_id;
    if (input.token) config.token = input.token;
    if (input.auth_ref) config.auth_ref = input.auth_ref;
  } else if (input.type === "video") {
    if (!input.url) throw new Error("video learn requires url");
    config.url = input.url;
    if (input.platform) config.platform = input.platform;
    if (input.language) config.language = input.language;
    if (input.options?.allow_stt_fallback !== undefined) config.allow_stt_fallback = input.options.allow_stt_fallback;
    if (input.options?.max_duration_minutes !== undefined) config.max_duration_minutes = input.options.max_duration_minutes;
    if (input.options?.max_chunks !== undefined) config.max_chunks = input.options.max_chunks;
  } else if (input.type === "local") {
    if (!input.path) throw new Error("local learn requires path");
    const rootPath = resolvePath(input.path);
    const gate = isLearnLocalPathAllowed(rootPath);
    if (!gate.allowed) {
      throw new Error(`Path not allowed by RETAINDB_LOCAL_ALLOWLIST. Allowed roots: ${gate.allowlist.join(", ")}`);
    }
    const maxFiles = Math.max(1, input.options?.max_files || 200);
    const maxBytesPerFile = 512 * 1024;
    const files: string[] = [];
    const collect = (dir: string) => {
      if (files.length >= maxFiles) return;
      let entries;
      try {
        entries = readdirSync(dir, { withFileTypes: true });
      } catch {
        return;
      }
      for (const entry of entries) {
        if (files.length >= maxFiles) return;
        const fullPath = join(dir, entry.name);
        if (shouldSkipLearnSensitivePath(fullPath)) continue;
        if (entry.isDirectory()) {
          collect(fullPath);
          continue;
        }
        if (entry.isFile()) {
          files.push(fullPath);
        }
      }
    };
    collect(rootPath);

    const directSource = await ensureDirectIngestSource(project.id, auth.orgId);
    let chunksIndexed = 0;
    for (const fullPath of files) {
      const stats = statSync(fullPath);
      if (stats.size > maxBytesPerFile) continue;
      const relPath = relative(rootPath, fullPath);
      const content = redactLearnSecrets(readFileSync(fullPath, "utf8"));
      await ingestDocument({
        sourceId: directSource.id,
        projectId: project.id,
        externalId: `local-${relPath}`,
        title: relPath,
        content,
        filePath: relPath,
        metadata: {
          source_type: "local",
          path: relPath,
          ...(input.metadata || {}),
        },
        ingestionProfile: input.options?.ingestion_profile,
        strategyOverride: input.options?.strategy_override,
        profileConfig: input.options?.profile_config as any,
      });
      chunksIndexed += 1;
    }
    return {
      success: true as const,
      mode: "source" as const,
      project: project.id,
      source_id: directSource.id,
      status: "ready",
      job_id: null,
      index_started: true,
      chunks_indexed: chunksIndexed,
    };
  } else {
    throw new Error(`Unsupported learn source type: ${input.type}`);
  }

  const sourceName =
    input.name?.trim()
    || (input.type === "video" && input.url
      ? `video:${new URL(input.url).hostname}`
      : `${input.type}-source-${Date.now()}`);
  const source = await prisma.source.create({
    data: {
      orgId: project.orgId,
      projectId: project.id,
      name: sourceName,
      type: connectorType,
      connectorType,
      config,
      status: "PENDING",
    },
  });

  let jobId: string | null = null;
  let status = "created";
  if (autoIndex) {
    const queued = await enqueueSync(source.id, {
      traceId,
      parentTraceId: traceId,
      reuseExisting: true,
    });
    jobId = queued.jobId;
    status = connectorType === "video" ? "processing" : "queued";
  }

  return {
    success: true as const,
    mode: "source" as const,
    project: project.id,
    source_id: source.id,
    status,
    job_id: jobId,
    index_started: autoIndex,
  };
}

function scheduleConversationAutoLearn(params: {
  auth: AuthContext;

  projectId: string;
  sessionId: string;
  userId?: string;
}) {
  const key = `${params.projectId}:${params.userId || "session_only"}:${params.sessionId}`;
  const existing = learnAutoTimers.get(key);
  if (existing) clearTimeout(existing);
  const timeout = setTimeout(async () => {
    learnAutoTimers.delete(key);
    try {
      const recentMessages = await prisma.message.findMany({
        where: { sessionId: params.sessionId },
        orderBy: { createdAt: "desc" },
        take: 10,
        select: {
          role: true,
          content: true,
          createdAt: true,
        },
      });
      if (recentMessages.length === 0) return;
      await ingestMemorySession({
        sessionId: params.sessionId,
        projectId: params.projectId,
        orgId: params.auth.orgId,
        userId: params.userId,
        messages: recentMessages.reverse().map((message) => ({
          role: message.role,
          content: message.content,
          timestamp: message.createdAt,
        })),
      });
    } catch (error) {
      console.warn("[Learn] conversation auto-learn failed:", error instanceof Error ? error.message : String(error));
    }
  }, LEARN_AUTO_DEBOUNCE_MS);
  learnAutoTimers.set(key, timeout);
}


function simpleClawProjectSlug(userId: string) {
  const base = slugify(`sc-${userId}`).slice(0, 56);
  return base || `sc-${nanoid(10).toLowerCase()}`;
}

function normalizeMemoryType(memoryType?: string) {
  const value = (memoryType || "").toLowerCase();
  const mapped: Record<string, "factual" | "preference" | "event" | "relationship" | "opinion" | "goal" | "instruction"> = {
    factual: "factual",
    episodic: "event",
    semantic: "factual",
    procedural: "instruction",
    preference: "preference",
    event: "event",
    relationship: "relationship",
    opinion: "opinion",
    goal: "goal",
    instruction: "instruction",
  };
  return mapped[value] || "factual";
}


// ─── Query Context ───────────────────────────────────────────

api.post(
  "/v1/context/query",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      query: z.string().min(1).max(5000),
      top_k: z.number().int().min(1).max(50).optional().default(10),
      threshold: z.number().min(0).max(1).optional().default(0.3),
      chunk_types: z.array(z.string()).optional(),
      source_ids: z.array(z.string()).optional(),
      metadata_filter: z.record(z.any()).optional(),
      // Oracle prefilter (structure-aware scope selection)
      oracle_mode: z.enum(["off", "auto", "force"]).optional(),
      oracle_max_seed_hits: z.number().int().min(10).max(400).optional(),
      oracle_max_documents: z.number().int().min(1).max(20).optional(),
      oracle_max_sections_per_doc: z.number().int().min(1).max(30).optional(),
      oracle_max_candidate_chunks: z.number().int().min(50).max(5000).optional(),
      // Hybrid search
      hybrid: z.boolean().optional().default(true),
      use_vector: z.boolean().optional().default(true),
      vector_weight: z.number().min(0).max(1).optional(),
      bm25_weight: z.number().min(0).max(1).optional(),
      // Reranking
      rerank: z.boolean().optional().default(true),
      // Memory
      include_memories: z.boolean().optional().default(false),
      user_id: z.string().optional(),
      session_id: z.string().optional(),
      agent_id: z.string().optional(),
      // Graph
      include_graph: z.boolean().optional().default(false),
      graph_depth: z.number().int().min(1).max(3).optional(),
      // Context packing
      max_tokens: z.number().int().optional(),
      // Compression (token reduction)
      compress: z.boolean().optional().default(false),
      compression_strategy: z.enum(["summarize", "extract", "delta", "adaptive"]).optional(),
      previous_context_hash: z.string().optional(),
      // Cache
      use_cache: z.boolean().optional().default(true),
      // Precision rollout
      include_parent_content: z.boolean().optional().default(false),
      retrieval_profile: z.enum(["legacy", "precision_v1"]).optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const body = c.req.valid("json");
    c.header("x-retaindb-stack", STACK_NAME);
    c.header("x-whisper-stack", STACK_NAME);

    const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);

    console.log(`[Context] Query project: ${project.name} (${project.id})`);

    const result = await retrieve({
      projectId: project.id,
      query: body.query,
      topK: body.top_k,
      threshold: body.threshold,
      chunkTypes: body.chunk_types,
      sourceIds: body.source_ids,
      metadataFilter: body.metadata_filter,
      oracleMode: body.oracle_mode,
      oracleMaxSeedHits: body.oracle_max_seed_hits,
      oracleMaxDocuments: body.oracle_max_documents,
      oracleMaxSectionsPerDoc: body.oracle_max_sections_per_doc,
      oracleMaxCandidateChunks: body.oracle_max_candidate_chunks,
      hybridSearch: body.hybrid,
      useVector: body.use_vector,
      vectorWeight: body.vector_weight,
      bm25Weight: body.bm25_weight,
      rerank: body.rerank,
      includeMemories: body.include_memories,
      userId: body.user_id,
      sessionId: body.session_id,
      agentId: body.agent_id,
      includeGraph: body.include_graph,
      graphDepth: body.graph_depth,
      maxTokens: body.max_tokens,
      compress: body.compress,
      compressionStrategy: body.compression_strategy,
      previousContextHash: body.previous_context_hash,
      useCache: body.use_cache,
      includeParentContent: body.include_parent_content,
      retrievalProfile: body.retrieval_profile,
    });

    // usage tracking not available in OSS

    const consistencyState = body.include_memories ? "eventual" : "fresh";

    return c.json({
      results: result.results.map((r) => ({
        id: r.id,
        content: r.content,
        score: Math.round(r.score * 1000) / 1000,
        metadata: r.metadata,
        source: r.sourceName,
        document: r.documentTitle,
        type: r.chunkType,
        retrieval_source: r.source,
      })),
      context: result.context,
      meta: {
        query: body.query,
        total: result.meta.totalResults,
        latency_ms: result.meta.latencyMs,
        cache_hit: result.meta.cacheHit,
        tokens_used: result.meta.tokensUsed,
        context_hash: result.meta.contextHash,
        source_scope: result.meta.sourceScope
          ? {
            mode: result.meta.sourceScope.mode,
            source_ids: result.meta.sourceScope.sourceIds,
            host: result.meta.sourceScope.host,
            matched_sources: result.meta.sourceScope.matchedSources,
          }
          : undefined,
        compression: result.meta.compression,
        profile: result.meta.profile || "balanced",
        retrieval_profile: result.meta.retrievalProfile,
        region: DEPLOY_REGION,
        consistency_state: consistencyState,
        timing: result.meta.timing,
      },
    });
  }
);

api.post(
  "/v1/learn",
  rateLimitMiddleware(RateLimits.ingest),
  zValidator("json", learnRequestSchema),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const body = c.req.valid("json");
    const traceId = c.get("traceId");
    const endpoint = "/v1/learn";
    const replay = await loadMutationReplay(c, auth, endpoint, body as Record<string, any>);
    if (replay.replay?.type === "conflict") {
      return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
    }
    if (replay.replay?.type === "hit") {
      c.header("x-idempotency-replay", "true");
      return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
    }

    const project = await ensureLearnProject(auth, body.project);
    try {
      let responseBody: Record<string, any>;
      if (body.mode === "conversation") {
        const result = await ingestMemorySession({
          sessionId: body.session_id,
          projectId: project.id,
          orgId: auth.orgId,
          userId: body.user_id,
          agentId: body.agent_id,
          taskId: body.task_id,
          events: body.events,
          promotionMode: body.promotion_mode,
          messages: body.messages.map((message) => ({
            role: message.role,
            content: message.content,
            timestamp: message.timestamp ? new Date(message.timestamp) : new Date(),
          })),
        });
        responseBody = {
          success: true,
          mode: "conversation",
          project: project.id,
          scope_mode: body.user_id ? "user_session" : "session_only",
          memories_created: result.memoriesCreated,
          relations_created: result.relationsCreated,
          memories_invalidated: result.memoriesInvalidated,
          scope_counts: result.scopeCounts,
          scopes_touched: result.scopesTouched,
          errors: result.errors,
        };
      } else if (body.mode === "text") {
        responseBody = await learnTextContent({
          auth,
          project,
          traceId,
          input: body,
        });
      } else {
        responseBody = await learnSourceContent({
          auth,
          project,
          traceId,
          input: body,
        });
      }

      const statusCode =
        body.mode === "text" && responseBody.status === "processing"
          ? 202
          : body.mode === "source" && responseBody.index_started
            ? 202
            : 200;
      await storeMutationReplay({
        auth,
        endpoint,
        idempotencyKey: replay.idempotencyKey,
        requestHash: replay.requestHash,
        statusCode,
        body: responseBody,
      });
      return c.json(bigIntJson(responseBody), statusCode as any);
    } catch (error: any) {
      return c.json({
        success: false,
        error: error?.message || "Learn failed",
        trace_id: traceId,
      }, 400);
    }
  }
);

// ─── Batch Learn (up to 600 mixed items) ────────────────────────────────────

const batchLearnItemSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("text"),
    title: z.string().min(1),
    content: z.string().min(1),
    metadata: z.record(z.any()).optional(),
    tags: z.array(z.string()).optional(),
    custom_id: z.string().max(100).optional(),
  }),
  z.object({
    type: z.literal("url"),
    url: z.string().url(),
    title: z.string().optional(),
    metadata: z.record(z.any()).optional(),
    tags: z.array(z.string()).optional(),
    custom_id: z.string().max(100).optional(),
  }),
  z.object({
    type: z.literal("conversation"),
    session_id: z.string().min(1),
    messages: z.array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().min(1),
        timestamp: z.string().optional(),
      })
    ).min(1),
    metadata: z.record(z.any()).optional(),
    tags: z.array(z.string()).optional(),
    custom_id: z.string().max(100).optional(),
  }),
]);

const batchLearnSchema = z.object({
  items: z.array(batchLearnItemSchema).min(1).max(600),
  project: z.string().optional(),
  namespace: z.string().optional(),
  tags: z.array(z.string()).optional(),
  async: z.boolean().optional().default(true),
});

api.post(
  "/v1/learn/batch",
  rateLimitMiddleware(RateLimits.ingest),
  zValidator("json", batchLearnSchema),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const traceId = c.get("traceId");
    const body = c.req.valid("json");

    const project = await ensureLearnProject(auth, body.project);
    const accepted: Array<{ index: number; type: string; custom_id?: string; status: string; job_id?: string; error?: string }> = [];

    // Process all items in parallel — each fires independently
    await Promise.all(
      body.items.map(async (item, index) => {
        const itemTags = [...(body.tags || []), ...(("tags" in item && item.tags) ? item.tags : [])];
        try {
          if (item.type === "text") {
            const result = await learnTextContent({
              auth,
              project,
              traceId,
              input: {
                mode: "text",
                project: project.id,
                title: item.title,
                content: item.content,
                metadata: { ...(item.metadata || {}), ...(item.custom_id ? { custom_id: item.custom_id } : {}) },
                namespace: body.namespace,
                tags: itemTags,
                options: { async: body.async !== false },
              },
            });
            accepted.push({ index, type: "text", custom_id: item.custom_id, status: "queued", job_id: result.job_id });

          } else if (item.type === "url") {
            const result = await learnSourceContent({
              auth,
              project,
              traceId,
              input: {
                mode: "source",
                project: project.id,
                type: "web",
                url: item.url,
                name: item.title,
                options: {},
              } as any,
            });
            accepted.push({ index, type: "url", custom_id: item.custom_id, status: "queued", job_id: result.job_id ?? undefined });

          } else if (item.type === "conversation") {
            // Ingest conversation sessions directly into memory
            const { ingestSession: ingestMemSession } = await import("../engine/memory/index.js");
            void ingestMemSession({
              sessionId: item.session_id,
              projectId: project.id,
              orgId: auth.orgId,
              userId: auth.userId ?? "batch",
              messages: item.messages.map((m) => ({
                role: m.role as "user" | "assistant",
                content: m.content,
                timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
              })),
            }).catch((err) => console.warn(`[Batch] session ${item.session_id} ingest failed:`, err.message));
            accepted.push({ index, type: "conversation", custom_id: item.custom_id, status: "queued" });
          }
        } catch (error: any) {
          accepted.push({ index, type: item.type, custom_id: ("custom_id" in item ? item.custom_id : undefined), status: "error", error: error.message });
        }
      })
    );

    const succeeded = accepted.filter((a) => a.status !== "error").length;
    const failed = accepted.filter((a) => a.status === "error").length;


    return c.json({
      success: true,
      accepted: succeeded,
      failed,
      total: body.items.length,
      items: accepted,
      trace_id: traceId,
    }, 202);
  }
);

// ─── Projects ────────────────────────────────────────────────

// ─── Index API (high-level marketing facade) ───────────────────────────────────

api.post(
  "/v1/index",
  rateLimitMiddleware(RateLimits.ingest),
  zValidator("json", indexApiRequestSchema),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const traceId = c.get("traceId");
    const body = c.req.valid("json");
    const endpoint = "/v1/index";

    const replay = await loadMutationReplay(c, auth, endpoint, body as Record<string, any>);
    if (replay.replay?.type === "conflict") {
      return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
    }
    if (replay.replay?.type === "hit") {
      c.header("x-idempotency-replay", "true");
      return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
    }

    const project = await ensureLearnProject(auth, body.project);
    const results: Array<Record<string, any>> = [];
    const topLevelAutoIndex = body.auto_index ?? true;

    for (const src of body.sources) {
      const effectiveAutoIndex = src.auto_index ?? topLevelAutoIndex;

      if (src.type === "local") {
        const res = await learnSourceContent({
          auth,
          project,
          traceId,
          input: {
            mode: "source",
            project: project.id,
            type: "local",
            name: src.name,
            metadata: src.metadata,
            path: src.path,
            options: src.options,
          } as any,
        });
        results.push({ type: src.type, ...res });
        continue;
      }

      if (src.type === "github-tarball") {
        if (!src.owner || !src.repo) {
          return c.json({ error: "github-tarball requires owner and repo", trace_id: traceId }, 400);
        }
        const source = await prisma.source.create({
          data: {
            projectId: project.id,
            orgId: auth.orgId,
            name: src.name?.trim() || `${src.owner}/${src.repo}`,
            connectorType: "github-tarball",
            type: "github-tarball",
            config: {
              owner: src.owner,
              repo: src.repo,
              branch: src.branch || "main",
              ...(src.token ? { token: src.token } : {}),
            },
            syncSchedule: src.sync_schedule,
            status: "PENDING",
          },
        });
        let jobId: string | null = null;
        let status = "created";
        if (effectiveAutoIndex) {
          const queued = await enqueueSync(source.id, {
            traceId,
            parentTraceId: traceId,
            reuseExisting: true,
          });
          jobId = queued.jobId;
          status = "queued";
        }
        results.push({
          type: src.type,
          success: true,
          mode: "source",
          project: project.id,
          source_id: source.id,
          status,
          job_id: jobId,
          index_started: effectiveAutoIndex,
        });
        continue;
      }

      if (["github", "web", "url", "playwright", "pdf", "slack", "video"].includes(src.type)) {
        const res = await learnSourceContent({
          auth,
          project,
          traceId,
          input: {
            mode: "source",
            project: project.id,
            type: src.type as any,
            name: src.name,
            metadata: src.metadata as any,
            owner: src.owner,
            repo: src.repo,
            branch: src.branch,
            paths: src.paths,
            url: src.url,
            file_path: src.file_path,
            channel_ids: src.channel_ids,
            since: src.since,
            token: src.token,
            auth_ref: src.auth_ref,
            platform: src.platform,
            language: src.language,
            options: {
              ...(src.options || {}),
              auto_index: effectiveAutoIndex,
            } as any,
          } as any,
        });
        results.push({ type: src.type, ...res });
        continue;
      }

      const baseName = src.name?.trim() || `${src.type}-${nanoid(8).toLowerCase()}`;
      const config = mergeIndexSourceConfig(src);

      let createdSource: { id: string };
      try {
        createdSource = await prisma.source.create({
          data: {
            orgId: project.orgId,
            projectId: project.id,
            name: baseName,
            type: src.type,
            connectorType: src.type,
            config,
            syncSchedule: src.sync_schedule,
            status: "PENDING",
          },
          select: { id: true },
        });
      } catch (error: any) {
        if (error?.code === "P2002") {
          createdSource = await prisma.source.create({
            data: {
              orgId: project.orgId,
              projectId: project.id,
              name: `${baseName}-${nanoid(6).toLowerCase()}`,
              type: src.type,
              connectorType: src.type,
              config,
              syncSchedule: src.sync_schedule,
              status: "PENDING",
            },
            select: { id: true },
          });
        } else {
          throw error;
        }
      }

      let jobId: string | null = null;
      let status = "created";
      if (effectiveAutoIndex) {
        const queued = await enqueueSync(createdSource.id, {
          traceId,
          parentTraceId: traceId,
          reuseExisting: true,
        });
        jobId = queued.jobId;
        status = "queued";
      }

      results.push({
        type: src.type,
        success: true,
        mode: "source",
        project: project.id,
        source_id: createdSource.id,
        status,
        job_id: jobId,
        index_started: effectiveAutoIndex,
      });
    }

    let bundle: any = null;
    if (body.return_bundle) {
      bundle = await exportIndexBundle({
        orgId: auth.orgId,
        projectId: project.id,
        options: body.bundle
          ? {
              include: body.bundle.include as any,
              limits: body.bundle.limits as any,
              redactSecrets: body.bundle.redact_secrets,
            }
          : undefined,
      });
    }

    const responseBody = bigIntJson({
      success: true,
      project: {
        id: project.id,
        name: project.name,
        slug: project.slug,
      },
      results,
      ...(bundle ? { bundle } : {}),
      bundle_endpoint: "/v1/index/bundle",
      trace_id: traceId,
    });

    await storeMutationReplay({
      auth,
      endpoint,
      idempotencyKey: replay.idempotencyKey,
      requestHash: replay.requestHash,
      statusCode: 202,
      body: responseBody,
      ttlSeconds: 7 * 24 * 60 * 60,
    });

    return c.json(responseBody, 202);
  }
);

api.post(
  "/v1/index/bundle",
  rateLimitMiddleware(RateLimits.query),
  zValidator("json", indexBundleRequestSchema),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const traceId = c.get("traceId");
    const body = c.req.valid("json");

    const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);

    const bundle = await exportIndexBundle({
      orgId: auth.orgId,
      projectId: project.id,
      options: {
        include: body.include as any,
        limits: body.limits as any,
        redactSecrets: body.redact_secrets,
      },
    });

    const filename = `retaindb-index-${String(project.slug || project.name || project.id).replace(/[^a-zA-Z0-9-_]+/g, "-")}.json`;
    c.header("content-type", "application/json; charset=utf-8");
    c.header("content-disposition", `attachment; filename=\"${filename}\"`);
    return c.json(bigIntJson({ success: true, bundle, trace_id: traceId }));
  }
);


api.get("/v1/projects", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const orgFilter = c.req.query("orgId");
  const results = await prisma.project.findMany({
    where: auth.isAdmin
      ? (orgFilter ? { orgId: orgFilter } : {})
      : { orgId: auth.orgId },
  });
  return c.json(bigIntJson({ projects: results }));
});

api.get("/v1/projects/resolve", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const projectRef = (c.req.query("project") || "").trim();

  if (!projectRef) {
    return c.json({ error: "project query param is required" }, 400);
  }

  const project = await resolveProjectReference(auth.orgId, projectRef, auth.isAdmin);
  if (!project) return c.json({ error: "Project not found" }, 404);

  return c.json(bigIntJson({
    input: projectRef,
    resolved: {
      id: project.id,
      name: project.name,
      slug: project.slug,
      orgId: project.orgId,
    },
  }));
});

api.get("/v1/projects/:id", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const projectRef = c.req.param("id");

  const project = await resolveProjectReference(auth.orgId, projectRef, auth.isAdmin);
  if (!project) return c.json(bigIntJson({ error: "Project not found" }), 404);

  const hydratedProject = await prisma.project.findFirst({
    where: { id: project.id },
    include: {
      sources: {
        select: sourceSummarySelect,
      },
    },
  });

  if (!hydratedProject) return c.json(bigIntJson({ error: "Project not found" }), 404);

  const { sources, ...projectData } = hydratedProject;
  return c.json(bigIntJson({ ...projectData, sources }));
});

api.delete("/v1/projects/:id", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const projectRef = c.req.param("id");

  const project = await resolveProjectReference(auth.orgId, projectRef, auth.isAdmin);

  if (!project) return c.json(bigIntJson({ error: "Project not found" }), 404);

  await prisma.project.delete({ where: { id: project.id } });
  return c.json(bigIntJson({ deleted: true }));
});

api.get("/v1/projects/:id/stats", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const projectRef = c.req.param("id");

  const project = await resolveProjectReference(auth.orgId, projectRef, auth.isAdmin);

  if (!project) return c.json(bigIntJson({ error: "Project not found" }), 404);

  const [
    documentCount,
    chunkCount,
    sourceCount,
    memoryCount,
    entityCount,
  ] = await Promise.all([
    prisma.document.count({ where: { projectId: project.id } }),
    prisma.chunk.count({ where: { document: { projectId: project.id } } }),
    prisma.source.count({ where: { projectId: project.id } }),
    prisma.memory.count({ where: { projectId: project.id, isActive: true } }),
    prisma.entity.count({ where: { projectId: project.id } }),
  ]);

  return c.json({
    documents: Number(documentCount),
    chunks: Number(chunkCount),
    sources: Number(sourceCount),
    memories: Number(memoryCount),
    entities: Number(entityCount),
  });
});

// ─── Sources ─────────────────────────────────────────────────

api.get("/v1/sources", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const projectRef = (c.req.query("project") || "").trim();

  if (!projectRef) {
    return c.json({ error: "project query param is required" }, 400);
  }

  const project = await resolveProjectReference(auth.orgId, projectRef, auth.isAdmin);
  if (!project) return c.json({ error: "Project not found" }, 404);

  const srcs = await prisma.source.findMany({
    where: { projectId: project.id },
    select: sourceSummarySelect,
  });

  return c.json(bigIntJson({
    project: {
      id: project.id,
      name: project.name,
      slug: project.slug,
    },
    sources: srcs.map((s) => serializeSource(s)),
  }));
});

api.post(
  "/v1/projects/:projectId/sources",
  zValidator(
    "json",
    z.object({
      name: z.string().min(1),
      connector_type: z.enum([
        "github", "gitlab", "url", "sitemap", "text", "pdf", "web",
        "playwright", "api_spec", "dataset", "database", "confluence", "notion",
        "slack", "discord", "arxiv", "huggingface",
        "npm_package", "pypi_package", "custom", "video",
      ]),
      config: z.record(z.any()),
      sync_schedule: z.string().optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const projectRef = c.req.param("projectId");
    const body = c.req.valid("json");
    const traceId = c.get("traceId");
    const endpoint = "/v1/projects/:projectId/add_source";
    const replay = await loadMutationReplay(c, auth, endpoint, {
      projectId: projectRef,
      ...body,
    });
    if (replay.replay?.type === "conflict") {
      return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
    }
    if (replay.replay?.type === "hit") {
      c.header("x-idempotency-replay", "true");
      return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
    }

    const project = await resolveProjectReference(auth.orgId, projectRef, auth.isAdmin);

    if (!project) return c.json(bigIntJson({ error: "Project not found" }), 404);

    try {
      if (body.connector_type === "video") {
        const videoUrl = String((body.config as any)?.url || "").trim();
        if (!videoUrl) {
          return c.json(bigIntJson({ error: "video config.url is required" }), 400);
        }
        try {
          const parsed = new URL(videoUrl);
          if (!["http:", "https:"].includes(parsed.protocol)) {
            return c.json(bigIntJson({ error: "video config.url must be http(s)" }), 400);
          }
        } catch {
          return c.json(bigIntJson({ error: "video config.url must be a valid URL" }), 400);
        }
      }

      const source = await prisma.source.create({
        data: {
          orgId: project.orgId,
          projectId: project.id,
          name: body.name,
          type: body.connector_type,
          connectorType: body.connector_type,
          config: body.config,
          syncSchedule: body.sync_schedule,
          status: "PENDING",
        },
        select: sourceSummarySelect,
      });

      const responseBody = bigIntJson(serializeSource(source));
      await storeMutationReplay({
        auth,
        endpoint,
        idempotencyKey: replay.idempotencyKey,
        requestHash: replay.requestHash,
        statusCode: 201,
        body: responseBody,
      });
      return c.json(responseBody, 201);
    } catch (error: any) {
      if (error.code === 'P2002') {
        return c.json(bigIntJson({ error: `A source with name "${body.name}" already exists` }), 409);
      }
      throw error;
    }
  }
);

api.post(
  "/v1/projects/:projectId/add_source",
  zValidator(
    "json",
    z.object({
      type: z.enum(["video"]),
      url: z.string().url(),
      auto_sync: z.boolean().optional().default(true),
      tags: z.array(z.string()).optional(),
      platform: z.enum(["youtube", "loom", "generic"]).optional(),
      language: z.string().optional(),
      allow_stt_fallback: z.boolean().optional().default(true),
      max_duration_minutes: z.number().int().min(1).max(600).optional().default(180),
      name: z.string().optional(),
      ingestion_profile: z.enum(["auto", "repo", "web_docs", "pdf_layout", "video_transcript", "plain_text"]).optional(),
      strategy_override: z.enum(["fixed", "recursive", "semantic", "hierarchical", "adaptive"]).optional(),
      profile_config: z.record(z.any()).optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const projectRef = c.req.param("projectId");
    const body = c.req.valid("json");
    const traceId = c.get("traceId");
    const endpoint = "/v1/projects/:projectId/sources";
    const replay = await loadMutationReplay(c, auth, endpoint, {
      projectId: projectRef,
      ...body,
    });
    if (replay.replay?.type === "conflict") {
      return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
    }
    if (replay.replay?.type === "hit") {
      c.header("x-idempotency-replay", "true");
      return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
    }

    const project = await resolveProjectReference(auth.orgId, projectRef, auth.isAdmin);
    if (!project) return c.json(bigIntJson({ error: "Project not found" }), 404);

    try {
      const result = await learnSourceContent({
        auth,
        project,
        traceId,
        input: {
          mode: "source",
          project: project.id,
          type: "video",
          name: body.name,
          url: body.url,
          platform: body.platform,
          language: body.language,
          metadata: body.tags?.length ? { tags: body.tags } as Record<string, any> : undefined,
          options: {
            async: true,
            auto_index: body.auto_sync,
            ingestion_profile: body.ingestion_profile,
            strategy_override: body.strategy_override,
            profile_config: body.profile_config,
            allow_stt_fallback: body.allow_stt_fallback,
            max_duration_minutes: body.max_duration_minutes,
          },
        },
      });

      const responseBody = bigIntJson({
        source_id: result.source_id,
        sync_job_id: result.job_id ?? null,
        status:
          result.status === "queued" || result.status === "processing"
            ? result.status
            : "created",
      });
      await storeMutationReplay({
        auth,
        endpoint,
        idempotencyKey: replay.idempotencyKey,
        requestHash: replay.requestHash,
        statusCode: body.auto_sync ? 202 : 201,
        body: responseBody,
        ttlSeconds: body.auto_sync ? 7 * 24 * 60 * 60 : undefined,
      });
      return c.json(responseBody, body.auto_sync ? 202 : 201);
    } catch (error: any) {
      const parsed = new URL(body.url);
      const sourceName =
        body.name?.trim() ||
        `video:${parsed.hostname}${parsed.pathname === "/" ? "" : parsed.pathname.slice(0, 60)}`;
      if (error.code === "P2002") {
        return c.json(bigIntJson({ error: `A source with name "${sourceName}" already exists` }), 409);
      }
      throw error;
    }
  }
);

api.get("/v1/projects/:projectId/sources", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const projectRef = c.req.param("projectId");

  const project = await resolveProjectReference(auth.orgId, projectRef, auth.isAdmin);

  if (!project) return c.json({ error: "Project not found" }, 404);

  const srcs = await prisma.source.findMany({
    where: { projectId: project.id },
    select: sourceSummarySelect,
  });

  return c.json(bigIntJson({ sources: srcs.map((s) => serializeSource(s)) }));
});

api.put(
  "/v1/sources/:sourceId",
  zValidator(
    "json",
    z.object({
      name: z.string().min(1).optional(),
      config: z.record(z.any()).optional(),
      sync_schedule: z.string().optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const sourceId = c.req.param("sourceId");
    const body = c.req.valid("json");

    const source = await prisma.source.findFirst({
      where: { id: sourceId },
      include: { project: true },
    });

    if (!source) return c.json(bigIntJson({ error: "Source not found" }), 404);
    if (!auth.isAdmin && (!source.project || source.project.orgId !== auth.orgId)) {
      return c.json(bigIntJson({ error: "Not authorized" }), 403);
    }

    const updated = await prisma.source.update({
      where: { id: sourceId },
      data: {
        name: body.name,
        config: body.config,
        syncSchedule: body.sync_schedule,
        updatedAt: new Date(),
      },
    });

    return c.json(bigIntJson(updated));
  }
);

api.delete("/v1/sources/:sourceId", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const sourceId = c.req.param("sourceId");
  const traceId = c.get("traceId");

  const source = await prisma.source.findFirst({
    where: { id: sourceId },
    include: { project: true },
  });

  if (!source) return c.json(bigIntJson({ error: "Source not found" }), 404);
  if (!auth.isAdmin && (!source.project || source.project.orgId !== auth.orgId)) {
    return c.json(bigIntJson({ error: "Not authorized" }), 403);
  }

  await softDeleteSource({
    sourceId,
    retentionDays: 90,
  });
  fireWebhookEvent(auth.orgId, "source.deleted", {
    sourceId,
    projectId: source.projectId,
    sourceName: source.name,
    connectorType: source.connectorType,
    deleted_at: new Date().toISOString(),
  }, {
    traceId,
    parentTraceId: traceId,
  });
  return c.json(bigIntJson({
    deleted: true,
    trace_id: traceId,
  }));
});

api.post("/v1/sources/:sourceId/restore", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const sourceId = c.req.param("sourceId");
  const traceId = c.get("traceId");
  const endpoint = "/v1/sources/:sourceId/restore";
  const replay = await loadMutationReplay(c, auth, endpoint, { sourceId });
  if (replay.replay?.type === "conflict") {
    return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
  }
  if (replay.replay?.type === "hit") {
    c.header("x-idempotency-replay", "true");
    return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
  }

  const source = await prisma.source.findFirst({
    where: { id: sourceId },
    include: { project: true },
  });
  if (!source) return c.json(bigIntJson({ error: "Source not found" }), 404);
  if (!auth.isAdmin && (!source.project || source.project.orgId !== auth.orgId)) {
    return c.json(bigIntJson({ error: "Not authorized" }), 403);
  }

  const restored = await restoreSource(sourceId);
  fireWebhookEvent(auth.orgId, "source.restored", {
    sourceId,
    projectId: source.projectId,
    sourceName: source.name,
    connectorType: source.connectorType,
    restored_at: new Date().toISOString(),
  }, {
    traceId,
    parentTraceId: traceId,
  });
  const responseBody = bigIntJson({
    restored: true,
    source: serializeSource(restored),
    trace_id: traceId,
  });
  await storeMutationReplay({
    auth,
    endpoint,
    idempotencyKey: replay.idempotencyKey,
    requestHash: replay.requestHash,
    statusCode: 200,
    body: responseBody,
  });
  return c.json(responseBody);
});

// ─── Sync ────────────────────────────────────────────────────

import { enqueueSync, getJobStatus, cancelJob, getActiveJobForSource } from "../engine/sync-queue.js";

function mapSyncStage(source: any, job: any): "extracting" | "transcribing" | "segmenting" | "enriching" | "indexing" | "completed" | "failed" {
  const progressMessage = String(job?.progress?.message || "");
  const stageMatch = progressMessage.match(/\[stage:([a-z_]+)\]/i);
  if (stageMatch?.[1]) {
    const mapped = stageMatch[1].toLowerCase();
    if (["extracting", "transcribing", "segmenting", "enriching", "indexing", "completed", "failed"].includes(mapped)) {
      return mapped as any;
    }
  }
  if (job?.status === "failed" || source?.status === "ERROR") return "failed";
  if (job?.status === "completed" || source?.status === "READY") return "completed";
  if (source?.status === "INDEXING" || source?.status === "CONNECTING") return "indexing";
  return "extracting";
}

const syncRequestSchema = z.object({
  mode: z.enum(["incremental", "full"]).optional().default("incremental"),
});

api.post(
  "/v1/sources/:sourceId/sync",
  rateLimitMiddleware(RateLimits.sync),
  async (c) => {
  const auth = c.get("auth") as AuthContext;
  const sourceId = c.req.param("sourceId");
  const traceId = c.get("traceId");
  const rawBody = await c.req.json().catch(() => ({}));
  const parsed = syncRequestSchema.safeParse(rawBody || {});
  if (!parsed.success) {
    return c.json({ error: "Invalid sync request", details: parsed.error.flatten(), trace_id: traceId }, 400);
  }
  const body = parsed.data;
  const mode = body.mode || "incremental";
  if (!sourceId) return c.json({ error: "Source ID is required", trace_id: traceId }, 400);
  const endpoint = "/v1/sources/:sourceId/sync";
  const replay = await loadMutationReplay(c, auth, endpoint, { sourceId, mode });
  if (replay.replay?.type === "conflict") {
    return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
  }
  if (replay.replay?.type === "hit") {
    c.header("x-idempotency-replay", "true");
    return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
  }

  const source = await prisma.source.findFirst({
    where: { id: sourceId },
    include: { project: true },
  });

  if (!source) return c.json(bigIntJson({ error: "Source not found" }), 404);
  if (!auth.isAdmin && (!source.project || source.project.orgId !== auth.orgId)) {
    return c.json(bigIntJson({ error: "Not authorized" }), 403);
  }

  const projectId = source?.project?.id;

  try {
    // Enqueue async sync job
    const queued = await enqueueSync(sourceId, {
      traceId,
      parentTraceId: traceId,
      reuseExisting: true,
      mode,
    });
    const jobId = queued.jobId;

    // Update source status to pending
    await prisma.source.update({
      where: { id: sourceId },
      data: { 
        status: "INDEXING", 
        updatedAt: new Date(),
      },
    });

    fireWebhookEvent(auth.orgId, "source.sync_queued", {
      sourceId,
      projectId,
      connectorType: source.connectorType,
      jobId,
      mode: queued.mode,
    }, {
      traceId,
      parentTraceId: traceId,
    });

    const responseBody = bigIntJson({ 
      status: "queued", 
      job_id: jobId,
      source_version_id: queued.sourceVersionId,
      mode: queued.mode,
      message: queued.reused
        ? "Existing sync job already running for this source."
        : "Sync job queued and will run asynchronously. Check source status for progress."
    });
    await storeMutationReplay({
      auth,
      endpoint,
      idempotencyKey: replay.idempotencyKey,
      requestHash: replay.requestHash,
      statusCode: 202,
      body: responseBody,
      ttlSeconds: 7 * 24 * 60 * 60,
    });
    return c.json(responseBody, 202);
  } catch (err: any) {
    const message = err?.message || "Failed to queue sync";
    const duplicateMatch = typeof message === "string" ? message.match(/^SYNC_ALREADY_RUNNING:(.+)$/) : null;
    const modeConflictMatch = typeof message === "string" ? message.match(/^SYNC_MODE_CONFLICT:(.+?):(.+)$/) : null;
    if (duplicateMatch) {
      return c.json(
        bigIntJson({
          error: "Sync already running for this source",
          job_id: duplicateMatch[1],
        }),
        409
      );
    }
    if (modeConflictMatch) {
      return c.json(
        bigIntJson({
          error: "A sync job with a different mode is already running for this source",
          job_id: modeConflictMatch[1],
          active_mode: modeConflictMatch[2],
        }),
        409
      );
    }

    await prisma.source.update({
      where: { id: sourceId },
      data: {
        status: "ERROR",
        syncError: message,
        updatedAt: new Date(),
      },
    });

    fireWebhookEvent(auth.orgId, "source.failed", {
      sourceId,
      projectId,
      sourceName: source.name,
      connectorType: source.connectorType,
      error: message,
    }, {
      traceId,
      parentTraceId: traceId,
    });

    return c.json(bigIntJson({ error: message }), 500);
  }
  }
);

api.get("/v1/sources/:sourceId/versions", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const sourceId = c.req.param("sourceId");
  const source = await prisma.source.findFirst({
    where: { id: sourceId },
    include: { project: true },
  });
  if (!source) return c.json(bigIntJson({ error: "Source not found" }), 404);
  if (!auth.isAdmin && (!source.project || source.project.orgId !== auth.orgId)) {
    return c.json(bigIntJson({ error: "Not authorized" }), 403);
  }
  return c.json(bigIntJson({
    source_id: sourceId,
    active_version: source.activeVersionId || null,
    versions: await listSourceVersions(sourceId),
  }));
});

api.get("/v1/sources/:sourceId/versions/:versionId", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const sourceId = c.req.param("sourceId");
  const versionId = c.req.param("versionId");
  const source = await prisma.source.findFirst({
    where: { id: sourceId },
    include: { project: true },
  });
  if (!source) return c.json(bigIntJson({ error: "Source not found" }), 404);
  if (!auth.isAdmin && (!source.project || source.project.orgId !== auth.orgId)) {
    return c.json(bigIntJson({ error: "Not authorized" }), 403);
  }
  const version = await getSourceVersion(sourceId, versionId);
  if (!version) return c.json(bigIntJson({ error: "Source version not found" }), 404);
  return c.json(bigIntJson({
    source_id: sourceId,
    active_version: source.activeVersionId || null,
    version,
  }));
});

api.get("/v1/sync-jobs/:jobId", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const jobId = c.req.param("jobId");
  const job = getJobStatus(jobId);
  if (!job) {
    return c.json(bigIntJson({ error: "Sync job not found" }), 404);
  }

  const source = await prisma.source.findFirst({
    where: { id: job.sourceId },
    include: { project: true },
  });

  if (!source) {
    return c.json(bigIntJson({ error: "Source not found" }), 404);
  }
  if (!auth.isAdmin && (!source.project || source.project.orgId !== auth.orgId)) {
    return c.json(bigIntJson({ error: "Not authorized" }), 403);
  }

  return c.json(bigIntJson(serializeSyncJob(job)));
});

api.post("/v1/sync-jobs/:jobId/cancel", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const jobId = c.req.param("jobId");
  const job = getJobStatus(jobId);
  if (!job) {
    return c.json(bigIntJson({ error: "Sync job not found" }), 404);
  }

  const source = await prisma.source.findFirst({
    where: { id: job.sourceId },
    include: { project: true },
  });

  if (!source) {
    return c.json(bigIntJson({ error: "Source not found" }), 404);
  }
  if (!auth.isAdmin && (!source.project || source.project.orgId !== auth.orgId)) {
    return c.json(bigIntJson({ error: "Not authorized" }), 403);
  }

  const cancelled = cancelJob(jobId);
  if (!cancelled) {
    return c.json(bigIntJson({ error: "Sync job is no longer running" }), 409);
  }

  return c.json(bigIntJson({
    status: "cancellation_requested",
    job_id: jobId,
  }));
});

api.get("/v1/sources/:sourceId/status", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const sourceId = c.req.param("sourceId");

  const source = await prisma.source.findFirst({
    where: { id: sourceId },
    select: {
      ...sourceSummarySelect,
      project: {
        select: { orgId: true },
      },
    },
  });
  if (!source) return c.json(bigIntJson({ error: "Source not found" }), 404);
  if (!auth.isAdmin && (!source.project || source.project.orgId !== auth.orgId)) {
    return c.json(bigIntJson({ error: "Not authorized" }), 403);
  }

  const activeJob = getActiveJobForSource(sourceId);
  const stage = mapSyncStage(source, activeJob);
  const config = (source.config || {}) as Record<string, any>;
  const result = (activeJob?.result || {}) as Record<string, any>;

  return c.json(
    bigIntJson({
      source_id: source.id,
      status: String(source.status || ""),
      stage,
      sync_job_id: activeJob?.id || null,
      progress: activeJob?.progress || null,
      duration_seconds: result.durationSeconds ?? config.duration_seconds ?? null,
      chunks_indexed: result.chunksIndexed ?? source.chunkCount ?? null,
      decisions_detected: result.decisionsDetected ?? null,
      entities_extracted: result.entitiesExtracted ?? [],
      last_error: source.syncError || source.lastSyncError || activeJob?.error || null,
      updated_at: source.updatedAt,
      partial_failure: result.partialFailure ?? source.sourceVersions?.[0]?.partialFailure ?? false,
      documents_total: result.documentsTotal ?? source.sourceVersions?.[0]?.documentCount ?? source.documentCount ?? null,
      documents_indexed: result.documentsIndexed ?? source.sourceVersions?.[0]?.documentCount ?? source.documentCount ?? null,
      documents_failed: result.documentsFailed ?? 0,
      warning_codes: result.warningCodes ?? source.sourceVersions?.[0]?.warningCodes ?? [],
      error_code: result.errorCode ?? source.sourceVersions?.[0]?.errorCode ?? null,
      outcome:
        result.outcome ||
        ((result.partialFailure ?? source.sourceVersions?.[0]?.partialFailure)
          ? "partial_failure"
          : (result.errorCode ?? source.sourceVersions?.[0]?.errorCode)
            ? "failed"
            : "success"),
      mode: result.mode ?? activeJob?.mode ?? null,
      effective_mode: result.effectiveMode ?? activeJob?.effectiveMode ?? null,
      active_version: source.activeVersionId ?? null,
      restore_until: source.restoreUntil ?? null,
      latest_version: source.sourceVersions?.[0] ? serializeSourceVersion(source.sourceVersions[0]) : null,
    })
  );
});

// ─── GitHub Tarball Sync ───────────────────────────────────────

api.post(
  "/v1/projects/:projectId/sources/github-tarball",
  rateLimitMiddleware(RateLimits.sync),
  zValidator(
    "json",
    z.object({
      owner: z.string().describe("GitHub owner/org"),
      repo: z.string().describe("GitHub repository name"),
      branch: z.string().optional().default("main"),
      token: z.string().optional().describe("GitHub token for private repos"),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const projectId = c.req.param("projectId");
    const body = c.req.valid("json");
    const traceId = c.get("traceId");
    const endpoint = "/v1/projects/:projectId/sources/github-tarball";
    const replay = await loadMutationReplay(c, auth, endpoint, {
      projectId,
      ...body,
    });
    if (replay.replay?.type === "conflict") {
      return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
    }
    if (replay.replay?.type === "hit") {
      c.header("x-idempotency-replay", "true");
      return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
    }

  const project = await resolveProjectReference(auth.orgId, projectId, auth.isAdmin);
    if (!project) {
      return c.json(bigIntJson({ error: "Project not found" }), 404);
    }

    // Create source with type github-tarball
    const source = await prisma.source.create({
      data: {
        projectId: project.id,
        orgId: auth.orgId,
        name: `${body.owner}/${body.repo}`,
        connectorType: "github-tarball",
        type: "github-tarball",
        config: {
          owner: body.owner,
          repo: body.repo,
          branch: body.branch,
          token: body.token,
        },
        status: "PENDING",
      },
    });

    // Use the async queue system for background processing
    const queued = await enqueueSync(source.id, {
      traceId,
      parentTraceId: traceId,
      reuseExisting: true,
    });

    const responseBody = bigIntJson({
      source_id: source.id,
      job_id: queued.jobId,
      source_version_id: queued.sourceVersionId,
      status: "queued",
      message: "Sync job queued. Use /v1/sources/:sourceId to check progress."
    });
    await storeMutationReplay({
      auth,
      endpoint,
      idempotencyKey: replay.idempotencyKey,
      requestHash: replay.requestHash,
      statusCode: 202,
      body: responseBody,
      ttlSeconds: 7 * 24 * 60 * 60,
    });
    return c.json(responseBody, 202);
  }
);

// ─── Direct Ingest ───────────────────────────────────────────

api.post(
  "/v1/projects/:projectId/ingest",
  rateLimitMiddleware(RateLimits.ingest),
  zValidator(
    "json",
    z.object({
      documents: z.array(
        z.object({
          id: z.string().optional(),
          title: z.string(),
          content: z.string(),
          metadata: z.record(z.any()).optional(),
          file_path: z.string().optional(),
          namespace: z.string().optional(),
          tags: z.array(z.string()).optional(),
          ingestion_profile: z.enum(["auto", "repo", "web_docs", "pdf_layout", "video_transcript", "plain_text"]).optional(),
          strategy_override: z.enum(["fixed", "recursive", "semantic", "hierarchical", "adaptive"]).optional(),
          profile_config: z.record(z.any()).optional(),
        })
      ),
      webhook_url: z.string().url().optional(),
      namespace: z.string().optional(),
      tags: z.array(z.string()).optional(),
      ingestion_profile: z.enum(["auto", "repo", "web_docs", "pdf_layout", "video_transcript", "plain_text"]).optional(),
      strategy_override: z.enum(["fixed", "recursive", "semantic", "hierarchical", "adaptive"]).optional(),
      profile_config: z.record(z.any()).optional(),
      async: z.boolean().optional().default(true), // Default to async
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const projectRef = c.req.param("projectId");
    const body = c.req.valid("json");
    const traceId = c.get("traceId");
    const endpoint = "/v1/projects/:projectId/ingest";
    const replay = await loadMutationReplay(c, auth, endpoint, {
      projectId: projectRef,
      ...body,
    });
    if (replay.replay?.type === "conflict") {
      return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
    }
    if (replay.replay?.type === "hit") {
      c.header("x-idempotency-replay", "true");
      return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
    }

    const project = await resolveProjectReference(auth.orgId, projectRef, auth.isAdmin);

    if (!project) return c.json(bigIntJson({ error: "Project not found" }), 404);

    // ASYNC BY DEFAULT (unless explicitly disabled)
    if (body.async !== false) {
      try {
        const jobId = await ingestionQueue.createJob({
          orgId: auth.orgId,
          projectId: project.id,
          userId: auth.userId || 'system',
          documents: body.documents.map(doc => ({
            title: doc.title,
            content: doc.content,
            url: doc.file_path,
            metadata: {
              ...(doc.metadata || {}),
              file_path: doc.file_path,
            },
            namespace: doc.namespace || body.namespace,
            tags: [...(doc.tags || []), ...(body.tags || [])],
            ingestion_profile: doc.ingestion_profile || body.ingestion_profile,
            strategy_override: doc.strategy_override || body.strategy_override,
            profile_config: doc.profile_config || body.profile_config,
          })),
          webhookUrl: body.webhook_url,
          namespace: body.namespace,
          tags: body.tags,
        });

        const responseBody = {
          success: true,
          mode: 'async',
          jobId,
          status: 'PROCESSING',
          statusUrl: `/v1/jobs/${jobId}`,
          webhookUrl: body.webhook_url || null,
          trace_id: traceId,
        };
        await storeMutationReplay({
          auth,
          endpoint,
          idempotencyKey: replay.idempotencyKey,
          requestHash: replay.requestHash,
          statusCode: 202,
          body: responseBody,
        });
        return c.json(responseBody, 202);

      } catch (error: any) {
        console.error("[Ingest] Async job creation failed:", error);
        return c.json({
          error: "Failed to queue ingestion job",
          details: error.message
        }, 500);
      }
    }

    // LEGACY SYNC MODE (only if explicitly requested with async: false)
    try {
      const results: any[] = [];
      const errors: Array<{ title: string; error: string }> = [];

      for (const doc of body.documents) {
        try {
          const result = await learnTextContent({
            auth,
            project,
            traceId,
            input: {
              mode: "text",
              project: project.id,
              title: doc.title,
              content: doc.content,
              metadata: {
                ...(doc.metadata || {}),
                ...(doc.file_path ? { file_path: doc.file_path } : {}),
              },
              namespace: doc.namespace || body.namespace,
              tags: [...(doc.tags || []), ...(body.tags || [])],
              options: {
                async: false,
                ingestion_profile: doc.ingestion_profile || body.ingestion_profile,
                strategy_override: doc.strategy_override || body.strategy_override,
                profile_config: (doc.profile_config || body.profile_config) as any,
              },
            },
          });
          results.push({
            title: doc.title,
            chunks_indexed: result.chunks_indexed || 0,
          });
        } catch (error: any) {
          errors.push({
            title: doc.title,
            error: error?.message || "Unknown ingest error",
          });
        }
      }


      return c.json(bigIntJson({
        mode: 'sync',
        ingested: results.length,
        failed: errors.length,
        errors,
      }));
    } catch (error: any) {
      console.error("[Ingest] Fatal error:", error?.message || error);
      return c.json(bigIntJson({
        error: "Ingest failed",
        details: error?.message || "Unknown error",
      }), 500);
    }
  }
);

// ─── Memories (DEPRECATED - Use /v1/memory instead) ──────────
// ⚠️ LEGACY ENDPOINTS - Use SOTA Memory API at /v1/memory/* instead
// These endpoints are deprecated and will be removed in v3.0
// Migration guide: Use POST /v1/memory for single memory creation

api.post(
  "/v1/memories",
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      content: z.string().min(1).max(10000),
      memory_type: z.enum(["factual", "episodic", "semantic", "procedural"]).optional().default("factual"),
      user_id: z.string().optional(),
      session_id: z.string().optional(),
      agent_id: z.string().optional(),
      importance: z.number().min(0).max(1).optional().default(0.5),
      metadata: z.record(z.any()).optional(),
      expires_in_seconds: z.number().int().positive().optional(),
      webhook_url: z.string().url().optional(),
      namespace: z.string().optional(),
      tags: z.array(z.string()).optional(),
      async: z.boolean().optional().default(true), // Default to async
    })
  ),
  async (c) => {
    markLegacyMemoryRoute(c);
    try {
      const auth = c.get("auth") as AuthContext;
      const body = c.req.valid("json");

      const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);

      // ASYNC BY DEFAULT (unless explicitly disabled)
      if (body.async !== false) {
        try {
          const jobId = await ingestionQueue.createJob({
            orgId: auth.orgId,
            projectId: project.id,
            userId: auth.userId || 'system',
            memories: [{
              content: body.content,
              memory_type: body.memory_type,
              user_id: body.user_id,
              session_id: body.session_id,
              agent_id: body.agent_id,
              importance: body.importance,
              metadata: {
                ...body.metadata,
                namespace: body.namespace,
                tags: body.tags,
              },
              expires_in_seconds: body.expires_in_seconds,
            }],
            webhookUrl: body.webhook_url,
            namespace: body.namespace,
            tags: body.tags,
          });

          return c.json({
            success: true,
            mode: 'async',
            jobId,
            status: 'PROCESSING',
            statusUrl: `/v1/jobs/${jobId}`,
            webhookUrl: body.webhook_url || null,
          }, 202);

        } catch (error: any) {
          console.error("[Memory] Async job creation failed:", error);
          return c.json({
            error: "Failed to queue memory creation",
            details: error.message
          }, 500);
        }
      }

      const writeResult = await writeMemoryCanonical({
        projectId: project.id,
        orgId: auth.orgId,
        userId: body.user_id,
        sessionId: body.session_id,
        agentId: body.agent_id,
        content: body.content,
        memoryType: body.memory_type,
        importance: body.importance,
        confidenceRaw: 0.9,
        metadata: {
          ...(body.metadata || {}),
          namespace: body.namespace,
          tags: body.tags,
        },
        expiresAt: body.expires_in_seconds
          ? new Date(Date.now() + body.expires_in_seconds * 1000)
          : null,
        writeSource: "api.legacy.memories",
        writeMode: "direct_write",
        extractionMethod: "manual",
      });

      if (!writeResult.memory || writeResult.outcome === "dropped") {
        return c.json(bigIntJson({
          error: "Memory write rejected by canonical validation policy",
          validator_issues: writeResult.validatorIssues,
        }), 422);
      }


      return c.json(bigIntJson({
        mode: 'sync',
        memory: writeResult.memory,
        write_outcome: writeResult.outcome,
        scope_decision: writeResult.scopeDecision,
      }), writeResult.outcome === "created" ? 201 : 200);
    } catch (error: any) {
      console.error("Memory creation error:", error);
      console.error("Error details:", error.message, error.stack);
      return c.json(bigIntJson({ error: "Failed to create memory", details: error.message }), 500);
    }
  }
);

api.post(
  "/v1/memories/search",
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      query: z.string().min(1).max(5000),
      user_id: z.string().optional(),
      session_id: z.string().optional(),
      agent_id: z.string().optional(),
      memory_type: z.enum(["factual", "episodic", "semantic", "procedural"]).optional(),
      top_k: z.number().int().min(1).max(50).optional().default(10),
    })
  ),
  async (c) => {
    markLegacyMemoryRoute(c);
    const auth = c.get("auth") as AuthContext;
    const body = c.req.valid("json");

    const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);

    const queryEmbedding = await embedSingle(body.query);

    // Parameterized SQL (avoid interpolating user input into WHERE clauses).
    const conditions: any[] = [
      Prisma.sql`"projectId" = ${project.id}`,
      Prisma.sql`"isActive" = true`,
      Prisma.sql`("expiresAt" IS NULL OR "expiresAt" > NOW())`,
    ];
    if (body.user_id) conditions.push(Prisma.sql`"userId" = ${body.user_id}`);
    if (body.session_id) conditions.push(Prisma.sql`"sessionId" = ${body.session_id}`);
    if (body.agent_id) conditions.push(Prisma.sql`"agentId" = ${body.agent_id}`);
    if (body.memory_type) conditions.push(Prisma.sql`"memoryType" = ${body.memory_type}`);

    const whereSql = Prisma.join(conditions, " AND ");

    const memories = await prisma.$queryRaw(Prisma.sql`
      SELECT
        id, content, "memoryType", "userId",
        "sessionId", "agentId", importance,
        metadata, "accessCount", "createdAt",
        1 - (embedding <=> ${queryEmbedding}::vector) as similarity
      FROM memories
      WHERE ${whereSql}
      ORDER BY embedding <=> ${queryEmbedding}::vector
      LIMIT ${body.top_k}
    `);

    // Update access counts
    const memoryIds = (memories as any[]).map((r: any) => r.id);
    if (memoryIds.length > 0) {
      await prisma.memory.updateMany({
        where: { id: { in: memoryIds } },
        data: {
          accessCount: { increment: 1 },
          lastAccessedAt: new Date(),
        },
      });
    }


    return c.json({
      memories: (memories as any[]).map((r: any) => ({
        ...r,
        score: Math.round(r.similarity * 1000) / 1000,
      })),
    });
  }
);

api.get("/v1/memories", async (c) => {
  markLegacyMemoryRoute(c);
  const auth = c.get("auth") as AuthContext;
  const projectName = c.req.query("project");
  const userId = c.req.query("user_id");
  const sessionId = c.req.query("session_id");
  const agentId = c.req.query("agent_id");
  const limit = Math.min(parseInt(c.req.query("limit") || "50"), 200);

  const project = await ensureProject(auth.orgId, projectName, auth.isAdmin);

  const memories = await prisma.memory.findMany({
    where: {
      projectId: project.id,
      isActive: true,
      ...(userId && { userId }),
      ...(sessionId && { sessionId }),
      ...(agentId && { agentId }),
    },
    orderBy: { createdAt: 'desc' },
    take: limit,
    select: {
      id: true,
      content: true,
      memoryType: true,
      userId: true,
      sessionId: true,
      agentId: true,
      importance: true,
      metadata: true,
      accessCount: true,
      createdAt: true,
      updatedAt: true,
    },
  });

  return c.json({ memories });
});

api.get("/v1/memories/:id", async (c) => {
  markLegacyMemoryRoute(c);
  const auth = c.get("auth") as AuthContext;
  const id = c.req.param("id");

  const memory = await prisma.memory.findFirst({
    where: { id, orgId: auth.orgId },
    select: {
      id: true,
      projectId: true,
      orgId: true,
      content: true,
      memoryType: true,
      userId: true,
      sessionId: true,
      agentId: true,
      importance: true,
      confidence: true,
      metadata: true,
      accessCount: true,
      createdAt: true,
      updatedAt: true,
      documentDate: true,
      eventDate: true,
      validFrom: true,
      validUntil: true,
      isActive: true,
    },
  });

  if (!memory) return c.json({ error: "Memory not found" }, 404);
  return c.json({ memory: bigIntJson(memory) });
});

api.put(
  "/v1/memories/:id",
  zValidator(
    "json",
    z.object({
      content: z.string().optional(),
      importance: z.number().min(0).max(1).optional(),
      metadata: z.record(z.any()).optional(),
    })
  ),
  async (c) => {
    markLegacyMemoryRoute(c);
    const auth = c.get("auth") as AuthContext;
    const id = c.req.param("id");
    const body = c.req.valid("json");

    // Verify ownership via project -> org
    const memory = await prisma.memory.findFirst({
      where: { id },
      include: { project: true },
    });

    if (!memory) return c.json({ error: "Memory not found" }, 404);
    if (!memory.project || memory.project.orgId !== auth.orgId) return c.json({ error: "Not authorized" }, 403);

    const updateData: any = { updatedAt: new Date() };
    if (body.content) {
      updateData.content = body.content;
      // Note: We don't update embedding for Unsupported("vector") type
    }
    if (body.importance !== undefined) updateData.importance = body.importance;
    if (body.metadata) updateData.metadata = body.metadata;

    const updated = await prisma.memory.update({
      where: { id },
      data: updateData,
    });

    return c.json(updated);
  }
);

api.delete("/v1/memories/:id", async (c) => {
  markLegacyMemoryRoute(c);
  const auth = c.get("auth") as AuthContext;
  const id = c.req.param("id");

  const memory = await prisma.memory.findFirst({
    where: { id },
    include: { project: true },
  });

  if (!memory) return c.json({ error: "Memory not found" }, 404);
  if (!memory.project || memory.project.orgId !== auth.orgId) return c.json({ error: "Not authorized" }, 403);

  await prisma.memory.update({
    where: { id },
    data: { isActive: false, updatedAt: new Date() },
  });

  return c.json({ deleted: true });
});

// ─── Conversations ───────────────────────────────────────────

api.post(
  "/v1/conversations",
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      session_id: z.string().optional(),
      user_id: z.string().optional(),
      agent_id: z.string().optional(),
      title: z.string().optional(),
      metadata: z.record(z.any()).optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const body = c.req.valid("json");

    const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);

    const conv = await prisma.session.create({
      data: {
        projectId: project.id,
        sessionId: body.session_id,
        userId: body.user_id,
        agentId: body.agent_id,
        title: body.title,
        metadata: body.metadata || {},
      },
    });

    return c.json(conv, 201);
  }
);

api.get("/v1/conversations", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const projectName = c.req.query("project");
  const userId = c.req.query("user_id");
  const sessionId = c.req.query("session_id");
  const limit = Math.min(Math.max(parseInt(c.req.query("limit") || "50", 10), 1), 100);

  const project = await ensureProject(auth.orgId, projectName, auth.isAdmin);

  const results = await prisma.session.findMany({
    where: {
      projectId: project.id,
      ...(userId && { userId }),
      ...(sessionId && { sessionId }),
    },
    orderBy: { updatedAt: 'desc' },
    take: limit,
  });

  return c.json({ conversations: results });
});

api.get("/v1/conversations/:id", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const id = c.req.param("id");

  const conv = await prisma.session.findFirst({
    where: { id },
    include: { messages: { orderBy: { createdAt: 'asc' } } },
  });

  if (!conv) return c.json({ error: "Conversation not found" }, 404);
  if (!conv.projectId) return c.json({ error: "Conversation not found" }, 404);

  const project = await prisma.project.findFirst({
    where: {
      id: conv.projectId,
      orgId: auth.orgId,
    },
  });

  if (!project) return c.json({ error: "Not authorized" }, 403);

  return c.json(conv);
});

// ─── Messages ────────────────────────────────────────────────

api.post(
  "/v1/conversations/:conversationId/messages",
  zValidator(
    "json",
    z.object({
      role: z.enum(["user", "assistant", "system", "tool"]),
      content: z.string().min(1),
      metadata: z.record(z.any()).optional(),
      auto_learn: z.boolean().optional().default(true),
      auto_extract_memories: z.boolean().optional().default(false),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const conversationId = c.req.param("conversationId");
    const body = c.req.valid("json");

    const conv = await prisma.session.findFirst({
      where: { id: conversationId },
      include: { project: true },
    });

    if (!conv) return c.json({ error: "Conversation not found" }, 404);

    if (!conv.project || conv.project.orgId !== auth.orgId) return c.json({ error: "Not authorized" }, 403);

    const msg = await prisma.message.create({
      data: {
        sessionId: conversationId,
        role: body.role,
        content: body.content,
        metadata: body.metadata || {},
      },
    });

    // Update conversation timestamp
    await prisma.session.update({
      where: { id: conversationId },
      data: {
        updatedAt: new Date(),
      },
    });

    const shouldAutoLearn = body.auto_learn ?? body.auto_extract_memories ?? true;
    if (shouldAutoLearn && conv.projectId) {
      scheduleConversationAutoLearn({
        auth,
        projectId: conv.projectId,
        sessionId: conv.sessionId || conversationId,
        userId: conv.userId || undefined,
      });
    }

    return c.json(msg, 201);
  }
);

// ─── Graph: Entities & Relations ─────────────────────────────

api.post(
  "/v1/entities",
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      name: z.string().min(1),
      entity_type: z.string().min(1),
      description: z.string().optional(),
      metadata: z.record(z.any()).optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const body = c.req.valid("json");

    const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);

    const embedding = await embedSingle(`${body.entity_type}: ${body.name}${body.description ? ` - ${body.description}` : ""}`);

    const entity = await prisma.entity.upsert({
      where: {
        projectId_name_entityType: {
          projectId: project.id,
          name: body.name,
          entityType: body.entity_type,
        },
      },
      update: {
        description: body.description,
        metadata: body.metadata || {},
        embedding,
        updatedAt: new Date(),
      } as any,
      create: {
        projectId: project.id,
        name: body.name,
        entityType: body.entity_type,
        description: body.description,
        metadata: body.metadata || {},
        embedding,
      } as any,
    });

    return c.json(entity, 201);
  }
);

api.post(
  "/v1/relations",
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      from_entity: z.string(),
      from_type: z.string(),
      to_entity: z.string(),
      to_type: z.string(),
      relation_type: z.enum([
        "imports", "exports", "calls", "implements", "extends",
        "references", "depends_on", "related_to", "part_of",
        "contradicts", "supersedes",
      ]),
      weight: z.number().min(0).max(1).optional(),
      metadata: z.record(z.any()).optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const body = c.req.valid("json");

    const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);

    // Find or create entities
    const findOrCreateEntity = async (name: string, type: string) => {
      const existing = await prisma.entity.findFirst({
        where: {
          projectId: project.id,
          name,
          entityType: type,
        },
      });

      if (existing) return existing;

      const embedding = await embedSingle(`${type}: ${name}`);
      return await prisma.entity.create({
        data: { projectId: project.id, name, entityType: type, embedding } as any,
      });
    };

    const fromEntity = await findOrCreateEntity(body.from_entity, body.from_type);
    const toEntity = await findOrCreateEntity(body.to_entity, body.to_type);

    const relation = await prisma.entityRelation.upsert({
      where: {
        fromEntityId_toEntityId_relationType: {
          fromEntityId: fromEntity.id,
          toEntityId: toEntity.id,
          relationType: body.relation_type,
        },
      },
      update: {
        weight: body.weight,
        metadata: body.metadata || {},
      },
      create: {
        projectId: project.id,
        fromEntityId: fromEntity.id,
        toEntityId: toEntity.id,
        relationType: body.relation_type,
        weight: body.weight,
        metadata: body.metadata || {},
      },
    });

    return c.json(relation, 201);
  }
);

api.post(
  "/v1/graph/search",
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      query: z.string().min(1),
      entity_types: z.array(z.string()).optional(),
      depth: z.number().int().min(1).max(3).optional().default(1),
      top_k: z.number().int().min(1).max(50).optional().default(10),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const body = c.req.valid("json");

    const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);

    const queryEmbedding = await embedSingle(body.query);

    // Entity search with proper parameterization
    const relevantEntities = await prisma.$queryRaw(Prisma.sql`
      SELECT
        id, name, "entityType", description, metadata,
        1 - (embedding <=> ${queryEmbedding}::vector) as similarity
      FROM entities
      WHERE "projectId" = ${project.id}
        ${body.entity_types && body.entity_types.length > 0 
          ? Prisma.sql`AND "entityType" IN (${Prisma.join(body.entity_types)})` 
          : Prisma.sql``}
      ORDER BY embedding <=> ${queryEmbedding}::vector
      LIMIT ${body.top_k}
    `);

    // Get relations for found entities
    const entityIds = (relevantEntities as any[]).map((e: any) => e.id);
    let rels: any[] = [];

    if (entityIds.length > 0) {
      rels = await prisma.entityRelation.findMany({
        where: {
          projectId: project.id,
          OR: [
            { fromEntityId: { in: entityIds } },
            { toEntityId: { in: entityIds } },
          ],
        },
        select: {
          id: true,
          fromEntityId: true,
          toEntityId: true,
          relationType: true,
          weight: true,
        },
      });
    }

    return c.json({
      entities: (relevantEntities as any[]).map((e: any) => ({
        ...e,
        score: Math.round(e.similarity * 1000) / 1000,
      })),
      relations: rels,
    });
  }
);

// ─── Webhooks ────────────────────────────────────────────────

api.post(
  "/v1/webhooks",
  zValidator(
    "json",
    z.object({
      url: z.string().url(),
      events: z.array(z.string()).optional(),
      secret: z.string().optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const body = c.req.valid("json");

    const webhook = await prisma.webhook.create({
      data: {
        orgId: auth.orgId,
        url: body.url,
        secret: body.secret || nanoid(32),
        events: body.events || ["source.synced", "document.indexed", "memory.created"],
      },
    });

    return c.json(webhook, 201);
  }
);

api.get("/v1/webhooks", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const results = await prisma.webhook.findMany({
    where: { orgId: auth.orgId },
  });
  return c.json({ webhooks: results });
});

api.delete("/v1/webhooks/:id", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const id = c.req.param("id");

  const webhook = await prisma.webhook.findFirst({
    where: {
      id,
      orgId: auth.orgId,
    },
  });

  if (!webhook) return c.json({ error: "Webhook not found" }, 404);

  await prisma.webhook.delete({ where: { id } });
  return c.json({ deleted: true });
});

api.post("/v1/webhooks/:id/redeliver", async (c) => {
  const auth = c.get("auth") as AuthContext;
  const traceId = c.get("traceId");
  if (!auth.isAdmin) return c.json({ error: "Admin access required", trace_id: traceId }, 403);
  const webhookId = c.req.param("id");
  const endpoint = "/v1/webhooks/:id/redeliver";
  const replay = await loadMutationReplay(c, auth, endpoint, { webhookId });
  if (replay.replay?.type === "conflict") {
    return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
  }
  if (replay.replay?.type === "hit") {
    c.header("x-idempotency-replay", "true");
    return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
  }

  // Verify the webhook belongs to this org before touching its deliveries (IDOR guard)
  const webhook = await prisma.webhook.findFirst({ where: { id: webhookId, orgId: auth.orgId } });
  if (!webhook) {
    return c.json({ error: "Webhook not found", trace_id: traceId }, 404);
  }

  const delivery = await prisma.webhookDelivery.findFirst({
    where: { webhookId },
    orderBy: { deliveredAt: "desc" },
  });
  if (!delivery) {
    return c.json({ error: "Webhook delivery not found", trace_id: traceId }, 404);
  }

  const result = await redeliverWebhookDelivery(delivery.id, {
    traceId,
    parentTraceId: delivery.traceId || traceId,
  });
  const responseBody = {
    redelivered: true,
    delivery_id: delivery.id,
    result,
    trace_id: traceId,
  };
  await storeMutationReplay({
    auth,
    endpoint,
    idempotencyKey: replay.idempotencyKey,
    requestHash: replay.requestHash,
    statusCode: 202,
    body: responseBody,
  });
  return c.json(responseBody, 202);
});

// ─── Usage / Stats ───────────────────────────────────────────

// ─── Organizations (self-service setup) ──────────────────────

// ─── API Keys ────────────────────────────────────────────────


// ─── Async Jobs ──────────────────────────────────────────────

import { ingestionQueue } from "../engine/ingestion-queue.js";

api.get("/v1/jobs/:jobId", async (c) => {
  try {
    const auth = c.get("auth") as AuthContext;
    const jobId = c.req.param("jobId");

    // Get job from database
    const job = await ingestionQueue.getJobStatus(jobId);

    if (!job) {
      return c.json({ error: "Job not found" }, 404);
    }

    if (!auth.isAdmin && job.orgId !== auth.orgId) {
      return c.json({ error: "Not authorized" }, 403);
    }

    return c.json({
      success: true,
      job: job,
    });

  } catch (error: any) {
    console.error("Job status error:", error);
    return c.json({
      error: "Failed to fetch job status",
      details: error.message
    }, 500);
  }
});

// ─── Billing Routes ──────────────────────────────────────────


api.get("/v1/admin/latency/stats", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);

  const top = Math.min(Math.max(parseInt(c.req.query("top") || "50", 10), 1), 500);
  const minCount = Math.min(Math.max(parseInt(c.req.query("min_count") || "1", 10), 1), 10000);
  const includeSlowEvents = /^true$/i.test(c.req.query("include_slow") || "false");

  return c.json(
    getLatencySummary({
      top,
      minCount,
      includeSlowEvents,
    })
  );
});

api.get("/v1/admin/latency/gates", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);

  const minCount = Math.min(Math.max(parseInt(c.req.query("min_count") || "20", 10), 1), 100000);
  return c.json(getLatencyGateStatus({ minCount }));
});

api.get("/v1/admin/latency/config", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  return c.json(getLatencyTraceConfig());
});

api.post("/v1/admin/latency/reset", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  return c.json(resetLatencySummary());
});

api.get("/v1/admin/extraction/config", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  return c.json(getExtractionPhase0Config());
});

api.get("/v1/admin/extraction/stats", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);

  const lookbackDays = Math.min(Math.max(parseInt(c.req.query("lookback_days") || "14", 10), 1), 90);
  const tenantId = c.req.query("tenant_id") || undefined;
  const projectId = c.req.query("project_id") || undefined;
  return c.json(await getExtractionStats({ lookbackDays, tenantId, projectId }));
});

api.get("/v1/admin/extraction/gates", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);

  const tenantId = c.req.query("tenant_id") || undefined;
  const minDays = Math.min(Math.max(parseInt(c.req.query("min_days") || "7", 10), 1), 60);
  const minSamples = Math.min(Math.max(parseInt(c.req.query("min_samples") || "10000", 10), 1), 1_000_000);
  const lookbackDays = Math.min(Math.max(parseInt(c.req.query("lookback_days") || "30", 10), 1), 365);

  return c.json(await getExtractionGateStatus({ tenantId, minDays, minSamples, lookbackDays }));
});

api.get("/v1/admin/extraction/alerts", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);

  const lookbackHours = Math.min(Math.max(parseInt(c.req.query("lookback_hours") || "24", 10), 1), 168);
  return c.json(await getExtractionAlerts({ lookbackHours }));
});

api.post("/v1/admin/extraction/reset", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  return c.json(resetExtractionObservability());
});

api.get("/v1/admin/ops/routes", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  return c.json({
    routes: getNodeRouteMatrix(),
    latency: getLatencySummary({ top: 100, minCount: 1, includeSlowEvents: false }),
  });
});

api.get("/v1/admin/ops/retrieval", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  return c.json(getRetrievalHealthSummary());
});

api.get("/v1/admin/ops/queues", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  return c.json(await getQueueHealthSummary());
});

api.get("/v1/admin/ops/connectors", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  const lookbackHours = Math.min(Math.max(parseInt(c.req.query("lookback_hours") || "24", 10), 1), 168);
  return c.json(await getConnectorHealthSummary({ lookbackHours }));
});

api.get("/v1/admin/ops/sources", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  const [statusCounts, deletedSources, partialFailures] = await Promise.all([
    prisma.source.groupBy({
      by: ["status"],
      _count: { _all: true },
    }),
    prisma.source.count({ where: { deletedAt: { not: null } } }),
    prisma.sourceVersion.count({ where: { partialFailure: true } }),
  ]);
  return c.json({
    statuses: statusCounts,
    deleted_sources: deletedSources,
    partial_failure_versions: partialFailures,
  });
});

api.get("/v1/admin/ops/counters", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  return c.json(await getOperationalCounters());
});

api.get("/v1/admin/ops/webhooks", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  const lookbackHours = Math.min(Math.max(parseInt(c.req.query("lookback_hours") || "24", 10), 1), 168);
  return c.json(await getWebhookFailureSummary({ lookbackHours }));
});

api.get("/v1/admin/ops/alerts", async (c) => {
  const auth = c.get("auth") as AuthContext;
  if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
  const lookbackHours = Math.min(Math.max(parseInt(c.req.query("lookback_hours") || "24", 10), 1), 168);
  return c.json(await evaluateOperationalAlerts({ lookbackHours }));
});



api.post(
  "/v1/admin/sources/rehydrate",
  zValidator(
    "json",
    z.object({
      org_id: z.string().optional(),
      project_id: z.string().optional(),
      source_ids: z.array(z.string()).optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    if (!auth.isAdmin) return c.json({ error: "Admin access required" }, 403);
    const body = c.req.valid("json");
    const traceId = c.get("traceId");
    const endpoint = "/v1/admin/sources/rehydrate";
    const replay = await loadMutationReplay(c, auth, endpoint, body);
    if (replay.replay?.type === "conflict") {
      return c.json({ error: "Idempotency payload mismatch", trace_id: traceId }, 409);
    }
    if (replay.replay?.type === "hit") {
      c.header("x-idempotency-replay", "true");
      return c.json(bigIntJson(replay.replay.body), replay.replay.statusCode as any);
    }

    const staleRecovered = await markStaleSourceVersionsFailed();
    const sources = await prisma.source.findMany({
      where: {
        orgId: body.org_id || auth.orgId,
        ...(body.project_id ? { projectId: body.project_id } : {}),
        ...(body.source_ids?.length ? { id: { in: body.source_ids } } : {}),
        deletedAt: null,
      },
    });
    const jobs: any[] = [];
    for (const source of sources) {
      const queued = await enqueueSync(source.id, {
        traceId,
        parentTraceId: traceId,
        reuseExisting: true,
      });
      jobs.push({
        source_id: source.id,
        job_id: queued.jobId,
        source_version_id: queued.sourceVersionId,
        reused: queued.reused,
      });
      fireWebhookEvent(source.orgId, "source.rehydrated", {
        sourceId: source.id,
        sourceName: source.name,
        connectorType: source.connectorType,
        projectId: source.projectId,
        jobId: queued.jobId,
      }, {
        traceId,
        parentTraceId: traceId,
      });
    }

    const responseBody = {
      rehydrated: jobs.length,
      stale_versions_failed: staleRecovered,
      jobs,
      trace_id: traceId,
    };
    await storeMutationReplay({
      auth,
      endpoint,
      idempotencyKey: replay.idempotencyKey,
      requestHash: replay.requestHash,
      statusCode: 202,
      body: responseBody,
      ttlSeconds: 7 * 24 * 60 * 60,
    });
    return c.json(responseBody, 202);
  }
);

// ─── SimpleClaw Provisioning (Admin only) ────────────────────
