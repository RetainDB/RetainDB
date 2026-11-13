/**
 * SOTA Memory API Routes
 * Enhanced memory endpoints for temporal reasoning, relational graphs, and knowledge updates
 */

import { Hono } from "hono";
import { zValidator } from "@hono/zod-validator";
import { z } from "zod";
import crypto from "crypto";
import { prisma } from "../db/index.js";
import type { AuthContext } from "../middleware/auth.js";
import { rateLimitMiddleware, RateLimits } from "../middleware/rate-limit.js";
import { dialecticQuery } from "../engine/memory/dialectic.js";
import { synthesizeAgentModel, seedAgentIdentity } from "../engine/memory/agent-model.js";
import { extractExplicitMemory } from "../engine/memory/patterns.js";
import {
  searchMemories,
  ingestSession,
  updateMemory,
  getSessionMemories,
  getUserProfile,
  getVersionChain,
  synthesizeUserModel,
  detectUserGaps,
} from "../engine/memory/index.js";
import { extractMemories, extractMemoriesForSession, shouldExtractMemory } from "../engine/memory/extractor-unified.js";
import { ingestionQueue, type IngestionMemory } from "../engine/ingestion-queue.js";
import { classifyRetryableWriteFailure } from "../engine/memory/write-reliability.js";
import { addPendingOverlayEntry, getPendingOverlayEntries } from "../engine/pending-overlay.js";
import type { ExtractedMemory, MemorySearchResult } from "../engine/memory/types.js";
import { toMemorySearchResult } from "../engine/memory/search.js";
import {
  calibrateConfidence,
  countConfirmEventsForTenant,
  decideShadowScope,
  emitMemoryConfirmAnomaly,
  emitMemoryConfirmEvent,
  evaluateConfirmRateLimits,
  getExtractionPhase0Config,
  getTenantExtractionPolicy,
  recordExtractionInvocation,
} from "../engine/extraction-observability.js";
import { writeMemoryCanonical } from "../engine/memory/write.js";
import { ensureProject } from "./helpers.js";
import {
  getIdempotencyKey,
  hashIdempotencyPayload,
  loadIdempotentResponse,
  storeIdempotentResponse,
} from "./idempotency.js";
import {
  expandMemorySearchQueries,
  getMemorySemanticStatus,
  mergeMemoryNormalizationMetadata,
} from "../lib/memory-normalization.js";
import type { MemoryScopeTarget, PromotionMode, SessionWorkEvent } from "../engine/memory/types.js";

type Variables = {
  auth: AuthContext;
};

export const memoryRoutes = new Hono<{ Variables: Variables }>();

const MEMORY_TYPE_VALUES = [
  "factual",
  "episodic",
  "semantic",
  "procedural",
  "preference",
  "event",
  "relationship",
  "opinion",
  "goal",
  "instruction",
  "decision",
  "constraint",
  "solution",
  "project_state",
  "correction",
  "workflow",
] as const;

const LEGACY_TO_SOTA_MEMORY_TYPE: Record<string, "factual" | "preference" | "event" | "relationship" | "opinion" | "goal" | "instruction" | "decision" | "constraint" | "solution" | "project_state" | "correction" | "workflow"> = {
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
  decision: "decision",
  constraint: "constraint",
  solution: "solution",
  project_state: "project_state",
  correction: "correction",
  workflow: "workflow",
};

const SCOPE_TARGET_VALUES = ["USER", "SESSION", "PROJECT", "AGENT", "TASK", "DOCUMENT"] as const;
const PROMOTION_MODE_VALUES = ["session_state_v1", "user_specific_legacy"] as const;

function normalizeMemoryType(memoryType?: string) {
  if (!memoryType) return "factual" as const;
  return LEGACY_TO_SOTA_MEMORY_TYPE[memoryType] || "factual";
}

function resolveSessionStateMode(project: { settings?: unknown } | null | undefined): PromotionMode {
  const settings =
    project?.settings && typeof project.settings === "object" && !Array.isArray(project.settings)
      ? project.settings as Record<string, any>
      : {};
  const memory =
    settings.memory && typeof settings.memory === "object" && !Array.isArray(settings.memory)
      ? settings.memory
      : {};
  const mode = memory.session_state_mode;
  return mode === "user_specific_legacy" ? "user_specific_legacy" : "session_state_v1";
}

function normalizeScopeCounts(value: unknown): Partial<Record<Exclude<MemoryScopeTarget, "DROPPED">, number>> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const counts: Partial<Record<Exclude<MemoryScopeTarget, "DROPPED">, number>> = {};
  for (const [scope, count] of Object.entries(value)) {
    if (!SCOPE_TARGET_VALUES.includes(scope as typeof SCOPE_TARGET_VALUES[number])) continue;
    if (typeof count !== "number" || !Number.isFinite(count)) continue;
    counts[scope as Exclude<MemoryScopeTarget, "DROPPED">] = count;
  }
  return counts;
}

function getTraceId(c: any) {
  const fromContext = typeof c.get === "function" ? c.get("traceId") : undefined;
  if (typeof fromContext === "string" && fromContext.length > 0) {
    return fromContext;
  }
  const fromHeader =
    c.req.header("x-trace-id") ||
    c.req.header("x-request-id");
  return fromHeader || crypto.randomUUID();
}

type MemoryErrorCode =
  | "VALIDATION_ERROR"
  | "NOT_FOUND"
  | "PROJECT_NOT_FOUND"
  | "MEMORY_NOT_FOUND"
  | "NOT_AUTHORIZED"
  | "SYNC_WRITE_RESTRICTED"
  | "SEARCH_FAILED"
  | "TIMEOUT"
  | "TEMPORARY_UNAVAILABLE"
  | "WRITE_FAILED"
  | "INGEST_FAILED"
  | "EXTRACT_FAILED"
  | "INTERNAL_ERROR";

function memoryError(
  c: any,
  status: number,
  code: MemoryErrorCode,
  message: string,
  traceId: string,
  details?: string
) {
  return c.json(
    {
      success: false,
      error: {
        code,
        message,
        ...(details ? { details } : {}),
      },
      trace_id: traceId,
    },
    status as any
  );
}

function classifyWriteFailure(error: unknown): { status: number; code: "TIMEOUT" | "TEMPORARY_UNAVAILABLE"; message: string } {
  const message = error instanceof Error ? error.message : String(error || "Write failed");
  const normalized = message.toLowerCase();
  if (normalized.includes("timeout") || normalized.includes("timed out") || normalized.includes("abort")) {
    return {
      status: 504,
      code: "TIMEOUT",
      message: "Memory write timed out before commit",
    };
  }
  return {
    status: 503,
    code: "TEMPORARY_UNAVAILABLE",
    message: "Memory write could not be committed",
  };
}

function semanticStatusFromMetadata(metadata: unknown): "pending" | "ready" {
  return getMemorySemanticStatus(metadata);
}

const MEMORY_SYNC_INFERENCE_ENABLED = /^true$/i.test(process.env.MEMORY_SYNC_INFERENCE || "false");
const MEMORY_SYNC_INFERENCE_BUDGET_MS = parseInt(process.env.MEMORY_SYNC_INFERENCE_BUDGET_MS || "60", 10);
const MEMORY_VISIBILITY_SLA_MS = parseInt(process.env.MEMORY_VISIBILITY_SLA_MS || "2000", 10);
const PENDING_OVERLAY_WRITE_TTL_MS = parseInt(process.env.PENDING_OVERLAY_WRITE_TTL_MS || "10000", 10);
const DEFAULT_WRITE_MODE = (process.env.MEMORY_WRITE_MODE_DEFAULT || "async").toLowerCase();
const SYNC_WRITE_RESTRICTED = /^true$/i.test(process.env.SYNC_WRITE_RESTRICTED || "false");
const SYNC_WRITE_ALLOWED_ORGS = new Set(
  (process.env.SYNC_WRITE_ALLOWED_ORGS || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
);

function canUseSyncWrite(auth: AuthContext): boolean {
  if (!SYNC_WRITE_RESTRICTED) return true;
  return Boolean(auth.isAdmin) || SYNC_WRITE_ALLOWED_ORGS.has(auth.orgId);
}

function resolveInferenceTier(
  memory: Pick<ExtractedMemory, "inferred">,
  extractionMethod?: string
): "pattern" | "mini" | "strong" {
  if (memory.inferred === true) return "mini";
  if (memory.inferred === false) return "pattern";
  if (extractionMethod === "inference" || extractionMethod === "hybrid") return "mini";
  return "pattern";
}

function toExtractionDecisions(
  memories: ExtractedMemory[],
  extractionMethod?: string
) {
  return memories.map((memory) => ({
    memoryType: memory.memoryType,
    inferenceTier: resolveInferenceTier(memory, extractionMethod),
    confidenceRaw: Number.isFinite(memory.confidence) ? memory.confidence : 0,
    modelUsed: memory.inferred ? "inference" : "pattern",
    tokenUsage: Math.ceil((memory.content || "").length / 4),
  }));
}

function applyExtractionPolicyToList(
  memories: ExtractedMemory[],
  policy: Awaited<ReturnType<typeof getTenantExtractionPolicy>>
): ExtractedMemory[] {
  if (!policy.orchestrator_v2_enabled) {
    return memories;
  }
  return memories.filter((memory) => {
    const calibrated = calibrateConfidence(memory.confidence, memory.memoryType);
    const scope = decideShadowScope(calibrated, {
      userProfileThreshold: policy.user_profile_threshold,
      sessionOnlyThreshold: policy.session_only_threshold,
    });
    if (!policy.threshold_enforcement_active) {
      return true;
    }
    return scope !== "dropped";
  });
}

async function resolveTenantExtractionPolicySafe(
  tenantId: string
): Promise<Awaited<ReturnType<typeof getTenantExtractionPolicy>>> {
  try {
    return await getTenantExtractionPolicy(tenantId);
  } catch {
    return {
      tenant_id: tenantId,
      orchestrator_v2_enabled: false,
      tiered_escalation_enabled: false,
      threshold_enforcement_requested: false,
      threshold_enforcement_active: false,
      threshold_enforcement_reason: "disabled",
      user_profile_threshold: 0.78,
      session_only_threshold: 0.58,
      session_only_retention_days: 14,
      gate: {
        pass: false,
        sample_count: 0,
        observed_days: 0,
        waived: false,
      },
    };
  }
}

function isBackendConfirmAuth(auth: AuthContext): boolean {
  if (auth.isAdmin) return true;
  const scopes = auth.scopes || [];
  return scopes.includes("memory:confirm") || scopes.includes("backend");
}

function getClientIp(c: any): string | null {
  const forwarded = c.req.header("x-forwarded-for");
  if (forwarded) {
    return forwarded.split(",")[0]?.trim() || null;
  }
  return (
    c.req.header("cf-connecting-ip")
    || c.req.header("x-real-ip")
    || null
  );
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T | null> {
  let timeoutHandle: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<null>((resolve) => {
        timeoutHandle = setTimeout(() => resolve(null), timeoutMs);
      }),
    ]);
  } finally {
    if (timeoutHandle) clearTimeout(timeoutHandle);
  }
}

async function keywordSearchMemories(params: {
  projectId: string;
  orgId: string;
  query: string;
  userId?: string;
  sessionId?: string;
  agentId?: string;
  taskId?: string;
  scopes?: Array<typeof SCOPE_TARGET_VALUES[number]>;
  includeInactive?: boolean;
  memoryTypes?: Array<"factual" | "preference" | "event" | "relationship" | "opinion" | "goal" | "instruction" | "decision" | "constraint" | "solution" | "project_state" | "correction" | "workflow">;
  namespace?: string;
  tags?: string[];
  topK?: number;
}): Promise<MemorySearchResult[]> {
  const {
    projectId,
    orgId,
    query,
    userId,
    sessionId,
    agentId,
    taskId,
    scopes,
    includeInactive = false,
    memoryTypes,
    namespace,
    tags,
    topK = 10,
  } = params;

  const queryVariants = expandMemorySearchQueries(query);
  if (queryVariants.length === 0) {
    return [];
  }
  const metadataFilters = namespace
    ? {
        path: ["namespace"],
        equals: namespace,
      }
    : undefined;

  const memories = await prisma.memory.findMany({
    where: {
      projectId,
      orgId,
      ...(scopes && scopes.length > 0 ? { scope: { in: scopes } } : {}),
      OR: [
        { scope: "PROJECT" },
        ...(userId ? [{ scope: "USER", userId }] : []),
        ...(sessionId ? [{ scope: "SESSION", sessionId }] : []),
        ...(agentId ? [{ scope: "AGENT", agentId }] : []),
        ...(taskId ? [{ scope: "TASK", taskId }] : []),
      ],
      ...(includeInactive ? {} : { isActive: true }),
      ...(memoryTypes && memoryTypes.length > 0 ? { memoryType: { in: memoryTypes } } : {}),
      AND: [
        ...(namespace ? [{ metadata: metadataFilters }] : []),
        {
          OR: queryVariants.flatMap((variant) => ([
            { content: { contains: variant, mode: "insensitive" as const } },
            { metadata: { path: ["search_text"], string_contains: variant } },
            { metadata: { path: ["canonical_content"], string_contains: variant } },
            { metadata: { path: ["normalized_content"], string_contains: variant } },
          ])),
        },
      ],
    },
    orderBy: [
      { importance: "desc" },
      { updatedAt: "desc" },
    ],
    take: topK,
    select: {
      id: true,
      content: true,
      memoryType: true,
      entityMentions: true,
      confidence: true,
      version: true,
      scope: true,
      userId: true,
      sessionId: true,
      agentId: true,
      taskId: true,
      documentDate: true,
      eventDate: true,
      validFrom: true,
      validUntil: true,
      metadata: true,
    },
  });

  // Filter by tags if specified
  let filteredMemories = memories;
  if (tags && tags.length > 0) {
    filteredMemories = memories.filter(m => {
      const memTags = (m.metadata as any)?.tags || [];
      return tags.some(tag => memTags.includes(tag));
    });
  }

  return filteredMemories.map((m) => ({
    ...toMemorySearchResult({ ...m, memoryType: normalizeMemoryType(m.memoryType), similarity: 0.55 }),
    chunk: undefined,
    relations: [],
  }));
}

// ──────────────────────────────────────────────────────────────
// Memory Extraction Endpoint - Local + LLM hybrid
// ──────────────────────────────────────────────────────────────

memoryRoutes.post(
  "/v1/memory/extract",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "json",
    z.object({
      message: z.string().min(1).max(5000),
      context: z.string().optional(),
      session_id: z.string().optional(),
      project: z.string().optional(),
      user_id: z.string().optional(),
      enable_pattern: z.boolean().optional().default(true),
      enable_inference: z.boolean().optional().default(true),
      min_confidence: z.number().min(0).max(1).optional().default(0.5),
    })
  ),
  async (c) => {
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);
    const auth = c.get("auth");
    const body = c.req.valid("json");
    const startTime = Date.now();

    try {
      if (!shouldExtractMemory(body.message)) {
        const latencyMs = Date.now() - startTime;
        try {
          await recordExtractionInvocation({
            tenantId: auth.orgId,
            projectId: body.project || null,
            route: "/v1/memory/extract",
            invocationId: traceId,
            latencyMs,
            decisions: [],
          });
        } catch (shadowError) {
          console.warn(`[${traceId}] Extraction shadow logging skipped:`, shadowError);
        }
        return c.json({
          explicit: [],
          implicit: [],
          all: [],
          extractionMethod: "skipped",
          reason: "Message too short or is greeting",
          latencyMs,
        });
      }

      const policy = await resolveTenantExtractionPolicySafe(auth.orgId);
      const result = await extractMemories(body.message, body.context || "", {
        enablePattern: body.enable_pattern,
        enableInference: body.enable_inference,
        minConfidence: body.min_confidence,
        tieredEscalation: policy.orchestrator_v2_enabled && policy.tiered_escalation_enabled,
      });
      const latencyMs = Date.now() - startTime;
      const filteredExplicit = applyExtractionPolicyToList(result.explicit, policy);
      const filteredImplicit = applyExtractionPolicyToList(result.implicit, policy);
      const filteredAll = applyExtractionPolicyToList(result.all, policy);
      try {
        await recordExtractionInvocation({
          tenantId: auth.orgId,
          projectId: body.project || null,
          route: "/v1/memory/extract",
          invocationId: traceId,
          latencyMs,
          decisions: toExtractionDecisions(result.all, result.extractionMethod),
        });
      } catch (shadowError) {
        console.warn(`[${traceId}] Extraction shadow logging skipped:`, shadowError);
      }

      return c.json({
        ...result,
        explicit: filteredExplicit,
        implicit: filteredImplicit,
        all: filteredAll,
        policy: {
          orchestrator_v2_enabled: policy.orchestrator_v2_enabled,
          threshold_enforcement_active: policy.threshold_enforcement_active,
          user_profile_threshold: policy.user_profile_threshold,
          session_only_threshold: policy.session_only_threshold,
          threshold_enforcement_reason: policy.threshold_enforcement_reason,
        },
        latencyMs,
      });
    } catch (error) {
      console.error(`[${traceId}] Memory extraction error:`, error);
      return memoryError(
        c,
        500,
        "EXTRACT_FAILED",
        "Memory extraction failed",
        traceId,
        error instanceof Error ? error.message : String(error)
      );
    }
  }
);

// Extract memories from session messages
memoryRoutes.post(
  "/v1/memory/extract/session",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "json",
    z.object({
      messages: z.array(
        z.object({
          role: z.enum(["user", "assistant", "system"]),
          content: z.string().min(1).max(10000),
          timestamp: z.string().datetime().optional(),
        })
      ).min(1).max(100),
      project: z.string().optional(),
      user_id: z.string().optional(),
      enable_pattern: z.boolean().optional().default(true),
      enable_inference: z.boolean().optional().default(true),
    })
  ),
  async (c) => {
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);
    const auth = c.get("auth");
    const body = c.req.valid("json");
    const startTime = Date.now();

    try {
      const policy = await resolveTenantExtractionPolicySafe(auth.orgId);
      const result = await extractMemoriesForSession(body.messages, {
        enablePattern: body.enable_pattern,
        enableInference: body.enable_inference,
        tieredEscalation: policy.orchestrator_v2_enabled && policy.tiered_escalation_enabled,
      });
      const filteredResult = applyExtractionPolicyToList(result, policy);
      const latencyMs = Date.now() - startTime;
      try {
        await recordExtractionInvocation({
          tenantId: auth.orgId,
          projectId: body.project || null,
          route: "/v1/memory/extract/session",
          invocationId: traceId,
          latencyMs,
          decisions: toExtractionDecisions(result),
        });
      } catch (shadowError) {
        console.warn(`[${traceId}] Session extraction shadow logging skipped:`, shadowError);
      }

      return c.json({
        memories: filteredResult,
        count: filteredResult.length,
        policy: {
          orchestrator_v2_enabled: policy.orchestrator_v2_enabled,
          threshold_enforcement_active: policy.threshold_enforcement_active,
          user_profile_threshold: policy.user_profile_threshold,
          session_only_threshold: policy.session_only_threshold,
          threshold_enforcement_reason: policy.threshold_enforcement_reason,
        },
        latencyMs,
      });
    } catch (error) {
      console.error(`[${traceId}] Session memory extraction error:`, error);
      return memoryError(
        c,
        500,
        "EXTRACT_FAILED",
        "Session extraction failed",
        traceId,
        error instanceof Error ? error.message : String(error)
      );
    }
  }
);

// SOTA Memory Search - The main query endpoint
// ──────────────────────────────────────────────────────────────

memoryRoutes.post(
  "/v1/memory/search",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "json",
    z.object({
      query: z.string().min(1).max(5000),
      project: z.string().optional(),
      user_id: z.string().optional(),
      session_id: z.string().optional(),
      agent_id: z.string().optional(),
      task_id: z.string().trim().min(1).max(128).optional(),
      question_date: z.string().datetime().optional(), // ISO date string
      top_k: z.number().int().min(1).max(50).optional().default(10),
      memory_types: z.array(z.enum(MEMORY_TYPE_VALUES)).optional(),
      scope_targets: z.array(z.enum(SCOPE_TARGET_VALUES)).optional(),
      include_inactive: z.boolean().optional().default(false),
      include_chunks: z.boolean().optional().default(true),
      include_relations: z.boolean().optional().default(true),
      namespace: z.string().optional(),
      tags: z.array(z.string()).optional(),
      fast_mode: z.boolean().optional(),
      profile: z.enum(["fast", "balanced", "quality"]).optional().default("fast"),
      include_pending: z.boolean().optional().default(true),
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const body = c.req.valid("json");
    const startTime = Date.now();
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);

    try {
      const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);

      // Parse question date
      const questionDate = body.question_date ? new Date(body.question_date) : new Date();
      const normalizedMemoryTypes = body.memory_types?.map((t) => normalizeMemoryType(t));
      const profile = body.profile || (body.fast_mode === false ? "balanced" : "fast");
      const fastMode = profile === "fast";
      const includePending = body.include_pending !== false;

      let usedLexicalFallback = false;
      let results: MemorySearchResult[] = [];
      let lexicalLatencyMs = 0;
      let pendingMergeMs = 0;
      let pendingOverlayCount = 0;
      let diagnostics = {
        cache_ms: 0,
        embed_ms: 0,
        vector_ms: 0,
        lexical_ms: 0,
        merge_ms: 0,
        total_ms: 0,
        cache_hit: false,
        cache_hit_type: "none" as "none" | "simple" | "semantic",
        fast_mode: fastMode,
      };

      // Search memories (vector-first, then lexical fallback)
      try {
        results = await searchMemories({
          query: body.query,
          questionDate,
          projectId: project.id,
          orgId: auth.orgId,
          userId: body.user_id,
          sessionId: body.session_id,
          agentId: body.agent_id,
          taskId: body.task_id,
          topK: body.top_k,
          memoryTypes: normalizedMemoryTypes,
          scopes: body.scope_targets as MemoryScopeTarget[] | undefined,
          includeInactive: body.include_inactive,
          namespace: body.namespace,
          tags: body.tags,
          fastMode,
          diagnosticsCollector: (value) => {
            diagnostics = value;
          },
        });
        const shouldMergeScopedFallback = Boolean(body.user_id || body.session_id);
        if (results.length === 0 || shouldMergeScopedFallback) {
          usedLexicalFallback = true;
          const lexicalStart = Date.now();
          const lexicalResults = await keywordSearchMemories({
            projectId: project.id,
            orgId: auth.orgId,
            query: body.query,
            userId: body.user_id,
            sessionId: body.session_id,
            agentId: body.agent_id,
            taskId: body.task_id,
            scopes: body.scope_targets,
            includeInactive: body.include_inactive,
            memoryTypes: normalizedMemoryTypes,
            namespace: body.namespace,
            tags: body.tags,
            topK: body.top_k,
          });
          lexicalLatencyMs = Date.now() - lexicalStart;
          if (results.length === 0) {
            results = lexicalResults;
          } else if (lexicalResults.length > 0) {
            const merged: MemorySearchResult[] = [];
            const seen = new Set<string>();
            for (const item of [...results, ...lexicalResults]) {
              const id = String(item?.memory?.id || "").trim();
              if (!id || seen.has(id)) continue;
              seen.add(id);
              merged.push(item);
              if (merged.length >= (body.top_k || 10)) break;
            }
            results = merged;
          }
          diagnostics = {
            ...diagnostics,
            lexical_ms: diagnostics.lexical_ms + lexicalLatencyMs,
          };
        }
      } catch (vectorError: any) {
        usedLexicalFallback = true;
        console.warn("Memory vector search failed, falling back to lexical search:", vectorError?.message || vectorError);
        const lexicalStart = Date.now();
        results = await keywordSearchMemories({
          projectId: project.id,
          orgId: auth.orgId,
          query: body.query,
          userId: body.user_id,
          sessionId: body.session_id,
          agentId: body.agent_id,
          taskId: body.task_id,
          scopes: body.scope_targets,
          includeInactive: body.include_inactive,
          memoryTypes: normalizedMemoryTypes,
          namespace: body.namespace,
          tags: body.tags,
          topK: body.top_k,
        });
        lexicalLatencyMs = Date.now() - lexicalStart;
        diagnostics = {
          ...diagnostics,
          lexical_ms: lexicalLatencyMs,
          fast_mode: true,
        };
      }

      if (includePending && (body.user_id || body.session_id)) {
        const pendingFetchStart = Date.now();
        const pendingEntries = await getPendingOverlayEntries({
          orgId: auth.orgId,
          projectId: project.id,
          userId: body.user_id,
          sessionId: body.session_id,
          limit: body.top_k,
        });
        const pendingFetchMs = Date.now() - pendingFetchStart;
        pendingOverlayCount = pendingEntries.length;

        if (pendingEntries.length > 0) {
          const mergeStart = Date.now();
          const pendingResults = pendingEntries.map<MemorySearchResult>((entry, idx) => ({
            memory: {
              id: `pending:${entry.job_id}:${idx}`,
              content: entry.content,
              memoryType: "event",
              entityMentions: [],
              confidence: 0.7,
              version: 0,
              scope: body.session_id ? "SESSION" : body.user_id ? "USER" : "PROJECT",
              scopeTarget: body.session_id ? "SESSION" : body.user_id ? "USER" : "PROJECT",
              userId: body.user_id ?? null,
              sessionId: body.session_id ?? null,
              agentId: body.agent_id ?? null,
              taskId: body.task_id ?? null,
              metadata: { semantic_status: "pending" },
              temporal: {
                documentDate: new Date(entry.created_at),
                eventDate: new Date(entry.created_at),
                validFrom: new Date(entry.created_at),
                validUntil: new Date(entry.expires_at),
              },
            },
            chunk: undefined,
            similarity: 1,
            relations: [],
            pending: true,
            job_id: entry.job_id,
          }));

          const merged: MemorySearchResult[] = [];
          const seen = new Set<string>();
          for (const item of [...pendingResults, ...results]) {
            const memoryId = item?.memory?.id || "";
            const content = String(item?.memory?.content || "").trim().toLowerCase();
            const key = memoryId || `content:${content}`;
            if (!key || seen.has(key)) continue;
            seen.add(key);
            merged.push(item);
            if (merged.length >= body.top_k) break;
          }
          results = merged;
          pendingMergeMs += Date.now() - mergeStart;
        }

        diagnostics.lexical_ms += pendingFetchMs;
      }

      // Track usage
      const latency = Date.now() - startTime;
      const latencyBreakdown = {
        cache_ms: diagnostics.cache_ms,
        embed_ms: usedLexicalFallback ? 0 : diagnostics.embed_ms,
        vector_ms: usedLexicalFallback ? 0 : diagnostics.vector_ms,
        lexical_ms: diagnostics.lexical_ms,
        merge_ms: diagnostics.merge_ms + pendingMergeMs,
        total_ms: latency,
      };

      // Format response
      const formatDate = (d: Date | string | null | undefined): string | null => {
        if (!d || d === null || d === undefined) return null;
        try {
          if (typeof d === 'string') {
            // Validate it's a valid date string
            const parsed = new Date(d);
            return isNaN(parsed.getTime()) ? null : d;
          }
          const parsed = new Date(d);
          return isNaN(parsed.getTime()) ? null : parsed.toISOString();
        } catch {
          return null;
        }
      };

      const formattedResults = results.map((r) => ({
        memory: {
          id: r.memory.id,
          content: r.memory.content,
          type: r.memory.memoryType,
          entities: r.memory.entityMentions || [],
          confidence: r.memory.confidence,
          scope: r.memory.scope,
          scope_target: r.memory.scopeTarget || r.memory.scope,
          user_id: r.memory.userId ?? null,
          session_id: r.memory.sessionId ?? null,
          agent_id: r.memory.agentId ?? null,
          task_id: r.memory.taskId ?? null,
          semantic_status: semanticStatusFromMetadata(r.memory.metadata),
          version: r.memory.version,
          temporal: {
            document_date: formatDate(r.memory.temporal?.documentDate),
            event_date: formatDate(r.memory.temporal?.eventDate),
            valid_from: formatDate(r.memory.temporal?.validFrom),
            valid_until: formatDate(r.memory.temporal?.validUntil),
          },
        },
        chunk: body.include_chunks && r.chunk ? {
          id: r.chunk.id,
          content: r.chunk.content,
          metadata: r.chunk.metadata,
        } : undefined,
        similarity: r.similarity,
        relations: body.include_relations ? r.relations : undefined,
      }));

      return c.json({
        results: formattedResults,
        count: results.length,
        scope_counts: normalizeScopeCounts(
          formattedResults.reduce((acc, item) => {
            const scope = item.memory.scope_target;
            if (scope && SCOPE_TARGET_VALUES.some((value) => value === scope)) {
              acc[scope] = (acc[scope] || 0) + 1;
            }
            return acc;
          }, {} as Record<string, number>)
        ),
        query: body.query,
        trace_id: traceId,
        question_date: questionDate.toISOString(),
        latency_ms: latency,
        latency_breakdown: latencyBreakdown,
        fallback: usedLexicalFallback ? "lexical" : "vector",
        mode: profile,
        profile,
        include_pending: includePending,
        pending_overlay_count: pendingOverlayCount,
        scopes_touched: Array.from(new Set(formattedResults.map((item) => item.memory.scope_target).filter(Boolean))),
      });
    } catch (error) {
      console.error(`[${traceId}] Memory search error:`, error);
      try {
      } catch {
        // no-op
      }
      return memoryError(c, 500, "SEARCH_FAILED", "Memory search failed", traceId);
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Create Single Memory - Simple memory creation
// ──────────────────────────────────────────────────────────────

memoryRoutes.post(
  "/v1/memory",
  rateLimitMiddleware(RateLimits.mutation),
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      content: z.string().min(1).max(10000),
      memory_type: z.enum(MEMORY_TYPE_VALUES).optional(),
      auto_detect: z.boolean().optional().default(true), // Auto-detect memory type
      user_id: z.string().optional(),
      session_id: z.string().optional(),
      agent_id: z.string().optional(),
      task_id: z.string().trim().min(1).max(128).optional(),
      scope_target: z.enum(SCOPE_TARGET_VALUES).optional(),
      promotion_mode: z.enum(PROMOTION_MODE_VALUES).optional(),
      importance: z.number().min(0).max(1).optional().default(0.5),
      confidence: z.number().min(0).max(1).optional().default(0.8),
      metadata: z.record(z.any()).optional(),
      entity_mentions: z.array(z.string()).optional(),
      document_date: z.string().datetime().optional(),
      event_date: z.string().datetime().optional(),
      namespace: z.string().optional(),
      tags: z.array(z.string()).optional(),
      async: z.boolean().optional(),
      write_mode: z.enum(["async", "sync"]).optional(),
      webhook_url: z.string().url().optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const body = c.req.valid("json");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);

    try {
      const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);
      const promotionMode = body.promotion_mode || resolveSessionStateMode(project);

      const idempotencyKey = getIdempotencyKey({
        "Idempotency-Key": c.req.header("Idempotency-Key"),
        "idempotency-key": c.req.header("idempotency-key"),
      });
      const idempotencyRequestHash = idempotencyKey
        ? hashIdempotencyPayload({
            route: "/v1/memory",
            projectId: project.id,
            body: {
              ...body,
              async: undefined,
              write_mode: undefined,
            },
          })
        : null;

      if (idempotencyKey && idempotencyRequestHash) {
        const prior = await loadIdempotentResponse({
          orgId: auth.orgId,
          endpoint: "/v1/memory",
          idempotencyKey,
          requestHash: idempotencyRequestHash,
        });

        if (prior.type === "conflict") {
          return memoryError(
            c,
            409,
            "VALIDATION_ERROR",
            "Idempotency key reuse with different payload",
            traceId
          );
        }

        if (prior.type === "hit") {
          c.header("x-idempotency-replay", "true");
          return c.json(prior.body, prior.statusCode as any);
        }
      }

      // Auto-detect memory type if not specified or auto_detect is true
      let normalizedMemoryType = normalizeMemoryType(body.memory_type);
      let detectionMethod: "explicit" | "inference" | "manual" = "manual";
      let detectionConfidence = body.confidence;

      if (body.auto_detect && !body.memory_type) {
        const explicitMatch = extractExplicitMemory(body.content);
        
        if (explicitMatch.length > 0) {
          normalizedMemoryType = explicitMatch[0].type;
          detectionMethod = "explicit";
          detectionConfidence = explicitMatch[0].confidence;
        } else if (MEMORY_SYNC_INFERENCE_ENABLED) {
          // Budget-gated inference to protect sync write latency.
          try {
            const { extractImplicitMemories } = await import("../engine/memory/inference.js");
            const implicitMemories = await withTimeout(
              extractImplicitMemories(body.content, ""),
              MEMORY_SYNC_INFERENCE_BUDGET_MS
            );
            if (implicitMemories && implicitMemories.length > 0) {
              normalizedMemoryType = implicitMemories[0].memoryType;
              detectionMethod = "inference";
              detectionConfidence = implicitMemories[0].confidence;
            }
          } catch (inferenceError) {
            console.error("Memory type inference failed:", inferenceError);
            // Keep default "factual" type
          }
        }
      }

      const requestedWriteMode =
        body.write_mode ||
        (body.async === false
          ? "sync"
          : body.async === true
            ? "async"
            : "sync");
      const syncWriteRequested = requestedWriteMode === "sync";

      const shouldAsync = !syncWriteRequested;

      // Async processing mode
      if (shouldAsync) {
        const acceptedAt = new Date().toISOString();
        const jobId = await ingestionQueue.createJob({
          orgId: auth.orgId,
          projectId: project.id,
          userId: auth.userId ?? "",
          memories: [{
            content: body.content,
            memory_type: normalizedMemoryType,
            user_id: body.user_id,
            session_id: body.session_id,
            agent_id: body.agent_id,
            task_id: body.task_id,
            importance: body.importance,
            metadata: {
              ...mergeMemoryNormalizationMetadata(body.metadata, body.content),
              namespace: body.namespace,
              tags: body.tags,
              scope_target: body.scope_target,
              promotion_mode: promotionMode,
              entity_mentions: body.entity_mentions,
              document_date: body.document_date,
              event_date: body.event_date,
              confidence_raw: body.confidence,
              write_source: "api.memory.create",
              write_mode: "direct_write",
              extraction_method:
                detectionMethod === "explicit"
                  ? "pattern"
                  : detectionMethod === "inference"
                    ? "inference"
                    : "manual",
              traceId,
            },
          }],
          webhookUrl: body.webhook_url,
          namespace: body.namespace,
          tags: body.tags,
        });


        const pendingVisibility = await addPendingOverlayEntry({
          orgId: auth.orgId,
          projectId: project.id,
          userId: body.user_id,
          sessionId: body.session_id,
          content: body.content,
          jobId,
          ttlMs: Math.max(MEMORY_VISIBILITY_SLA_MS, PENDING_OVERLAY_WRITE_TTL_MS),
          createdAt: acceptedAt,
        });

        const responseBody = {
          success: true,
          mode: 'async',
          trace_id: traceId,
          job_id: jobId,
          status_url: `/v1/memory/jobs/${jobId}`,
          legacy_status_url: `/v1/jobs/${jobId}`,
          consistency: "eventual",
          visibility_sla_ms: MEMORY_VISIBILITY_SLA_MS,
          accepted_at: acceptedAt,
          pending_visibility: pendingVisibility,
          message: "Memory creation queued for processing",
        };

        if (idempotencyKey && idempotencyRequestHash) {
          await storeIdempotentResponse({
            orgId: auth.orgId,
            endpoint: "/v1/memory",
            idempotencyKey,
            requestHash: idempotencyRequestHash,
            statusCode: 202,
            body: responseBody,
          });
        }

        return c.json(responseBody, 202);
      }

      const policy = await resolveTenantExtractionPolicySafe(auth.orgId);
      const writeResult = await writeMemoryCanonical({
        projectId: project.id,
        orgId: auth.orgId,
        userId: body.user_id,
        sessionId: body.session_id,
        agentId: body.agent_id,
        taskId: body.task_id,
        content: body.content,
        memoryType: normalizedMemoryType,
        importance: body.importance,
        confidenceRaw: detectionConfidence,
        entityMentions: body.entity_mentions,
        documentDate: body.document_date ? new Date(body.document_date) : null,
        eventDate: body.event_date ? new Date(body.event_date) : null,
        metadata: {
          ...mergeMemoryNormalizationMetadata(body.metadata, body.content),
          namespace: body.namespace,
          tags: body.tags,
          traceId,
        },
        writeSource: "api.memory.create",
        writeMode: "direct_write",
        extractionMethod:
          detectionMethod === "explicit"
            ? "pattern"
            : detectionMethod === "inference"
              ? "inference"
              : "manual",
        scopeHint: body.scope_target,
        promotionMode,
        pendingOverlayTtlMs: Math.max(MEMORY_VISIBILITY_SLA_MS, PENDING_OVERLAY_WRITE_TTL_MS),
        sessionRetentionDays: policy.session_only_retention_days,
      });

      if (writeResult.outcome === "dropped" || !writeResult.memory) {
        return memoryError(
          c,
          422,
          "VALIDATION_ERROR",
          "Memory write was rejected by canonical validation policy",
          traceId,
          writeResult.validatorIssues.join(", ")
        );
      }

      // Track usage

      const responseBody = {
        success: true,
        mode: 'sync',
        trace_id: traceId,
        memory_id: writeResult.memory.id,
        semantic_status: semanticStatusFromMetadata(writeResult.memory.metadata),
        consistency: "eventual",
        visibility_sla_ms: MEMORY_VISIBILITY_SLA_MS,
        pending_visibility: Boolean(body.user_id || body.session_id),
        memory: {
          id: writeResult.memory.id,
          content: writeResult.memory.content,
          type: writeResult.memory.memoryType,
          project_id: writeResult.memory.projectId,
          user_id: writeResult.memory.userId,
          session_id: writeResult.memory.sessionId,
          agent_id: writeResult.memory.agentId,
          task_id: writeResult.memory.taskId,
          importance: writeResult.memory.importance,
          confidence: writeResult.memory.confidence,
          created_at: writeResult.memory.createdAt,
          semantic_status: semanticStatusFromMetadata(writeResult.memory.metadata),
        },
        auto_detected: {
          method: detectionMethod,
          confidence: detectionConfidence,
        },
        write_outcome: writeResult.outcome,
        scope_decision: writeResult.scopeDecision,
        scope_target: writeResult.scopeTarget,
        validator_issues: writeResult.validatorIssues,
      };

      if (idempotencyKey && idempotencyRequestHash) {
        await storeIdempotentResponse({
          orgId: auth.orgId,
          endpoint: "/v1/memory",
          idempotencyKey,
          requestHash: idempotencyRequestHash,
          statusCode: 201,
          body: responseBody,
        });
      }

      const statusCode = writeResult.outcome === "created" ? 201 : 200;
      return c.json(responseBody, statusCode as any);
    } catch (error) {
      console.error(`[${traceId}] Memory creation error:`, error);
      try {
      } catch {
        // no-op
      }
      const classified = classifyWriteFailure(error);
      return memoryError(
        c,
        classified.status,
        classified.code,
        classified.message,
        traceId,
        "retryable=true;write_state=not_committed"
      );
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Bulk Memory Creation - Create multiple memories at once
// ──────────────────────────────────────────────────────────────

memoryRoutes.post(
  "/v1/memory/bulk",
  rateLimitMiddleware(RateLimits.mutation),
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      memories: z.array(
        z.object({
          content: z.string().min(1).max(10000),
          memory_type: z.enum(MEMORY_TYPE_VALUES).optional().default("factual"),
          user_id: z.string().optional(),
          session_id: z.string().optional(),
          agent_id: z.string().optional(),
          task_id: z.string().trim().min(1).max(128).optional(),
          scope_target: z.enum(SCOPE_TARGET_VALUES).optional(),
          importance: z.number().min(0).max(1).optional().default(0.5),
          confidence: z.number().min(0).max(1).optional().default(0.8),
          metadata: z.record(z.any()).optional(),
          entity_mentions: z.array(z.string()).optional(),
          document_date: z.string().datetime().optional(),
          event_date: z.string().datetime().optional(),
        })
      ).min(1).max(1000),
      namespace: z.string().optional(),
      tags: z.array(z.string()).optional(),
      promotion_mode: z.enum(PROMOTION_MODE_VALUES).optional(),
      async: z.boolean().optional(),
      write_mode: z.enum(["async", "sync"]).optional(),
      webhook_url: z.string().url().optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const body = c.req.valid("json");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);

    try {
      const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);
      const promotionMode = body.promotion_mode || resolveSessionStateMode(project);

      const requestedWriteMode =
        body.write_mode ||
        (body.async === false
          ? "sync"
          : body.async === true
            ? "async"
            : body.memories.length > 10
              ? "async"
              : DEFAULT_WRITE_MODE === "sync"
                ? "sync"
                : "async");
      const syncWriteRequested = requestedWriteMode === "sync";

      if (syncWriteRequested && !canUseSyncWrite(auth)) {
        return memoryError(
          c,
          403,
          "SYNC_WRITE_RESTRICTED",
          "write_mode=sync is restricted to internal/admin tenants",
          traceId
        );
      }

      // Large batches default to async unless sync is explicitly requested.
      const shouldAsync = requestedWriteMode === "async";

      // Async processing mode (recommended for bulk)
      if (shouldAsync) {
        const acceptedAt = new Date().toISOString();
        const normalizedMemories = body.memories.map(mem => ({
          content: mem.content,
          memory_type: normalizeMemoryType(mem.memory_type),
          user_id: mem.user_id,
          session_id: mem.session_id,
          agent_id: mem.agent_id,
          task_id: mem.task_id,
          importance: mem.importance,
          metadata: {
            ...(mem.metadata || {}),
            scope_target: mem.scope_target,
            promotion_mode: promotionMode,
            entity_mentions: mem.entity_mentions,
            document_date: mem.document_date,
            event_date: mem.event_date,
            confidence_raw: mem.confidence,
            write_source: "api.memory.bulk",
            write_mode: "direct_write",
            extraction_method: "manual",
            traceId,
          },
        }));

        const jobId = await ingestionQueue.createJob({
          orgId: auth.orgId,
          projectId: project.id,
          userId: auth.userId ?? "",
          memories: normalizedMemories,
          webhookUrl: body.webhook_url,
          namespace: body.namespace,
          tags: body.tags,
        });


        let pendingVisibility = false;
        for (const mem of body.memories) {
          const visibility = await addPendingOverlayEntry({
            orgId: auth.orgId,
            projectId: project.id,
            userId: mem.user_id,
            sessionId: mem.session_id,
            content: mem.content,
            jobId,
            ttlMs: Math.max(MEMORY_VISIBILITY_SLA_MS, PENDING_OVERLAY_WRITE_TTL_MS),
            createdAt: acceptedAt,
          });
          pendingVisibility = pendingVisibility || visibility;
        }

        return c.json({
          success: true,
          mode: 'async',
          trace_id: traceId,
          job_id: jobId,
          status_url: `/v1/memory/jobs/${jobId}`,
          legacy_status_url: `/v1/jobs/${jobId}`,
          consistency: "eventual",
          visibility_sla_ms: MEMORY_VISIBILITY_SLA_MS,
          accepted_at: acceptedAt,
          pending_visibility: pendingVisibility,
          memories_queued: body.memories.length,
          message: "Bulk memory creation queued for processing",
        }, 202);
      }

      const policy = await resolveTenantExtractionPolicySafe(auth.orgId);
      const createdMemories: any[] = [];
      const rejectedMemories: any[] = [];
      for (const mem of body.memories) {
        const normalizedMemoryType = normalizeMemoryType(mem.memory_type);
        const writeResult = await writeMemoryCanonical({
          projectId: project.id,
          orgId: auth.orgId,
          userId: mem.user_id,
          sessionId: mem.session_id,
          agentId: mem.agent_id,
          taskId: mem.task_id,
          content: mem.content,
          memoryType: normalizedMemoryType,
          importance: mem.importance,
          confidenceRaw: mem.confidence,
          entityMentions: mem.entity_mentions,
          documentDate: mem.document_date ? new Date(mem.document_date) : null,
          eventDate: mem.event_date ? new Date(mem.event_date) : null,
          metadata: {
            ...(mem.metadata || {}),
            namespace: body.namespace,
            tags: body.tags,
            traceId,
          },
          writeSource: "api.memory.bulk",
          writeMode: "direct_write",
          extractionMethod: "manual",
          scopeHint: mem.scope_target,
          promotionMode,
          pendingOverlayTtlMs: Math.max(MEMORY_VISIBILITY_SLA_MS, PENDING_OVERLAY_WRITE_TTL_MS),
          sessionRetentionDays: policy.session_only_retention_days,
        });

        if (!writeResult.memory || writeResult.outcome === "dropped") {
          rejectedMemories.push({
            content: mem.content,
            reason: writeResult.validatorIssues,
          });
          continue;
        }

        createdMemories.push({
          id: writeResult.memory.id,
          content: writeResult.memory.content,
          type: writeResult.memory.memoryType,
          user_id: writeResult.memory.userId,
          session_id: writeResult.memory.sessionId,
          agent_id: writeResult.memory.agentId,
          task_id: writeResult.memory.taskId,
          importance: writeResult.memory.importance,
          confidence: writeResult.memory.confidence,
          created_at: writeResult.memory.createdAt,
          write_outcome: writeResult.outcome,
          scope_decision: writeResult.scopeDecision,
          scope_target: writeResult.scopeTarget,
        });
      }


      return c.json({
        success: true,
        mode: 'sync',
        trace_id: traceId,
        memories: createdMemories,
        count: createdMemories.length,
        scope_counts: normalizeScopeCounts(
          createdMemories.reduce((acc, memory) => {
            const scope = memory.scope_target;
            if (scope && scope !== "DROPPED" && SCOPE_TARGET_VALUES.includes(scope as typeof SCOPE_TARGET_VALUES[number])) {
              acc[scope] = (acc[scope] || 0) + 1;
            }
            return acc;
          }, {} as Record<string, number>)
        ),
        rejected: rejectedMemories,
      }, createdMemories.length > 0 ? 201 : 200);
    } catch (error) {
      console.error(`[${traceId}] Bulk memory creation error:`, error);
      return memoryError(c, 500, "WRITE_FAILED", "Failed to create memories in bulk", traceId);
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Ingest Session - Create memories from conversation
// ──────────────────────────────────────────────────────────────

memoryRoutes.post(
  "/v1/memory/ingest/session",
  rateLimitMiddleware(RateLimits.ingest),
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      session_id: z.string(),
      user_id: z.string().optional(),
      agent_id: z.string().optional(),
      task_id: z.string().trim().min(1).max(128).optional(),
      messages: z.array(
        z.object({
          role: z.string(),
          content: z.string(),
          timestamp: z.string().datetime(),
        })
      ),
      events: z.array(z.object({
        kind: z.enum(["decision", "constraint", "outcome", "failure", "task_update", "file_edit", "tool_result"]),
        summary: z.string().min(1).max(1000),
        details: z.string().max(5000).optional(),
        salience: z.enum(["low", "medium", "high"]).optional(),
        timestamp: z.string().datetime().optional(),
        filePaths: z.array(z.string()).optional(),
        toolName: z.string().optional(),
        success: z.boolean().optional(),
      })).optional(),
      namespace: z.string().optional(),
      tags: z.array(z.string()).optional(),
      promotion_mode: z.enum(PROMOTION_MODE_VALUES).optional(),
      async: z.boolean().optional(),
      write_mode: z.enum(["async", "sync"]).optional(),
      webhook_url: z.string().url().optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const body = c.req.valid("json");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);

    try {
      const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);
      const promotionMode = body.promotion_mode || resolveSessionStateMode(project);

      const policy = await resolveTenantExtractionPolicySafe(auth.orgId);

      const emitIngestShadowTelemetry = async () => {
        try {
          const extracted = await extractMemoriesForSession(body.messages, {
            enablePattern: true,
            enableInference: true,
            tieredEscalation: policy.orchestrator_v2_enabled && policy.tiered_escalation_enabled,
          });
          await recordExtractionInvocation({
            tenantId: auth.orgId,
            projectId: project.id,
            route: "/v1/memory/ingest/session",
            invocationId: traceId,
            decisions: toExtractionDecisions(extracted),
          });
        } catch (shadowError) {
          console.warn(`[${traceId}] Ingest shadow extraction logging failed:`, shadowError);
        }
      };
      void emitIngestShadowTelemetry();

      const requestedWriteMode =
        body.write_mode ||
        (body.async === false
          ? "sync"
          : body.async === true
            ? "async"
            : DEFAULT_WRITE_MODE === "sync"
              ? "sync"
              : "async");
      const syncWriteRequested = requestedWriteMode === "sync";

      if (syncWriteRequested && !canUseSyncWrite(auth)) {
        return memoryError(
          c,
          403,
          "SYNC_WRITE_RESTRICTED",
          "write_mode=sync is restricted to internal/admin tenants",
          traceId
        );
      }

      const shouldAsync = !syncWriteRequested;

      // Async processing mode with job tracking
      if (shouldAsync) {
        const acceptedAt = new Date().toISOString();
        const jobId = await ingestionQueue.createJob({
          orgId: auth.orgId,
          projectId: project.id,
          userId: auth.userId ?? "",
          conversations: [{
            session_id: body.session_id,
            user_id: body.user_id,
            title: `Session ${body.session_id}`,
            messages: body.messages,
            metadata: {
              agent_id: body.agent_id,
              task_id: body.task_id,
              events: body.events || [],
              promotion_mode: promotionMode,
              namespace: body.namespace,
              tags: body.tags,
              traceId,
            },
          }],
          webhookUrl: body.webhook_url,
          namespace: body.namespace,
          tags: body.tags,
        });


        const overlayContent = body.messages
          .filter((m) => m.role === "user" || m.role === "assistant")
          .slice(-3)
          .map((m) => `${m.role}: ${m.content}`)
          .join("\n");
        const pendingVisibility = await addPendingOverlayEntry({
          orgId: auth.orgId,
          projectId: project.id,
          userId: body.user_id,
          sessionId: body.session_id,
          content: overlayContent.length > 0 ? overlayContent : `Session ${body.session_id}`,
          jobId,
          ttlMs: Math.max(MEMORY_VISIBILITY_SLA_MS, PENDING_OVERLAY_WRITE_TTL_MS),
          createdAt: acceptedAt,
        });

        return c.json({
          success: true,
          mode: 'async',
          trace_id: traceId,
          job_id: jobId,
          status_url: `/v1/memory/jobs/${jobId}`,
          legacy_status_url: `/v1/jobs/${jobId}`,
          consistency: "eventual",
          visibility_sla_ms: MEMORY_VISIBILITY_SLA_MS,
          accepted_at: acceptedAt,
          pending_visibility: pendingVisibility,
          session_id: body.session_id,
          messages_queued: body.messages.length,
          message: "Session ingestion queued for processing",
        }, 202);
      }

      // Synchronous mode - wait for committed ingestion result before returning.
      const result = await ingestSession({
        sessionId: body.session_id,
        projectId: project.id,
        orgId: auth.orgId,
        userId: body.user_id,
        agentId: body.agent_id,
        taskId: body.task_id,
        events: body.events as SessionWorkEvent[] | undefined,
        promotionMode,
        messages: body.messages.map((m) => ({
          role: m.role,
          content: m.content,
          timestamp: new Date(m.timestamp),
        })),
      });

      // Track usage

      return c.json({
        success: true,
        mode: 'sync',
        trace_id: traceId,
        message: "Session ingestion completed",
        session_id: body.session_id,
        messages_processed: body.messages.length,
        memories_created: result.memoriesCreated,
        relations_created: result.relationsCreated,
        memories_invalidated: result.memoriesInvalidated,
        scope_counts: result.scopeCounts,
        scopes_touched: result.scopesTouched,
        errors: result.errors,
      });
    } catch (error) {
      console.error(`[${traceId}] Session ingestion error:`, error);
      return memoryError(c, 500, "INGEST_FAILED", "Failed to ingest session", traceId);
    }
  }
);

// ──────────────────────────────────────────────────────────────

memoryRoutes.get(
  "/v1/memory/jobs/:jobId",
  rateLimitMiddleware(RateLimits.query),
  async (c) => {
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);
    const auth = c.get("auth");
    const jobId = c.req.param("jobId") ?? "";

    try {
      const job = await ingestionQueue.getJobStatus(jobId);
      if (!job) {
        return memoryError(c, 404, "NOT_FOUND", "Job not found", traceId);
      }

      if (!auth.isAdmin && job.orgId !== auth.orgId) {
        return memoryError(c, 403, "NOT_AUTHORIZED", "Not authorized", traceId);
      }

      return c.json({
        success: true,
        job,
      });
    } catch (error: any) {
      console.error(`[${traceId}] [Memory] Job status error:`, error);
      return memoryError(
        c,
        500,
        "INTERNAL_ERROR",
        "Failed to fetch memory job status",
        traceId,
        error?.message || String(error)
      );
    }
  }
);
// Get Session Memories - All memories from a conversation
// ──────────────────────────────────────────────────────────────

memoryRoutes.get(
  "/v1/memory/session/:sessionId",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "query",
    z.object({
      project: z.string().optional(),
      include_inactive: z.string().optional(), // "true" or "false"
      include_pending: z.string().optional(), // "true" or "false"
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const sessionId = c.req.param("sessionId");
    const query = c.req.valid("query");

    try {
      const project = await ensureProject(auth.orgId, query.project, auth.isAdmin);

      const includeInactive = query.include_inactive === "true";
      const includePending = query.include_pending !== "false";

      const memories = await getSessionMemories({
        sessionId,
        projectId: project.id,
      });

      let pendingEntries: Awaited<ReturnType<typeof getPendingOverlayEntries>> = [];
      if (includePending) {
        pendingEntries = await getPendingOverlayEntries({
          orgId: auth.orgId,
          projectId: project.id,
          sessionId,
          limit: 50,
        });
      }

      // Robust date formatter
      const formatDateSafe = (d: any): string | null => {
        if (d === null || d === undefined) return null;
        if (typeof d === 'string') {
          const date = new Date(d);
          return isNaN(date.getTime()) ? null : d;
        }
        if (d instanceof Date) {
          return isNaN(d.getTime()) ? null : d.toISOString();
        }
        return null;
      };

      const mergedMemories = [
        ...pendingEntries.map((entry, idx) => ({
          id: `pending:${entry.job_id}:${idx}`,
          content: entry.content,
          type: "event",
          scope: "SESSION",
          scope_target: "SESSION",
          importance: 0.5,
          confidence: 0.7,
          entities: [],
          created_at: entry.created_at,
          document_date: entry.created_at,
          event_date: entry.created_at,
          pending: true,
          job_id: entry.job_id,
        })),
        ...memories.map((m) => ({
          id: m.id,
          content: m.content,
          type: m.memoryType,
          scope: m.scope,
          scope_target: m.scope,
          user_id: m.userId ?? null,
          session_id: m.sessionId ?? null,
          agent_id: m.agentId ?? null,
          task_id: (m as any).taskId ?? null,
          importance: m.importance,
          confidence: m.confidence,
          entities: m.entityMentions || [],
          semantic_status: semanticStatusFromMetadata(m.metadata),
          created_at: formatDateSafe(m.createdAt),
          document_date: formatDateSafe(m.documentDate),
          event_date: formatDateSafe(m.eventDate),
        })),
      ];

      return c.json({
        session_id: sessionId,
        memories: mergedMemories,
        count: mergedMemories.length,
        scope_counts: normalizeScopeCounts(
          mergedMemories.reduce((acc, memory) => {
            const scope = memory.scope_target;
            if (scope && SCOPE_TARGET_VALUES.includes(scope)) {
              acc[scope] = (acc[scope] || 0) + 1;
            }
            return acc;
          }, {} as Record<string, number>)
        ),
        include_pending: includePending,
        pending_overlay_count: pendingEntries.length,
      });
    } catch (error) {
      console.error("Get session memories error:", error);
      return c.json({ error: "Failed to get session memories" }, 500);
    }
  }
);

// Memory Graph - Node/edge graph for visualization and debugging
memoryRoutes.get(
  "/v1/memory/graph",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "query",
    z.object({
      project: z.string().optional(),
      user_id: z.string().optional(),
      session_id: z.string().optional(),
      include_inactive: z.string().optional(),
      limit: z.string().optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const query = c.req.valid("query");

    try {
      const project = await ensureProject(auth.orgId, query.project, auth.isAdmin);

      const includeInactive = query.include_inactive === "true";
      const limit = Math.min(Math.max(parseInt(query.limit || "300", 10), 1), 1000);

      const memories = await prisma.memory.findMany({
        where: {
          projectId: project.id,
          ...(query.user_id ? { userId: query.user_id } : {}),
          ...(query.session_id ? { sessionId: query.session_id } : {}),
          ...(includeInactive ? {} : { isActive: true }),
        },
        orderBy: { createdAt: "desc" },
        take: limit,
        select: {
          id: true,
          content: true,
          memoryType: true,
          confidence: true,
          importance: true,
          userId: true,
          sessionId: true,
          isActive: true,
          createdAt: true,
          updatedAt: true,
        },
      });

      if (memories.length === 0) {
        return c.json({
          nodes: [],
          edges: [],
          stats: {
            node_count: 0,
            edge_count: 0,
            relation_edges: 0,
            sequence_edges: 0,
            memory_types: {},
          },
        });
      }

      const memoryIds = memories.map((m) => m.id);
      const idSet = new Set(memoryIds);

      const relationEdgesRaw = await prisma.memoryRelation.findMany({
        where: {
          OR: [
            { fromMemoryId: { in: memoryIds } },
            { toMemoryId: { in: memoryIds } },
          ],
        },
        select: {
          id: true,
          fromMemoryId: true,
          toMemoryId: true,
          relationType: true,
          confidence: true,
          reasoning: true,
        },
      });

      const relationEdges = relationEdgesRaw
        .filter((r) => idSet.has(r.fromMemoryId) && idSet.has(r.toMemoryId))
        .map((r) => ({
          id: r.id,
          source: r.fromMemoryId,
          target: r.toMemoryId,
          type: r.relationType,
          confidence: r.confidence,
          reasoning: r.reasoning,
          layer: "relation",
        }));

      // Build lightweight chronological "conversation flow" edges.
      const bySession = new Map<string, typeof memories>();
      for (const memory of memories) {
        const sid = memory.sessionId || "no-session";
        const bucket = bySession.get(sid) || [];
        bucket.push(memory);
        bySession.set(sid, bucket);
      }

      const sequenceEdges: Array<{
        id: string;
        source: string;
        target: string;
        type: string;
        confidence: number;
        layer: string;
      }> = [];

      for (const [sessionId, sessionMemories] of bySession) {
        if (sessionMemories.length < 2 || sessionId === "no-session") continue;
        const ordered = [...sessionMemories].sort(
          (a, b) => a.createdAt.getTime() - b.createdAt.getTime()
        );
        for (let i = 1; i < ordered.length; i++) {
          sequenceEdges.push({
            id: `sequence:${ordered[i - 1].id}:${ordered[i].id}`,
            source: ordered[i - 1].id,
            target: ordered[i].id,
            type: "sequence",
            confidence: 1,
            layer: "conversation",
          });
        }
      }

      const nodes = memories.map((m) => ({
        id: m.id,
        label: m.content.slice(0, 140),
        content: m.content,
        memory_type: m.memoryType,
        confidence: m.confidence,
        importance: m.importance,
        user_id: m.userId,
        session_id: m.sessionId,
        is_active: m.isActive,
        created_at: m.createdAt.toISOString(),
        updated_at: m.updatedAt.toISOString(),
      }));

      const memoryTypes = memories.reduce<Record<string, number>>((acc, m) => {
        acc[m.memoryType] = (acc[m.memoryType] || 0) + 1;
        return acc;
      }, {});

      return c.json({
        nodes,
        edges: [...relationEdges, ...sequenceEdges],
        stats: {
          node_count: nodes.length,
          edge_count: relationEdges.length + sequenceEdges.length,
          relation_edges: relationEdges.length,
          sequence_edges: sequenceEdges.length,
          memory_types: memoryTypes,
        },
      });
    } catch (error) {
      console.error("Memory graph error:", error);
      return c.json({ error: "Failed to build memory graph" }, 500);
    }
  }
);

// Conversation-specific graph shortcut
memoryRoutes.get(
  "/v1/memory/graph/conversation/:sessionId",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "query",
    z.object({
      project: z.string().optional(),
      include_inactive: z.string().optional(),
      limit: z.string().optional(),
    })
  ),
  async (c) => {
    const sessionId = c.req.param("sessionId");
    const query = c.req.valid("query");

    const target = new URL(c.req.url);
    target.pathname = "/v1/memory/graph";
    if (query.project) target.searchParams.set("project", query.project);
    target.searchParams.set("session_id", sessionId);
    if (query.include_inactive) target.searchParams.set("include_inactive", query.include_inactive);
    if (query.limit) target.searchParams.set("limit", query.limit);

    return c.redirect(target.toString(), 307);
  }
);

// ──────────────────────────────────────────────────────────────
// Get User Profile - Long-term user preferences and facts
// ──────────────────────────────────────────────────────────────

memoryRoutes.get(
  "/v1/memory/profile/:userId",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "query",
    z.object({
      project: z.string().optional(),
      memory_types: z.string().optional(), // Comma-separated
      include_pending: z.string().optional(), // "true" | "false"
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const userId = c.req.param("userId");
    const query = c.req.valid("query");

    try {
      const project = await ensureProject(auth.orgId, query.project, auth.isAdmin);

      const memoryTypes = query.memory_types ? query.memory_types.split(",") : undefined;
      const includePending = query.include_pending !== "false";

      const memories = await getUserProfile({
        userId,
        projectId: project.id,
        memoryTypes,
      });

      let pendingEntries: Awaited<ReturnType<typeof getPendingOverlayEntries>> = [];
      if (includePending) {
        pendingEntries = await getPendingOverlayEntries({
          orgId: auth.orgId,
          projectId: project.id,
          userId,
          limit: 100,
        });
      }

      // Robust date formatter
      const formatDateSafe = (d: any): string | null => {
        if (d === null || d === undefined) return null;
        if (typeof d === 'string') {
          const date = new Date(d);
          return isNaN(date.getTime()) ? null : d;
        }
        if (d instanceof Date) {
          return isNaN(d.getTime()) ? null : d.toISOString();
        }
        return null;
      };

      const mergedMemories = [
        ...pendingEntries.map((entry, idx) => ({
          id: `pending:${entry.job_id}:${idx}`,
          content: entry.content,
          type: "event",
          scope: "USER",
          scope_target: "USER",
          entities: [],
          importance: 0.5,
          confidence: 0.7,
          document_date: entry.created_at,
          pending: true,
          job_id: entry.job_id,
        })),
        ...memories.map((m) => ({
          id: m.id,
          content: m.content,
          type: m.memoryType,
          scope: m.scope,
          scope_target: m.scope,
          user_id: m.userId ?? null,
          session_id: m.sessionId ?? null,
          agent_id: m.agentId ?? null,
          task_id: (m as any).taskId ?? null,
          entities: m.entityMentions || [],
          importance: m.importance,
          confidence: m.confidence,
          semantic_status: semanticStatusFromMetadata(m.metadata),
          document_date: formatDateSafe(m.documentDate),
        })),
      ];

      return c.json({
        user_id: userId,
        memories: mergedMemories,
        count: mergedMemories.length,
        scope_counts: normalizeScopeCounts(
          mergedMemories.reduce((acc, memory) => {
            const scope = memory.scope_target;
            if (scope && SCOPE_TARGET_VALUES.includes(scope)) {
              acc[scope] = (acc[scope] || 0) + 1;
            }
            return acc;
          }, {} as Record<string, number>)
        ),
        include_pending: includePending,
        pending_overlay_count: pendingEntries.length,
      });
    } catch (error) {
      console.error("Get user profile error:", error);
      return c.json({ error: "Failed to get user profile" }, 500);
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Get Memory - Exact fetch by id
// ──────────────────────────────────────────────────────────────

memoryRoutes.get(
  "/v1/memory/:memoryId",
  rateLimitMiddleware(RateLimits.query),
  async (c) => {
    const auth = c.get("auth");
    const memoryId = c.req.param("memoryId");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);

    try {
      const memory = await prisma.memory.findFirst({
        where: {
          id: memoryId,
          orgId: auth.orgId,
        },
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

      if (!memory) {
        return memoryError(c, 404, "MEMORY_NOT_FOUND", "Memory not found", traceId);
      }

      return c.json({
        success: true,
        trace_id: traceId,
        memory_id: memory.id,
        semantic_status: semanticStatusFromMetadata(memory.metadata),
        memory: {
          id: memory.id,
          project_id: memory.projectId,
          org_id: memory.orgId,
          content: memory.content,
          type: memory.memoryType,
          user_id: memory.userId,
          session_id: memory.sessionId,
          agent_id: memory.agentId,
          importance: memory.importance,
          confidence: memory.confidence,
          semantic_status: semanticStatusFromMetadata(memory.metadata),
          metadata: memory.metadata,
          access_count: memory.accessCount,
          created_at: memory.createdAt,
          updated_at: memory.updatedAt,
          document_date: memory.documentDate,
          event_date: memory.eventDate,
          valid_from: memory.validFrom,
          valid_until: memory.validUntil,
          is_active: memory.isActive,
        },
      });
    } catch (error) {
      console.error("Get memory error:", error);
      return memoryError(c, 500, "INTERNAL_ERROR", "Failed to get memory", traceId);
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Get Version Chain - History of a memory's updates
// ──────────────────────────────────────────────────────────────

memoryRoutes.get(
  "/v1/memory/:memoryId/versions",
  rateLimitMiddleware(RateLimits.query),
  async (c) => {
    const auth = c.get("auth");
    const memoryId = c.req.param("memoryId");

    try {
      // Verify memory belongs to this org
      const memory = await prisma.memory.findFirst({
        where: {
          id: memoryId,
          orgId: auth.orgId,
        },
      });

      if (!memory) {
        return c.json({ error: "Memory not found" }, 404);
      }

      const versions = await getVersionChain(memoryId ?? "", prisma);

      return c.json({
        memory_id: memoryId,
        current_version: memory.version,
        versions: versions.map((v) => ({
          version: v.version,
          content: v.content,
          updated_at: (v as any).updatedAt ?? null,
          superseded_by: (v as any).supersededBy ?? null,
        })),
        count: versions.length,
      });
    } catch (error) {
      console.error("Get version chain error:", error);
      return c.json({ error: "Failed to get version chain" }, 500);
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Update Memory - Create new version
// ──────────────────────────────────────────────────────────────

memoryRoutes.put(
  "/v1/memory/:memoryId",
  rateLimitMiddleware(RateLimits.mutation),
  zValidator(
    "json",
    z.object({
      content: z.string().min(1).max(10000),
      confidence: z.number().min(0).max(1).optional(),
      reason: z.string().optional(),
      async: z.boolean().optional(),
      write_mode: z.enum(["async", "sync"]).optional(),
      webhook_url: z.string().url().optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const memoryId = c.req.param("memoryId");
    const body = c.req.valid("json");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);

    try {
      // Verify memory belongs to this org
      const existing = await prisma.memory.findFirst({
        where: {
          id: memoryId,
          orgId: auth.orgId,
        },
      });

      if (!existing) {
        return c.json({ error: "Memory not found" }, 404);
      }

      const requestedWriteMode =
        body.write_mode ||
        (body.async === false
          ? "sync"
          : body.async === true
            ? "async"
            : DEFAULT_WRITE_MODE === "sync"
              ? "sync"
              : "async");
      const syncWriteRequested = requestedWriteMode === "sync";

      if (syncWriteRequested && !canUseSyncWrite(auth)) {
        return c.json(
          {
            error: "write_mode=sync is restricted to internal/admin tenants",
            trace_id: traceId,
          },
          403
        );
      }

      const shouldAsync = !syncWriteRequested;

      // Async processing mode
      if (shouldAsync) {
        // For updates, we create a new memory version via the queue
        const jobId = await ingestionQueue.createJob({
          orgId: auth.orgId,
          projectId: existing.projectId ?? "",
          userId: auth.userId ?? "",
          memories: [{
            content: body.content,
            memory_type: existing.memoryType,
            user_id: existing.userId || undefined,
            session_id: existing.sessionId || undefined,
            agent_id: existing.agentId || undefined,
            importance: existing.importance,
            metadata: {
              ...(existing.metadata as any || {}),
              update_reason: body.reason,
              previous_memory_id: memoryId,
              previous_version: existing.version,
              traceId,
            },
          }],
          webhookUrl: body.webhook_url,
        });


        return c.json({
          success: true,
          mode: 'async',
          trace_id: traceId,
          job_id: jobId,
          status_url: `/v1/memory/jobs/${jobId}`,
          legacy_status_url: `/v1/jobs/${jobId}`,
          consistency: "eventual",
          visibility_sla_ms: MEMORY_VISIBILITY_SLA_MS,
          message: "Memory update queued for processing",
          previous_memory_id: memoryId,
        }, 202);
      }

      // Synchronous mode (legacy/fallback)
      const updated = await updateMemory({
        memoryId: memoryId ?? "",
        newContent: body.content,
        reasoning: body.reason,
      });


      return c.json({
        success: true,
        mode: 'sync',
        trace_id: traceId,
        memory: {
          id: updated.newMemoryId,
          previous_version: updated.oldMemoryId,
        },
      });
    } catch (error) {
      console.error(`[${traceId}] Update memory error:`, error);
      return c.json({
        error: "Failed to update memory",
        trace_id: traceId,
      }, 500);
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Delete Memory - Soft delete
// ──────────────────────────────────────────────────────────────

memoryRoutes.post(
  "/v1/memory/:memoryId/confirm",
  rateLimitMiddleware(RateLimits.mutation),
  zValidator(
    "json",
    z.object({
      reason: z.string().min(1).max(500).optional(),
      metadata: z.record(z.any()).optional(),
    }).optional().default({}),
  ),
  async (c) => {
    const auth = c.get("auth");
    const body = c.req.valid("json");
    const memoryId = c.req.param("memoryId");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);

    const config = getExtractionPhase0Config();
    if (!config.memory_confirm_endpoint_enabled) {
      return memoryError(c, 404, "NOT_FOUND", "Endpoint not enabled", traceId);
    }

    if (config.memory_confirm_backend_only && !isBackendConfirmAuth(auth)) {
      return memoryError(c, 403, "NOT_AUTHORIZED", "Backend credential required", traceId);
    }

    try {
      const memory = await prisma.memory.findFirst({
        where: {
          id: memoryId,
          orgId: auth.orgId,
        },
        select: {
          id: true,
          projectId: true,
          userId: true,
          sessionId: true,
        },
      });
      if (!memory) {
        return memoryError(c, 404, "MEMORY_NOT_FOUND", "Memory not found", traceId);
      }

      const now = Date.now();
      const [confirmCountLast5m, confirmCountLast1h] = await Promise.all([
        countConfirmEventsForTenant({
          tenantId: auth.orgId,
          since: new Date(now - 5 * 60 * 1000),
        }),
        countConfirmEventsForTenant({
          tenantId: auth.orgId,
          since: new Date(now - 60 * 60 * 1000),
        }),
      ]);
      const rateCheck = evaluateConfirmRateLimits({
        confirmCountLast5m,
        confirmCountLast1h,
        burstLimit: config.memory_confirm_burst_limit_5m,
        hourlyLimit: config.memory_confirm_rate_limit_per_hour,
      });
      if (!rateCheck.allowed) {
        await emitMemoryConfirmAnomaly({
          tenantId: auth.orgId,
          projectId: memory.projectId,
          reason: "rate_limit_exceeded",
          details: {
            trace_id: traceId,
            memory_id: memoryId,
            confirm_count_last_5m: confirmCountLast5m,
            confirm_count_last_1h: confirmCountLast1h,
            reason: rateCheck.reason,
          },
        });
        return c.json({
          success: false,
          error: {
            code: "RATE_LIMITED",
            message: "Confirm rate limit exceeded",
            reason: rateCheck.reason,
          },
          retry_after_seconds: rateCheck.retryAfterSeconds ?? 60,
          trace_id: traceId,
        }, 429);
      }

      const actorId = auth.keyId || auth.userId || auth.actorId || "backend";
      const serviceId = c.req.header("x-service-id") || auth.authType || "unknown";
      const sourceIp = getClientIp(c);
      await emitMemoryConfirmEvent({
        tenantId: auth.orgId,
        projectId: memory.projectId,
        memoryId,
        confirmedByType: "backend",
        confirmedById: actorId,
        reason: body.reason,
        metadata: {
          trace_id: traceId,
          source_ip: sourceIp,
          service_id: serviceId,
          user_id: memory.userId || null,
          session_id: memory.sessionId || null,
          confirm_count_last_5m: confirmCountLast5m + 1,
          confirm_count_last_1h: confirmCountLast1h + 1,
          ...(body.metadata || {}),
        },
      });

      // anomaly detection not available in OSS

      return c.json({
        success: true,
        memory_id: memoryId,
        tenant_id: auth.orgId,
        confirmed_by_type: "backend",
        confirmed_by_id: actorId,
        trace_id: traceId,
      });
    } catch (error) {
      console.error(`[${traceId}] Memory confirm failed:`, error);
      return memoryError(c, 500, "INTERNAL_ERROR", "Failed to confirm memory", traceId);
    }
  }
);

memoryRoutes.delete(
  "/v1/memory/:memoryId",
  rateLimitMiddleware(RateLimits.mutation),
  async (c) => {
    const auth = c.get("auth");
    const memoryId = c.req.param("memoryId");

    try {
      // Verify memory belongs to this org
      const memory = await prisma.memory.findFirst({
        where: {
          id: memoryId,
          orgId: auth.orgId,
        },
      });

      if (!memory) {
        return c.json({ error: "Memory not found" }, 404);
      }

      // Soft delete by setting expiresAt to now
      await prisma.memory.update({
        where: { id: memoryId },
        data: {
          expiresAt: new Date(),
          updatedAt: new Date(),
        },
      });

      return c.json({
        success: true,
        message: "Memory deleted",
        memory_id: memoryId,
      });
    } catch (error) {
      console.error("Delete memory error:", error);
      return c.json({ error: "Failed to delete memory" }, 500);
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Bulk Delete Memories
// ──────────────────────────────────────────────────────────────

memoryRoutes.post(
  "/v1/memory/bulk-delete",
  rateLimitMiddleware(RateLimits.mutation),
  zValidator(
    "json",
    z.object({
      memory_ids: z.array(z.string()).min(1).max(500),
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const { memory_ids } = c.req.valid("json");

    try {
      // Only soft-delete memories that belong to this org
      const result = await prisma.memory.updateMany({
        where: {
          id: { in: memory_ids },
          orgId: auth.orgId,
        },
        data: {
          expiresAt: new Date(),
          updatedAt: new Date(),
        },
      });

      return c.json({
        success: true,
        deleted: result.count,
        requested: memory_ids.length,
      });
    } catch (error) {
      console.error("Bulk delete memory error:", error);
      return c.json({ error: "Failed to bulk delete memories" }, 500);
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Cleanup Expired Memories (admin)
// Purges soft-deleted and TTL-expired memories from the database
// ──────────────────────────────────────────────────────────────

memoryRoutes.post(
  "/v1/admin/memory/cleanup",
  rateLimitMiddleware(RateLimits.mutation),
  zValidator(
    "json",
    z.object({
      project_id: z.string().optional(),
      dry_run: z.boolean().default(false),
    }).default({})
  ),
  async (c) => {
    const auth = c.get("auth");
    const { project_id, dry_run } = c.req.valid("json");

    // Admin-only
    if (!auth.isAdmin) {
      return c.json({ error: "Admin access required" }, 403);
    }

    try {
      const where = {
        orgId: auth.orgId,
        expiresAt: { lt: new Date() },
        ...(project_id ? { projectId: project_id } : {}),
      };

      const count = await prisma.memory.count({ where });

      if (dry_run) {
        return c.json({ dry_run: true, would_delete: count });
      }

      await prisma.memory.deleteMany({ where });

      return c.json({ success: true, deleted: count });
    } catch (error) {
      console.error("Memory cleanup error:", error);
      return c.json({ error: "Cleanup failed" }, 500);
    }
  }
);

// ─── Dialectic query ─────────────────────────────────────────────────────────
// Answer a natural-language question about a user from their stored memories

memoryRoutes.post(
  "/v1/memory/dialectic",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "json",
    z.object({
      user_id: z.string().min(1),
      question: z.string().min(1).max(1000),
      project: z.string().optional(),
      reasoning_level: z.enum(["minimal", "low", "medium", "high"]).optional().default("medium"),
    })
  ),
  async (c) => {
    const auth = c.get("auth") as AuthContext;
    const { user_id, question, project, reasoning_level } = c.req.valid("json");

    if (!project) {
      return c.json({ success: false, error: "project is required for dialectic queries" }, 400);
    }

    try {
      const result = await dialecticQuery({
        userId: user_id,
        projectId: project,
        query: question,
        reasoningLevel: reasoning_level,
      });
      return c.json({ success: true, ...result });
    } catch (err: any) {
      return c.json({ success: false, error: err?.message || "Dialectic query failed" }, 500);
    }
  }
);

// ─── User model synthesis ─────────────────────────────────────────────────────
// Build a structured profile of a user from their memories

memoryRoutes.get(
  "/v1/memory/user-model/:userId",
  rateLimitMiddleware(RateLimits.query),
  zValidator("query", z.object({ project: z.string().optional() })),
  async (c) => {
    const auth = c.get("auth");
    const userId = c.req.param("userId");
    const query = c.req.valid("query");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);

    try {
      const project = await ensureProject(auth.orgId, query.project, auth.isAdmin);
      const model = await synthesizeUserModel({
        userId,
        projectId: project.id,
      });
      return c.json({
        ...model,
        trace_id: traceId,
      });
    } catch (error) {
      console.error("Get user model error:", error);
      return memoryError(c, 500, "INTERNAL_ERROR", "Failed to build user model", traceId);
    }
  }
);

memoryRoutes.post(
  "/v1/memory/profile/:userId/gaps",
  rateLimitMiddleware(RateLimits.query),
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      context: z.string().min(1),
      limit: z.number().int().min(1).max(10).optional(),
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const userId = c.req.param("userId");
    const body = c.req.valid("json");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);

    try {
      const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);
      const gaps = await detectUserGaps({
        userId,
        projectId: project.id,
        context: body.context,
        limit: body.limit,
      });
      return c.json({
        ...gaps,
        trace_id: traceId,
      });
    } catch (error) {
      console.error("Get user gaps error:", error);
      return memoryError(c, 500, "INTERNAL_ERROR", "Failed to detect user gaps", traceId);
    }
  }
);

// ──────────────────────────────────────────────────────────────
// Agent Self-Model — what the agent knows about itself
// ──────────────────────────────────────────────────────────────

memoryRoutes.get(
  "/v1/memory/agent/:agentId/model",
  rateLimitMiddleware(RateLimits.query),
  zValidator("query", z.object({ project: z.string().optional() })),
  async (c) => {
    const auth = c.get("auth");
    const agentId = c.req.param("agentId");
    const query = c.req.valid("query");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);
    try {
      const project = await ensureProject(auth.orgId, query.project, auth.isAdmin);
      const model = await synthesizeAgentModel({ agentId, projectId: project.id });
      return c.json({ ...model, trace_id: traceId });
    } catch (error) {
      console.error("Agent model error:", error);
      return memoryError(c, 500, "INTERNAL_ERROR", "Failed to build agent model", traceId);
    }
  }
);

memoryRoutes.post(
  "/v1/memory/agent/:agentId/seed",
  rateLimitMiddleware(RateLimits.mutation),
  zValidator(
    "json",
    z.object({
      project: z.string().optional(),
      content: z.string().min(1).max(32000),
      source: z.string().optional().default("manual"),
    })
  ),
  async (c) => {
    const auth = c.get("auth");
    const agentId = c.req.param("agentId");
    const body = c.req.valid("json");
    const traceId = getTraceId(c);
    c.header("x-trace-id", traceId);
    try {
      const project = await ensureProject(auth.orgId, body.project, auth.isAdmin);
      const result = await seedAgentIdentity({
        agentId,
        projectId: project.id,
        orgId: auth.orgId,
        content: body.content,
        source: body.source,
      });
      return c.json({ agent_id: agentId, ...result, trace_id: traceId });
    } catch (error) {
      console.error("Agent seed error:", error);
      return memoryError(c, 500, "INTERNAL_ERROR", "Failed to seed agent identity", traceId);
    }
  }
);

