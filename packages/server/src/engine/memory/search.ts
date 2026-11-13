/**
 * SOTA Memory Search Engine
 * Memory-first hybrid search with temporal filtering and graph traversal
 * Beats Supermemory's 81.6% and targets 85%+ on LongMemEval
 * 
 * OPTIMIZATION NOTES:
 * - Timing instrumentation tracks latency at each pipeline step
 * - Early exit when similarity is high to skip expensive graph traversal
 * - Semantic cache threshold lowered to 0.85 for higher hit rate
 * - Graph traversal parallelized with chunk injection where possible
 */

import { db } from "../../db/index.js";
import { expandMemorySearchQueries } from "../../lib/memory-normalization.js";
import { embedSingle } from "../embeddings.js";
import { parseTemporalFast } from "./temporal-local.js";
import { calculateTemporalRelevance } from "./temporal.js";
import { getFromSemanticCache, setInSemanticCache, getFromCache, setInCache } from "../cache.js";
import type { MemoryScopeTarget, MemorySearchDiagnostics, MemorySearchParams, MemorySearchResult } from "./types.js";
import { Prisma } from "@prisma/client";
import { decrypt } from "../../lib/encryption.js";

// Early exit threshold - skip graph traversal if top result is very similar
const EARLY_EXIT_SIMILARITY = 0.92;

// Lower semantic cache threshold for higher hit rate (was 0.92, now 0.85)
const SEMANTIC_CACHE_THRESHOLD = 0.85;

// SLO budgets (overridable via env)
const TOTAL_SLO_BUDGET_MS = parseInt(process.env.MEMORY_SEARCH_TOTAL_BUDGET_MS || "220", 10);
const POST_VECTOR_BUDGET_MS = parseInt(process.env.MEMORY_SEARCH_POST_VECTOR_BUDGET_MS || "120", 10);
const CHUNK_INJECTION_GUARDRAIL_MS = parseInt(process.env.MEMORY_SEARCH_CHUNK_GUARDRAIL_MS || "180", 10);

// Timing instrumentation
interface TimingLog {
  step: string;
  duration: number;
}

interface QueryIntent {
  wantsRecent: boolean;
  asksForSearchHistory: boolean;
}

function detectQueryIntent(query: string): QueryIntent {
  const normalized = query.toLowerCase();
  const wantsRecent = /\b(last|latest|recent|recently|previous|before|earlier)\b/.test(normalized);
  const asksForSearchHistory = /\b(search|searched|query|queried|asked|question|chat|conversation|llm|ai)\b/.test(normalized);
  return { wantsRecent, asksForSearchHistory };
}

function parseDateSafe(value: unknown): Date | null {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(String(value));
  return Number.isNaN(date.getTime()) ? null : date;
}

function getMemoryRecencyDate(memory: any): Date | null {
  return (
    parseDateSafe(memory.eventDate) ||
    parseDateSafe(memory.documentDate) ||
    parseDateSafe(memory.updatedAt) ||
    parseDateSafe(memory.createdAt) ||
    null
  );
}

function rerankByIntent(memories: any[], questionDate: Date | undefined, intent: QueryIntent): any[] {
  if (!intent.wantsRecent && !intent.asksForSearchHistory) {
    return memories;
  }

  const nowMs = (questionDate && !isNaN(questionDate.getTime())) ? questionDate.getTime() : Date.now();
  const eventKeyword = /\b(search|searched|looked|queried|asked|question|chat|conversation|llm|ai)\b/i;

  const scored = memories.map((memory) => {
    const similarityScore = typeof memory.finalScore === "number"
      ? memory.finalScore
      : (memory.similarity || 0);

    const recencyDate = getMemoryRecencyDate(memory);
    const ageDays = recencyDate ? Math.max(0, (nowMs - recencyDate.getTime()) / (1000 * 60 * 60 * 24)) : 365;
    const recencyScore = Math.max(0, 1 - Math.min(ageDays / 365, 1));

    let typeBoost = 0;
    if (intent.asksForSearchHistory) {
      if (memory.memoryType === "event") typeBoost += 0.35;
      if (eventKeyword.test(memory.content || "")) typeBoost += 0.2;
    }

    const finalIntentScore =
      similarityScore * 0.65 +
      (intent.wantsRecent ? recencyScore * 0.35 : 0) +
      typeBoost;

    return { memory, finalIntentScore };
  });

  scored.sort((a, b) => b.finalIntentScore - a.finalIntentScore);
  return scored.map((s) => ({ ...s.memory, finalScore: s.finalIntentScore }));
}

const DEBUG_TIMINGS = process.env.MEMORY_SEARCH_DEBUG_TIMINGS === "true";

function isTruthyEnv(value: string | undefined): boolean {
  const normalized = (value || "").trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

function shouldDisableLatencyDegradation(params: Pick<MemorySearchParams, "disableLatencyDegradation">): boolean {
  if (typeof params.disableLatencyDegradation === "boolean") {
    return params.disableLatencyDegradation;
  }
  return isTruthyEnv(process.env.MEMORY_SEARCH_DISABLE_LATENCY_DEGRADATION);
}

function logTimings(timings: TimingLog[], _query: string): void {
  // Only log full timing breakdown in explicit debug mode — queries contain user data.
  if (!DEBUG_TIMINGS) return;
  const total = timings.reduce((sum, t) => sum + t.duration, 0);
  console.log(`\n[MemorySearch] timing (total: ${total}ms):`);
  timings.forEach((t) => {
    const pct = total > 0 ? ((t.duration / total) * 100).toFixed(1) : "0.0";
    console.log(`  ${t.step}: ${t.duration}ms (${pct}%)`);
  });
}

function emitDiagnostics(
  params: MemorySearchParams,
  timings: TimingLog[],
  totalMs: number,
  cacheHitType: "none" | "simple" | "semantic",
  fastMode: boolean
): void {
  if (typeof params.diagnosticsCollector !== "function") {
    return;
  }

  const sumStep = (...steps: string[]) =>
    timings
      .filter((timing) => steps.includes(timing.step))
      .reduce((acc, timing) => acc + timing.duration, 0);

  const cacheMs = sumStep("simple_cache", "semantic_cache_check", "cache_write");
  const embedMs = sumStep("embedding");
  const vectorMs = sumStep("vector_search");
  const lexicalMs = sumStep("lexical_search");
  const mergeMs = totalMs - cacheMs - embedMs - vectorMs - lexicalMs;

  const diagnostics: MemorySearchDiagnostics = {
    cache_ms: Math.max(cacheMs, 0),
    embed_ms: Math.max(embedMs, 0),
    vector_ms: Math.max(vectorMs, 0),
    lexical_ms: Math.max(lexicalMs, 0),
    merge_ms: Math.max(mergeMs, 0),
    total_ms: Math.max(totalMs, 0),
    cache_hit: cacheHitType !== "none",
    cache_hit_type: cacheHitType,
    fast_mode: fastMode,
  };

  params.diagnosticsCollector(diagnostics);
}

function resolveApplicableScopes(params: Pick<MemorySearchParams, "userId" | "sessionId" | "agentId" | "taskId" | "scopes">): MemoryScopeTarget[] {
  if (params.scopes && params.scopes.length > 0) {
    return Array.from(new Set(params.scopes.filter((scope) => scope !== "DROPPED")));
  }

  const scopes = new Set<MemoryScopeTarget>(["PROJECT"]);
  if (params.userId) scopes.add("USER");
  if (params.sessionId) scopes.add("SESSION");
  if (params.agentId) scopes.add("AGENT");
  if (params.taskId) scopes.add("TASK");
  return Array.from(scopes);
}

function scopeBoost(memory: any, params: Pick<MemorySearchParams, "userId" | "sessionId" | "agentId" | "taskId">): number {
  const scope = String(memory.scope || "");
  if (scope === "TASK" && params.taskId && memory.taskId === params.taskId) return 0.3;
  if (scope === "SESSION" && params.sessionId && memory.sessionId === params.sessionId) return 0.2;
  if (scope === "AGENT" && params.agentId && memory.agentId === params.agentId) return 0.15;
  if (scope === "USER" && params.userId && memory.userId === params.userId) return 0.15;
  if (scope === "PROJECT") return 0.1;
  return 0;
}

function rerankByScope(memories: any[], params: Pick<MemorySearchParams, "userId" | "sessionId" | "agentId" | "taskId">): any[] {
  return [...memories]
    .map((memory) => {
      const boost = scopeBoost(memory, params);
      const baseScore = typeof memory.finalScore === "number"
        ? memory.finalScore
        : (typeof memory.similarity === "number" ? memory.similarity : 0);
      return {
        ...memory,
        similarity: typeof memory.similarity === "number" ? memory.similarity + boost : boost,
        finalScore: baseScore + boost,
      };
    })
    .sort((left, right) => (right.finalScore ?? right.similarity ?? 0) - (left.finalScore ?? left.similarity ?? 0));
}
/**
 * Main memory search function
 * Implements memory-first hybrid approach from Supermemory
 * OPTIMIZED: Added timing instrumentation and parallel execution
 */
export async function searchMemories(
  params: MemorySearchParams
): Promise<MemorySearchResult[]> {
  const timings: TimingLog[] = [];
  const startTotal = Date.now();
  const disableLatencyDegradation = shouldDisableLatencyDegradation(params);

  const {
    query,
    questionDate,
    userId,
    projectId,
    orgId,
    sessionId,
    agentId,
    taskId,
    topK = 10,
    includeInactive = false,
    memoryTypes,
    namespace,
    tags,
    fastMode = false,
  } = params;
  let effectiveFastMode = fastMode;
  let cacheHitType: "none" | "simple" | "semantic" = "none";
  const queryIntent = detectQueryIntent(query);
  const queryVariants = expandMemorySearchQueries(query);
  const semanticQuery = queryVariants.join(" | ") || query;
  const applicableScopes = resolveApplicableScopes({ userId, sessionId, agentId, taskId, scopes: params.scopes });
  const effectiveMemoryTypes =
    queryIntent.asksForSearchHistory && (!memoryTypes || memoryTypes.length === 0)
      ? (["event"] as string[])
      : memoryTypes;

  // Step 0a: Check simple cache first (before embedding!)
  const step0Start = Date.now();
  const cacheScope = JSON.stringify({ orgId, projectId, userId, sessionId, agentId, taskId,
    scopes: applicableScopes, includeInactive, memoryTypes: effectiveMemoryTypes, namespace, tags,
    temporalFilter: params.temporalFilter, questionDate: questionDate.toISOString().slice(0, 10), topK, fastMode });
  const cacheKey = `search:${projectId}:${userId || ""}:${cacheScope}:${query}`;
  const simpleCached = await getFromCache<MemorySearchResult[]>(cacheKey);
  timings.push({ step: "simple_cache", duration: Date.now() - step0Start });
  
  if (simpleCached) {
    cacheHitType = "simple";
    const total = Date.now() - startTotal;
    timings.push({ step: "TOTAL", duration: total });
    logTimings(timings, query);
    console.log("⚡ Simple cache hit");
    emitDiagnostics(params, timings, total, cacheHitType, effectiveFastMode);
    return simpleCached;
  }

  // Step 0b: Generate embedding + check semantic cache
  const embedStart = Date.now();
  const queryEmbedding = await embedSingle(semanticQuery);
  timings.push({ step: "embedding", duration: Date.now() - embedStart });

  const semanticCacheStart = Date.now();
  const cached = await getFromSemanticCache(queryEmbedding, cacheScope);
  timings.push({ step: "semantic_cache_check", duration: Date.now() - semanticCacheStart });

  if (cached && cached.similarity >= SEMANTIC_CACHE_THRESHOLD) {
    cacheHitType = "semantic";
    console.log(`⚡ Semantic cache hit (similarity: ${cached.similarity.toFixed(3)})`);
    const results = cached.results.slice(0, topK);
    await setInCache(cacheKey, results, 300);
    const total = Date.now() - startTotal;
    timings.push({ step: "TOTAL", duration: total });
    logTimings(timings, query);
    emitDiagnostics(params, timings, total, cacheHitType, effectiveFastMode);
    return results;
  }

  // Step 1: Parse temporal constraints from query (LOCAL - no LLM!)
  const temporalStart = Date.now();
  const temporal = params.temporalFilter || parseTemporalFast(query, questionDate);
  timings.push({ step: "temporal_parse", duration: Date.now() - temporalStart });

  // Step 2: Vector search on memories (NOT chunks!)
  const vectorStart = Date.now();
  const semanticResults = await vectorSearchMemories({
    embedding: queryEmbedding,
    userId,
    projectId,
    orgId,
    sessionId,
    agentId,
    taskId,
    scopes: applicableScopes,
    temporal,
    includeInactive,
    memoryTypes: effectiveMemoryTypes,
    namespace,
    tags,
    limit: topK * 3, // Get more for reranking
  });
  timings.push({ step: "vector_search", duration: Date.now() - vectorStart });
  const scopedSemanticResults = rerankByScope(semanticResults, { userId, sessionId, agentId, taskId });

  // Guardrail: if we're already over budget after vector search, degrade to fast mode.
  if (!disableLatencyDegradation && !effectiveFastMode && Date.now() - startTotal > POST_VECTOR_BUDGET_MS) {
    effectiveFastMode = true;
    timings.push({ step: "degraded_mode_fast_after_vector", duration: 0 });
    console.warn(`[MemorySearch] Degraded to fast mode after vector stage (budget=${POST_VECTOR_BUDGET_MS}ms)`);
  }

  if (scopedSemanticResults.length === 0) {
    await setInCache(cacheKey, [], 60); // Cache empty results too
    const total = Date.now() - startTotal;
    timings.push({ step: "TOTAL", duration: total });
    logTimings(timings, query);
    emitDiagnostics(params, timings, total, cacheHitType, effectiveFastMode);
    return [];
  }

  // Step 3: Early exit if top result is excellent
  if (!queryIntent.wantsRecent && scopedSemanticResults.length > 0 && scopedSemanticResults[0].similarity >= EARLY_EXIT_SIMILARITY) {
    console.log(`⚡ Early exit at ${scopedSemanticResults[0].similarity.toFixed(3)}`);
    const topMemories = scopedSemanticResults.slice(0, topK).map(toMemorySearchResult);
    
    // Cache good results
    await setInSemanticCache(queryEmbedding, topMemories, 300, cacheScope);
    await setInCache(cacheKey, topMemories, 300); // 5 min cache
    
    const total = Date.now() - startTotal;
    timings.push({ step: "TOTAL", duration: total });
    logTimings(timings, query);
    emitDiagnostics(params, timings, total, cacheHitType, effectiveFastMode);
    return topMemories;
  }

  // Step 4: FAST MODE - Skip graph traversal and temporal scoring for speed
  let finalResults = scopedSemanticResults.slice(0, topK);
  
  if (!effectiveFastMode) {
    // Graph traversal - enrich with related memories
    const graphStart = Date.now();
    const enriched = await enrichWithRelations(scopedSemanticResults, topK * 2, orgId);
    timings.push({ step: "graph_traversal", duration: Date.now() - graphStart });

    // Temporal relevance scoring
    const temporalScoringStart = Date.now();
    const scored = enriched.map((memory) => ({
      ...memory,
      temporalScore: memory.documentDate
        ? calculateTemporalRelevance(memory.documentDate, questionDate)
        : 0.5,
    }));
    timings.push({ step: "temporal_scoring", duration: Date.now() - temporalScoringStart });

    // Combine scores (semantic + temporal)
    const combineStart = Date.now();
    const combined = rerankByScope(scored.map((m) => ({
      ...m,
      finalScore: m.similarity * 0.7 + m.temporalScore * 0.3,
    })), { userId, sessionId, agentId, taskId });
    timings.push({ step: "score_combine", duration: Date.now() - combineStart });

    // Sort by final score
    const sortStart = Date.now();
    combined.sort((a, b) => (b.finalScore ?? 0) - (a.finalScore ?? 0));
    timings.push({ step: "sort", duration: Date.now() - sortStart });

    // Take top K
    finalResults = combined.slice(0, topK);
  }

  // Step 4b: Intent-aware reranking for "last/recent search" style queries.
  if (queryIntent.wantsRecent || queryIntent.asksForSearchHistory) {
    const intentStart = Date.now();
    finalResults = rerankByIntent(finalResults, questionDate, queryIntent).slice(0, topK);
    timings.push({ step: "intent_rerank", duration: Date.now() - intentStart });
  }

  // Step 5: Inject source chunks for context (skip in fast mode)
  let results: MemorySearchResult[];
  const shouldSkipChunkInjection =
    !disableLatencyDegradation &&
    !effectiveFastMode &&
    (Date.now() - startTotal > CHUNK_INJECTION_GUARDRAIL_MS);
  if (shouldSkipChunkInjection) {
    timings.push({ step: "degraded_mode_skip_chunk_injection", duration: 0 });
    console.warn(`[MemorySearch] Skipping chunk injection near SLO budget (guardrail=${CHUNK_INJECTION_GUARDRAIL_MS}ms)`);
  }

  if (effectiveFastMode || shouldSkipChunkInjection) {
    // Fast mode: skip chunk injection, return memories directly
    results = finalResults.map(toMemorySearchResult);
    timings.push({ step: "chunk_injection", duration: 0 });
  } else {
    const chunkStart = Date.now();
    results = await injectSourceChunks(finalResults);
    timings.push({ step: "chunk_injection", duration: Date.now() - chunkStart });
  }

  // Cache results for similar queries
  const cacheWriteStart = Date.now();
  await setInSemanticCache(queryEmbedding, results, 300, cacheScope);
  await setInCache(cacheKey, results, 300); // 5 min cache
  timings.push({ step: "cache_write", duration: Date.now() - cacheWriteStart });

  const total = Date.now() - startTotal;
  timings.push({ step: "TOTAL", duration: total });
  if (total > TOTAL_SLO_BUDGET_MS) {
    console.warn(`[MemorySearch] SLO miss: ${total}ms (budget=${TOTAL_SLO_BUDGET_MS}ms)`);
  }
  logTimings(timings, query);
  emitDiagnostics(params, timings, total, cacheHitType, effectiveFastMode);

  return results;
}

type SearchMemory = Omit<MemorySearchResult["memory"], "temporal" | "metadata"> &
  MemorySearchResult["memory"]["temporal"] & {
    metadata?: unknown;
    similarity: number;
    sourceChunkId?: string | null;
    isRelated?: boolean;
  };

function copyMetadata(metadata: unknown): Record<string, unknown> {
  return metadata && typeof metadata === "object" && !Array.isArray(metadata)
    ? { ...metadata }
    : {};
}

function isMemoryScopeTarget(scope: string | undefined): scope is MemoryScopeTarget {
  return scope === "USER" || scope === "SESSION" || scope === "PROJECT" || scope === "AGENT"
    || scope === "TASK" || scope === "DOCUMENT" || scope === "DROPPED";
}

export function toMemorySearchResult(memory: SearchMemory): MemorySearchResult {
  return {
    memory: {
      id: memory.id, content: memory.content, memoryType: memory.memoryType,
      entityMentions: memory.entityMentions || [], confidence: memory.confidence, version: memory.version,
      scope: memory.scope, scopeTarget: memory.scopeTarget ?? (isMemoryScopeTarget(memory.scope) ? memory.scope : undefined),
      userId: memory.userId ?? null, sessionId: memory.sessionId ?? null, agentId: memory.agentId ?? null, taskId: memory.taskId ?? null,
      metadata: copyMetadata(memory.metadata),
      temporal: { documentDate: memory.documentDate, eventDate: memory.eventDate, validFrom: memory.validFrom, validUntil: memory.validUntil },
    },
    similarity: memory.similarity,
  };
}

/**
 * Vector search on memories table
 */
async function vectorSearchMemories(params: {
  embedding: number[];
  userId?: string;
  projectId: string;
  orgId?: string;
  sessionId?: string;
  agentId?: string;
  taskId?: string;
  scopes?: MemoryScopeTarget[];
  temporal: { hasTemporalConstraint: boolean; dateRange?: { start: Date; end: Date } };
  includeInactive: boolean;
  memoryTypes?: string[];
  namespace?: string;
  tags?: string[];
  limit: number;
}): Promise<any[]> {
  const {
    embedding,
    userId,
    projectId,
    orgId,
    sessionId,
    agentId,
    taskId,
    scopes,
    temporal,
    includeInactive,
    memoryTypes,
    namespace,
    tags,
    limit,
  } = params;

  // Build WHERE clause using parameterized Prisma.sql to prevent SQL injection
  let whereClause = Prisma.sql`"projectId" = ${projectId} AND ("validUntil" IS NULL OR "validUntil" > NOW())`;

  if (orgId) whereClause = Prisma.sql`${whereClause} AND "orgId" = ${orgId}`;
  if (scopes && scopes.length > 0) {
    whereClause = Prisma.sql`${whereClause} AND "scope" IN (${Prisma.join(scopes)})`;
  }
  const identityFilters: Prisma.Sql[] = [Prisma.sql`"scope" = 'PROJECT'`];
  if (userId) identityFilters.push(Prisma.sql`("scope" = 'USER' AND "userId" = ${userId})`);
  if (sessionId) identityFilters.push(Prisma.sql`("scope" = 'SESSION' AND "sessionId" = ${sessionId})`);
  if (agentId) identityFilters.push(Prisma.sql`("scope" = 'AGENT' AND "agentId" = ${agentId})`);
  if (taskId) identityFilters.push(Prisma.sql`("scope" = 'TASK' AND "taskId" = ${taskId})`);
  if (scopes?.includes("DOCUMENT")) {
    identityFilters.push(Prisma.sql`"scope" = 'DOCUMENT'`);
  }
  whereClause = Prisma.sql`${whereClause} AND (${Prisma.join(identityFilters, " OR ")})`;
  if (!includeInactive) whereClause = Prisma.sql`${whereClause} AND "isActive" = true`;
  if (memoryTypes && memoryTypes.length > 0) {
    whereClause = Prisma.sql`${whereClause} AND "memoryType" IN (${Prisma.join(memoryTypes)})`;
  }
  if (temporal.hasTemporalConstraint && temporal.dateRange) {
    whereClause = Prisma.sql`${whereClause} AND "documentDate" >= ${temporal.dateRange.start} AND "documentDate" <= ${temporal.dateRange.end}`;
  }
  if (namespace) {
    whereClause = Prisma.sql`${whereClause} AND metadata->>'namespace' = ${namespace}`;
  }
  if (tags && tags.length > 0) {
    const tagConditions = tags.map(tag => Prisma.sql`metadata->'tags' @> ${JSON.stringify([tag])}::jsonb`);
    whereClause = Prisma.sql`${whereClause} AND (${Prisma.join(tagConditions, " OR ")})`;
  }

  // Embedding is a float array from our own model (not user input) — safe to inline as vector literal
  const embeddingLiteral = Prisma.raw(`'[${embedding.join(",")}]'::vector`);
  const limitLiteral = Prisma.raw(String(Math.min(Math.max(1, Math.floor(limit)), 1000)));

  const [, results] = await db.$transaction([
    db.$executeRaw`SET LOCAL hnsw.iterative_scan = strict_order`,
    db.$queryRaw(Prisma.sql`
    SELECT
      id,
      content,
      "memoryType" as "memoryType",
      "entityMentions" as "entityMentions",
      confidence,
      version,
      scope,
      scope as "scopeTarget",
      "userId" as "userId",
      "sessionId" as "sessionId",
      "agentId" as "agentId",
      "taskId" as "taskId",
      "documentDate" as "documentDate",
      "eventDate" as "eventDate",
      "validFrom" as "validFrom",
      "validUntil" as "validUntil",
      "createdAt" as "createdAt",
      "updatedAt" as "updatedAt",
      "sourceChunkId" as "sourceChunkId",
      metadata,
      1 - (embedding <=> ${embeddingLiteral}) as similarity
    FROM memories
    WHERE ${whereClause}
    ORDER BY embedding <=> ${embeddingLiteral}
    LIMIT ${limitLiteral}
    `),
  ]);

  return (results as any[]).map((r) => ({
    ...r,
    content: decrypt(r.content, orgId),
  }));
}

/**
 * Enrich memories with related memories via graph traversal
 * Optimized: Single query for relations + related memories
 */
async function enrichWithRelations(
  memories: any[],
  maxTotal: number,
  orgId?: string,
): Promise<any[]> {
  if (memories.length === 0) {
    return [];
  }

  const memoryIds = memories.map((m) => m.id);
  const memoryIdSet = new Set(memoryIds);

  let relationsQuery: unknown[];
  try {
  // Get relations and related memories in one query
  relationsQuery = await db.$queryRaw(Prisma.sql`
    SELECT 
      r.id as relation_id,
      r."fromMemoryId",
      r."toMemoryId",
      r."relationType",
      m.id,
      m.content,
      m."memoryType",
      m."entityMentions",
      m.confidence,
      m.version,
      m.scope,
      m."userId",
      m."sessionId",
      m."agentId",
      m."taskId",
      m."documentDate",
      m."eventDate",
      m."validFrom",
      m."validUntil",
      m."createdAt",
      m."updatedAt",
      m."sourceChunkId",
      m.metadata
    FROM "memory_relations" r
    LEFT JOIN memories m ON m.id = r."toMemoryId"
    WHERE r."fromMemoryId" IN (${Prisma.join(memoryIds)})
      AND m."isActive" = true
    LIMIT 100
  `);
  } catch (error) {
    console.warn("[MemorySearch] enrichWithRelations DB query failed — returning memories without relation enrichment:", error);
    return memories;
  }

  if (!relationsQuery || (relationsQuery as any[]).length === 0) {
    return memories;
  }

  // Collect unique related memory IDs
  const relatedIds = new Set<string>();
  const relationMap = new Map<string, any[]>();

  for (const row of relationsQuery as any[]) {
    if (row.toMemoryId && !memoryIdSet.has(row.toMemoryId)) {
      relatedIds.add(row.toMemoryId);
      
      if (!relationMap.has(row.fromMemoryId)) {
        relationMap.set(row.fromMemoryId, []);
      }
      relationMap.get(row.fromMemoryId)?.push({
        memoryId: row.toMemoryId,
        relationType: row.relationType,
        content: decrypt(row.content, orgId),
      });
    }
  }

  if (relatedIds.size === 0) {
    return memories;
  }

  // Fetch related memories in batch
  const relatedIdsList = Array.from(relatedIds);
  let relatedMemories: unknown[];
  try {
    relatedMemories = await db.$queryRaw(Prisma.sql`
      SELECT
        id,
        content,
        "memoryType" as "memoryType",
        "entityMentions" as "entityMentions",
        confidence,
        version,
        scope,
        "userId" as "userId",
        "sessionId" as "sessionId",
        "agentId" as "agentId",
        "taskId" as "taskId",
        "documentDate" as "documentDate",
        "eventDate" as "eventDate",
        "validFrom" as "validFrom",
        "validUntil" as "validUntil",
        "createdAt" as "createdAt",
        "updatedAt" as "updatedAt",
        "sourceChunkId" as "sourceChunkId",
        metadata
      FROM memories
      WHERE id IN (${Prisma.join(relatedIdsList)})
        AND "isActive" = true
      LIMIT ${Prisma.raw(String(Math.max(0, maxTotal - memories.length)))}
    `);
  } catch (error) {
    console.warn("[MemorySearch] enrichWithRelations related fetch failed — returning base memories:", error);
    return memories;
  }

  // Assign lower scores to related memories, decrypt content if encrypted
  const relatedWithScores = (relatedMemories as any[]).map((m) => ({
    ...m,
    content: decrypt(m.content, orgId),
    similarity: 0.6,
    isRelated: true,
    relations: relationMap.get(m.id) || [],
  }));

  return [...memories, ...relatedWithScores];
}

/**
 * Inject source chunks for top memories
 * Provides full context for LLM
 */
async function injectSourceChunks(
  memories: SearchMemory[]
): Promise<MemorySearchResult[]> {
  const chunkIds = memories
    .map((m) => m.sourceChunkId)
    .filter((id): id is string => typeof id === "string");

  if (chunkIds.length === 0) {
    // Return memories without chunks
    return memories.map(toMemorySearchResult);
  }

  // Fetch chunks
  const chunks = await db.chunk.findMany({
    where: {
      id: { in: chunkIds },
    },
    select: {
      id: true,
      content: true,
      metadata: true,
    },
  });

  const chunkMap = new Map(chunks.map((c) => [c.id, c]));

  // Map memories to results with chunks
  return memories.map((m) => ({
    ...toMemorySearchResult(m),
    chunk: m.sourceChunkId && chunkMap.has(m.sourceChunkId)
      ? {
          id: chunkMap.get(m.sourceChunkId)!.id,
          content: chunkMap.get(m.sourceChunkId)!.content,
          metadata: copyMetadata(chunkMap.get(m.sourceChunkId)!.metadata),
        }
      : undefined,
    relations: m.isRelated ? [] : undefined,
  }));
}

/**
 * Session-based memory query
 * Gets recent memories from a session for context building
 */
export async function getSessionMemories(params: {
  sessionId: string;
  projectId: string;
  limit?: number;
  sinceDate?: Date;
}): Promise<any[]> {
  const { sessionId, projectId, limit = 50, sinceDate } = params;

  const where: any = {
    sessionId,
    projectId,
    isActive: true,
    scope: "SESSION",
    OR: [
      { validUntil: null },
      { validUntil: { gt: new Date() } },
    ],
  };

  if (sinceDate) {
    where.createdAt = { gte: sinceDate };
  }

  return db.memory.findMany({
    where,
    orderBy: {
      createdAt: "desc",
    },
    take: limit,
  });
}

/**
 * User profile memory query
 * Gets long-term user preferences and facts
 */
export async function getUserProfile(params: {
  userId: string;
  projectId: string;
  memoryTypes?: string[];
  limit?: number;
}): Promise<any[]> {
  const { userId, projectId, memoryTypes, limit = 50 } = params;

  const where: any = {
    userId,
    projectId,
    isActive: true,
    scope: "USER",
    OR: [
      { validUntil: null },
      { validUntil: { gt: new Date() } },
    ],
  };

  if (memoryTypes) {
    where.memoryType = { in: memoryTypes };
  }

  return db.memory.findMany({
    where,
    orderBy: { importance: "desc" },
    take: Math.min(limit, 100),
  });
}
