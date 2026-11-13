import { Prisma } from "@prisma/client";
import { prisma } from "../db/index.js";

export type ShadowScopeDecision = "user_profile" | "session_only" | "dropped";
export type InferenceTier = "pattern" | "mini" | "strong";
export type ConflictType = "contradicts" | "supersedes" | "stale_preference";
export type ConflictRecommendedAction = "demote" | "invalidate" | "needs_review";

export interface ExtractionDecisionInput {
  memoryType: string;
  inferenceTier: InferenceTier;
  confidenceRaw: number;
  confidenceCalibrated?: number;
  shadowScopeDecision?: ShadowScopeDecision;
  escalationUsed?: boolean;
  escalationReason?: string;
  modelUsed?: string;
  tokenUsage?: number;
}

export interface ExtractionInvocationInput {
  tenantId: string;
  projectId?: string | null;
  route: string;
  invocationId: string;
  timestamp?: string;
  latencyMs?: number;
  decisions: ExtractionDecisionInput[];
}

export interface ConfirmRateLimitInput {
  confirmCountLast5m: number;
  confirmCountLast1h: number;
  burstLimit: number;
  hourlyLimit: number;
}

export interface ConfirmRateLimitResult {
  allowed: boolean;
  reason?: "burst_limit" | "hourly_limit";
  retryAfterSeconds?: number;
}

export interface MemoryConflictEventInput {
  tenantId: string;
  projectId?: string | null;
  conflictTargetMemoryId: string;
  conflictEvidenceMemoryId: string;
  conflictType: ConflictType;
  recommendedAction: ConflictRecommendedAction;
  metadata?: Record<string, unknown>;
}

export interface ExtractionTenantPolicy {
  tenant_id: string;
  orchestrator_v2_enabled: boolean;
  tiered_escalation_enabled: boolean;
  threshold_enforcement_requested: boolean;
  threshold_enforcement_active: boolean;
  threshold_enforcement_reason: "disabled" | "gate_pass" | "gate_blocked";
  user_profile_threshold: number;
  session_only_threshold: number;
  session_only_retention_days: number;
  gate: {
    pass: boolean;
    sample_count: number;
    observed_days: number;
    waived: boolean;
  };
}

const EXTRACTION_ORCHESTRATOR_V2 = /^true$/i.test(process.env.EXTRACTION_ORCHESTRATOR_V2 || "false");
const EXTRACTION_SHADOW_MODE = /^true$/i.test(process.env.EXTRACTION_SHADOW_MODE || "false");
const EXTRACTION_TIERED_ESCALATION = /^true$/i.test(process.env.EXTRACTION_TIERED_ESCALATION || "false");
const EXTRACTION_THRESHOLD_ENFORCEMENT = /^true$/i.test(process.env.EXTRACTION_THRESHOLD_ENFORCEMENT || "false");
const SESSION_ONLY_RETENTION_DAYS = Math.max(parseInt(process.env.SESSION_ONLY_RETENTION_DAYS || "14", 10), 1);
const MEMORY_CONFIRM_ENDPOINT_ENABLED = /^true$/i.test(process.env.MEMORY_CONFIRM_ENDPOINT_ENABLED || "false");
const MEMORY_CONFIRM_BACKEND_ONLY = !/^false$/i.test(process.env.MEMORY_CONFIRM_BACKEND_ONLY || "true");
const MEMORY_CONFIRM_RATE_LIMIT_PER_HOUR = Math.max(parseInt(process.env.MEMORY_CONFIRM_RATE_LIMIT_PER_HOUR || "200", 10), 1);
const MEMORY_CONFIRM_BURST_LIMIT_5M = Math.max(parseInt(process.env.MEMORY_CONFIRM_BURST_LIMIT_5M || "50", 10), 1);
const MEMORY_CONFIRM_ANOMALY_SPIKE_MULTIPLIER = Math.max(
  parseFloat(process.env.MEMORY_CONFIRM_ANOMALY_SPIKE_MULTIPLIER || "2"),
  1
);
const MEMORY_CONFIRM_ANOMALY_MIN_VOLUME = Math.max(
  parseInt(process.env.MEMORY_CONFIRM_ANOMALY_MIN_VOLUME || "50", 10),
  1
);
const MEMORY_CONFIRM_TO_NEW_RATIO_THRESHOLD = Math.max(
  parseFloat(process.env.MEMORY_CONFIRM_TO_NEW_RATIO_THRESHOLD || "3"),
  1
);
const EXTRACTION_GATE_MIN_DAYS = Math.max(parseInt(process.env.EXTRACTION_GATE_MIN_DAYS || "7", 10), 1);
const EXTRACTION_GATE_MIN_SAMPLES = Math.max(parseInt(process.env.EXTRACTION_GATE_MIN_SAMPLES || "10000", 10), 1);
const EXTRACTION_GATE_WAIVED_TENANTS = new Set(
  (process.env.EXTRACTION_GATE_WAIVED_TENANTS || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)
);
const EXTRACTION_USER_PROFILE_THRESHOLD = clampThreshold(
  parseFloat(process.env.EXTRACTION_USER_PROFILE_THRESHOLD || "0.78"),
  0.78
);
const EXTRACTION_SESSION_ONLY_THRESHOLD = clampThreshold(
  parseFloat(process.env.EXTRACTION_SESSION_ONLY_THRESHOLD || "0.58"),
  0.58
);
const EXTRACTION_TENANT_THRESHOLDS_RAW = process.env.EXTRACTION_TENANT_THRESHOLDS || "{}";
const EXTRACTION_POLICY_CACHE_TTL_MS = Math.max(
  parseInt(process.env.EXTRACTION_POLICY_CACHE_TTL_MS || "60000", 10),
  1000
);
const EXPECTED_EXTRACTION_ROUTES = [
  "/v1/memory/extract",
  "/v1/memory/extract/session",
  "/v1/memory/ingest/session",
] as const;

const policyCache = new Map<string, { value: ExtractionTenantPolicy; expiresAt: number }>();

let resetAtIso: string | null = null;

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function clampThreshold(value: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function parseTenantThresholdOverrides(): Record<string, { user_profile?: number; session_only?: number }> {
  try {
    const parsed = JSON.parse(EXTRACTION_TENANT_THRESHOLDS_RAW || "{}");
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return parsed as Record<string, { user_profile?: number; session_only?: number }>;
  } catch {
    return {};
  }
}

export function getExtractionPhase0Config() {
  return {
    extraction_orchestrator_v2: EXTRACTION_ORCHESTRATOR_V2,
    extraction_shadow_mode: EXTRACTION_SHADOW_MODE,
    extraction_tiered_escalation: EXTRACTION_TIERED_ESCALATION,
    extraction_threshold_enforcement: EXTRACTION_THRESHOLD_ENFORCEMENT,
    session_only_retention_days: SESSION_ONLY_RETENTION_DAYS,
    memory_confirm_endpoint_enabled: MEMORY_CONFIRM_ENDPOINT_ENABLED,
    memory_confirm_backend_only: MEMORY_CONFIRM_BACKEND_ONLY,
    memory_confirm_rate_limit_per_hour: MEMORY_CONFIRM_RATE_LIMIT_PER_HOUR,
    memory_confirm_burst_limit_5m: MEMORY_CONFIRM_BURST_LIMIT_5M,
    memory_confirm_anomaly_spike_multiplier: MEMORY_CONFIRM_ANOMALY_SPIKE_MULTIPLIER,
    memory_confirm_anomaly_min_volume: MEMORY_CONFIRM_ANOMALY_MIN_VOLUME,
    memory_confirm_to_new_ratio_threshold: MEMORY_CONFIRM_TO_NEW_RATIO_THRESHOLD,
    extraction_gate_min_days: EXTRACTION_GATE_MIN_DAYS,
    extraction_gate_min_samples: EXTRACTION_GATE_MIN_SAMPLES,
    extraction_user_profile_threshold: EXTRACTION_USER_PROFILE_THRESHOLD,
    extraction_session_only_threshold: EXTRACTION_SESSION_ONLY_THRESHOLD,
    extraction_tenant_thresholds: parseTenantThresholdOverrides(),
    extraction_policy_cache_ttl_ms: EXTRACTION_POLICY_CACHE_TTL_MS,
    extraction_gate_waived_tenants: [...EXTRACTION_GATE_WAIVED_TENANTS],
    reset_at: resetAtIso,
  };
}

export function isExtractionShadowModeEnabled(): boolean {
  return EXTRACTION_SHADOW_MODE;
}

export function calibrateConfidence(confidenceRaw: number, memoryType: string): number {
  const raw = clamp01(confidenceRaw);
  const normalizedType = (memoryType || "").toLowerCase();
  const adjustmentByType: Record<string, number> = {
    factual: 0.02,
    event: 0.01,
    preference: 0,
    opinion: -0.01,
    goal: 0,
    relationship: 0,
    instruction: 0,
  };
  const adjusted = raw + (adjustmentByType[normalizedType] ?? 0);
  return clamp01(adjusted);
}

export function decideShadowScope(
  confidenceCalibrated: number,
  thresholds?: { userProfileThreshold?: number; sessionOnlyThreshold?: number }
): ShadowScopeDecision {
  const userProfileThreshold = clampThreshold(
    thresholds?.userProfileThreshold ?? EXTRACTION_USER_PROFILE_THRESHOLD,
    EXTRACTION_USER_PROFILE_THRESHOLD
  );
  const sessionOnlyThreshold = clampThreshold(
    thresholds?.sessionOnlyThreshold ?? EXTRACTION_SESSION_ONLY_THRESHOLD,
    EXTRACTION_SESSION_ONLY_THRESHOLD
  );
  if (confidenceCalibrated >= userProfileThreshold) return "user_profile";
  if (confidenceCalibrated >= sessionOnlyThreshold) return "session_only";
  return "dropped";
}

export function evaluateConfirmRateLimits(input: ConfirmRateLimitInput): ConfirmRateLimitResult {
  if (input.confirmCountLast5m >= input.burstLimit) {
    return {
      allowed: false,
      reason: "burst_limit",
      retryAfterSeconds: 300,
    };
  }
  if (input.confirmCountLast1h >= input.hourlyLimit) {
    return {
      allowed: false,
      reason: "hourly_limit",
      retryAfterSeconds: 3600,
    };
  }
  return { allowed: true };
}

function getSinceBoundary(lookbackDays: number): Date {
  const now = new Date();
  const lookback = new Date(now.getTime() - lookbackDays * 24 * 60 * 60 * 1000);
  if (!resetAtIso) return lookback;
  const reset = new Date(resetAtIso);
  return reset > lookback ? reset : lookback;
}

export async function recordExtractionInvocation(input: ExtractionInvocationInput): Promise<void> {
  if (!EXTRACTION_SHADOW_MODE) return;
  const timestamp = input.timestamp ? new Date(input.timestamp) : new Date();
  const tenantThresholds = getTenantThresholds(input.tenantId);
  const rows = input.decisions.map((decision) => {
    const confidenceCalibrated = clamp01(
      decision.confidenceCalibrated ?? calibrateConfidence(decision.confidenceRaw, decision.memoryType)
    );
    const shadowScope = decision.shadowScopeDecision || decideShadowScope(confidenceCalibrated, {
      userProfileThreshold: tenantThresholds.user_profile_threshold,
      sessionOnlyThreshold: tenantThresholds.session_only_threshold,
    });
    return {
      orgId: input.tenantId,
      projectId: input.projectId || null,
      eventType: "memory_extraction_shadow_decision",
      source: "phase0_shadow",
      tokensUsed: decision.tokenUsage || 0,
      latencyMs: input.latencyMs || 0,
      metadata: {
        tenant_id: input.tenantId,
        project_id: input.projectId || null,
        route: input.route,
        invocation_id: input.invocationId,
        timestamp: timestamp.toISOString(),
        memory_type: decision.memoryType,
        inference_tier: decision.inferenceTier,
        confidence_raw: clamp01(decision.confidenceRaw),
        confidence_calibrated: confidenceCalibrated,
        shadow_scope_decision: shadowScope,
        escalation_used: Boolean(decision.escalationUsed),
        escalation_reason: decision.escalationReason || null,
        model_used: decision.modelUsed || null,
        token_usage: decision.tokenUsage || 0,
      },
      timestamp,
      createdAt: timestamp,
    };
  });

  if (rows.length > 0) {
  }

  const summary = {
    user_profile_count: rows.filter((row) => (row.metadata as any).shadow_scope_decision === "user_profile").length,
    session_only_count: rows.filter((row) => (row.metadata as any).shadow_scope_decision === "session_only").length,
    dropped_count: rows.filter((row) => (row.metadata as any).shadow_scope_decision === "dropped").length,
    decisions_count: rows.length,
  };

}

export function evaluateExtractionGate(input: {
  firstSeenAt: Date | null;
  sampleCount: number;
  now: Date;
  tenantId: string;
  minDays: number;
  minSamples: number;
  waived: boolean;
}) {
  const observedDays = input.firstSeenAt
    ? (input.now.getTime() - input.firstSeenAt.getTime()) / (24 * 60 * 60 * 1000)
    : 0;
  const hasMinDays = observedDays >= input.minDays;
  const hasMinSamples = input.sampleCount >= input.minSamples;
  const pass = input.waived || (hasMinDays && hasMinSamples);
  return {
    tenant_id: input.tenantId,
    waived: input.waived,
    first_seen_at: input.firstSeenAt?.toISOString() || null,
    observed_days: observedDays,
    sample_count: input.sampleCount,
    min_days_required: input.minDays,
    min_samples_required: input.minSamples,
    has_min_days: hasMinDays,
    has_min_samples: hasMinSamples,
    pass,
  };
}

export function getTenantThresholds(tenantId: string): {
  user_profile_threshold: number;
  session_only_threshold: number;
} {
  const overrides = parseTenantThresholdOverrides()[tenantId] || {};
  return {
    user_profile_threshold: clampThreshold(
      overrides.user_profile ?? EXTRACTION_USER_PROFILE_THRESHOLD,
      EXTRACTION_USER_PROFILE_THRESHOLD
    ),
    session_only_threshold: clampThreshold(
      overrides.session_only ?? EXTRACTION_SESSION_ONLY_THRESHOLD,
      EXTRACTION_SESSION_ONLY_THRESHOLD
    ),
  };
}

export async function getTenantExtractionPolicy(tenantId: string): Promise<ExtractionTenantPolicy> {
  const now = Date.now();
  const cached = policyCache.get(tenantId);
  if (cached && cached.expiresAt > now) {
    return cached.value;
  }

  const thresholds = getTenantThresholds(tenantId);
  const requested = EXTRACTION_THRESHOLD_ENFORCEMENT;
  let gatePass = false;
  let gateSampleCount = 0;
  let gateObservedDays = 0;
  let gateWaived = EXTRACTION_GATE_WAIVED_TENANTS.has(tenantId);

  if (requested) {
    try {
      const gateStatus = await getExtractionGateStatus({
        tenantId,
        minDays: EXTRACTION_GATE_MIN_DAYS,
        minSamples: EXTRACTION_GATE_MIN_SAMPLES,
      });
      const row = gateStatus.tenant_cohorts.find((cohort) => cohort.tenant_id === tenantId);
      if (row) {
        gatePass = row.pass;
        gateSampleCount = row.sample_count;
        gateObservedDays = row.observed_days;
        gateWaived = row.waived;
      } else if (gateWaived) {
        gatePass = true;
      }
    } catch {
      gatePass = false;
    }
  }

  const enforcementActive = requested && gatePass;
  const policy: ExtractionTenantPolicy = {
    tenant_id: tenantId,
    orchestrator_v2_enabled: EXTRACTION_ORCHESTRATOR_V2,
    tiered_escalation_enabled: EXTRACTION_TIERED_ESCALATION,
    threshold_enforcement_requested: requested,
    threshold_enforcement_active: enforcementActive,
    threshold_enforcement_reason: !requested
      ? "disabled"
      : gatePass
        ? "gate_pass"
        : "gate_blocked",
    user_profile_threshold: thresholds.user_profile_threshold,
    session_only_threshold: thresholds.session_only_threshold,
    session_only_retention_days: SESSION_ONLY_RETENTION_DAYS,
    gate: {
      pass: gatePass,
      sample_count: gateSampleCount,
      observed_days: gateObservedDays,
      waived: gateWaived,
    },
  };

  policyCache.set(tenantId, {
    value: policy,
    expiresAt: now + EXTRACTION_POLICY_CACHE_TTL_MS,
  });
  return policy;
}

export async function getExtractionGateStatus(params?: {
  tenantId?: string;
  minDays?: number;
  minSamples?: number;
  lookbackDays?: number;
}) {
  const minDays = Math.max(params?.minDays ?? EXTRACTION_GATE_MIN_DAYS, 1);
  const minSamples = Math.max(params?.minSamples ?? EXTRACTION_GATE_MIN_SAMPLES, 1);
  const since = getSinceBoundary(params?.lookbackDays ?? 30);

  const filters: Prisma.Sql[] = [
    Prisma.sql`"eventType" = 'memory_extraction_shadow_decision'`,
    Prisma.sql`"createdAt" >= ${since}`,
  ];
  if (params?.tenantId) {
    filters.push(Prisma.sql`"orgId" = ${params.tenantId}`);
  }
  const whereSql = Prisma.join(filters, " AND ");

  const tenantRows = await prisma.$queryRaw<Array<{
    tenant_id: string;
    first_seen_at: Date | null;
    sample_count: bigint;
  }>>(Prisma.sql`
    SELECT
      "orgId" as tenant_id,
      MIN("createdAt") as first_seen_at,
      COUNT(*) as sample_count
    FROM usage_events
    WHERE ${whereSql}
    GROUP BY "orgId"
    ORDER BY COUNT(*) DESC
  `);

  const now = new Date();
  const cohorts = tenantRows.map((row) =>
    evaluateExtractionGate({
      tenantId: row.tenant_id,
      firstSeenAt: row.first_seen_at,
      sampleCount: Number(row.sample_count),
      now,
      minDays,
      minSamples,
      waived: EXTRACTION_GATE_WAIVED_TENANTS.has(row.tenant_id),
    })
  );

  const passCount = cohorts.filter((row) => row.pass).length;
  const failCount = cohorts.length - passCount;

  return {
    generated_at: now.toISOString(),
    since: since.toISOString(),
    min_days_required: minDays,
    min_samples_required: minSamples,
    tenant_cohorts: cohorts,
    summary: {
      tenant_count: cohorts.length,
      pass_count: passCount,
      fail_count: failCount,
      ready_for_threshold_enforcement: cohorts.length > 0 && failCount === 0,
    },
  };
}

function toNumber(value: bigint | number | null | undefined): number {
  if (value === null || value === undefined) return 0;
  return Number(value);
}

export async function getExtractionStats(params?: {
  lookbackDays?: number;
  tenantId?: string;
  projectId?: string;
}) {
  const since = getSinceBoundary(params?.lookbackDays ?? 14);
  const filters: Prisma.Sql[] = [
    Prisma.sql`"createdAt" >= ${since}`,
  ];
  if (params?.tenantId) {
    filters.push(Prisma.sql`"orgId" = ${params.tenantId}`);
  }
  if (params?.projectId) {
    filters.push(Prisma.sql`"projectId" = ${params.projectId}`);
  }
  const scopedWhere = Prisma.join(filters, " AND ");

  const perTenant = await prisma.$queryRaw<Array<{
    tenant_id: string;
    decision_count: bigint;
    avg_confidence: number | null;
    user_profile_count: bigint;
    session_only_count: bigint;
    dropped_count: bigint;
    escalation_count: bigint;
    bin_lt_40: bigint;
    bin_40_58: bigint;
    bin_58_78: bigint;
    bin_gte_78: bigint;
  }>>(Prisma.sql`
    SELECT
      "orgId" as tenant_id,
      COUNT(*) as decision_count,
      AVG(((metadata->>'confidence_calibrated')::numeric)) as avg_confidence,
      SUM(CASE WHEN metadata->>'shadow_scope_decision' = 'user_profile' THEN 1 ELSE 0 END) as user_profile_count,
      SUM(CASE WHEN metadata->>'shadow_scope_decision' = 'session_only' THEN 1 ELSE 0 END) as session_only_count,
      SUM(CASE WHEN metadata->>'shadow_scope_decision' = 'dropped' THEN 1 ELSE 0 END) as dropped_count,
      SUM(CASE WHEN metadata->>'escalation_used' = 'true' THEN 1 ELSE 0 END) as escalation_count,
      SUM(CASE WHEN (metadata->>'confidence_calibrated')::numeric < 0.40 THEN 1 ELSE 0 END) as bin_lt_40,
      SUM(CASE WHEN (metadata->>'confidence_calibrated')::numeric >= 0.40 AND (metadata->>'confidence_calibrated')::numeric < 0.58 THEN 1 ELSE 0 END) as bin_40_58,
      SUM(CASE WHEN (metadata->>'confidence_calibrated')::numeric >= 0.58 AND (metadata->>'confidence_calibrated')::numeric < 0.78 THEN 1 ELSE 0 END) as bin_58_78,
      SUM(CASE WHEN (metadata->>'confidence_calibrated')::numeric >= 0.78 THEN 1 ELSE 0 END) as bin_gte_78
    FROM usage_events
    WHERE "eventType" = 'memory_extraction_shadow_decision'
      AND ${scopedWhere}
    GROUP BY "orgId"
    ORDER BY COUNT(*) DESC
  `);

  const falsePositiveByTenant = await prisma.$queryRaw<Array<{ tenant_id: string; count: bigint }>>(Prisma.sql`
    SELECT "orgId" as tenant_id, COUNT(*) as count
    FROM usage_events
    WHERE "eventType" = 'memory_false_positive_feedback'
      AND ${scopedWhere}
    GROUP BY "orgId"
  `);
  const falsePositiveMap = new Map(falsePositiveByTenant.map((row) => [row.tenant_id, toNumber(row.count)]));

  const confirmHourly = await prisma.$queryRaw<Array<{
    tenant_id: string;
    hour_bucket: Date;
    count: bigint;
  }>>(Prisma.sql`
    SELECT
      "orgId" as tenant_id,
      date_trunc('hour', "createdAt") as hour_bucket,
      COUNT(*) as count
    FROM usage_events
    WHERE "eventType" = 'memory_confirm'
      AND ${scopedWhere}
    GROUP BY "orgId", date_trunc('hour', "createdAt")
    ORDER BY hour_bucket DESC
    LIMIT 1000
  `);

  const confirmDaily = await prisma.$queryRaw<Array<{
    tenant_id: string;
    day_bucket: Date;
    count: bigint;
  }>>(Prisma.sql`
    SELECT
      "orgId" as tenant_id,
      date_trunc('day', "createdAt") as day_bucket,
      COUNT(*) as count
    FROM usage_events
    WHERE "eventType" = 'memory_confirm'
      AND ${scopedWhere}
    GROUP BY "orgId", date_trunc('day', "createdAt")
    ORDER BY day_bucket DESC
    LIMIT 1000
  `);

  const tenantRows = perTenant.map((row) => {
    const decisionCount = toNumber(row.decision_count);
    const escalationCount = toNumber(row.escalation_count);
    const falsePositiveCount = falsePositiveMap.get(row.tenant_id) || 0;
    return {
      tenant_id: row.tenant_id,
      decision_count: decisionCount,
      avg_confidence: row.avg_confidence ? Number(row.avg_confidence) : 0,
      scope_ratio: {
        user_profile: decisionCount > 0 ? toNumber(row.user_profile_count) / decisionCount : 0,
        session_only: decisionCount > 0 ? toNumber(row.session_only_count) / decisionCount : 0,
        dropped: decisionCount > 0 ? toNumber(row.dropped_count) / decisionCount : 0,
      },
      confidence_histogram: {
        lt_40: toNumber(row.bin_lt_40),
        bin_40_58: toNumber(row.bin_40_58),
        bin_58_78: toNumber(row.bin_58_78),
        gte_78: toNumber(row.bin_gte_78),
      },
      escalation_rate: decisionCount > 0 ? escalationCount / decisionCount : 0,
      false_positive_feedback_rate: decisionCount > 0 ? falsePositiveCount / decisionCount : 0,
    };
  });

  return {
    generated_at: new Date().toISOString(),
    since: since.toISOString(),
    config: getExtractionPhase0Config(),
    panels: {
      confidence_histogram_by_tenant: tenantRows.map((row) => ({
        tenant_id: row.tenant_id,
        histogram: row.confidence_histogram,
      })),
      scope_decision_ratio_by_tenant: tenantRows.map((row) => ({
        tenant_id: row.tenant_id,
        scope_ratio: row.scope_ratio,
      })),
      escalation_rate_by_tenant: tenantRows.map((row) => ({
        tenant_id: row.tenant_id,
        escalation_rate: row.escalation_rate,
      })),
      false_positive_feedback_rate_by_tenant: tenantRows.map((row) => ({
        tenant_id: row.tenant_id,
        false_positive_feedback_rate: row.false_positive_feedback_rate,
      })),
      confirm_volume_by_tenant: {
        hourly: confirmHourly.map((row) => ({
          tenant_id: row.tenant_id,
          hour: row.hour_bucket instanceof Date ? row.hour_bucket.toISOString() : String(row.hour_bucket),
          count: toNumber(row.count),
        })),
        daily: confirmDaily.map((row) => ({
          tenant_id: row.tenant_id,
          day: row.day_bucket instanceof Date ? row.day_bucket.toISOString() : String(row.day_bucket),
          count: toNumber(row.count),
        })),
      },
    },
    per_tenant: tenantRows,
    global: {
      decision_count: tenantRows.reduce((acc, row) => acc + row.decision_count, 0),
      avg_confidence:
        tenantRows.length > 0
          ? tenantRows.reduce((acc, row) => acc + row.avg_confidence, 0) / tenantRows.length
          : 0,
    },
  };
}

export async function getExtractionAlerts(params?: { lookbackHours?: number }) {
  const lookbackHours = Math.max(params?.lookbackHours ?? 24, 1);
  const since = new Date(Date.now() - lookbackHours * 60 * 60 * 1000);

  const anomalies = await prisma.$queryRaw<Array<{
    tenant_id: string;
    count: bigint;
  }>>`
    SELECT "orgId" as tenant_id, COUNT(*) as count
    FROM usage_events
    WHERE "eventType" = 'memory_confirm_anomaly'
      AND "createdAt" >= ${since}
    GROUP BY "orgId"
  `;

  const escalationAnomalies = await prisma.$queryRaw<Array<{
    tenant_id: string;
    decision_count: bigint;
    escalation_count: bigint;
  }>>`
    SELECT
      "orgId" as tenant_id,
      COUNT(*) as decision_count,
      SUM(CASE WHEN metadata->>'escalation_used' = 'true' THEN 1 ELSE 0 END) as escalation_count
    FROM usage_events
    WHERE "eventType" = 'memory_extraction_shadow_decision'
      AND "createdAt" >= ${since}
    GROUP BY "orgId"
    HAVING COUNT(*) >= 100
      AND (SUM(CASE WHEN metadata->>'escalation_used' = 'true' THEN 1 ELSE 0 END)::numeric / COUNT(*)) > 0.35
  `;

  const routeCoverage = await prisma.$queryRaw<Array<{
    route: string | null;
    count: bigint;
  }>>`
    SELECT
      metadata->>'route' as route,
      COUNT(*) as count
    FROM usage_events
    WHERE "eventType" = 'memory_extraction_shadow_invocation'
      AND "createdAt" >= ${since}
    GROUP BY metadata->>'route'
  `;
  const seenRoutes = new Set(
    routeCoverage
      .map((row) => (row.route || "").trim())
      .filter(Boolean)
  );
  const missingRoutes = EXPECTED_EXTRACTION_ROUTES.filter((route) => !seenRoutes.has(route));

  const tenantRouteCoverage = await prisma.$queryRaw<Array<{
    tenant_id: string;
    route: string | null;
    count: bigint;
  }>>`
    SELECT
      "orgId" as tenant_id,
      metadata->>'route' as route,
      COUNT(*) as count
    FROM usage_events
    WHERE "eventType" = 'memory_extraction_shadow_invocation'
      AND "createdAt" >= ${since}
    GROUP BY "orgId", metadata->>'route'
  `;
  const perTenantRoutes = new Map<string, Set<string>>();
  for (const row of tenantRouteCoverage) {
    const route = (row.route || "").trim();
    if (!route) continue;
    const set = perTenantRoutes.get(row.tenant_id) || new Set<string>();
    set.add(route);
    perTenantRoutes.set(row.tenant_id, set);
  }
  const perTenantGaps = [...perTenantRoutes.entries()]
    .map(([tenantId, routes]) => ({
      tenant_id: tenantId,
      missing_routes: EXPECTED_EXTRACTION_ROUTES.filter((route) => !routes.has(route)),
    }))
    .filter((row) => row.missing_routes.length > 0);

  return {
    generated_at: new Date().toISOString(),
    lookback_hours: lookbackHours,
    alerts: {
      confirm_abuse_spikes: anomalies.map((row) => ({
        tenant_id: row.tenant_id,
        anomaly_count: toNumber(row.count),
      })),
      escalation_anomalies: escalationAnomalies.map((row) => ({
        tenant_id: row.tenant_id,
        decision_count: toNumber(row.decision_count),
        escalation_count: toNumber(row.escalation_count),
        escalation_rate:
          toNumber(row.decision_count) > 0
            ? toNumber(row.escalation_count) / toNumber(row.decision_count)
            : 0,
      })),
      telemetry_gaps: {
        missing_routes_global: missingRoutes,
        missing_routes_by_tenant: perTenantGaps,
      },
    },
  };
}

export function resetExtractionObservability() {
  resetAtIso = new Date().toISOString();
  policyCache.clear();
  return {
    reset: true,
    reset_at: resetAtIso,
  };
}

export async function countConfirmEventsForTenant(params: {
  tenantId: string;
  since: Date;
}): Promise<number> {
  const rows = await prisma.$queryRaw<Array<{ count: bigint }>>`
    SELECT COUNT(*)::bigint AS count
    FROM usage_events
    WHERE "orgId" = ${params.tenantId}
      AND "createdAt" >= ${params.since}
      AND "eventType" = 'memory_confirm_event'
  `;
  return Number(rows[0]?.count ?? 0);
}

export async function emitMemoryConfirmEvent(params: {
  tenantId: string;
  projectId?: string | null;
  memoryId: string;
  confirmedByType: "backend";
  confirmedById: string;
  reason?: string;
  metadata?: Record<string, unknown>;
}) {
}

export async function emitMemoryConfirmAnomaly(params: {
  tenantId: string;
  projectId?: string | null;
  reason: string;
  details: Record<string, unknown>;
}) {
}

export async function emitMemoryConflictDetectedEvent(input: MemoryConflictEventInput) {
}
