import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { prisma } from "../db/index.js";

export interface RetrievalFamilySummary {
  source_family: string;
  latest: {
    timestamp: string | null;
    result_path: string | null;
    precision_at_k: number | null;
    recall_at_k: number | null;
    mrr_at_k: number | null;
    ndcg_at_k: number | null;
  } | null;
  previous: {
    timestamp: string | null;
    result_path: string | null;
    precision_at_k: number | null;
    recall_at_k: number | null;
    mrr_at_k: number | null;
    ndcg_at_k: number | null;
  } | null;
  drift: {
    precision_at_k: number | null;
    recall_at_k: number | null;
    mrr_at_k: number | null;
    ndcg_at_k: number | null;
  } | null;
}

export interface OperationalAlert {
  code: string;
  severity: "warning" | "critical";
  category: "retrieval" | "connector" | "queue" | "webhook" | "lifecycle";
  summary: string;
  metadata: Record<string, any>;
  created_at: string;
}

const ROOT = process.cwd();
const RESULTS_DIR = resolve(ROOT, "benchmarks", "results");
const RETRIEVAL_DRIFT_THRESHOLD = Math.max(
  Number(process.env.RETRIEVAL_DRIFT_THRESHOLD || "0.08"),
  0.01
);
const CONNECTOR_FAILURE_RATE_THRESHOLD = Math.max(
  Number(process.env.CONNECTOR_FAILURE_RATE_THRESHOLD || "0.35"),
  0.01
);
const CONNECTOR_MIN_SAMPLE = Math.max(Number(process.env.CONNECTOR_ALERT_MIN_SAMPLE || "3"), 1);
const QUEUE_BACKLOG_THRESHOLD = Math.max(Number(process.env.QUEUE_BACKLOG_THRESHOLD || "20"), 1);
const WEBHOOK_FAILURE_THRESHOLD = Math.max(Number(process.env.WEBHOOK_FAILURE_THRESHOLD || "10"), 1);
const PARTIAL_FAILURE_THRESHOLD = Math.max(Number(process.env.PARTIAL_FAILURE_THRESHOLD || "5"), 1);

function toMetricView(report: any) {
  return {
    timestamp: report?.benchmark_info?.timestamp || null,
    result_path: report?.result_path || null,
    precision_at_k: report?.summary?.avg_metrics?.precision_at_k ?? null,
    recall_at_k: report?.summary?.avg_metrics?.recall_at_k ?? null,
    mrr_at_k: report?.summary?.avg_metrics?.mrr_at_k ?? null,
    ndcg_at_k: report?.summary?.avg_metrics?.ndcg_at_k ?? null,
  };
}

function safeReadJson(filePath: string) {
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function listRetrievalReports() {
  if (!existsSync(RESULTS_DIR)) return [];
  return readdirSync(RESULTS_DIR)
    .filter((name) => /^retrieval_gold_.*\.json$/i.test(name))
    .map((name) => join(RESULTS_DIR, name))
    .map((absPath) => {
      const json = safeReadJson(absPath);
      if (!json) return null;
      return {
        ...json,
        result_path: absPath.replace(`${ROOT}\\`, "").replace(/\\/g, "/"),
      };
    })
    .filter(Boolean)
    .sort((a: any, b: any) => {
      const aTime = new Date(a?.benchmark_info?.timestamp || 0).getTime();
      const bTime = new Date(b?.benchmark_info?.timestamp || 0).getTime();
      return bTime - aTime;
    });
}

export function getRetrievalHealthSummary(): { families: RetrievalFamilySummary[]; latest_reports: number } {
  const reports = listRetrievalReports();
  const byFamily = new Map<string, any[]>();
  for (const report of reports) {
    const family = String(report?.benchmark_info?.dataset?.source_family || "unknown");
    const existing = byFamily.get(family) || [];
    existing.push(report);
    byFamily.set(family, existing);
  }

  const families = [...byFamily.entries()].map(([sourceFamily, familyReports]) => {
    const latest = familyReports[0] || null;
    const previous = familyReports[1] || null;
    const latestMetrics = latest ? toMetricView(latest) : null;
    const previousMetrics = previous ? toMetricView(previous) : null;
    return {
      source_family: sourceFamily,
      latest: latestMetrics,
      previous: previousMetrics,
      drift:
        latestMetrics && previousMetrics
          ? {
              precision_at_k:
                latestMetrics.precision_at_k !== null && previousMetrics.precision_at_k !== null
                  ? latestMetrics.precision_at_k - previousMetrics.precision_at_k
                  : null,
              recall_at_k:
                latestMetrics.recall_at_k !== null && previousMetrics.recall_at_k !== null
                  ? latestMetrics.recall_at_k - previousMetrics.recall_at_k
                  : null,
              mrr_at_k:
                latestMetrics.mrr_at_k !== null && previousMetrics.mrr_at_k !== null
                  ? latestMetrics.mrr_at_k - previousMetrics.mrr_at_k
                  : null,
              ndcg_at_k:
                latestMetrics.ndcg_at_k !== null && previousMetrics.ndcg_at_k !== null
                  ? latestMetrics.ndcg_at_k - previousMetrics.ndcg_at_k
                  : null,
            }
          : null,
    };
  });

  return {
    families,
    latest_reports: reports.length,
  };
}

export async function getConnectorHealthSummary(params?: { lookbackHours?: number }) {
  const since = new Date(Date.now() - Math.max(params?.lookbackHours || 24, 1) * 60 * 60 * 1000);
  const jobs = await prisma.syncJob.findMany({
    where: { createdAt: { gte: since } },
    select: {
      id: true,
      status: true,
      type: true,
      partialFailure: true,
      errorCode: true,
      warningCodes: true,
      documentsIndexed: true,
      documentsFailed: true,
      source: {
        select: {
          connectorType: true,
          type: true,
          orgId: true,
          projectId: true,
        },
      },
    },
  });

  const byConnector = new Map<string, any>();
  for (const job of jobs) {
    const key = String(job.source?.connectorType || job.source?.type || "unknown");
    const bucket =
      byConnector.get(key) ||
      {
        connector_type: key,
        total_jobs: 0,
        completed_jobs: 0,
        failed_jobs: 0,
        partial_failures: 0,
        documents_indexed: 0,
        documents_failed: 0,
        error_codes: {} as Record<string, number>,
        warning_codes: {} as Record<string, number>,
      };
    bucket.total_jobs += 1;
    if (String(job.status).toUpperCase() === "COMPLETED") bucket.completed_jobs += 1;
    if (String(job.status).toUpperCase() === "FAILED") bucket.failed_jobs += 1;
    if (job.partialFailure) bucket.partial_failures += 1;
    bucket.documents_indexed += job.documentsIndexed || 0;
    bucket.documents_failed += job.documentsFailed || 0;
    if (job.errorCode) {
      bucket.error_codes[job.errorCode] = (bucket.error_codes[job.errorCode] || 0) + 1;
    }
    for (const warningCode of job.warningCodes || []) {
      bucket.warning_codes[warningCode] = (bucket.warning_codes[warningCode] || 0) + 1;
    }
    byConnector.set(key, bucket);
  }

  return {
    lookback_hours: Math.max(params?.lookbackHours || 24, 1),
    connectors: [...byConnector.values()]
      .map((bucket) => ({
        ...bucket,
        failure_rate: bucket.total_jobs > 0 ? bucket.failed_jobs / bucket.total_jobs : 0,
        partial_failure_rate: bucket.total_jobs > 0 ? bucket.partial_failures / bucket.total_jobs : 0,
      }))
      .sort((a, b) => b.total_jobs - a.total_jobs),
  };
}

export async function getQueueHealthSummary() {
  const [syncSummary, staleVersions, queuedJobs, oldestPending, oldestStaged] = await Promise.all([
    prisma.syncJob.groupBy({
      by: ["status"],
      _count: { _all: true },
    }),
    prisma.sourceVersion.count({
      where: { status: { in: ["STAGED", "PROMOTING"] } },
    }),
    prisma.syncJob.count({
      where: { status: { in: ["PENDING", "RUNNING"] } },
    }),
    prisma.syncJob.findFirst({
      where: { status: { in: ["PENDING", "RUNNING"] } },
      orderBy: { createdAt: "asc" },
      select: { id: true, createdAt: true, sourceId: true, type: true, status: true },
    }),
    prisma.sourceVersion.findFirst({
      where: { status: { in: ["STAGED", "PROMOTING"] } },
      orderBy: { createdAt: "asc" },
      select: { id: true, createdAt: true, sourceId: true, status: true },
    }),
  ]);

  return {
    sync_jobs: syncSummary,
    active_backlog: queuedJobs,
    stale_source_versions: staleVersions,
    oldest_active_job: oldestPending,
    oldest_staged_version: oldestStaged,
  };
}

export async function getOperationalCounters() {
  const [sourcesByOrg, sourcesByProject, docsByProject, chunksByProject, sources] = await Promise.all([
    prisma.source.groupBy({ by: ["orgId"], _count: { _all: true } }),
    prisma.source.groupBy({ by: ["projectId"], _count: { _all: true } }),
    prisma.document.groupBy({
      by: ["projectId"],
      where: { deletedAt: null },
      _count: { _all: true },
    }),
    prisma.chunk.groupBy({
      by: ["projectId"],
      _count: { _all: true },
    }),
    prisma.source.findMany({
      select: {
        id: true,
        orgId: true,
        projectId: true,
        type: true,
        connectorType: true,
        status: true,
        documentCount: true,
        chunkCount: true,
      },
      orderBy: { updatedAt: "desc" },
      take: 200,
    }),
  ]);

  const docsByProjectMap = new Map(docsByProject.map((row) => [String(row.projectId || "null"), row._count._all]));
  const chunksByProjectMap = new Map(chunksByProject.map((row) => [String(row.projectId || "null"), row._count._all]));

  return {
    by_source: sources.map((source) => ({
      source_id: source.id,
      org_id: source.orgId,
      project_id: source.projectId,
      connector_type: source.connectorType || source.type,
      status: source.status,
      document_count: source.documentCount,
      chunk_count: source.chunkCount,
    })),
    by_project: sourcesByProject.map((row) => ({
      project_id: row.projectId,
      source_count: row._count._all,
      document_count: docsByProjectMap.get(String(row.projectId || "null")) || 0,
      chunk_count: chunksByProjectMap.get(String(row.projectId || "null")) || 0,
    })),
    by_org: sourcesByOrg.map((row) => ({
      org_id: row.orgId,
      source_count: row._count._all,
    })),
  };
}

export async function getWebhookFailureSummary(params?: { lookbackHours?: number }) {
  const since = new Date(Date.now() - Math.max(params?.lookbackHours || 24, 1) * 60 * 60 * 1000);
  const [recentFailures, deliveryCounts] = await Promise.all([
    prisma.webhookDelivery.findMany({
      where: {
        deliveredAt: { gte: since },
        errorCode: { not: null },
      },
      orderBy: { deliveredAt: "desc" },
      take: 25,
    }),
    prisma.webhookDelivery.groupBy({
      by: ["event"],
      where: { deliveredAt: { gte: since } },
      _count: { _all: true },
    }),
  ]);
  return {
    lookback_hours: Math.max(params?.lookbackHours || 24, 1),
    deliveries: deliveryCounts,
    recent_failures: recentFailures,
  };
}

export async function evaluateOperationalAlerts(params?: { lookbackHours?: number }) {
  const createdAt = new Date().toISOString();
  const [retrieval, connectors, queue, webhooks] = await Promise.all([
    Promise.resolve(getRetrievalHealthSummary()),
    getConnectorHealthSummary(params),
    getQueueHealthSummary(),
    getWebhookFailureSummary(params),
  ]);

  const alerts: OperationalAlert[] = [];

  for (const family of retrieval.families) {
    const drift = family.drift?.ndcg_at_k ?? family.drift?.recall_at_k ?? null;
    if (family.source_family === "pdf" || family.source_family === "video") {
      if (drift !== null && drift < -RETRIEVAL_DRIFT_THRESHOLD) {
        alerts.push({
          code: "RETRIEVAL_DRIFT",
          severity: "warning",
          category: "retrieval",
          summary: `${family.source_family} retrieval quality regressed`,
          metadata: {
            source_family: family.source_family,
            drift,
            latest: family.latest,
            previous: family.previous,
          },
          created_at: createdAt,
        });
      }
    }
  }

  for (const connector of connectors.connectors) {
    if (connector.total_jobs >= CONNECTOR_MIN_SAMPLE && connector.failure_rate >= CONNECTOR_FAILURE_RATE_THRESHOLD) {
      alerts.push({
        code: "CONNECTOR_FAILURE_SPIKE",
        severity: connector.failure_rate >= 0.6 ? "critical" : "warning",
        category: "connector",
        summary: `${connector.connector_type} failure rate exceeded threshold`,
        metadata: connector,
        created_at: createdAt,
      });
    }
    if (connector.partial_failures >= PARTIAL_FAILURE_THRESHOLD) {
      alerts.push({
        code: "CONNECTOR_PARTIAL_FAILURE_REPEAT",
        severity: "warning",
        category: "connector",
        summary: `${connector.connector_type} partial failures are repeating`,
        metadata: connector,
        created_at: createdAt,
      });
    }
  }

  if (queue.active_backlog >= QUEUE_BACKLOG_THRESHOLD || queue.stale_source_versions > 0) {
    alerts.push({
      code: "QUEUE_BACKLOG",
      severity: queue.stale_source_versions > 0 ? "critical" : "warning",
      category: "queue",
      summary: "Sync backlog or stale staged versions exceeded threshold",
      metadata: queue,
      created_at: createdAt,
    });
  }

  if (webhooks.recent_failures.length >= WEBHOOK_FAILURE_THRESHOLD) {
    alerts.push({
      code: "WEBHOOK_FAILURE_BURST",
      severity: "warning",
      category: "webhook",
      summary: "Webhook delivery failures exceeded threshold",
      metadata: {
        lookback_hours: webhooks.lookback_hours,
        recent_failures: webhooks.recent_failures.length,
      },
      created_at: createdAt,
    });
  }

  return {
    generated_at: createdAt,
    alerts,
    retrieval,
    connectors,
    queue,
    webhooks,
  };
}
