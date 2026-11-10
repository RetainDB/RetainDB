import { prisma } from "../db/index.js";
import { syncGitHub } from "../connectors/github.js";
import { syncGitHubTarball } from "../connectors/github-tarball.js";
import { syncGitLab } from "../connectors/gitlab.js";
import { syncWeb } from "../connectors/web.js";
import { syncUrl } from "../connectors/url.js";
import { syncSitemap } from "../connectors/sitemap.js";
import { syncArxiv } from "../connectors/arxiv.js";
import { syncNpmPackage } from "../connectors/npm_package.js";
import { syncPyPIPackage } from "../connectors/pypi_package.js";
import { syncHuggingFace } from "../connectors/huggingface.js";
import { syncApiSpec } from "../connectors/api_spec.js";
import { syncNotion } from "../connectors/notion.js";
import { syncConfluence } from "../connectors/confluence.js";
import { syncSlack } from "../connectors/slack.js";
import { syncDiscord } from "../connectors/discord.js";
import { syncPdf } from "../connectors/pdf.js";
import { syncText } from "../connectors/text.js";
import { syncDatabase } from "../connectors/database.js";
import { syncPlaywright } from "../connectors/playwright.js";
import { syncVideo } from "../connectors/video.js";
import { syncDataset } from "../connectors/dataset.js";
import {
  deriveConnectorOutcome,
  classifyErrorCode,
  createStagedSourceVersion,
  failSourceVersion,
  promoteSourceVersion,
  summarizeWarningCodes,
} from "./source-versions.js";

export type SyncStatus = "pending" | "running" | "completed" | "failed" | "cancelled";
export type SyncMode = "incremental" | "full";
export type SyncEffectiveMode = "incremental" | "full";

export interface SyncJob {
  id: string;
  sourceId: string;
  sourceVersionId?: string | null;
  mode?: SyncMode;
  effectiveMode?: SyncEffectiveMode;
  traceId?: string | null;
  parentTraceId?: string | null;
  status: SyncStatus;
  progress: {
    current: number;
    total: number;
    message: string;
  };
  result?: {
    filesIndexed?: number;
    pagesIndexed?: number;
    papersIndexed?: number;
    documentsIndexed?: number;
    documentsTotal?: number;
    documentsFailed?: number;
    partialFailure?: boolean;
    warningCodes?: string[];
    errorCode?: string | null;
    activeVersion?: string | null;
    outcome?: "success" | "partial_failure" | "failed";
    mode?: SyncMode;
    effectiveMode?: SyncEffectiveMode;
    errors?: string[];
  };
  error?: string;
  startedAt?: Date;
  completedAt?: Date;
}

type SyncProgress = { current: number; total: number; message: string };

const JOB_RETENTION_MS = 30 * 60 * 1000;
const DB_PROGRESS_THROTTLE_MS = 1500;

const activeJobControllers = new Map<string, AbortController>();
const sourceToJobId = new Map<string, string>();
const jobs = new Map<string, SyncJob>();
const lastDbProgressWrite = new Map<string, number>();

function toDbStatus(status: SyncStatus): string {
  switch (status) {
    case "pending":
      return "PENDING";
    case "running":
      return "RUNNING";
    case "completed":
      return "COMPLETED";
    case "cancelled":
      return "CANCELLED";
    case "failed":
    default:
      return "FAILED";
  }
}

function updateJob(jobId: string, patch: Partial<SyncJob>) {
  const existing = jobs.get(jobId);
  if (!existing) return;
  jobs.set(jobId, { ...existing, ...patch });
}

async function persistJobCreate(
  jobId: string,
  sourceId: string,
  params?: {
    sourceVersionId?: string | null;
    traceId?: string | null;
    parentTraceId?: string | null;
    mode?: SyncMode;
  }
) {
  try {
    await prisma.syncJob.create({
      data: {
        id: jobId,
        sourceId,
        sourceVersionId: params?.sourceVersionId || null,
        type: (params?.mode || "incremental").toUpperCase(),
        status: "PENDING",
        startedAt: new Date(),
        traceId: params?.traceId || null,
        parentTraceId: params?.parentTraceId || null,
        progress: 0,
      },
    });
  } catch (error) {
    console.warn(`[SyncJob ${jobId}] Failed to persist create:`, (error as any)?.message || error);
  }
}

async function persistJobPatch(
  jobId: string,
  patch: {
    status?: string;
    progress?: number;
    completedAt?: Date;
    documentsTotal?: number;
    documentsIndexed?: number;
    documentsFailed?: number;
    partialFailure?: boolean;
    warningCodes?: string[];
    errorCode?: string | null;
    traceId?: string | null;
    parentTraceId?: string | null;
    errorMessage?: string | null;
    logs?: any;
  }
) {
  try {
    await prisma.syncJob.update({
      where: { id: jobId },
      data: patch,
    });
  } catch (error) {
    console.warn(`[SyncJob ${jobId}] Failed to persist patch:`, (error as any)?.message || error);
  }
}

async function persistProgressThrottled(jobId: string, progress: SyncProgress) {
  const now = Date.now();
  const last = lastDbProgressWrite.get(jobId) || 0;
  if (now - last < DB_PROGRESS_THROTTLE_MS) return;
  lastDbProgressWrite.set(jobId, now);
  const ratio = progress.total > 0 ? Math.min(100, Math.round((progress.current / progress.total) * 100)) : 0;
  await persistJobPatch(jobId, { progress: ratio });
}

function scheduleJobCleanup(jobId: string) {
  setTimeout(() => {
    jobs.delete(jobId);
    lastDbProgressWrite.delete(jobId);
  }, JOB_RETENTION_MS).unref?.();
}

function reportJobProgress(jobId: string, progress: SyncProgress) {
  const job = jobs.get(jobId);
  if (!job) return;
  const nextProgress: SyncProgress = {
    current: Math.max(0, progress.current || 0),
    total: Math.max(progress.total || 0, 0),
    message: progress.message || job.progress.message || "Syncing...",
  };
  updateJob(jobId, { progress: nextProgress });
  void persistProgressThrottled(jobId, nextProgress);
}

export async function enqueueSync(sourceId: string): Promise<string>;
export async function enqueueSync(
  sourceId: string,
  opts: {
    traceId?: string | null;
    parentTraceId?: string | null;
    reuseExisting?: boolean;
    mode?: SyncMode;
  }
): Promise<{ jobId: string; reused: boolean; sourceVersionId: string | null; mode: SyncMode }>;
export async function enqueueSync(
  sourceId: string,
  opts?: {
    traceId?: string | null;
    parentTraceId?: string | null;
    reuseExisting?: boolean;
    mode?: SyncMode;
  }
): Promise<string | { jobId: string; reused: boolean; sourceVersionId: string | null; mode: SyncMode }> {
  const requestedMode = opts?.mode || "incremental";
  const existingJobId = sourceToJobId.get(sourceId);
  if (existingJobId) {
    const existing = jobs.get(existingJobId);
    if (existing && (existing.status === "pending" || existing.status === "running")) {
      const existingMode = existing.mode || "incremental";
      const compatibleReuse =
        requestedMode === existingMode ||
        (requestedMode === "incremental" && existingMode === "full");
      if (opts?.reuseExisting) {
        if (compatibleReuse) {
          return opts
          ? {
              jobId: existingJobId,
              reused: true,
              sourceVersionId: existing.sourceVersionId || null,
              mode: existingMode,
            }
          : existingJobId;
        }
        throw new Error(`SYNC_MODE_CONFLICT:${existingJobId}:${existingMode}`);
      }
      throw new Error(`SYNC_ALREADY_RUNNING:${existingJobId}`);
    }
    sourceToJobId.delete(sourceId);
  }

  const stagedVersion = await createStagedSourceVersion(sourceId, {
    traceId: opts?.traceId || null,
  });
  const jobId = `sync_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
  const abortController = new AbortController();

  const job: SyncJob = {
    id: jobId,
    sourceId,
    sourceVersionId: stagedVersion.id,
    mode: requestedMode,
    effectiveMode: requestedMode,
    traceId: opts?.traceId || null,
    parentTraceId: opts?.parentTraceId || null,
    status: "pending",
    progress: {
      current: 0,
      total: 0,
      message: "Queued",
    },
    startedAt: new Date(),
  };

  jobs.set(jobId, job);
  sourceToJobId.set(sourceId, jobId);
  activeJobControllers.set(jobId, abortController);
  await prisma.sourceVersion.update({
    where: { id: stagedVersion.id },
    data: { syncJobId: jobId },
  });
  await persistJobCreate(jobId, sourceId, {
    sourceVersionId: stagedVersion.id,
    traceId: opts?.traceId || null,
    parentTraceId: opts?.parentTraceId || null,
    mode: requestedMode,
  });

  void processSync(jobId, sourceId, abortController.signal).catch((error) => {
    console.error(`[SyncJob ${jobId}] Unhandled error:`, error);
  });

  return opts
      ? {
          jobId,
          reused: false,
          sourceVersionId: stagedVersion.id,
          mode: requestedMode,
        }
    : jobId;
}

export function getJobStatus(jobId: string): SyncJob | null {
  const job = jobs.get(jobId);
  if (!job) return null;
  return {
    ...job,
    progress: { ...job.progress },
    result: job.result ? { ...job.result } : undefined,
  };
}

export function getActiveJobForSource(sourceId: string): SyncJob | null {
  const jobId = sourceToJobId.get(sourceId);
  if (!jobId) return null;
  return getJobStatus(jobId);
}

export function cancelJob(jobId: string): boolean {
  const controller = activeJobControllers.get(jobId);
  if (!controller) return false;
  controller.abort();
  activeJobControllers.delete(jobId);
  return true;
}

function normalizeConfig(type: string, rawConfig: any, sourceName: string): any {
  const cfg: any = { ...(rawConfig || {}) };

  switch (type) {
    case "github":
    case "github-tarball": {
      if (cfg.repository && (!cfg.owner || !cfg.repo)) {
        const match = String(cfg.repository).match(/github\.com[/:]([^/]+)\/([^/]+?)(?:\.git)?(?:\/.*)?$/);
        if (match) {
          cfg.owner = match[1];
          cfg.repo = match[2].replace(/\.git$/, "");
        }
      }
      break;
    }

    case "gitlab": {
      if (cfg.repository && !cfg.projectPath) {
        cfg.projectPath = cfg.repository;
      }
      if (cfg.url && !cfg.host) {
        try {
          cfg.host = new URL(cfg.url).host;
        } catch {}
      }
      break;
    }

    case "notion": {
      if (typeof cfg.pages === "string") {
        cfg.pageIds = cfg.pages.split(",").map((s: string) => s.trim()).filter(Boolean);
        delete cfg.pages;
      }
      break;
    }

    case "slack": {
      if (typeof cfg.channels === "string") {
        cfg.channelIds = cfg.channels.split(",").map((s: string) => s.trim()).filter(Boolean);
        delete cfg.channels;
      }
      break;
    }

    case "discord": {
      if (typeof cfg.channelIds === "string") {
        cfg.channelIds = cfg.channelIds.split(",").map((s: string) => s.trim()).filter(Boolean);
      }
      break;
    }

    case "confluence": {
      if (cfg.url && !cfg.baseUrl) {
        cfg.baseUrl = cfg.url;
        delete cfg.url;
      }
      if (cfg.token && !cfg.apiToken) {
        cfg.apiToken = cfg.token;
        delete cfg.token;
      }
      break;
    }

    case "huggingface": {
      if (cfg.hfType && !cfg.repoType) {
        cfg.repoType = cfg.hfType;
        delete cfg.hfType;
      }
      if (cfg.name && !cfg.repoId) {
        cfg.repoId = cfg.name;
        delete cfg.name;
      }
      break;
    }

    case "npm_package":
    case "npm": {
      if (cfg.name && !cfg.packageName) {
        cfg.packageName = cfg.name;
        delete cfg.name;
      }
      break;
    }

    case "pypi_package":
    case "pypi": {
      if (cfg.name && !cfg.packageName) {
        cfg.packageName = cfg.name;
        delete cfg.name;
      }
      break;
    }

    case "database": {
      if (!cfg.connectionString && cfg.host) {
        const dialect = cfg.type || "postgresql";
        const user = encodeURIComponent(cfg.user || "");
        const pass = cfg.password ? `:${encodeURIComponent(cfg.password)}` : "";
        const host = cfg.host || "localhost";
        const port = cfg.port || (dialect === "mysql" ? "3306" : "5432");
        const db = cfg.database || "";
        cfg.connectionString = `${dialect}://${user}${pass}@${host}:${port}/${db}`;
      }
      break;
    }

    case "text": {
      if (!cfg.title) cfg.title = sourceName;
      break;
    }

    case "web":
    case "url":
    case "playwright": {
      if (typeof cfg.maxPages === "string") {
        cfg.maxPages = parseInt(cfg.maxPages, 10) || (type === "playwright" ? 10 : 100);
      }
      break;
    }

    case "arxiv": {
      if (typeof cfg.maxResults === "string") {
        cfg.maxResults = parseInt(cfg.maxResults, 10) || 20;
      }
      break;
    }

    case "video": {
      cfg.allow_stt_fallback = cfg.allow_stt_fallback !== false;
      if (!cfg.max_duration_minutes || Number(cfg.max_duration_minutes) <= 0) {
        cfg.max_duration_minutes = 180;
      }
      if (!cfg.max_chunks || Number(cfg.max_chunks) <= 0) {
        cfg.max_chunks = 2000;
      }
      break;
    }
  }

  return cfg;
}

function canonicalType(type: string): string {
  switch (type) {
    case "npm_package":
      return "npm";
    case "pypi_package":
      return "pypi";
    default:
      return type;
  }
}

function summarizeResult(result: any) {
  const explicitTotal = Number(result?.documentsTotal ?? result?.totalFiles ?? result?.totalUrls ?? 0);
  const explicitIndexed = Number(result?.documentsIndexed ?? 0);
  const docs =
    explicitIndexed +
    Number(result?.filesIndexed || 0) +
    Number(result?.pagesIndexed || 0) +
    Number(result?.papersIndexed || 0);
  const total = explicitTotal || docs || 0;
  const failures = Number(result?.documentsFailed ?? (Array.isArray(result?.errors) ? result.errors.length : 0));
  const warningCodes = summarizeWarningCodes(Array.isArray(result?.errors) ? result.errors : []);
  const partialFailure = Boolean(result?.partialFailure) || failures > 0;
  const errorCode = result?.errorCode ?? (failures > 0 ? warningCodes[0] || "PARTIAL_FAILURE" : null);
  return {
    documentsIndexed: docs,
    documentsTotal: total,
    documentsFailed: failures,
    partialFailure,
    warningCodes,
    errorCode,
    outcome: deriveConnectorOutcome({
      partialFailure,
      errorCode,
      documentsIndexed: docs,
    }),
  };
}

function connectorSupportsIncremental(type: string): boolean {
  return ["github", "github-tarball", "gitlab", "notion", "slack", "discord", "video"].includes(type);
}

async function processSync(jobId: string, sourceId: string, signal: AbortSignal) {
  const startTime = Date.now();
  const job = jobs.get(jobId);
  if (!job) return;
  const sourceVersionId = job.sourceVersionId || null;

  updateJob(jobId, {
    status: "running",
    progress: { current: 0, total: 0, message: "Preparing source sync..." },
    startedAt: new Date(),
  });
  await persistJobPatch(jobId, {
    status: toDbStatus("running"),
    progress: 0,
    errorMessage: null,
    traceId: job.traceId || null,
    parentTraceId: job.parentTraceId || null,
  });

  try {
    const source = await prisma.source.findUnique({ where: { id: sourceId } });
    if (!source) throw new Error(`Source ${sourceId} not found`);

    await prisma.source.update({
      where: { id: sourceId },
      data: { status: "INDEXING", lastSyncAt: new Date() },
    });

    const type = canonicalType(source.type);
    const requestedMode = job.mode || "incremental";
    const effectiveMode: SyncEffectiveMode = connectorSupportsIncremental(type)
      ? requestedMode
      : "full";
    const connectorWarnings =
      requestedMode === "incremental" && effectiveMode === "full"
        ? ["INCREMENTAL_UNSUPPORTED"]
        : [];
    updateJob(jobId, {
      effectiveMode,
    });
    await prisma.source.update({
      where: { id: sourceId },
      data: {
        syncMode: requestedMode === "full" ? "MANUAL" : source.syncMode,
      },
    });
    if (!source.projectId) {
      throw new Error(`Source ${sourceId} is missing projectId`);
    }
    const projectId = source.projectId;
    const config = normalizeConfig(source.type, source.config as any, source.name);
    console.log(`[SyncJob ${jobId}] Starting sync: ${source.name} (${source.type} -> ${type})`);

    const onProgress = (progress: SyncProgress) => reportJobProgress(jobId, progress);
    let result: any;

    switch (type) {
      case "github":
        result = await syncGitHub(sourceId, projectId, config, onProgress, signal);
        break;
      case "github-tarball":
        result = await syncGitHubTarball(sourceId, projectId, config, onProgress, signal);
        break;
      case "gitlab":
        result = await syncGitLab(sourceId, projectId, config, onProgress, signal);
        break;
      case "web":
        result = await syncWeb(sourceId, projectId, config);
        break;
      case "url":
        result = await syncUrl(sourceId, projectId, config);
        break;
      case "sitemap":
        result = await syncSitemap(sourceId, projectId, config);
        break;
      case "arxiv":
        result = await syncArxiv(sourceId, projectId, config);
        break;
      case "npm":
        result = await syncNpmPackage(sourceId, projectId, config);
        break;
      case "pypi":
        result = await syncPyPIPackage(sourceId, projectId, config);
        break;
      case "huggingface":
        result = await syncHuggingFace(sourceId, projectId, config);
        break;
      case "api_spec":
        result = await syncApiSpec(sourceId, projectId, config);
        break;
      case "notion":
        result = await syncNotion(sourceId, projectId, config, onProgress, signal);
        break;
      case "confluence":
        result = await syncConfluence(sourceId, projectId, config);
        break;
      case "slack":
        result = await syncSlack(sourceId, projectId, config, onProgress, signal);
        break;
      case "discord":
        result = await syncDiscord(sourceId, projectId, config, onProgress, signal);
        break;
      case "pdf":
        result = await syncPdf(sourceId, projectId, config);
        break;
      case "text":
        result = await syncText(sourceId, projectId, config);
        break;
      case "database":
        result = await syncDatabase(sourceId, projectId, config);
        break;
      case "playwright":
        result = await syncPlaywright(sourceId, projectId, config);
        break;
      case "video":
        result = await syncVideo(sourceId, projectId, config, onProgress, signal);
        break;
      case "dataset":
        result = await syncDataset(sourceId, projectId, config, onProgress, signal);
        break;
      default:
        throw new Error(`Unknown connector type: ${source.type}`);
    }

    const duration = Date.now() - startTime;
    const summary = summarizeResult(result);
    const mergedWarningCodes = [...new Set([...(summary.warningCodes || []), ...connectorWarnings])];
    if (sourceVersionId) {
      await promoteSourceVersion({
        sourceId,
        sourceVersionId,
        partialFailure: summary.partialFailure,
        warningCodes: mergedWarningCodes,
        errorCode: summary.errorCode,
      });
    }
    const refreshedSource = await prisma.source.findUnique({
      where: { id: sourceId },
      select: { activeVersionId: true, documentCount: true, chunkCount: true },
    });
    updateJob(jobId, {
      status: "completed",
      completedAt: new Date(),
      progress: {
        current: summary.documentsTotal || summary.documentsIndexed || 1,
        total: summary.documentsTotal || summary.documentsIndexed || 1,
        message: "Sync completed",
      },
      result: {
        ...result,
        documentsTotal: summary.documentsTotal,
        documentsIndexed: summary.documentsIndexed,
        documentsFailed: summary.documentsFailed,
        partialFailure: summary.partialFailure,
        warningCodes: mergedWarningCodes,
        errorCode: summary.errorCode,
        activeVersion: refreshedSource?.activeVersionId || sourceVersionId,
        outcome: summary.outcome,
        mode: requestedMode,
        effectiveMode,
      },
    });

    await persistJobPatch(jobId, {
      status: toDbStatus("completed"),
      progress: 100,
      completedAt: new Date(),
      documentsTotal: summary.documentsTotal,
      documentsIndexed: summary.documentsIndexed,
      documentsFailed: summary.documentsFailed,
      partialFailure: summary.partialFailure,
      warningCodes: mergedWarningCodes,
      errorCode: summary.errorCode,
      errorMessage: null,
    });

    console.log(`[SyncJob ${jobId}] Completed in ${duration}ms:`, result);
  } catch (error: any) {
    const cancelled = signal.aborted || error?.message === "SYNC_ABORTED";
    const status: SyncStatus = cancelled ? "cancelled" : "failed";
    const message = cancelled ? "Sync cancelled by user" : error?.message || "Sync failed";
    const duration = Date.now() - startTime;
    const errorCode = classifyErrorCode(message);
    console.error(`[SyncJob ${jobId}] ${status} after ${duration}ms:`, message);

    if (sourceVersionId) {
      await failSourceVersion({
        sourceId,
        sourceVersionId,
        errorMessage: message,
        errorCode,
      });
    } else {
      await prisma.source.update({
        where: { id: sourceId },
        data: {
          status: "ERROR",
          lastSyncStatus: status,
          lastSyncDurationMs: duration,
          lastSyncError: message,
          syncError: message,
          syncErrorCount: { increment: cancelled ? 0 : 1 },
        },
      });
    }

    updateJob(jobId, {
      status,
      error: message,
      completedAt: new Date(),
      progress: {
        ...((jobs.get(jobId)?.progress as SyncProgress) || { current: 0, total: 0, message: "" }),
        message,
      },
    });

    await persistJobPatch(jobId, {
      status: toDbStatus(status),
      completedAt: new Date(),
      errorCode,
      errorMessage: message,
    });
  } finally {
    activeJobControllers.delete(jobId);
    sourceToJobId.delete(sourceId);
    scheduleJobCleanup(jobId);
  }
}

setInterval(() => {
  const now = Date.now();
  for (const [jobId, job] of jobs.entries()) {
    if (!job.completedAt) continue;
    if (now - job.completedAt.getTime() > JOB_RETENTION_MS) {
      jobs.delete(jobId);
      lastDbProgressWrite.delete(jobId);
    }
  }
}, 60_000).unref();

/**
 * Called once at server startup to recover from crashes/restarts.
 *
 * - RUNNING jobs: were interrupted mid-sync. Mark them FAILED so the source
 *   isn't stuck in a perpetual INDEXING state. They can be manually re-triggered.
 * - PENDING jobs: never started. Re-enqueue them so they run normally.
 */
export async function recoverInterruptedJobs(): Promise<void> {
  try {
    const stale = await prisma.syncJob.findMany({
      where: { status: { in: ["RUNNING", "PENDING"] } },
      select: { id: true, sourceId: true, status: true, type: true },
    });

    if (stale.length === 0) return;
    console.log(`[SyncQueue] Recovering ${stale.length} interrupted jobs from previous run`);

    for (const dbJob of stale) {
      if (dbJob.status === "RUNNING") {
        // Mark job as failed — we can't resume mid-sync
        await prisma.syncJob.update({
          where: { id: dbJob.id },
          data: {
            status: "FAILED",
            completedAt: new Date(),
            errorMessage: "Server restarted while sync was in progress",
            errorCode: "SERVER_RESTART",
          },
        }).catch(() => {});

        // Reset source status so it can be re-synced
        await prisma.source.update({
          where: { id: dbJob.sourceId },
          data: { status: "ERROR", lastSyncError: "Server restarted during sync" },
        }).catch(() => {});

        console.log(`[SyncQueue] Marked interrupted job ${dbJob.id} (source ${dbJob.sourceId}) as FAILED`);
      } else if (dbJob.status === "PENDING") {
        // Re-enqueue: check source still exists and isn't already being synced
        const existing = sourceToJobId.get(dbJob.sourceId);
        if (existing) continue;

        const source = await prisma.source.findUnique({
          where: { id: dbJob.sourceId },
          select: { id: true, status: true },
        }).catch(() => null);

        if (!source) continue;

        try {
          await prisma.syncJob.update({
            where: { id: dbJob.id },
            data: { status: "FAILED", errorMessage: "Re-queued as new job after server restart" },
          });
          // Trigger a fresh sync
          void enqueueSync(dbJob.sourceId, {
            mode: (dbJob.type?.toLowerCase() as SyncMode) || "incremental",
          }).catch(() => {});
          console.log(`[SyncQueue] Re-queued pending job for source ${dbJob.sourceId}`);
        } catch {
          // Non-fatal
        }
      }
    }
  } catch (error: any) {
    console.warn("[SyncQueue] Job recovery failed (non-fatal):", error.message);
  }
}

