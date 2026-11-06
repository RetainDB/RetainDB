import { Prisma } from "@prisma/client";
import { prisma } from "../db/index.js";

const DEFAULT_RESTORE_WINDOW_DAYS = 7;
const STALE_VERSION_TIMEOUT_MS = 30 * 60 * 1000;
const DEFAULT_SUPERSEDED_RETENTION_DAYS = 30;
const DEFAULT_SUPERSEDED_VERSIONS_TO_KEEP = 3;

export type ConnectorOutcome = "success" | "partial_failure" | "failed";

export async function ensureActiveSourceVersion(sourceId: string) {
  return prisma.$transaction(async tx => {
    await tx.$queryRaw`SELECT id FROM sources WHERE id = ${sourceId} FOR UPDATE`;
    const source = await tx.source.findUnique({
      where: { id: sourceId },
      select: {
        id: true,
        orgId: true,
        projectId: true,
        activeVersionId: true,
      },
    });
    if (!source) throw new Error(`Source ${sourceId} not found`);
    if (source.activeVersionId) {
      return tx.sourceVersion.findUnique({ where: { id: source.activeVersionId } });
    }

    const aggregate = await tx.sourceVersion.aggregate({
      where: { sourceId },
      _max: { versionNumber: true },
    });
    const created = await tx.sourceVersion.create({
      data: {
        sourceId,
        orgId: source.orgId,
        projectId: source.projectId,
        versionNumber: (aggregate._max.versionNumber ?? 0) + 1,
        status: "ACTIVE",
        promotedAt: new Date(),
        restoreUntil: computeRestoreUntil(90),
      },
    });
    await tx.source.update({
      where: { id: sourceId },
      data: { activeVersionId: created.id },
    });
    return created;
  });
}

export async function resolveSourceVersionForWrite(sourceId: string): Promise<string | null> {
  const source = await prisma.source.findUnique({
    where: { id: sourceId },
    select: {
      id: true,
      status: true,
      activeVersionId: true,
    },
  });
  if (!source) return null;

  if (source.status === "INDEXING") {
    const staged = await prisma.sourceVersion.findFirst({
      where: {
        sourceId,
        status: { in: ["STAGED", "PROMOTING"] },
      },
      orderBy: { createdAt: "desc" },
      select: { id: true },
    });
    if (staged?.id) return staged.id;
  }

  if (source.activeVersionId) return source.activeVersionId;
  const active = await ensureActiveSourceVersion(sourceId);
  return active?.id || null;
}

export async function createStagedSourceVersion(sourceId: string, params?: {
  syncJobId?: string | null;
  traceId?: string | null;
}) {
  return prisma.$transaction(async tx => {
    await tx.$queryRaw`SELECT id FROM sources WHERE id = ${sourceId} FOR UPDATE`;
    const source = await tx.source.findUnique({
      where: { id: sourceId },
      select: { id: true, orgId: true, projectId: true },
    });
    if (!source) throw new Error(`Source ${sourceId} not found`);
    const aggregate = await tx.sourceVersion.aggregate({
      where: { sourceId },
      _max: { versionNumber: true },
    });
    return tx.sourceVersion.create({
      data: {
        sourceId,
        orgId: source.orgId,
        projectId: source.projectId,
        versionNumber: (aggregate._max.versionNumber ?? 0) + 1,
        status: "STAGED",
        syncJobId: params?.syncJobId || null,
        restoreUntil: computeRestoreUntil(90),
        errorMessage: params?.traceId ? `trace:${params.traceId}` : null,
      },
    });
  });
}

export async function refreshSourceVersionCounts(sourceVersionId: string) {
  const version = await prisma.sourceVersion.findUnique({
    where: { id: sourceVersionId },
    select: { sourceId: true },
  });
  if (!version) return null;
  const [documentCount, chunkCount] = await Promise.all([
    prisma.document.count({ where: { sourceVersionId, deletedAt: null } }),
    prisma.chunk.count({
      where: {
        document: {
          sourceVersionId,
          deletedAt: null,
        },
      },
    }),
  ]);
  return prisma.sourceVersion.update({
    where: { id: sourceVersionId },
    data: {
      documentCount,
      chunkCount,
    },
  });
}

export async function promoteSourceVersion(params: {
  sourceId: string;
  sourceVersionId: string;
  partialFailure?: boolean;
  warningCodes?: string[];
  errorCode?: string | null;
}) {
  const now = new Date();
  return prisma.$transaction(async (tx) => {
    const source = await tx.source.findUnique({
      where: { id: params.sourceId },
      select: {
        id: true,
        activeVersionId: true,
      },
    });
    if (!source) throw new Error(`Source ${params.sourceId} not found`);

    const counts = await tx.document.count({
      where: { sourceVersionId: params.sourceVersionId, deletedAt: null },
    });
    const chunkCount = await tx.chunk.count({
      where: {
        document: {
          sourceVersionId: params.sourceVersionId,
          deletedAt: null,
        },
      },
    });

    await tx.sourceVersion.update({
      where: { id: params.sourceVersionId },
      data: {
        status: "PROMOTING",
      },
    });

    if (source.activeVersionId && source.activeVersionId !== params.sourceVersionId) {
      await tx.sourceVersion.update({
        where: { id: source.activeVersionId },
        data: {
          status: "SUPERSEDED",
          supersededAt: now,
        },
      });
    }

    const promoted = await tx.sourceVersion.update({
      where: { id: params.sourceVersionId },
      data: {
        status: "ACTIVE",
        promotedAt: now,
        partialFailure: Boolean(params.partialFailure),
        warningCodes: params.warningCodes || [],
        errorCode: params.errorCode || null,
        documentCount: counts,
        chunkCount,
      },
    });

    await tx.source.update({
      where: { id: params.sourceId },
      data: {
        activeVersionId: params.sourceVersionId,
        documentCount: counts,
        chunkCount,
        status: "READY",
        syncError: params.errorCode || null,
        lastSyncError: params.errorCode || null,
        lastSyncAt: now,
      },
    });

    return promoted;
  });
}

export async function failSourceVersion(params: {
  sourceId: string;
  sourceVersionId: string;
  errorMessage: string;
  errorCode?: string | null;
  partialFailure?: boolean;
  warningCodes?: string[];
}) {
  const now = new Date();
  await prisma.sourceVersion.update({
    where: { id: params.sourceVersionId },
    data: {
      status: "FAILED",
      failedAt: now,
      errorMessage: params.errorMessage,
      errorCode: params.errorCode || null,
      partialFailure: Boolean(params.partialFailure),
      warningCodes: params.warningCodes || [],
    },
  });

  await prisma.source.update({
    where: { id: params.sourceId },
    data: {
      status: "ERROR",
      syncError: params.errorMessage,
      lastSyncError: params.errorMessage,
      lastSyncAt: now,
    },
  });
}

export async function markStaleSourceVersionsFailed(): Promise<number> {
  const cutoff = new Date(Date.now() - STALE_VERSION_TIMEOUT_MS);
  const updated = await prisma.sourceVersion.updateMany({
    where: {
      status: { in: ["STAGED", "PROMOTING"] },
      updatedAt: { lt: cutoff },
    },
    data: {
      status: "FAILED",
      failedAt: new Date(),
      errorCode: "STALE_VERSION_TIMEOUT",
      errorMessage: "Staged source version exceeded promotion timeout",
    },
  });
  return updated.count;
}

export async function softDeleteSource(params: {
  sourceId: string;
  retentionDays: number;
}) {
  const deletedAt = new Date();
  const restoreUntil = computeRestoreUntil(params.retentionDays, deletedAt);
  return prisma.source.update({
    where: { id: params.sourceId },
    data: {
      deletedAt,
      restoreUntil,
      status: "DISABLED",
    },
  });
}

export async function restoreSource(sourceId: string) {
  return prisma.source.update({
    where: { id: sourceId },
    data: {
      deletedAt: null,
      restoreUntil: null,
      status: "READY",
    },
  });
}

export function computeRestoreUntil(retentionDays: number, from = new Date()): Date {
  const restoreWindowDays = Math.min(DEFAULT_RESTORE_WINDOW_DAYS, Math.max(retentionDays, 1));
  return new Date(from.getTime() + restoreWindowDays * 24 * 60 * 60 * 1000);
}

export function deriveConnectorOutcome(input?: {
  partialFailure?: boolean | null;
  errorCode?: string | null;
  documentsIndexed?: number | null;
}): ConnectorOutcome {
  if (input?.errorCode && Number(input?.documentsIndexed || 0) <= 0 && !input?.partialFailure) {
    return "failed";
  }
  return input?.partialFailure ? "partial_failure" : "success";
}

export function serializeSourceVersion(version: any) {
  return {
    id: version.id,
    source_id: version.sourceId,
    org_id: version.orgId,
    project_id: version.projectId || null,
    version_number: version.versionNumber,
    status: String(version.status || "").toLowerCase(),
    partial_failure: Boolean(version.partialFailure),
    outcome: deriveConnectorOutcome({
      partialFailure: version.partialFailure,
      errorCode: version.errorCode,
      documentsIndexed: version.documentCount,
    }),
    warning_codes: Array.isArray(version.warningCodes) ? version.warningCodes : [],
    error_code: version.errorCode || null,
    error_message: version.errorMessage || null,
    document_count: version.documentCount ?? 0,
    chunk_count: version.chunkCount ?? 0,
    sync_job_id: version.syncJobId || null,
    promoted_at: version.promotedAt || null,
    superseded_at: version.supersededAt || null,
    failed_at: version.failedAt || null,
    restore_until: version.restoreUntil || null,
    created_at: version.createdAt,
    updated_at: version.updatedAt,
  };
}

export async function listSourceVersions(sourceId: string) {
  const versions = await prisma.sourceVersion.findMany({
    where: { sourceId },
    orderBy: [{ versionNumber: "desc" }, { createdAt: "desc" }],
  });
  return versions.map(serializeSourceVersion);
}

export async function getSourceVersion(sourceId: string, versionId: string) {
  const version = await prisma.sourceVersion.findFirst({
    where: {
      id: versionId,
      sourceId,
    },
  });
  return version ? serializeSourceVersion(version) : null;
}

export async function purgeExpiredDeletedSources(now = new Date()) {
  const expired = await prisma.source.findMany({
    where: {
      deletedAt: { not: null },
      restoreUntil: { lt: now },
    },
    select: {
      id: true,
      orgId: true,
      projectId: true,
      activeVersionId: true,
    },
    take: 100,
  });
  for (const source of expired) {
    await prisma.source.delete({
      where: { id: source.id },
    });
  }
  return expired;
}

export async function pruneSupersededSourceVersions(params?: {
  retentionDays?: number;
  keepLatest?: number;
}) {
  const retentionDays = Math.max(params?.retentionDays || DEFAULT_SUPERSEDED_RETENTION_DAYS, 1);
  const keepLatest = Math.max(params?.keepLatest || DEFAULT_SUPERSEDED_VERSIONS_TO_KEEP, 1);
  const cutoff = new Date(Date.now() - retentionDays * 24 * 60 * 60 * 1000);
  const candidates = await prisma.sourceVersion.findMany({
    where: {
      status: "SUPERSEDED",
      supersededAt: { lt: cutoff },
    },
    orderBy: [{ sourceId: "asc" }, { versionNumber: "desc" }],
    select: {
      id: true,
      sourceId: true,
      versionNumber: true,
    },
  });

  const candidatesBySource = new Map<string, typeof candidates>();
  for (const version of candidates) {
    const existing = candidatesBySource.get(version.sourceId) || [];
    existing.push(version);
    candidatesBySource.set(version.sourceId, existing);
  }

  let pruned = 0;
  for (const versions of candidatesBySource.values()) {
    const deletable = versions.slice(keepLatest);
    for (const version of deletable) {
      await prisma.$transaction(async (tx) => {
        await tx.chunk.deleteMany({
          where: {
            document: {
              sourceVersionId: version.id,
            },
          },
        });
        await tx.document.deleteMany({
          where: { sourceVersionId: version.id },
        });
        await tx.sourceVersion.delete({
          where: { id: version.id },
        });
      });
      pruned += 1;
    }
  }

  return pruned;
}

export function summarizeWarningCodes(errors: string[]): string[] {
  return [...new Set(errors.map((error) => classifyErrorCode(error)).filter(Boolean))];
}

export function classifyErrorCode(message: string | null | undefined): string {
  const normalized = String(message || "").toLowerCase();
  if (!normalized) return "";
  if (normalized.includes("timeout")) return "TIMEOUT";
  if (normalized.includes("429") || normalized.includes("rate limit")) return "RATE_LIMITED";
  if (normalized.includes("401") || normalized.includes("403") || normalized.includes("unauthorized")) {
    return "AUTH_FAILURE";
  }
  if (normalized.includes("cookie") || normalized.includes("captcha")) return "HTML_INTERSTITIAL";
  if (normalized.includes("transcript")) return "TRANSCRIPT_FAILURE";
  if (normalized.includes("ocr")) return "OCR_REQUIRED";
  if (normalized.includes("truncated")) return "TRUNCATED_SOURCE";
  if (normalized.includes("pdf")) return "PDF_PARSE_FAILURE";
  if (normalized.includes("html") || normalized.includes("dom") || normalized.includes("redirect")) {
    return "HTML_PARSE_FAILURE";
  }
  if (normalized.includes("repo") || normalized.includes("github") || normalized.includes("gitlab")) {
    return "REPO_LAYOUT_FAILURE";
  }
  if (normalized.includes("partial")) return "PARTIAL_FAILURE";
  return "PARTIAL_FAILURE";
}
