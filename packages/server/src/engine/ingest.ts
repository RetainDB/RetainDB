import { createHash, randomUUID } from "crypto";
import PQueue from "p-queue";
import { prisma } from "../db/index.js";
import { chunkText } from "./chunker.js";
import { embed } from "./embeddings.js";
import { extractEntities } from "./extractor.js";
import type { IngestionProfile, ProfileConfig, StrategyOverride } from "./ingestion-profiles.js";
import { refreshSourceVersionCounts, resolveSourceVersionForWrite } from "./source-versions.js";

const queue = new PQueue({ concurrency: 20 });
const ENABLE_AUTO_EXTRACTION = process.env.DISABLE_AUTO_EXTRACTION !== "true";
const EMBED_BATCH_SIZE = 500;

export interface IngestDocumentInput {
  sourceId: string;
  projectId: string;
  externalId: string;
  title: string;
  content: string;
  metadata?: Record<string, any>;
  filePath?: string;
  url?: string;
  webUrl?: string;
  sourceType?: string;
  ingestionProfile?: IngestionProfile;
  strategyOverride?: StrategyOverride;
  profileConfig?: ProfileConfig;
  skipEntityExtraction?: boolean;
  sourceVersionId?: string | null;
  skipSourceCountUpdate?: boolean;
  embedTexts?: (texts: string[]) => Promise<number[][]>;
}

type EmbeddedChunk = {
  tempId: string;
  vector?: number[];
};

export async function ingestDocument(input: IngestDocumentInput) {
  const {
    sourceId,
    projectId,
    externalId,
    title,
    content,
    metadata = {},
    filePath,
    url,
    webUrl,
    sourceType,
    ingestionProfile = "auto",
    strategyOverride,
    profileConfig,
    sourceVersionId: explicitSourceVersionId,
  } = input;

  const source = await prisma.source.findUnique({
    where: { id: sourceId },
    select: { id: true, activeVersionId: true, status: true },
  });
  if (!source) {
    throw new Error(`Source ${sourceId} not found for ingest.`);
  }
  const sourceVersionId =
    explicitSourceVersionId === undefined
      ? await resolveSourceVersionForWrite(sourceId)
      : explicitSourceVersionId;

  const contentHash = createHash("sha256").update(content).digest("hex");
  const chunking = chunkText(content, {
    filePath: filePath || externalId,
    metadata: { ...metadata, title },
    sourceType,
    ingestionProfile,
    strategyOverride,
    profileConfig,
  });

  const documentMetadata = {
    ...metadata,
    title,
    source_type: metadata.source_type || sourceType || metadata.source || null,
    ingestion_profile: chunking.plan.profile,
    chunk_strategy: chunking.plan.strategy,
    parser: chunking.plan.parser,
    parser_confidence: chunking.plan.parser_confidence,
    latency_budget_ms: chunking.plan.latency_budget_ms,
    duplicate_rate: chunking.stats.duplicateRate,
    duplicate_rate_scope: chunking.stats.duplicateRateScope,
    chunk_count_total: chunking.stats.totalChunks,
    chunk_count_searchable: chunking.stats.searchableChunks,
  };

  const existingDoc = sourceVersionId
    ? await prisma.document.findFirst({
        where: {
          sourceVersionId,
          externalId,
        },
        select: { id: true },
      })
    : await prisma.document.findFirst({
        where: {
          sourceId,
          externalId,
          sourceVersionId: null,
        },
        select: { id: true },
      });

  const doc = existingDoc
    ? await prisma.document.update({
        where: { id: existingDoc.id },
        data: {
          title,
          path: filePath || metadata.path || null,
          webUrl: url || webUrl || metadata.url || null,
          content,
          metadata: documentMetadata,
          contentHash,
          chunkingStrategy: chunking.plan.strategy,
          tokens: Math.ceil(content.length / 4),
          indexedAt: null,
          status: "PROCESSING",
          updatedAt: new Date(),
          deletedAt: null,
          sourceVersionId,
        },
      })
    : await prisma.document.create({
        data: {
          sourceId,
          sourceVersionId,
          projectId,
          externalId,
          title,
          path: filePath || metadata.path || null,
          webUrl: url || webUrl || metadata.url || null,
          content,
          metadata: documentMetadata,
          contentHash,
          chunkingStrategy: chunking.plan.strategy,
          tokens: Math.ceil(content.length / 4),
          indexedAt: null,
          status: "PROCESSING",
        },
      });

  const insertedChunkIds: string[] = [];
  const insertedLeafChunkIds: Array<{ tempId: string; chunkId: string }> = [];
  let readyDoc: typeof doc;

  try {
    await prisma.chunk.deleteMany({ where: { documentId: doc.id } });

    const embeddings = await embedSearchableChunks(chunking.chunks, input.embedTexts, Boolean(input.embedTexts));
    const embeddedByTempId = new Map(embeddings.map((entry) => [entry.tempId, entry.vector]));
    const chunkInsertErrors: unknown[] = [];

    // Pre-assign UUIDs so parent IDs are known before any insert runs.
    const chunkDbIds = new Map<string, string>();
    for (const chunk of chunking.chunks) {
      chunkDbIds.set(chunk.tempId, randomUUID());
    }

    const insertChunk = async (chunk: (typeof chunking.chunks)[number]) => {
      const id = chunkDbIds.get(chunk.tempId)!;
      const parentDbId = chunk.parentTempId ? chunkDbIds.get(chunk.parentTempId) || null : null;
      const enrichedMetadata = {
        ...chunk.metadata,
        section_path: chunk.sectionPath || chunk.metadata.section_path || null,
        heading_path: chunk.headingPath || chunk.metadata.heading_path || null,
        parent_chunk_id: chunk.parentTempId || null,
        content_kind: chunk.metadata.content_kind || (chunk.role === "parent" ? "parent_context" : chunk.role),
      };
      const metadataJson = JSON.stringify(enrichedMetadata);
      const vector = chunk.searchable ? embeddedByTempId.get(chunk.tempId) : undefined;
      const searchContent = buildSearchContent(title, chunk.headingPath, chunk.sectionPath, chunk.content);
      const contentHash = createHash("sha256").update(chunk.content).digest("hex");
      try {
        if (vector && vector.length > 0) {
          await prisma.$executeRaw`
            INSERT INTO "chunks" (
              id, "documentId", "projectId", content, "chunkType", "chunkIndex", metadata,
              "tokenCount", tokens, "searchContent", "contentHash", "parentChunkId", "sectionPath",
              "headingPath", embedding, "createdAt", "updatedAt"
            )
            VALUES (
              ${id}::uuid, ${doc.id}, ${projectId}, ${chunk.content}, ${chunk.chunkType}, ${chunk.chunkIndex},
              ${metadataJson}::jsonb, ${Math.ceil(chunk.content.length / 4)}, ${Math.ceil(chunk.content.length / 4)},
              ${searchContent}, ${contentHash}, ${parentDbId}, ${chunk.sectionPath || null}, ${chunk.headingPath || null},
              ${`[${vector.join(",")}]`}::vector, NOW(), NOW()
            )
          `;
        } else {
          await prisma.$executeRaw`
            INSERT INTO "chunks" (
              id, "documentId", "projectId", content, "chunkType", "chunkIndex", metadata,
              "tokenCount", tokens, "searchContent", "contentHash", "parentChunkId", "sectionPath",
              "headingPath", "createdAt", "updatedAt"
            )
            VALUES (
              ${id}::uuid, ${doc.id}, ${projectId}, ${chunk.content}, ${chunk.chunkType}, ${chunk.chunkIndex},
              ${metadataJson}::jsonb, ${Math.ceil(chunk.content.length / 4)}, ${Math.ceil(chunk.content.length / 4)},
              ${searchContent}, ${contentHash}, ${parentDbId}, ${chunk.sectionPath || null}, ${chunk.headingPath || null},
              NOW(), NOW()
            )
          `;
        }
        insertedChunkIds.push(id);
        if (chunk.searchable) {
          insertedLeafChunkIds.push({ tempId: chunk.tempId, chunkId: id });
        }
      } catch (error: any) {
        chunkInsertErrors.push(error);
      }
    };

    // Two-pass parallel insert: roots first, then children (preserves FK constraints).
    const roots = chunking.chunks.filter((c) => !c.parentTempId);
    const children = chunking.chunks.filter((c) => c.parentTempId);
    await Promise.all(roots.map(insertChunk));
    if (chunkInsertErrors.length === 0 && children.length > 0) await Promise.all(children.map(insertChunk));
    if (chunkInsertErrors.length > 0) {
      throw new AggregateError(chunkInsertErrors, "Failed to persist document chunks");
    }

    readyDoc = await prisma.document.update({
      where: { id: doc.id },
      data: { status: "READY", indexedAt: new Date(), parseError: null },
    });
  } catch (error) {
    try {
      await prisma.chunk.deleteMany({ where: { documentId: doc.id } });
      await prisma.document.update({
        where: { id: doc.id },
        data: { status: "ERROR", indexedAt: null, parseError: error instanceof Error ? error.message : String(error) },
      });
      if (!input.skipSourceCountUpdate) await updateSourceCounts(sourceId, sourceVersionId);
    } catch (cleanupError) {
      console.error(`[Ingest] Failed to record ingestion failure for document ${doc.id}:`, cleanupError);
    }
    throw error;
  }

  if (ENABLE_AUTO_EXTRACTION && !input.skipEntityExtraction) {
    const searchableChunks = chunking.chunks
      .filter((chunk) => chunk.searchable && chunk.content.length > 200)
      .slice(0, 5);

    for (const chunk of searchableChunks) {
      const stored = insertedLeafChunkIds.find((entry) => entry.tempId === chunk.tempId);
      extractEntities(projectId, chunk.content, chunk.chunkType, chunk.metadata, stored?.chunkId).catch((error) => {
        console.warn(`[Ingest] Entity extraction failed for chunk ${stored?.chunkId ?? chunk.tempId} (non-critical, continuing):`, error?.message ?? error);
      });
    }
  }

  if (!input.skipSourceCountUpdate) await updateSourceCounts(sourceId, sourceVersionId);

  return {
    ...readyDoc,
    documentId: doc.id,
    chunksCreated: insertedChunkIds.length,
    searchableChunksCreated: insertedLeafChunkIds.length,
    chunkPlan: chunking.plan,
    chunkStats: chunking.stats,
  };
}

async function embedSearchableChunks(chunks: ReturnType<typeof chunkText>["chunks"], embedTexts = embed, required = false): Promise<EmbeddedChunk[]> {
  const searchable = chunks.filter((chunk) => chunk.searchable);
  const embedded: EmbeddedChunk[] = [];

  for (let i = 0; i < searchable.length; i += EMBED_BATCH_SIZE) {
    const batch = searchable.slice(i, i + EMBED_BATCH_SIZE);
    let vectors: number[][] = [];
    try {
      vectors = await embedTexts(batch.map((chunk) => chunk.content));
    } catch (error: any) {
      if (required) throw error;
      console.error(`[Ingest] Embedding batch failed — skipping ${batch.length} chunks (no semantic search for these):`, error?.message || error);
      // Don't push chunks with undefined vectors; they'd be stored but never semantically searchable
      continue;
    }

    for (let j = 0; j < batch.length; j++) {
      if (!vectors[j]) {
        if (required) throw new Error("Missing required document embedding");
        console.warn(`[Ingest] Missing vector for chunk ${batch[j].tempId} — skipping`);
        continue;
      }
      embedded.push({
        tempId: batch[j].tempId,
        vector: vectors[j],
      });
    }
  }

  return embedded;
}

function buildSearchContent(title: string, headingPath: string | undefined, sectionPath: string | undefined, content: string): string {
  const parts = [title, headingPath, sectionPath, content]
    .filter(Boolean)
    .join("\n");
  return parts.slice(0, 20000);
}

async function updateSourceCounts(sourceId: string, sourceVersionId?: string | null) {
  if (sourceVersionId) {
    await refreshSourceVersionCounts(sourceVersionId);
  }

  const source = await prisma.source.findUnique({
    where: { id: sourceId },
    select: { activeVersionId: true },
  });
  if (!source) return;

  if (sourceVersionId && source.activeVersionId && source.activeVersionId !== sourceVersionId) {
    return;
  }

  const [docCount, chunkCount, errorCount] = await Promise.all([
    prisma.document.count({
      where: {
        sourceId,
        deletedAt: null,
        ...(source.activeVersionId ? { sourceVersionId: source.activeVersionId } : {}),
      },
    }),
    prisma.chunk.count({
      where: {
        document: {
          sourceId,
          deletedAt: null,
          ...(source.activeVersionId ? { sourceVersionId: source.activeVersionId } : {}),
        },
      },
    }),
    prisma.document.count({
      where: {
        sourceId,
        deletedAt: null,
        status: "ERROR",
        ...(source.activeVersionId ? { sourceVersionId: source.activeVersionId } : {}),
      },
    }),
  ]);

  await prisma.source.update({
    where: { id: sourceId },
    data: {
      documentCount: docCount,
      chunkCount,
      lastSyncAt: new Date(),
      status: errorCount > 0 ? "ERROR" : "READY",
      updatedAt: new Date(),
    },
  });
}

export async function finalizeSourceCounts(sourceId: string, sourceVersionId?: string | null) {
  return updateSourceCounts(sourceId, sourceVersionId);
}

export async function ingestDocuments(inputs: IngestDocumentInput[]) {
  const sourceVersions = new Map<string, Promise<string | null>>();
  const boundInputs = await Promise.all(inputs.map(async input => {
    let sourceVersionId = input.sourceVersionId;
    if (sourceVersionId === undefined) {
      let version = sourceVersions.get(input.sourceId);
      if (!version) {
        version = resolveSourceVersionForWrite(input.sourceId);
        sourceVersions.set(input.sourceId, version);
      }
      sourceVersionId = await version;
    }
    return { ...input, sourceVersionId, skipSourceCountUpdate: true };
  }));
  const tasks = boundInputs.map((input) => queue.add(() => ingestDocument(input)));
  const results = await Promise.all(tasks).catch(async error => {
    await Promise.allSettled(tasks);
    throw error;
  });

  // Skip if caller (e.g. GitHub connector) will handle counts itself after the full sync
  const allSkipped = inputs.every((i) => i.skipSourceCountUpdate);
  if (!allSkipped) {
    // One update per unique sourceId instead of one per document
    const uniqueSourceIds = [...new Set(boundInputs.map((i) => i.sourceId))];
    await Promise.all(
      uniqueSourceIds.map((sourceId) => {
        const sourceVersionId = boundInputs.find((i) => i.sourceId === sourceId)?.sourceVersionId;
        return updateSourceCounts(sourceId, sourceVersionId);
      })
    );
  }

  return results;
}
