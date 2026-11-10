import { createHash } from "crypto";
import { prisma } from "../db/index.js";

export type IndexBundleInclude = {
  sources: boolean;
  documents: boolean;
  chunks: boolean;
  memories: boolean;
  entities: boolean;
  relations: boolean;
};

export type IndexBundleLimits = {
  maxSources: number;
  maxDocuments: number;
  maxChunks: number;
  maxChunkChars: number;
  maxMemories: number;
  maxEntities: number;
  maxRelations: number;
};

export type ExportIndexBundleOptions = {
  include?: Partial<IndexBundleInclude>;
  limits?: Partial<IndexBundleLimits>;
  redactSecrets?: boolean;
};

type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue };

function toJsonSafe(value: unknown): JsonValue {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "bigint") return value.toString();
  if (value instanceof Date) return value.toISOString();
  if (Array.isArray(value)) return value.map(toJsonSafe);
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const out: Record<string, JsonValue> = {};
    for (const [key, entry] of Object.entries(obj)) {
      if (entry === undefined) continue;
      out[key] = toJsonSafe(entry);
    }
    return out;
  }
  return String(value);
}

function computeChecksum(value: unknown): string {
  const json = JSON.stringify(toJsonSafe(value));
  return createHash("sha256").update(json).digest("hex");
}

function redactSecretsDeep(value: unknown): unknown {
  if (!value) return value;
  if (typeof value === "string") {
    return value
      .replace(/(api[_-]?key\s*[=:]\s*)[^\s"'`]+/gi, "$1[REDACTED]")
      .replace(/(token\s*[=:]\s*)[^\s"'`]+/gi, "$1[REDACTED]")
      .replace(/(secret\s*[=:]\s*)[^\s"'`]+/gi, "$1[REDACTED]");
  }
  if (Array.isArray(value)) return value.map(redactSecretsDeep);
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(obj)) {
      if (entry === undefined) continue;
      const keyLower = key.toLowerCase();
      if (["token", "api_key", "apikey", "secret", "password"].some((needle) => keyLower.includes(needle))) {
        out[key] = "[REDACTED]";
        continue;
      }
      out[key] = redactSecretsDeep(entry);
    }
    return out;
  }
  return value;
}

export type IndexBundle = {
  bundle_version: "1.0";
  exported_at: string;
  project: {
    id: string;
    slug: string | null;
    name: string;
    description: string | null;
  };
  truncated: Partial<Record<keyof IndexBundleInclude, boolean>>;
  contents: {
    sources?: any[];
    documents?: any[];
    chunks?: any[];
    memories?: any[];
    entities?: any[];
    relations?: {
      entity_relations?: any[];
      memory_relations?: any[];
    };
  };
  checksum: string;
};

const DEFAULT_INCLUDE: IndexBundleInclude = {
  sources: true,
  documents: true,
  chunks: false,
  memories: true,
  entities: false,
  relations: false,
};

const DEFAULT_LIMITS: IndexBundleLimits = {
  maxSources: 200,
  maxDocuments: 500,
  maxChunks: 5000,
  maxChunkChars: 4000,
  maxMemories: 500,
  maxEntities: 2000,
  maxRelations: 4000,
};

export async function exportIndexBundle(params: {
  orgId: string;
  projectId: string;
  options?: ExportIndexBundleOptions;
}): Promise<IndexBundle> {
  const include: IndexBundleInclude = { ...DEFAULT_INCLUDE, ...(params.options?.include || {}) };
  const limits: IndexBundleLimits = { ...DEFAULT_LIMITS, ...(params.options?.limits || {}) };
  const redactSecrets = params.options?.redactSecrets !== false;

  const project = await prisma.project.findFirst({
    where: { id: params.projectId, orgId: params.orgId },
    select: { id: true, slug: true, name: true, description: true },
  });
  if (!project) {
    throw new Error("Project not found");
  }

  const truncated: Partial<Record<keyof IndexBundleInclude, boolean>> = {};
  const contents: IndexBundle["contents"] = {};

  if (include.sources) {
    const sources = await prisma.source.findMany({
      where: { projectId: project.id, deletedAt: null },
      orderBy: { createdAt: "asc" },
      take: limits.maxSources + 1,
      select: {
        id: true,
        name: true,
        type: true,
        connectorType: true,
        status: true,
        syncSchedule: true,
        lastSyncAt: true,
        lastSyncError: true,
        syncError: true,
        createdAt: true,
        updatedAt: true,
        config: true,
      },
    });
    truncated.sources = sources.length > limits.maxSources;
    const limited = sources.slice(0, limits.maxSources);
    contents.sources = limited.map((s) => ({
      ...s,
      config: redactSecrets ? redactSecretsDeep(s.config) : s.config,
    }));
  }

  let documentIds: string[] = [];
  if (include.documents || include.chunks) {
    const documents = await prisma.document.findMany({
      where: { projectId: project.id, deletedAt: null },
      orderBy: { createdAt: "asc" },
      take: limits.maxDocuments + 1,
      select: {
        id: true,
        sourceId: true,
        sourceVersionId: true,
        externalId: true,
        title: true,
        path: true,
        mimeType: true,
        language: true,
        metadata: true,
        tokens: true,
        status: true,
        parseError: true,
        indexedAt: true,
        lastModified: true,
        webUrl: true,
        createdAt: true,
        updatedAt: true,
      },
    });
    truncated.documents = documents.length > limits.maxDocuments;
    const limited = documents.slice(0, limits.maxDocuments);
    documentIds = limited.map((d) => d.id);
    if (include.documents) {
      contents.documents = limited;
    }
  }

  if (include.chunks && documentIds.length > 0) {
    const chunks = await prisma.chunk.findMany({
      where: { documentId: { in: documentIds } },
      orderBy: [{ documentId: "asc" }, { chunkIndex: "asc" }],
      take: limits.maxChunks + 1,
      select: {
        id: true,
        documentId: true,
        chunkOrder: true,
        chunkIndex: true,
        chunkType: true,
        content: true,
        tokens: true,
        metadata: true,
        sectionPath: true,
        headingPath: true,
        createdAt: true,
        updatedAt: true,
      },
    });
    truncated.chunks = chunks.length > limits.maxChunks;
    contents.chunks = chunks.slice(0, limits.maxChunks).map((c) => ({
      ...c,
      content:
        typeof c.content === "string"
          ? String(redactSecrets ? redactSecretsDeep(c.content) : c.content).slice(0, limits.maxChunkChars)
          : c.content,
    }));
  }

  if (include.memories) {
    const memories = await prisma.memory.findMany({
      where: { projectId: project.id, isActive: true },
      orderBy: { createdAt: "asc" },
      take: limits.maxMemories + 1,
      select: {
        id: true,
        userId: true,
        sessionId: true,
        agentId: true,
        memoryType: true,
        content: true,
        metadata: true,
        scope: true,
        importance: true,
        confidence: true,
        entityMentions: true,
        documentDate: true,
        eventDate: true,
        sourceChunkId: true,
        version: true,
        validFrom: true,
        validUntil: true,
        supersededBy: true,
        createdAt: true,
        updatedAt: true,
      },
    });
    truncated.memories = memories.length > limits.maxMemories;
    contents.memories = memories.slice(0, limits.maxMemories).map((m) => ({
      ...m,
      content: redactSecrets ? redactSecretsDeep(m.content) : m.content,
      metadata: redactSecrets ? redactSecretsDeep(m.metadata) : m.metadata,
    }));
  }

  if (include.entities) {
    const entities = await prisma.entity.findMany({
      where: { projectId: project.id },
      orderBy: { createdAt: "asc" },
      take: limits.maxEntities + 1,
      select: {
        id: true,
        name: true,
        entityType: true,
        description: true,
        metadata: true,
        createdAt: true,
      },
    });
    truncated.entities = entities.length > limits.maxEntities;
    contents.entities = entities.slice(0, limits.maxEntities).map((e) => ({
      ...e,
      metadata: redactSecrets ? redactSecretsDeep(e.metadata) : e.metadata,
    }));
  }

  if (include.relations) {
    const [entityRelations, memoryRelations] = await Promise.all([
      prisma.entityRelation.findMany({
        where: { projectId: project.id },
        orderBy: { createdAt: "asc" },
        take: limits.maxRelations + 1,
        select: {
          id: true,
          fromEntityId: true,
          toEntityId: true,
          relationType: true,
          weight: true,
          metadata: true,
          createdAt: true,
        },
      }),
      prisma.memoryRelation.findMany({
        where: {
          OR: [
            { fromMemory: { projectId: project.id } },
            { toMemory: { projectId: project.id } },
          ],
        },
        orderBy: { createdAt: "asc" },
        take: limits.maxRelations + 1,
        select: {
          id: true,
          fromMemoryId: true,
          toMemoryId: true,
          relationType: true,
          confidence: true,
          reasoning: true,
          metadata: true,
          createdAt: true,
        },
      }),
    ]);

    truncated.relations = entityRelations.length > limits.maxRelations || memoryRelations.length > limits.maxRelations;
    contents.relations = {
      entity_relations: entityRelations.slice(0, limits.maxRelations).map((r) => ({
        ...r,
        metadata: redactSecrets ? redactSecretsDeep(r.metadata) : r.metadata,
      })),
      memory_relations: memoryRelations.slice(0, limits.maxRelations).map((r) => ({
        ...r,
        metadata: redactSecrets ? redactSecretsDeep(r.metadata) : r.metadata,
      })),
    };
  }

  const unsignedBundle = {
    bundle_version: "1.0" as const,
    exported_at: new Date().toISOString(),
    project: {
      id: project.id,
      slug: project.slug || null,
      name: project.name,
      description: project.description || null,
    },
    truncated,
    contents,
  };
  const checksum = computeChecksum(unsignedBundle);
  return { ...unsignedBundle, checksum };
}

