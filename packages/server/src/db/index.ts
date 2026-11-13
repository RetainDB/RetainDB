import { PrismaClient } from "@prisma/client";

const globalForPrisma = globalThis as unknown as {
  prisma: ReturnType<typeof createPrismaClient> | undefined;
};

function buildDatabaseUrl(): string | undefined {
  const raw = process.env.DATABASE_URL;
  if (!raw) return raw;

  try {
    const url = new URL(raw);

    if (process.env.DB_CONNECTION_LIMIT && !url.searchParams.has("connection_limit")) {
      url.searchParams.set("connection_limit", process.env.DB_CONNECTION_LIMIT);
    }
    if (process.env.DB_POOL_TIMEOUT && !url.searchParams.has("pool_timeout")) {
      url.searchParams.set("pool_timeout", process.env.DB_POOL_TIMEOUT);
    }
    if (process.env.DB_MAX_CONNECTION_LIFETIME && !url.searchParams.has("max_connection_lifetime")) {
      url.searchParams.set("max_connection_lifetime", process.env.DB_MAX_CONNECTION_LIFETIME);
    }

    return url.toString();
  } catch {
    return raw;
  }
}

function createPrismaClient() {
  return new PrismaClient({
    datasources: {
      db: {
        url: buildDatabaseUrl(),
      },
    },
  }).$extends({
    name: "bigint-to-string",
    query: {
      async $allOperations({ args, query }) {
        const result = await query(args);
        return hasBigInt(result) ? convertBigIntToString(result) : result;
      },
    },
  });
}

export const db = globalForPrisma.prisma ?? createPrismaClient();
export const prisma = db;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasBigInt(obj: unknown): boolean {
  if (typeof obj === "bigint") return true;
  if (Array.isArray(obj)) return obj.some(hasBigInt);
  if (isPlainObject(obj)) return Object.values(obj).some(hasBigInt);
  return false;
}

function convertBigIntToString(obj: unknown): unknown {
  if (typeof obj === "bigint") return obj.toString();
  if (Array.isArray(obj)) return obj.map(convertBigIntToString);
  if (isPlainObject(obj)) {
    return Object.fromEntries(
      Object.entries(obj).map(([k, v]) => [k, convertBigIntToString(v)])
    );
  }
  return obj;
}

// Cache singleton in production to prevent exhausting connection pool on hot reloads
if (process.env.NODE_ENV === "production") {
  globalForPrisma.prisma = db;
}

// Re-export Prisma types for convenience
export { Prisma } from "@prisma/client";
export type {
  Project,
  Source,
  Document,
  Chunk,
  Embedding,
  Entity,
  EntityRelation,
  Memory,
  MemoryRelation,
  Session,
  Message,
  ChunkMemory,
  SyncJob,
  Webhook,
  WebhookDelivery,
  SharedFile,
  AgentTask,
  AgentRun,
  AgentRunStep,
} from "@prisma/client";
