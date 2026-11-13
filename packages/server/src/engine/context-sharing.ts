/**
 * Context Sharing - Save, share, and resume conversations
 * Like Nia's context sharing feature
 */

import { nanoid } from "nanoid";
import { db, Prisma } from "../db/index.js";
import { getSessionMemories } from "./memory/index.js";

function toIsoString(value: unknown): string | undefined {
  if (!value) return undefined;
  if (value instanceof Date) return value.toISOString();
  const parsed = new Date(String(value));
  return Number.isNaN(parsed.getTime()) ? undefined : parsed.toISOString();
}

function toDate(value: unknown): Date | null {
  const iso = toIsoString(value);
  return iso ? new Date(iso) : null;
}

async function ensureSharedContextsTable(): Promise<void> {
  const statements = SHARED_CONTEXTS_MIGRATION
    .split(";")
    .map((statement) => statement.trim())
    .filter(Boolean);
  for (const statement of statements) {
    await db.$executeRawUnsafe(statement);
  }
}

export interface SharedContext {
  id: string;
  title?: string;
  sessionId: string;
  projectId: string;
  orgId: string;
  userId?: string;
  shareUrl: string;
  memories: any[];
  messages: any[];
  chunks: any[];
  metadata: Record<string, any>;
  createdAt: Date;
  expiresAt: Date | null;
  accessCount: number;
}

/**
 * Create a shareable context from a session
 */
export async function createSharedContext(params: {
  sessionId: string;
  projectId: string;
  orgId: string;
  userId?: string;
  title?: string;
  includeMemories?: boolean;
  includeMessages?: boolean;
  includeChunks?: boolean;
  expiryDays?: number;
  metadata?: Record<string, any>;
}): Promise<SharedContext> {
  await ensureSharedContextsTable();
  const {
    sessionId,
    projectId,
    orgId,
    userId,
    title,
    includeMemories = true,
    includeMessages = true,
    includeChunks = false,
    expiryDays = 7,
    metadata = {},
  } = params;

  // Generate unique share ID — 21 chars = ~126 bits of entropy (unguessable)
  const shareId = nanoid(21);

  // Get session data
  const session = await db.session.findFirst({
    where: {
      projectId,
      orgId,
      OR: [{ id: sessionId }, { sessionId }],
    },
  });

  if (!session) {
    throw new Error("Session not found");
  }

  // Collect data to share
  const shareData: any = {
    sessionId: session.sessionId || sessionId,
    projectId,
    userId,
    title: title || `Session ${session.sessionId || session.id}`,
    metadata,
  };

  // Include memories
  if (includeMemories) {
    const memories = await getSessionMemories({
      sessionId,
      projectId,
      limit: 200,
    });

    shareData.memories = memories.map((m: any) => ({
      id: m.id,
      content: m.content,
      type: m.memoryType,
      entities: m.entityMentions,
      confidence: m.confidence,
      documentDate: toIsoString(m.documentDate),
      eventDate: toIsoString(m.eventDate),
    }));
  } else {
    shareData.memories = [];
  }

  // Include messages
  if (includeMessages) {
    const messages = await db.message.findMany({
      where: {
        sessionId: session.id,
      },
      orderBy: { createdAt: "asc" },
      take: 500,
    });

    shareData.messages = messages.map((m) => ({
      role: m.role,
      content: m.content,
      createdAt: toIsoString((m.metadata as Record<string, unknown> | null)?.original_timestamp) || toIsoString(m.createdAt),
    }));
  } else {
    shareData.messages = [];
  }

  // Include chunks (optional, can be large)
  if (includeChunks) {
    // Get chunks referenced by memories (limit to 50 to prevent memory bloat)
    const chunkIds = shareData.memories
      .map((m: any) => m.sourceChunkId)
      .filter(Boolean)
      .slice(0, 50);

    if (chunkIds.length > 0) {
      const chunks = await db.chunk.findMany({
        where: {
          id: { in: chunkIds },
          projectId,
        },
        select: {
          id: true,
          content: true,
          metadata: true,
          chunkType: true,
        },
        take: 50,
      });

      shareData.chunks = chunks;
    } else {
      shareData.chunks = [];
    }
  } else {
    shareData.chunks = [];
  }

  // Calculate expiry
  const expiresAt = expiryDays
    ? new Date(Date.now() + expiryDays * 24 * 60 * 60 * 1000)
    : null;

  // Store in database
  await db.$executeRaw`
    INSERT INTO shared_contexts (
      id, session_id, project_id, org_id, user_id,
      share_data, expires_at, created_at, access_count
    ) VALUES (
      ${shareId}, ${sessionId}, ${projectId}, ${orgId}, ${userId || null},
      ${JSON.stringify(shareData)}::jsonb, ${expiresAt}, NOW(), 0
    )
    ON CONFLICT (id) DO NOTHING
  `;

  const baseUrl = process.env.BASE_URL || "https://api.retaindb.com";
  const shareUrl = `${baseUrl}/shared/${shareId}`;

  return {
    id: shareId,
    orgId: "default",
    title: shareData.title,
    sessionId: session.sessionId || sessionId,
    projectId,
    userId,
    shareUrl,
    memories: shareData.memories,
    messages: shareData.messages,
    chunks: shareData.chunks,
    metadata,
    createdAt: new Date(),
    expiresAt,
    accessCount: 0,
  };
}

/**
 * Load a shared context
 */
export async function loadSharedContext(shareId: string): Promise<SharedContext | null> {
  await ensureSharedContextsTable();
  const result = await db.$queryRaw<any[]>`
    SELECT
      id,
      session_id,
      project_id,
      org_id,
      user_id,
      share_data,
      created_at::text AS created_at,
      expires_at::text AS expires_at,
      access_count
    FROM shared_contexts
    WHERE id = ${shareId}
    AND (expires_at IS NULL OR expires_at > NOW())
  `;

  if (result.length === 0) {
    return null;
  }

  const row = result[0];

  // Increment access count
  await db.$executeRaw`
    UPDATE shared_contexts
    SET access_count = access_count + 1,
        last_accessed_at = NOW()
    WHERE id = ${shareId}
  `;

  const baseUrl = process.env.BASE_URL || "https://api.retaindb.com";

  return {
    id: row.id,
    title: row.share_data.title || row.share_data.metadata?.title || null,
    sessionId: row.session_id,
    projectId: row.project_id,
    orgId: row.org_id,
    userId: row.user_id,
    shareUrl: `${baseUrl}/shared/${shareId}`,
    memories: row.share_data.memories || [],
    messages: row.share_data.messages || [],
    chunks: row.share_data.chunks || [],
    metadata: row.share_data.metadata || {},
    createdAt: toDate(row.created_at) || new Date(),
    expiresAt: toDate(row.expires_at),
    accessCount: row.access_count,
  };
}

/**
 * Resume a session from shared context
 * Creates a new session with the same memories/context
 */
export async function resumeFromSharedContext(params: {
  shareId: string;
  projectId: string;
  orgId: string;
  userId?: string;
  newSessionId?: string;
}): Promise<{ sessionId: string; memoriesRestored: number; messagesRestored: number; chunksRestored: number }> {
  const { shareId, projectId, orgId, userId, newSessionId } = params;

  // Load shared context
  const sharedContext = await loadSharedContext(shareId);

  if (!sharedContext) {
    throw new Error("Shared context not found or expired");
  }

  // Ensure the caller can only resume contexts that belong to their own org
  if (sharedContext.orgId !== orgId) {
    throw new Error("Shared context not found or expired");
  }

  // Create new session
  const sessionId = newSessionId || nanoid();

  await db.session.create({
    data: {
      id: sessionId,
      projectId,
      orgId,
      userId,
      title: `Resumed from ${sharedContext.id}`,
      metadata: {
        resumedFrom: shareId,
        originalSessionId: sharedContext.sessionId,
        ...sharedContext.metadata,
      },
    },
  });

  // Restore memories
  let memoriesRestored = 0;
  let messagesRestored = 0;

  for (const memory of sharedContext.memories) {
    try {
      // Re-create memory in new session
      await db.memory.create({
        data: {
          projectId,
          orgId,
          userId,
          sessionId,
          memoryType: memory.type,
          content: memory.content,
          entityMentions: memory.entities || [],
          confidence: memory.confidence || 0.8,
          documentDate: memory.documentDate ? new Date(memory.documentDate) : null,
          eventDate: memory.eventDate ? new Date(memory.eventDate) : null,
          validFrom: new Date(),
          metadata: {
            restoredFrom: shareId,
          },
        },
      });

      memoriesRestored++;
    } catch (error) {
      console.error("Failed to restore memory:", error);
    }
  }

  // Restore messages (as context, not full message history)
  for (const msg of sharedContext.messages) {
    try {
      await db.message.create({
        data: {
          sessionId,
          role: msg.role,
          content: msg.content,
          metadata: {
            restoredFrom: shareId,
            originalTimestamp: msg.createdAt,
          },
        },
      });
      messagesRestored++;
    } catch (error) {
      console.error("Failed to restore message:", error);
    }
  }

  return {
    sessionId,
    memoriesRestored,
    messagesRestored,
    chunksRestored: sharedContext.chunks.length,
  };
}

/**
 * Delete expired shared contexts (cleanup job)
 */
export async function cleanupExpiredContexts(): Promise<number> {
  await ensureSharedContextsTable();
  const result = await db.$executeRaw`
    DELETE FROM shared_contexts
    WHERE expires_at IS NOT NULL
    AND expires_at < NOW()
  `;

  return result as number;
}

/**
 * List shared contexts for a user
 */
export async function listSharedContexts(params: {
  userId?: string;
  projectId?: string;
  orgId: string;
  limit?: number;
}): Promise<Array<{
  id: string;
  shareUrl: string;
  createdAt: Date;
  expiresAt: Date | null;
  accessCount: number;
  metadata: Record<string, any>;
}>> {
  await ensureSharedContextsTable();
  const { userId, projectId, orgId, limit = 50 } = params;
  const maxLimit = Math.min(limit, 100);

  const conditions = [Prisma.sql`org_id = ${orgId}`];
  if (userId) conditions.push(Prisma.sql`user_id = ${userId}`);
  if (projectId) conditions.push(Prisma.sql`project_id = ${projectId}`);
  const whereClause = Prisma.join(conditions, " AND ");

  const results = await db.$queryRaw<any[]>`
    SELECT id, session_id, created_at, expires_at, access_count, share_data
    FROM shared_contexts
    WHERE ${whereClause}
    ORDER BY created_at DESC
    LIMIT ${maxLimit}
  `;

  const baseUrl = process.env.BASE_URL || "https://api.retaindb.com";

  return results.map((row) => ({
    id: row.id,
    shareUrl: `${baseUrl}/shared/${row.id}`,
    createdAt: row.created_at,
    expiresAt: row.expires_at,
    accessCount: row.access_count,
    metadata: row.share_data?.metadata || {},
  }));
}

/**
 * Create shared_contexts table migration
 * Run this to add the table
 */
export const SHARED_CONTEXTS_MIGRATION = `
CREATE TABLE IF NOT EXISTS shared_contexts (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  project_id TEXT NOT NULL,
  org_id TEXT NOT NULL,
  user_id TEXT,
  share_data JSONB NOT NULL DEFAULT '{}',
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  access_count INTEGER NOT NULL DEFAULT 0,
  last_accessed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_shared_contexts_org ON shared_contexts(org_id);
CREATE INDEX IF NOT EXISTS idx_shared_contexts_user ON shared_contexts(user_id);
CREATE INDEX IF NOT EXISTS idx_shared_contexts_expires ON shared_contexts(expires_at);
CREATE INDEX IF NOT EXISTS idx_shared_contexts_session ON shared_contexts(session_id);
`;
