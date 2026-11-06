import { createHash } from "node:crypto";
import { prisma } from "../db/index.js";
import { hashIdempotencyPayload } from "../api/idempotency.js";
import { ingestDocument } from "./ingest.js";
import { embedNativeCapture, stopNativeEmbeddingProcess } from "./native-embeddings.js";

export function redactCapture(value: unknown): unknown {
  if (typeof value === "string") {
    try {
      const parsed: unknown = JSON.parse(value);
      if (parsed && typeof parsed === "object") return JSON.stringify(redactCapture(parsed));
    } catch {}
    return value
      .replace(/\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})\b/g, "[REDACTED]")
      .replace(/(\bBearer\s+)[A-Za-z0-9._~+\/-]+/gi, "$1[REDACTED]")
      .replace(/(["']?(?:[a-z][a-z0-9]*[_-])*(?:api[_-]?key|(?:access|refresh|auth)[_-]?token|token|secret(?:[_-]?access[_-]?key)?|password|private[_-]?key)["']?\s*[:=]\s*)(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^"',\s}]+)/gi, "$1[REDACTED]");
  }
  if (Array.isArray(value)) return value.map(redactCapture);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, entry]) => [
      key,
      /(?:^|[_-])(?:authorization|api[_-]?key|(?:access|refresh|auth)[_-]?token|token|secret(?:[_-]?access[_-]?key)?|password|private[_-]?key)$/i.test(key)
        ? "[REDACTED]" : redactCapture(entry),
    ]));
  }
  return value;
}

function stableId(value: string): string {
  const hex = createHash("sha256").update(value).digest("hex");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-5${hex.slice(13, 16)}-a${hex.slice(17, 20)}-${hex.slice(20, 32)}`;
}

export interface NativeCaptureInput {
  orgId: string;
  projectId: string;
  sessionId: string;
  agentId: string;
  hookType: string;
  data: Record<string, unknown>;
  timestamp?: string;
}

export async function acceptNativeCapture(input: NativeCaptureInput) {
  const { orgId, projectId, sessionId, agentId, hookType } = input;
  const digest = hashIdempotencyPayload(input);
  const id = stableId(`native-event:${digest}`);
  const sessionKey = stableId(JSON.stringify([orgId, projectId, sessionId]));
  const safeData = redactCapture(input.data) as Record<string, unknown>;
  const content = `${hookType}\n${JSON.stringify(safeData)}`;
  const timestamp = input.timestamp ? new Date(input.timestamp) : new Date();
  const message = await prisma.$transaction(async tx => {
    await tx.session.upsert({
      where: { id: sessionKey },
      create: { id: sessionKey, orgId, projectId, sessionId, agentId, title: `Agent session ${sessionId}` },
      update: { sessionId },
    });
    const existing = await tx.message.findUnique({ where: { id } });
    if (existing) return existing;
    const stored = await tx.message.create({
      data: {
        id, sessionId: sessionKey, role: "event", content, createdAt: timestamp,
        metadata: {
          source: "native_agent_hook", external_session_id: sessionId, agent_id: agentId,
          hook_type: hookType, data: safeData as object, native_capture_state: "pending", capture_attempts: 0,
        },
      },
    });
    await tx.session.update({
      where: { id: sessionKey },
      data: { messageCount: { increment: 1 }, lastMessageAt: timestamp },
    });
    return stored;
  });
  void drainNativeCaptures().catch(error => console.error("[NativeCapture] Drain failed:", error));
  return { id: message.id, status: (message.metadata as Record<string, unknown>).native_capture_state };
}

interface PendingCapture {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  projectId: string;
  orgId: string;
  externalSessionId: string;
}

async function indexCapture(message: PendingCapture): Promise<string> {
  const deadline = Date.now() + 120_000;
  const sourceName = `native-agent-hooks-${message.projectId}`;
  const source = await prisma.source.upsert({
    where: { projectId_name: { projectId: message.projectId, name: sourceName } },
    create: {
      orgId: message.orgId, projectId: message.projectId, name: sourceName,
      type: "text", connectorType: "native_agent_hook", config: {}, status: "INDEXING",
    },
    update: {},
  });
  const document = await ingestDocument({
    sourceId: source.id, projectId: message.projectId, externalId: message.id,
    title: `${message.metadata.hook_type} (${message.externalSessionId})`, content: message.content,
    ingestionProfile: "plain_text", strategyOverride: "fixed", skipEntityExtraction: true,
    embedTexts: texts => embedNativeCapture(texts, deadline - Date.now()),
    metadata: {
      ...message.metadata, native_capture_state: undefined,
      event_id: message.id, session_id: message.externalSessionId, source: "native_agent_hook",
    },
  });
  const [{ total, vectors }] = await prisma.$queryRaw<Array<{ total: number; vectors: number }>>`
    SELECT COUNT(*)::int AS total, COUNT(embedding)::int AS vectors
    FROM chunks WHERE "documentId" = ${document.id} AND COALESCE(metadata->>'searchable', 'true') != 'false'
  `;
  if (!total || total !== vectors || total !== document.chunkStats.searchableChunks) {
    throw new Error("Native capture requires vectors for every searchable chunk");
  }
  return document.id;
}

async function processNextCapture(): Promise<boolean> {
  return prisma.$transaction(async tx => {
    const rows = await tx.$queryRaw<PendingCapture[]>`
      SELECT m.id, m.content, m.metadata, s."projectId", s."orgId", s."sessionId" AS "externalSessionId"
      FROM messages m JOIN sessions s ON s.id = m."sessionId"
      WHERE s."projectId" IS NOT NULL AND s."orgId" IS NOT NULL
        AND m.metadata->>'source' = 'native_agent_hook'
        AND m.metadata->>'native_capture_state' IN ('pending', 'processing')
        AND COALESCE((m.metadata->>'retry_at')::timestamptz, '-infinity') <= NOW()
      ORDER BY m."createdAt" LIMIT 1 FOR UPDATE OF m SKIP LOCKED
    `;
    const message = rows[0];
    if (!message) return false;
    const attempts = Number(message.metadata.capture_attempts || 0) + 1;
    try {
      const documentId = await indexCapture(message);
      await tx.$executeRaw`
        UPDATE messages SET metadata = (metadata - 'last_error' - 'retry_at') || jsonb_build_object(
          'native_capture_state', 'indexed', 'document_id', ${documentId}::text,
          'capture_attempts', ${attempts}::int, 'indexed_at', NOW()
        ) WHERE id = ${message.id}
      `;
    } catch (error) {
      const errorText = String(redactCapture(error instanceof Error ? error.message : String(error))).slice(0, 1000);
      const delaySeconds = Math.min(300, 2 ** Math.min(8, attempts));
      await tx.$executeRaw`
        UPDATE messages SET metadata = metadata || jsonb_build_object(
          'native_capture_state', 'pending', 'last_error', ${errorText}::text,
          'capture_attempts', ${attempts}::int, 'retry_at', NOW() + ${delaySeconds} * INTERVAL '1 second'
        ) WHERE id = ${message.id}
      `;
      console.error(`[NativeCapture] Event ${message.id} will retry: ${errorText}`);
    }
    return true;
  }, { timeout: 180_000, maxWait: 5000 });
}

let draining: Promise<void> | undefined;

export function drainNativeCaptures(): Promise<void> {
  if (!draining) draining = drainBatch().finally(() => { draining = undefined; });
  return draining;
}

async function drainBatch(): Promise<void> {
  for (let index = 0; index < 20; index++) {
    if (!await processNextCapture()) break;
  }
}

export function startNativeCaptureWorker(): () => void {
  const timer = setInterval(() => {
    void drainNativeCaptures().catch(error => console.error("[NativeCapture] Drain failed:", error));
  }, 1000);
  timer.unref();
  void drainNativeCaptures().catch(error => console.error("[NativeCapture] Recovery failed:", error));
  return () => {
    clearInterval(timer);
    void stopNativeEmbeddingProcess();
  };
}
