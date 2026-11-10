/**
 * High-performance ingestion queue with parallel processing
 * Handles large file uploads asynchronously with webhook notifications
 */

import { prisma } from "../db/index.js";
import { ingestDocument } from "./ingest.js";
import { ingestSession } from "./memory/index.js";
import { writeMemoryCanonical } from "./memory/write.js";
import { withRetryableWriteRetries } from "./memory/write-reliability.js";

export type IngestionJobStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export interface IngestionJob {
  id: string;
  orgId: string;
  projectId: string;
  userId: string;
  status: IngestionJobStatus;
  totalDocuments: number;
  processedDocuments: number;
  totalChunks: number;
  processedChunks: number;
  webhookUrl?: string;
  metadata: Record<string, any>;
  startedAt?: Date;
  completedAt?: Date;
  error?: string;
}

export interface IngestionDocument {
  title: string;
  content: string;
  url?: string;
  metadata?: Record<string, any>;
  namespace?: string;
  tags?: string[];
  ingestion_profile?: "auto" | "repo" | "web_docs" | "pdf_layout" | "video_transcript" | "plain_text";
  strategy_override?: "fixed" | "recursive" | "semantic" | "hierarchical" | "adaptive";
  profile_config?: Record<string, any>;
}

export interface IngestionMemory {
  content: string;
  memory_type?: string;
  user_id?: string;
  session_id?: string;
  agent_id?: string;
  task_id?: string;
  importance?: number;
  metadata?: Record<string, any>;
  expires_in_seconds?: number;
}

const INGESTION_RETRY_MAX_ATTEMPTS = parseInt(process.env.INGESTION_RETRY_MAX_ATTEMPTS || "5", 10);
const INGESTION_RETRY_BACKOFF_MS = parseInt(process.env.INGESTION_RETRY_BACKOFF_MS || "750", 10);

export interface IngestionConversation {
  session_id?: string;
  user_id?: string;
  agent_id?: string;
  task_id?: string;
  title?: string;
  messages: Array<{ role: string; content: string }>;
  events?: Array<Record<string, any>>;
  metadata?: Record<string, any>;
}

class IngestionQueue {
  private processing = new Map<string, boolean>();
  private maxConcurrent = 50;

  async createJob(params: {
    orgId: string;
    projectId: string;
    userId: string;
    documents?: IngestionDocument[];
    memories?: IngestionMemory[];
    conversations?: IngestionConversation[];
    webhookUrl?: string;
    chunkSize?: number;
    chunkOverlap?: number;
    namespace?: string;
    tags?: string[];
  }): Promise<string> {
    const { randomUUID } = await import('crypto');
    const jobId = randomUUID();

    const totalItems =
      (params.documents?.length || 0) +
      (params.memories?.length || 0) +
      (params.conversations?.length || 0);

    // Create job record
    await prisma.$executeRaw`
      INSERT INTO ingestion_jobs (
        id, org_id, project_id, user_id, status,
        total_documents, processed_documents, total_chunks, processed_chunks,
        webhook_url, metadata, created_at, updated_at
      ) VALUES (
        ${jobId}, ${params.orgId}, ${params.projectId}, ${params.userId}, 'PENDING',
        ${totalItems}, 0, 0, 0,
        ${params.webhookUrl || null}, ${JSON.stringify({
          chunkSize: params.chunkSize || 1000,
          chunkOverlap: params.chunkOverlap || 200,
          namespace: params.namespace,
          tags: params.tags || [],
          hasDocuments: (params.documents?.length || 0) > 0,
          hasMemories: (params.memories?.length || 0) > 0,
          hasConversations: (params.conversations?.length || 0) > 0,
          ingestion_profile: params.documents?.[0]?.ingestion_profile,
          strategy_override: params.documents?.[0]?.strategy_override,
          profile_config: params.documents?.[0]?.profile_config,
        })}::jsonb, NOW(), NOW()
      )
    `;

    // Store documents for processing
    const itemsToStore: any[] = [];

    // Add documents
    if (params.documents && params.documents.length > 0) {
      const { randomUUID } = await import('crypto');
      params.documents.forEach((doc, idx) => {
        itemsToStore.push({
          id: randomUUID(),
          jobId,
          projectId: params.projectId,
          title: doc.title,
          content: doc.content,
          url: doc.url,
          metadata: {
            ...(doc.metadata || {}),
            type: 'document',
            namespace: doc.namespace || params.namespace,
            tags: [...(doc.tags || []), ...(params.tags || [])],
            ingestion_profile: doc.ingestion_profile,
            strategy_override: doc.strategy_override,
            profile_config: doc.profile_config,
            index: idx,
          },
          status: 'PENDING',
        });
      });
    }

    // Add memories
    if (params.memories && params.memories.length > 0) {
      const { randomUUID } = await import('crypto');
      params.memories.forEach((mem, idx) => {
        itemsToStore.push({
          id: randomUUID(),
          jobId,
          projectId: params.projectId,
          title: `Memory ${idx + 1}`,
          content: mem.content,
          url: null,
          metadata: {
            type: 'memory',
            memory_type: mem.memory_type,
            user_id: mem.user_id,
            session_id: mem.session_id,
            agent_id: mem.agent_id,
            importance: mem.importance,
            namespace: mem.metadata?.namespace || params.namespace,
            tags: [...(mem.metadata?.tags || []), ...(params.tags || [])],
            index: idx,
          },
          status: 'PENDING',
        });
      });
    }

    // Add conversations
    if (params.conversations && params.conversations.length > 0) {
      const { randomUUID } = await import('crypto');
      params.conversations.forEach((conv, idx) => {
        // Flatten messages into content
        const content = conv.messages.map((m: any) =>
          `${m.role}: ${m.content}`
        ).join('\n\n');

        itemsToStore.push({
          id: randomUUID(),
          jobId,
          projectId: params.projectId,
          title: conv.title || `Conversation ${idx + 1}`,
          content,
          url: null,
          metadata: {
            type: 'conversation',
            session_id: conv.session_id,
            user_id: conv.user_id,
            agent_id: conv.agent_id || conv.metadata?.agent_id,
            task_id: conv.task_id || conv.metadata?.task_id,
            events: conv.events || conv.metadata?.events || [],
            promotion_mode: conv.metadata?.promotion_mode,
            messages: conv.messages,
            namespace: conv.metadata?.namespace || params.namespace,
            tags: [...(conv.metadata?.tags || []), ...(params.tags || [])],
            index: idx,
          },
          status: 'PENDING',
        });
      });
    }

    // Batch insert documents
    if (itemsToStore.length > 0) {
      for (const item of itemsToStore) {
        await prisma.$executeRaw`
          INSERT INTO ingestion_documents (
            id, job_id, project_id, title, content, url, metadata, status, created_at, updated_at
          ) VALUES (
            ${item.id}, ${item.jobId}, ${item.projectId}, ${item.title}, ${item.content},
            ${item.url}, ${JSON.stringify(item.metadata)}::jsonb, ${item.status}, NOW(), NOW()
          )
        `;
      }
    }

    // Start processing asynchronously (fire and forget)
    this.processJob(jobId).catch(err => {
      console.error(`[IngestionQueue] Job ${jobId} failed:`, err);
    });

    return jobId;
  }

  private async processJob(jobId: string) {
    // Prevent duplicate processing
    if (this.processing.get(jobId)) {
      return;
    }
    this.processing.set(jobId, true);

    try {
      // Get job details
      const jobRows = await prisma.$queryRaw<any[]>`
        SELECT * FROM ingestion_jobs WHERE id = ${jobId}
      `;

      if (jobRows.length === 0) {
        throw new Error(`Job ${jobId} not found`);
      }

      const job = jobRows[0];

      // Get pending documents
      const documents = await prisma.$queryRaw<any[]>`
        SELECT * FROM ingestion_documents
        WHERE job_id = ${jobId} AND status = 'PENDING'
        ORDER BY created_at ASC
      `;

      // Update status to processing
      await prisma.$executeRaw`
        UPDATE ingestion_jobs
        SET status = 'PROCESSING', started_at = NOW(), updated_at = NOW()
        WHERE id = ${jobId}
      `;

      // Send webhook notification
      await this.sendWebhook(job.webhook_url, {
        event: 'ingestion.started',
        jobId: job.id,
        totalDocuments: job.total_documents,
        timestamp: new Date().toISOString(),
      });

      // Process documents in parallel batches
      const batchSize = this.maxConcurrent;
      for (let i = 0; i < documents.length; i += batchSize) {
        const batch = documents.slice(i, i + batchSize);

        const batchResults = await Promise.allSettled(
          batch.map(doc => this.processDocument(job, doc))
        );
        for (const r of batchResults) {
          if (r.status === "rejected") {
            console.error(`[IngestionQueue] Batch document failed:`, r.reason);
          }
        }

        // Update progress
        const progress = Math.min(i + batchSize, documents.length);
        await prisma.$executeRaw`
          UPDATE ingestion_jobs
          SET processed_documents = ${progress}, updated_at = NOW()
          WHERE id = ${jobId}
        `;

        // Send progress webhook
        await this.sendWebhook(job.webhook_url, {
          event: 'ingestion.progress',
          jobId: job.id,
          processedDocuments: progress,
          totalDocuments: job.total_documents,
          progress: (progress / job.total_documents) * 100,
          timestamp: new Date().toISOString(),
        });
      }

      const failures = await prisma.$queryRaw<Array<{ count: bigint | string }>>`
        SELECT COUNT(*) AS count FROM ingestion_documents WHERE job_id = ${jobId} AND status = 'FAILED'
      `;
      if (Number(failures[0]?.count || 0) > 0) {
        throw new Error("One or more ingestion documents failed");
      }

      // Get final chunk count
      const chunkCounts = await prisma.$queryRaw<any[]>`
        SELECT total_chunks, processed_chunks FROM ingestion_jobs WHERE id = ${jobId}
      `;

      // Mark job as completed
      await prisma.$executeRaw`
        UPDATE ingestion_jobs
        SET status = 'COMPLETED', completed_at = NOW(), updated_at = NOW()
        WHERE id = ${jobId}
      `;

      // Send completion webhook
      await this.sendWebhook(job.webhook_url, {
        event: 'ingestion.completed',
        jobId: job.id,
        totalDocuments: job.total_documents,
        totalChunks: chunkCounts[0]?.total_chunks || 0,
        timestamp: new Date().toISOString(),
      });

    } catch (error: any) {
      console.error(`[IngestionQueue] Job ${jobId} failed:`, error);

      await prisma.$executeRaw`
        UPDATE ingestion_jobs
        SET status = 'FAILED', error = ${error.message}, completed_at = NOW(), updated_at = NOW()
        WHERE id = ${jobId}
      `;

      const jobRows = await prisma.$queryRaw<any[]>`
        SELECT webhook_url FROM ingestion_jobs WHERE id = ${jobId}
      `;

      // Send failure webhook
      await this.sendWebhook(jobRows[0]?.webhook_url, {
        event: 'ingestion.failed',
        jobId: jobId,
        error: error.message,
        timestamp: new Date().toISOString(),
      });
    } finally {
      this.processing.delete(jobId);
    }
  }

  private async processDocument(job: any, doc: any) {
    try {
      const metadata = typeof doc.metadata === 'string' ? JSON.parse(doc.metadata) : doc.metadata;
      const jobMetadata = typeof job.metadata === 'string' ? JSON.parse(job.metadata) : job.metadata;

      const chunkSize = jobMetadata.chunkSize || 1000;
      const chunkOverlap = jobMetadata.chunkOverlap || 200;
      const namespace = metadata?.namespace || jobMetadata.namespace;
      const tags = metadata?.tags || jobMetadata.tags || [];

      const type = metadata?.type || 'document';

      if (type === 'memory') {
        const expiresAt = metadata.expires_in_seconds
          ? new Date(Date.now() + metadata.expires_in_seconds * 1000)
          : null;
        const documentDate = metadata.document_date ? new Date(metadata.document_date) : null;
        const eventDate = metadata.event_date ? new Date(metadata.event_date) : null;
        const writeResult = await withRetryableWriteRetries(
          () =>
            writeMemoryCanonical({
              projectId: job.project_id,
              orgId: job.org_id,
              userId: metadata.user_id || null,
              sessionId: metadata.session_id || null,
              agentId: metadata.agent_id || null,
              taskId: metadata.task_id || null,
              content: doc.content,
              memoryType: metadata.memory_type || "factual",
              importance: metadata.importance || 0.5,
              confidenceRaw: metadata.confidence_raw || metadata.confidence || 0.8,
              entityMentions: metadata.entity_mentions || [],
              documentDate,
              eventDate,
              expiresAt,
              metadata: {
                ...(metadata || {}),
                namespace,
                tags,
              },
              writeSource: metadata.write_source || "ingestion_queue.memory",
              writeMode: metadata.write_mode || "direct_write",
              extractionMethod: metadata.extraction_method || "manual",
              scopeHint: metadata.scope_target || undefined,
              promotionMode: metadata.promotion_mode || undefined,
              sessionRetentionDays: 14,
            }),
          {
            maxAttempts: INGESTION_RETRY_MAX_ATTEMPTS,
            baseDelayMs: INGESTION_RETRY_BACKOFF_MS,
            label: "IngestionQueue.writeMemory",
          }
        );

        if (writeResult.outcome === "dropped") {
          await prisma.$executeRaw`
            UPDATE ingestion_documents
            SET status = 'FAILED', error = ${`memory dropped: ${writeResult.validatorIssues.join(", ")}`}, updated_at = NOW()
            WHERE id = ${doc.id}
          `;
          return { success: false };
        }

        // Update document status
        await prisma.$executeRaw`
          UPDATE ingestion_documents
          SET status = 'COMPLETED', updated_at = NOW()
          WHERE id = ${doc.id}
        `;

        // Increment chunks (1 per memory)
        await prisma.$executeRaw`
          UPDATE ingestion_jobs
          SET total_chunks = total_chunks + 1,
              processed_chunks = processed_chunks + 1,
              updated_at = NOW()
          WHERE id = ${job.id}
        `;

      } else if (type === 'conversation') {
        // Process as conversation using ingestSession
        const messages = Array.isArray(metadata.messages)
          ? metadata.messages.map((message: any) => ({
              role: String(message?.role || "user"),
              content: String(message?.content || ""),
              timestamp: message?.timestamp ? new Date(message.timestamp) : new Date(),
            }))
          : [];

        await ingestSession({
          sessionId: metadata.session_id || `session_${doc.id}`,
          projectId: job.project_id,
          orgId: job.org_id,
          userId: metadata.user_id,
          agentId: metadata.agent_id,
          taskId: metadata.task_id,
          events: Array.isArray(metadata.events) ? metadata.events : [],
          promotionMode: metadata.promotion_mode,
          messages: messages,
        });

        // Update document status
        await prisma.$executeRaw`
          UPDATE ingestion_documents
          SET status = 'COMPLETED', updated_at = NOW()
          WHERE id = ${doc.id}
        `;

        // Increment chunks (estimate ~1 chunk per message)
        await prisma.$executeRaw`
          UPDATE ingestion_jobs
          SET total_chunks = total_chunks + ${messages.length},
              processed_chunks = processed_chunks + ${messages.length},
              updated_at = NOW()
          WHERE id = ${job.id}
        `;

      } else {
        // Process as document using existing ingest function
        const sourceId = await this.ensureAsyncJobSource(job);
        const result = await ingestDocument({
          sourceId,
          projectId: job.project_id,
          externalId: doc.id,
          title: doc.title,
          content: doc.content,
          metadata: metadata || {},
          url: doc.url,
          filePath: metadata?.file_path,
          ingestionProfile: metadata?.ingestion_profile || jobMetadata.ingestion_profile,
          strategyOverride: metadata?.strategy_override || jobMetadata.strategy_override,
          profileConfig: metadata?.profile_config || jobMetadata.profile_config,
        });

        // Update document status
        await prisma.$executeRaw`
          UPDATE ingestion_documents
          SET status = 'COMPLETED', document_id = ${result.documentId}, updated_at = NOW()
          WHERE id = ${doc.id}
        `;

        // Increment chunks
        await prisma.$executeRaw`
          UPDATE ingestion_jobs
          SET total_chunks = total_chunks + ${result.chunksCreated || 1},
              processed_chunks = processed_chunks + ${result.chunksCreated || 1},
              updated_at = NOW()
          WHERE id = ${job.id}
        `;
      }

      return { success: true };

    } catch (error: any) {
      console.error(`[IngestionQueue] Document ${doc.id} failed:`, error);

      await prisma.$executeRaw`
        UPDATE ingestion_documents
        SET status = 'FAILED', error = ${error.message}, updated_at = NOW()
        WHERE id = ${doc.id}
      `.catch(() => { /* best-effort status update */ });

      return { success: false, error: error.message };
    }
  }

  private async sendWebhook(url: string | null | undefined, payload: any, attempt = 0) {
    if (!url) return;

    const MAX_ATTEMPTS = 4;
    const BACKOFF_BASE_MS = 500;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Webhook-Event': payload.event,
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(10_000),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }
    } catch (error: any) {
      if (attempt < MAX_ATTEMPTS - 1) {
        const delay = BACKOFF_BASE_MS * Math.pow(2, attempt);
        console.warn(`[Webhook] Attempt ${attempt + 1} failed for ${url}, retrying in ${delay}ms:`, error.message);
        await new Promise(r => setTimeout(r, delay));
        return this.sendWebhook(url, payload, attempt + 1);
      }
      console.error(`[Webhook] All ${MAX_ATTEMPTS} attempts failed for ${url}:`, error.message);
    }
  }

  async getJobStatus(jobId: string): Promise<any> {
    const jobs = await prisma.$queryRaw<any[]>`
      SELECT * FROM ingestion_jobs WHERE id = ${jobId}
    `;

    if (jobs.length === 0) {
      return null;
    }

    const job = jobs[0];

    return {
      id: job.id,
      orgId: job.org_id,
      status: job.status,
      totalDocuments: job.total_documents,
      processedDocuments: job.processed_documents,
      totalChunks: job.total_chunks,
      processedChunks: job.processed_chunks,
      progress: job.total_documents > 0
        ? (job.processed_documents / job.total_documents) * 100
        : 0,
      startedAt: job.started_at,
      completedAt: job.completed_at,
      error: job.error,
      metadata: typeof job.metadata === 'string' ? JSON.parse(job.metadata) : job.metadata,
    };
  }

  private async ensureAsyncJobSource(job: any): Promise<string> {
    const sourceName = `async-ingest-${job.id}`;
    const existing = await prisma.source.findFirst({
      where: {
        orgId: job.org_id,
        projectId: job.project_id,
        connectorType: "custom",
        name: sourceName,
      },
      select: { id: true },
    });
    if (existing?.id) return existing.id;

    const created = await prisma.source.create({
      data: {
        orgId: job.org_id,
        projectId: job.project_id,
        name: sourceName,
        type: "custom",
        connectorType: "custom",
        config: { async_job_id: job.id },
        status: "INDEXING",
      },
      select: { id: true },
    });
    return created.id;
  }
}

export const ingestionQueue = new IngestionQueue();
