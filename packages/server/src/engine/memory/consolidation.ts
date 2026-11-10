/**
 * Memory Consolidation - Deduplicate and merge similar memories
 * Prevents memory bloat and improves search accuracy
 */

import OpenAI from "openai";
import { db } from "../../db/index.js";
import { embedSingle } from "../embeddings.js";

function getOpenAI() {
  return new OpenAI({
    apiKey: process.env.OPENAI_API_KEY || "",
    ...(process.env.OPENAI_BASE_URL ? { baseURL: process.env.OPENAI_BASE_URL } : {}),
  });
}

function getConsolidationModel(): string {
  return process.env.CONSOLIDATION_MODEL || process.env.OPENAI_MODEL || "gpt-5.4-mini";
}

function getMaxOutputTokensParam(model: string, maxTokens: number) {
  return /^gpt-5/i.test(model)
    ? { max_completion_tokens: maxTokens }
    : { max_tokens: maxTokens };
}

interface MemoryCluster {
  representative: any; // Most important memory in cluster
  duplicates: any[]; // Similar memories to merge
  similarity: number; // Average similarity score
}

/**
 * Find duplicate/similar memories using vector similarity
 */
export async function findDuplicateMemories(params: {
  projectId: string;
  userId?: string;
  similarityThreshold?: number;
  limit?: number;
}): Promise<MemoryCluster[]> {
  const {
    projectId,
    userId,
    similarityThreshold = 0.95,
    limit = 50,
  } = params;

  const maxMemories = Math.min(Math.max(limit, 10), 100);

  const memories = await db.memory.findMany({
    where: {
      projectId,
      userId,
      isActive: true,
      validUntil: null,
    },
    orderBy: { importance: "desc" },
    take: maxMemories,
  });

  const clusters: MemoryCluster[] = [];
  const processed = new Set<string>();

  for (let i = 0; i < memories.length; i++) {
    const memory = memories[i];

    if (processed.has(memory.id)) continue;

    const similar: any[] = [];
    const candidates = memories.slice(i + 1);

    const batchSimilarities = await calculateBatchSimilarity(memory.id, candidates.map(c => c.id));

    for (let j = 0; j < candidates.length; j++) {
      const other = candidates[j];

      if (processed.has(other.id)) continue;

      const similarity = batchSimilarities[j];

      if (similarity >= similarityThreshold) {
        similar.push({ ...other, similarity });
        processed.add(other.id);
      }
    }

    if (similar.length > 0) {
      clusters.push({
        representative: memory,
        duplicates: similar,
        similarity: similar.reduce((sum, m) => sum + m.similarity, 0) / similar.length,
      });
      processed.add(memory.id);
    }
  }

  return clusters;
}

async function calculateBatchSimilarity(memoryId: string, otherIds: string[]): Promise<number[]> {
  if (otherIds.length === 0) return [];

  const placeholders = otherIds.map((_, i) => `(m1.embedding <=> $${i + 2}::vector)`).join(' + ');
  const conditions = otherIds.map((id, i) => `m2.id = $${i + 2}`).join(' OR ');

  const result = await db.$queryRaw<any[]>`
    SELECT
      1 - (m1.embedding <=> m2.embedding) as similarity,
      m2.id as id
    FROM memories m1, memories m2
    WHERE m1.id = ${memoryId} AND (${conditions})
  `;

  const similarityMap = new Map(result.map(r => [r.id, r.similarity]));
  return otherIds.map(id => similarityMap.get(id) || 0);
}

/**
 * Calculate cosine similarity between two memory embeddings
 */
async function calculateSimilarity(memoryId1: string, memoryId2: string): Promise<number> {
  const result = await db.$queryRaw<any[]>`
    SELECT
      1 - (m1.embedding <=> m2.embedding) as similarity
    FROM memories m1, memories m2
    WHERE m1.id = ${memoryId1} AND m2.id = ${memoryId2}
  `;

  return result[0]?.similarity || 0;
}

/**
 * Merge duplicate memories using LLM
 */
export async function mergeDuplicateMemories(cluster: MemoryCluster): Promise<string> {
  const memories = [cluster.representative, ...cluster.duplicates];

  const prompt = `You are merging duplicate memories into a single, comprehensive memory.

**Memories to merge:**
${memories
  .map(
    (m, i) => `${i + 1}. "${m.content}" (confidence: ${m.confidence}, date: ${m.documentDate?.toISOString() || "unknown"})`
  )
  .join("\n")}

**Instructions:**
1. Combine all unique information from these memories
2. Resolve any contradictions by keeping the most recent or most confident information
3. Extract all unique entity mentions
4. Use the highest confidence score
5. Keep the most recent document date

Return JSON:
{
  "merged_content": "comprehensive merged memory",
  "entity_mentions": ["list", "of", "entities"],
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of how you merged"
}`;

  const model = getConsolidationModel();
  const response = await getOpenAI().chat.completions.create({
    model,
    ...getMaxOutputTokensParam(model, 2048),
    temperature: 0.0,
    messages: [{ role: "user", content: prompt }],
    response_format: { type: "json_object" },
  });

  const text = response.choices[0]?.message?.content?.trim();
  if (!text) {
    throw new Error("Failed to merge memories");
  }

  const jsonMatch = text.match(/```json\n?([\s\S]*?)\n?```/) || text.match(/\{[\s\S]*\}/);
  const jsonStr = jsonMatch ? (jsonMatch[1] || jsonMatch[0]) : text;

  const result = JSON.parse(jsonStr);

  // Create merged memory
  const embedding = await embedSingle(result.merged_content);

  const mergedMemory = await db.memory.create({
    data: {
      projectId: cluster.representative.projectId,
      orgId: cluster.representative.orgId,
      userId: cluster.representative.userId,
      sessionId: cluster.representative.sessionId,
      memoryType: cluster.representative.memoryType,
      content: result.merged_content,
      entityMentions: result.entity_mentions || [],
      confidence: result.confidence || cluster.representative.confidence,
      documentDate: cluster.representative.documentDate,
      eventDate: cluster.representative.eventDate,
      validFrom: new Date(),
      importance: Math.max(...memories.map((m) => m.importance || 0.5)),
      metadata: {
        mergedFrom: memories.map((m) => m.id),
        mergeReasoning: result.reasoning,
        mergedAt: new Date().toISOString(),
      },
    },
  });

  const embeddingLiteral = `[${embedding.join(",")}]`;
  await db.$executeRaw`
    UPDATE memories SET embedding = ${embeddingLiteral}::vector WHERE id = ${mergedMemory.id}
  `;

  // Deactivate old memories
  for (const memory of memories) {
    await db.memory.update({
      where: { id: memory.id },
      data: {
        isActive: false,
        validUntil: new Date(),
        supersededBy: mergedMemory.id,
      },
    });
  }

  return mergedMemory.id;
}

/**
 * Run consolidation job on a project
 */
export async function consolidateMemories(params: {
  projectId: string;
  userId?: string;
  similarityThreshold?: number;
  dryRun?: boolean;
}): Promise<{
  clustersFound: number;
  memoriesMerged: number;
  memoriesDeactivated: number;
}> {
  const { projectId, userId, similarityThreshold = 0.95, dryRun = false } = params;

  console.log(`🔍 Finding duplicate memories in project ${projectId}...`);

  const clusters = await findDuplicateMemories({
    projectId,
    userId,
    similarityThreshold,
  });

  console.log(`📊 Found ${clusters.length} memory clusters`);

  if (dryRun) {
    for (const cluster of clusters) {
      console.log(`\nCluster (similarity: ${cluster.similarity.toFixed(2)}):`);
      console.log(`  Representative: "${cluster.representative.content}"`);
      console.log(`  Duplicates: ${cluster.duplicates.length}`);
      cluster.duplicates.forEach((d) => {
        console.log(`    - "${d.content}"`);
      });
    }

    return {
      clustersFound: clusters.length,
      memoriesMerged: 0,
      memoriesDeactivated: 0,
    };
  }

  // Merge clusters
  let memoriesMerged = 0;
  let memoriesDeactivated = 0;

  for (const cluster of clusters) {
    try {
      console.log(`🔗 Merging cluster with ${cluster.duplicates.length + 1} memories...`);

      await mergeDuplicateMemories(cluster);

      memoriesMerged++;
      memoriesDeactivated += cluster.duplicates.length + 1; // All memories in cluster

      console.log(`✅ Merged successfully`);
    } catch (error) {
      console.error(`❌ Failed to merge cluster:`, error);
    }
  }

  console.log(
    `\n✅ Consolidation complete: ${memoriesMerged} clusters merged, ${memoriesDeactivated} memories deactivated`
  );

  return {
    clustersFound: clusters.length,
    memoriesMerged,
    memoriesDeactivated,
  };
}

/**
 * Find memories that need consolidation (scheduled job)
 */
export async function scheduledConsolidation(orgId: string): Promise<void> {
  console.log(`🔄 Running scheduled consolidation for org ${orgId}...`);

  const projects = await db.project.findMany({
    where: { orgId },
  });

  for (const project of projects) {
    try {
      const result = await consolidateMemories({
        projectId: project.id,
        similarityThreshold: 0.92, // Slightly lower for scheduled runs
      });

      if (result.memoriesMerged > 0) {
        console.log(
          `📊 Project ${project.name}: merged ${result.memoriesMerged} clusters`
        );
      }
    } catch (error) {
      console.error(`Failed to consolidate project ${project.name}:`, error);
    }
  }

  console.log("✅ Scheduled consolidation complete");
}
