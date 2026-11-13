import { getOpenAIClient } from "./openai-client.js";
import { createHash } from "crypto";

// ─── Types ───────────────────────────────────────────────────

export interface CompressedContext {
  context: string;
  originalTokens: number;
  compressedTokens: number;
  reductionPercent: number;
  strategy: string;
}

export interface CompressionOptions {
  maxTokens?: number;
  strategy?: "summarize" | "extract" | "delta" | "adaptive";
  previousContextHash?: string; // for delta mode
  previousContext?: string; // for delta mode
  targetReduction?: number; // 0-1, e.g. 0.6 = reduce by 60%
}

// ─── In-memory delta cache ──────────────────────────────────

const deltaCache = new Map<string, { context: string; hash: string; timestamp: number }>();
const DELTA_CACHE_TTL = 600_000; // 10 min

function hashContext(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 16);
}

// ─── Main Compression Function ──────────────────────────────

export async function compressContext(
  rawContext: string,
  opts: CompressionOptions = {}
): Promise<CompressedContext> {
  const {
    maxTokens = 4000,
    strategy = "adaptive",
    previousContextHash,
    previousContext,
    targetReduction = 0.5,
  } = opts;

  const originalTokens = estimateTokens(rawContext);

  // If already under budget, return as-is
  if (originalTokens <= maxTokens) {
    return {
      context: rawContext,
      originalTokens,
      compressedTokens: originalTokens,
      reductionPercent: 0,
      strategy: "none",
    };
  }

  switch (strategy) {
    case "delta":
      return deltaCompress(rawContext, originalTokens, maxTokens, previousContextHash, previousContext);
    case "summarize":
      return summarizeCompress(rawContext, originalTokens, maxTokens);
    case "extract":
      return extractCompress(rawContext, originalTokens, maxTokens);
    case "adaptive":
    default:
      return adaptiveCompress(rawContext, originalTokens, maxTokens, previousContextHash, previousContext);
  }
}

// ─── Adaptive Strategy ──────────────────────────────────────
// Picks the best strategy based on context characteristics

async function adaptiveCompress(
  rawContext: string,
  originalTokens: number,
  maxTokens: number,
  previousHash?: string,
  previousCtx?: string
): Promise<CompressedContext> {
  const ratio = originalTokens / maxTokens;

  // If we have previous context and it's similar, use delta
  if (previousHash || previousCtx) {
    const delta = await deltaCompress(rawContext, originalTokens, maxTokens, previousHash, previousCtx);
    if (delta.compressedTokens <= maxTokens) return delta;
  }

  // Light compression (under 2x budget): extract key info
  if (ratio < 2) {
    return extractCompress(rawContext, originalTokens, maxTokens);
  }

  // Heavy compression (over 2x budget): summarize
  return summarizeCompress(rawContext, originalTokens, maxTokens);
}

// ─── Delta Compression ──────────────────────────────────────
// Only sends what changed since the last context

async function deltaCompress(
  rawContext: string,
  originalTokens: number,
  maxTokens: number,
  previousHash?: string,
  previousCtx?: string
): Promise<CompressedContext> {
  const currentHash = hashContext(rawContext);

  // Check if identical
  if (previousHash && previousHash === currentHash) {
    return {
      context: "[No changes since last context]",
      originalTokens,
      compressedTokens: 8,
      reductionPercent: 99,
      strategy: "delta-identical",
    };
  }

  // Get previous context from cache or parameter
  let prevCtx = previousCtx;
  if (!prevCtx && previousHash) {
    const cached = deltaCache.get(previousHash);
    if (cached && Date.now() - cached.timestamp < DELTA_CACHE_TTL) {
      prevCtx = cached.context;
    }
  }

  if (!prevCtx) {
    // No previous context to diff against, fall back to extract
    return extractCompress(rawContext, originalTokens, maxTokens);
  }

  // Split into blocks and find differences
  const prevBlocks = new Set(prevCtx.split("\n---\n").map((b) => b.trim()));
  const currentBlocks = rawContext.split("\n---\n").map((b) => b.trim());

  const newBlocks: string[] = [];
  const unchangedCount = { count: 0 };

  for (const block of currentBlocks) {
    if (prevBlocks.has(block)) {
      unchangedCount.count++;
    } else {
      newBlocks.push(block);
    }
  }

  let deltaContext: string;
  if (newBlocks.length === 0) {
    deltaContext = "[No new information since last query]";
  } else {
    const header = `[${unchangedCount.count} unchanged results omitted, ${newBlocks.length} new/updated]\n\n`;
    deltaContext = header + newBlocks.join("\n---\n");
  }

  // Trim if still over budget
  const deltaTokens = estimateTokens(deltaContext);
  if (deltaTokens > maxTokens) {
    const truncated = truncateToTokens(deltaContext, maxTokens);
    deltaCache.set(currentHash, { context: rawContext, hash: currentHash, timestamp: Date.now() });

    return {
      context: truncated,
      originalTokens,
      compressedTokens: estimateTokens(truncated),
      reductionPercent: Math.round((1 - estimateTokens(truncated) / originalTokens) * 100),
      strategy: "delta-truncated",
    };
  }

  // Cache current context for next delta
  deltaCache.set(currentHash, { context: rawContext, hash: currentHash, timestamp: Date.now() });

  return {
    context: deltaContext,
    originalTokens,
    compressedTokens: estimateTokens(deltaContext),
    reductionPercent: Math.round((1 - estimateTokens(deltaContext) / originalTokens) * 100),
    strategy: "delta",
  };
}

// ─── Extract Compression ────────────────────────────────────
// LLM extracts only the most relevant parts

async function extractCompress(
  rawContext: string,
  originalTokens: number,
  maxTokens: number
): Promise<CompressedContext> {
  try {
    const res = await getOpenAIClient().chat.completions.create({
      model: "gpt-4o-mini", // Fixed: was "gpt-4.1-nano" which doesn't exist
      messages: [
        {
          role: "system",
          content: `You are a context compressor. Extract and preserve ONLY the most important information from the provided context. Remove redundancy, boilerplate, and low-value content. Keep code snippets, key facts, API signatures, and important relationships. Output should be ${maxTokens} tokens or less. Do NOT add commentary — just output the compressed context.`,
        },
        { role: "user", content: rawContext },
      ],
      max_tokens: maxTokens,
      temperature: 0,
    });

    const compressed = res.choices[0]?.message?.content?.trim() || rawContext;
    const compressedTokens = estimateTokens(compressed);

    return {
      context: compressed,
      originalTokens,
      compressedTokens,
      reductionPercent: Math.round((1 - compressedTokens / originalTokens) * 100),
      strategy: "extract",
    };
  } catch {
    // Fallback to truncation
    const truncated = truncateToTokens(rawContext, maxTokens);
    return {
      context: truncated,
      originalTokens,
      compressedTokens: estimateTokens(truncated),
      reductionPercent: Math.round((1 - estimateTokens(truncated) / originalTokens) * 100),
      strategy: "truncate-fallback",
    };
  }
}

// ─── Summarize Compression ──────────────────────────────────
// For heavy compression — summarizes each block then combines

async function summarizeCompress(
  rawContext: string,
  originalTokens: number,
  maxTokens: number
): Promise<CompressedContext> {
  const blocks = rawContext.split("\n---\n").filter((b) => b.trim());

  // If few blocks, summarize the whole thing
  if (blocks.length <= 3) {
    return extractCompress(rawContext, originalTokens, maxTokens);
  }

  // Summarize each block individually, then combine
  const budgetPerBlock = Math.floor(maxTokens / blocks.length);

  try {
    const summaries = await Promise.all(
      blocks.map(async (block) => {
        if (estimateTokens(block) <= budgetPerBlock) return block;

        const res = await getOpenAIClient().chat.completions.create({
          model: "gpt-4o-mini", // Fixed: was "gpt-4.1-nano" which doesn't exist
          messages: [
            {
              role: "system",
              content: `Summarize this context block in ${budgetPerBlock} tokens or less. Preserve code signatures, key facts, and important details. Output only the summary.`,
            },
            { role: "user", content: block },
          ],
          max_tokens: budgetPerBlock,
          temperature: 0,
        });

        return res.choices[0]?.message?.content?.trim() || block.slice(0, budgetPerBlock * 4);
      })
    );

    const compressed = summaries.join("\n\n---\n\n");
    let compressedTokens = estimateTokens(compressed);

    // Final trim if still over
    let finalContext = compressed;
    if (compressedTokens > maxTokens) {
      finalContext = truncateToTokens(compressed, maxTokens);
      compressedTokens = estimateTokens(finalContext);
    }

    return {
      context: finalContext,
      originalTokens,
      compressedTokens,
      reductionPercent: Math.round((1 - compressedTokens / originalTokens) * 100),
      strategy: "summarize",
    };
  } catch {
    const truncated = truncateToTokens(rawContext, maxTokens);
    return {
      context: truncated,
      originalTokens,
      compressedTokens: estimateTokens(truncated),
      reductionPercent: Math.round((1 - estimateTokens(truncated) / originalTokens) * 100),
      strategy: "truncate-fallback",
    };
  }
}

// ─── Pre-computed Chunk Summaries ────────────────────────────
// Summarize chunks at ingest time so retrieval is cheaper

export async function summarizeChunk(content: string, chunkType: string): Promise<string> {
  // Only summarize large chunks
  if (estimateTokens(content) < 200) return content;

  try {
    const prompt = chunkType === "code"
      ? "Summarize this code in 2-3 sentences. Include: function/class names, what it does, parameters, return type."
      : "Summarize this text in 2-3 sentences. Preserve key facts, names, and important details.";

    const res = await getOpenAIClient().chat.completions.create({
      model: "gpt-4o-mini", // Fixed: was "gpt-4.1-nano" which doesn't exist
      messages: [
        { role: "system", content: prompt },
        { role: "user", content: content.slice(0, 3000) },
      ],
      max_tokens: 150,
      temperature: 0,
    });

    return res.choices[0]?.message?.content?.trim() || content;
  } catch {
    return content;
  }
}

// ─── Helpers ─────────────────────────────────────────────────

function estimateTokens(text: string): number {
  // ~3.5 chars/token is more conservative than the common "4 chars" estimate.
  // English prose ≈ 4 chars/token; code can be 2-3 chars/token.
  // Erring slightly low (more tokens estimated) means we truncate earlier, which is safer.
  return Math.ceil(text.length / 3.5);
}

function truncateToTokens(text: string, maxTokens: number): string {
  const maxChars = Math.floor(maxTokens * 3.5);
  if (text.length <= maxChars) return text;
  return text.slice(0, maxChars) + "\n\n[...truncated]";
}

// MEMORY FIX: Store interval reference for cleanup
let cacheCleanupInterval: NodeJS.Timeout | null = null;

function startCacheCleanup() {
  if (cacheCleanupInterval) return;

  cacheCleanupInterval = setInterval(() => {
    const now = Date.now();
    for (const [key, val] of deltaCache) {
      if (now - val.timestamp > DELTA_CACHE_TTL) deltaCache.delete(key);
    }
  }, 60_000);
  cacheCleanupInterval.unref();
}

export function stopCacheCleanup() {
  if (cacheCleanupInterval) {
    clearInterval(cacheCleanupInterval);
    cacheCleanupInterval = null;
  }
}

// Start cleanup on module load
startCacheCleanup();

// Cleanup on process exit
process.on('SIGTERM', stopCacheCleanup);
process.on('SIGINT', stopCacheCleanup);
process.on('beforeExit', stopCacheCleanup);
