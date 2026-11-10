/**
 * Cache Layer - Redis/in-memory caching for frequent queries
 * Reduces latency and costs
 */

import crypto from "crypto";

// In-memory cache fallback (use Redis in production)
const memoryCache = new Map<string, { data: any; expiry: number; accessTime: number }>();
const MAX_CACHE_SIZE = 1000; // Prevent memory exhaustion

// Semantic cache for vector search (stores query embeddings + results)
const semanticCache = new Map<string, { embedding: number[]; results: any; expiry: number; scope: string }>();
const MAX_SEMANTIC_CACHE_SIZE = 1000; // Increased from 500
const SEMANTIC_THRESHOLD = 0.85; // Lowered from 0.92 for higher hit rate

// Index for faster semantic lookup - bucket embeddings by quantizing to grid
const SEMANTIC_BUCKET_SIZE = 0.1; // Bucket size for approximate nearest neighbor
const semanticCacheIndex = new Map<string, string[]>(); // bucket key -> cache keys

export interface CacheConfig {
  ttl: number; // Time to live in seconds
  enabled: boolean;
  keyPrefix: string;
}

const DEFAULT_CONFIG: CacheConfig = {
  ttl: 3600, // 1 hour
  enabled: true,
  keyPrefix: "whisper:context:",
};

/**
 * Redis client (optional, falls back to memory)
 * MUST be exported for queue module
 */
let _redisClient: any = null;

export function getRedisClient() {
  return _redisClient;
}

function isRedisAvailable(): boolean {
  return _redisClient !== null && _redisClient.isOpen && _redisClient.isReady;
}

export async function initializeRedis(redisUrl?: string): Promise<void> {
  if (!redisUrl && !process.env.REDIS_URL) {
    console.log("⚠️  No Redis URL, using in-memory cache");
    return;
  }

  try {
    const redisModule = "redis";
    const { createClient } = (await import(redisModule)) as {
      createClient: (options: Record<string, unknown>) => any;
    };
    _redisClient = createClient({
      url: redisUrl || process.env.REDIS_URL,
      socket: {
        reconnectStrategy: (retries: number) => {
          if (retries > 10) {
            console.error("Redis max reconnect attempts reached, using in-memory cache");
            return new Error("Max reconnection attempts reached");
          }
          const delay = Math.min(retries * 100, 3000);
          console.log(`Redis reconnecting in ${delay}ms (attempt ${retries})`);
          return delay;
        },
      },
    });

    _redisClient.on("error", (err: Error) => {
      console.error("Redis error:", err.message);
    });

    _redisClient.on("connect", () => {
      console.log("✅ Redis connected");
    });

    _redisClient.on("ready", () => {
      console.log("✅ Redis ready");
    });

    _redisClient.on("end", () => {
      console.log("⚠️  Redis connection closed");
    });

    await _redisClient.connect();
  } catch (error) {
    console.error("❌ Redis connection failed, using in-memory cache:", error);
    _redisClient = null;
  }
}

/**
 * Generate cache key from parameters
 */
export function generateCacheKey(
  operation: string,
  params: Record<string, any>
): string {
  const paramStr = JSON.stringify(params, Object.keys(params).sort());
  const hash = crypto.createHash("sha256").update(paramStr).digest("hex").substring(0, 16);
  return `${DEFAULT_CONFIG.keyPrefix}${operation}:${hash}`;
}

/**
 * Get from cache
 */
export async function getFromCache<T>(key: string): Promise<T | null> {
  if (!DEFAULT_CONFIG.enabled) return null;

  try {
    if (isRedisAvailable()) {
      const data = await _redisClient.get(key);
      return data ? JSON.parse(data) : null;
    } else {
      // In-memory cache
      const cached = memoryCache.get(key);
      if (cached && cached.expiry > Date.now()) {
        // Update access time for LRU
        cached.accessTime = Date.now();
        return cached.data;
      } else {
        memoryCache.delete(key);
        return null;
      }
    }
  } catch (error) {
    console.error("Cache get error:", error);
    return null;
  }
}

/**
 * Set in cache
 */
export async function setInCache(
  key: string,
  data: any,
  ttl: number = DEFAULT_CONFIG.ttl
): Promise<void> {
  if (!DEFAULT_CONFIG.enabled) return;

  try {
    if (isRedisAvailable()) {
      // Upstash Redis uses set with ex option
      await _redisClient.set(key, JSON.stringify(data), { ex: ttl });
    } else {
      // In-memory cache with LRU eviction
      memoryCache.set(key, {
        data,
        expiry: Date.now() + ttl * 1000,
        accessTime: Date.now(),
      });

      // Enforce max size with LRU eviction
      if (memoryCache.size > MAX_CACHE_SIZE) {
        const now = Date.now();
        let oldestKey: string | null = null;
        let oldestAccessTime = now;

        // Find and remove oldest entries (expired first, then LRU)
        const entriesToDelete: string[] = [];
        for (const [k, v] of memoryCache.entries()) {
          if (v.expiry < now) {
            entriesToDelete.push(k);
          } else if (v.accessTime < oldestAccessTime) {
            oldestAccessTime = v.accessTime;
            oldestKey = k;
          }
        }

        // Delete expired entries first
        for (const k of entriesToDelete) {
          memoryCache.delete(k);
        }

        // If still over limit, delete oldest LRU entry
        if (memoryCache.size > MAX_CACHE_SIZE && oldestKey) {
          memoryCache.delete(oldestKey);
        }
      }
    }
  } catch (error) {
    console.error("Cache set error:", error);
  }
}

/**
 * Delete from cache
 */
export async function deleteFromCache(key: string): Promise<void> {
  try {
    if (isRedisAvailable()) {
      await _redisClient.del(key);
    } else {
      memoryCache.delete(key);
    }
  } catch (error) {
    console.error("Cache delete error:", error);
  }
}

/**
 * Clear cache by pattern
 */
export async function clearCacheByPattern(pattern: string): Promise<number> {
  try {
    if (isRedisAvailable()) {
      const keys = await _redisClient.keys(`${DEFAULT_CONFIG.keyPrefix}${pattern}*`);
      if (keys.length > 0) {
        await _redisClient.del(keys);
      }
      return keys.length;
    } else {
      let count = 0;
      const regex = new RegExp(`^${DEFAULT_CONFIG.keyPrefix}${pattern}`);
      for (const key of memoryCache.keys()) {
        if (regex.test(key)) {
          memoryCache.delete(key);
          count++;
        }
      }
      return count;
    }
  } catch (error) {
    console.error("Cache clear error:", error);
    return 0;
  }
}

/**
 * Cache decorator for functions
 */
export function cached<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  options: {
    keyGenerator: (...args: Parameters<T>) => string;
    ttl?: number;
  }
): T {
  return (async (...args: Parameters<T>) => {
    const key = options.keyGenerator(...args);

    // Try cache
    const cached = await getFromCache(key);
    if (cached !== null) {
      return cached;
    }

    // Execute function
    const result = await fn(...args);

    // Store in cache
    await setInCache(key, result, options.ttl);

    return result;
  }) as T;
}

/**
 * Cache statistics
 */
export interface CacheStats {
  hits: number;
  misses: number;
  size: number;
  hitRate: number;
}

let cacheHits = 0;
let cacheMisses = 0;

export function recordCacheHit(): void {
  cacheHits++;
}

export function recordCacheMiss(): void {
  cacheMisses++;
}

export async function getCacheStats(): Promise<CacheStats> {
  let size = 0;

  if (_redisClient) {
    size = await _redisClient.dbSize();
  } else {
    size = memoryCache.size;
  }

  const total = cacheHits + cacheMisses;
  const hitRate = total > 0 ? cacheHits / total : 0;

  return {
    hits: cacheHits,
    misses: cacheMisses,
    size,
    hitRate,
  };
}

/**
 * Warm cache with common queries
 * OPTIMIZED: Pre-generate embeddings and cache results
 */
export async function warmCache(params: {
  projectId: string;
  commonQueries: string[];
  searchFn?: (query: string, projectId: string) => Promise<any>;
}): Promise<{ cached: number; failed: number }> {
  const { projectId, commonQueries, searchFn } = params;
  
  if (!searchFn) {
    console.log("⚠️  No search function provided, skipping cache warm");
    return { cached: 0, failed: 0 };
  }

  console.log(`🔥 Warming cache for ${commonQueries.length} queries...`);
  
  let cached = 0;
  let failed = 0;

  // Process in parallel (batch of 5)
  const batchSize = 5;
  for (let i = 0; i < commonQueries.length; i += batchSize) {
    const batch = commonQueries.slice(i, i + batchSize);
    
    await Promise.allSettled(
      batch.map(async (query) => {
        try {
          await searchFn(query, projectId);
          cached++;
        } catch (error) {
          failed++;
        }
      })
    );
    
    console.log(`  Progress: ${Math.min(i + batchSize, commonQueries.length)}/${commonQueries.length}`);
  }

  console.log(`✅ Cache warmed: ${cached} cached, ${failed} failed`);
  return { cached, failed };
}

/**
 * Get hot memories for pre-caching
 * Returns the most recently accessed/created memories
 */
export async function getHotMemories(params: {
  projectId: string;
  limit?: number;
}): Promise<Array<{ id: string; content: string; embedding?: number[] }>> {
  const { limit = 50 } = params;
  
  // This would query your database for hot memories
  // You may want to track access frequency in your database
  // For now, return recent memories
  try {
    const { db } = await import("../db/index.js");
    const memories = await db.memory.findMany({
      where: {
        projectId: params.projectId,
        isActive: true,
      },
      orderBy: { createdAt: 'desc' },
      take: limit,
      select: {
        id: true,
        content: true,
      },
    });
    return memories as any[];
  } catch (error) {
    console.error("Failed to get hot memories:", error);
    return [];
  }
}

/**
 * Pre-warm semantic cache with hot memories
 * This directly populates the semantic cache with pre-computed embeddings
 */
export async function prewarmSemanticCache(params: {
  memories: Array<{ id: string; content: string; embedding?: number[] }>;
}): Promise<number> {
  const { memories } = params;
  let cached = 0;
  
  for (const memory of memories) {
    if (!memory.embedding) continue;
    
    // Store the memory content as a "result" in semantic cache
    // This way, similar queries will hit the cache
    const cacheKey = `sem:${crypto.createHash("md5").update(JSON.stringify(memory.embedding.slice(0, 10))).digest("hex").substring(0, 8)}`;
    
    semanticCache.set(cacheKey, {
      embedding: memory.embedding,
      results: [{
        memory: {
          id: memory.id,
          content: memory.content,
          memoryType: 'factual',
        },
        similarity: 1.0, // Perfect match for itself
      }],
      expiry: Date.now() + DEFAULT_CONFIG.ttl * 1000,
      scope: "",
    });
    
    cached++;
  }
  
  console.log(`🔥 Pre-warmed semantic cache with ${cached} memories`);
  return cached;
}

/**
 * Semantic Cache for Vector Search
 * Uses embedding similarity to find cached results for similar queries
 */

// Cosine similarity for embedding comparison
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Get bucket key for embedding (quantized for approximate lookup)
 */
function getEmbeddingBucket(embedding: number[]): string {
  return embedding.slice(0, 8).map(v => Math.floor(v / SEMANTIC_BUCKET_SIZE)).join(',');
}

// Returns the exact bucket + 26 adjacent buckets (±1 on first 3 dims, rest fixed).
// 27 total — enough to catch boundary queries without exploding to 3^8=6561.
function getNeighboringBuckets(embedding: number[]): string[] {
  const quantized = embedding.slice(0, 8).map(v => Math.floor(v / SEMANTIC_BUCKET_SIZE));
  const neighbors: string[] = [];
  for (let d0 = -1; d0 <= 1; d0++) {
    for (let d1 = -1; d1 <= 1; d1++) {
      for (let d2 = -1; d2 <= 1; d2++) {
        neighbors.push([
          quantized[0] + d0,
          quantized[1] + d1,
          quantized[2] + d2,
          ...quantized.slice(3),
        ].join(','));
      }
    }
  }
  return neighbors;
}

/**
 * Get from semantic cache using embedding similarity
 * OPTIMIZED: Uses bucket index for O(1) lookup instead of O(n) iteration
 */
export async function getFromSemanticCache(
  queryEmbedding: number[],
  scope = ""
): Promise<{ results: any; similarity: number } | null> {
  if (!DEFAULT_CONFIG.enabled) return null;
  
  const now = Date.now();

  // Collect candidate keys from the exact bucket + 26 neighboring buckets (±1 on first 3 dims).
  // This catches queries that fall near bucket boundaries — the most common cache-miss scenario.
  const neighborBuckets = getNeighboringBuckets(queryEmbedding);
  const candidateKeySet = new Set<string>();
  for (const bucket of neighborBuckets) {
    for (const key of semanticCacheIndex.get(bucket) ?? []) {
      candidateKeySet.add(key);
    }
  }

  let bestMatch: { key: string; embedding: number[]; results: any; similarity: number } | null = null;
  let bestSimilarity = 0;

  for (const key of candidateKeySet) {
    const entry = semanticCache.get(key);
    if (!entry || entry.scope !== scope) continue;

    if (entry.expiry < now) {
      semanticCache.delete(key);
      continue;
    }

    const similarity = cosineSimilarity(queryEmbedding, entry.embedding);
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity;
      bestMatch = { key, ...entry, similarity };
    }
  }
  
  // Return best match if above threshold
  if (bestMatch && bestMatch.similarity >= SEMANTIC_THRESHOLD) {
    recordCacheHit();
    return { results: bestMatch.results, similarity: bestMatch.similarity };
  }
  
  recordCacheMiss();
  return null;
}

/**
 * Store results in semantic cache with embedding
 * OPTIMIZED: Also updates bucket index for faster lookup
 */
export async function setInSemanticCache(
  queryEmbedding: number[],
  results: any,
  ttl: number = DEFAULT_CONFIG.ttl,
  scope = ""
): Promise<void> {
  if (!DEFAULT_CONFIG.enabled) return;

  const now = Date.now();
  const bucketKey = getEmbeddingBucket(queryEmbedding);
  const key = `sem:${crypto.createHash("sha256").update(JSON.stringify([scope, queryEmbedding])).digest("hex")}`;

  semanticCache.set(key, {
    embedding: queryEmbedding,
    results,
    expiry: now + ttl * 1000,
    scope,
  });
  
  // Update bucket index
  if (!semanticCacheIndex.has(bucketKey)) {
    semanticCacheIndex.set(bucketKey, []);
  }
  const bucket = semanticCacheIndex.get(bucketKey)!;
  if (!bucket.includes(key)) {
    bucket.push(key);
  }
  
  // Enforce max size with LRU-like eviction
  if (semanticCache.size > MAX_SEMANTIC_CACHE_SIZE) {
    let oldestKey: string | null = null;
    let oldestExpiry = Infinity;
    
    for (const [k, v] of semanticCache.entries()) {
      if (v.expiry < oldestExpiry) {
        oldestExpiry = v.expiry;
        oldestKey = k;
      }
    }
    
    if (oldestKey) {
      // Also remove from index
      const oldEntry = semanticCache.get(oldestKey);
      if (oldEntry) {
        const oldBucket = getEmbeddingBucket(oldEntry.embedding);
        const oldBucketKeys = semanticCacheIndex.get(oldBucket);
        if (oldBucketKeys) {
          const idx = oldBucketKeys.indexOf(oldestKey);
          if (idx > -1) oldBucketKeys.splice(idx, 1);
        }
      }
      semanticCache.delete(oldestKey);
    }
  }
}

/**
 * Get semantic cache stats
 */
export function getSemanticCacheStats(): { size: number; threshold: number; buckets: number } {
  return {
    size: semanticCache.size,
    threshold: SEMANTIC_THRESHOLD,
    buckets: semanticCacheIndex.size,
  };
}
