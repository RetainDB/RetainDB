/**
 * Local Embeddings using Transformers.js (BGE-large)
 * Fast, free, and high-quality embeddings without API calls
 */

let pipeline: any = null;
let env: any = null;
let transformersLoadPromise: Promise<void> | null = null;

async function ensureTransformersLoaded() {
  if (pipeline && env) return;
  if (!transformersLoadPromise) {
    transformersLoadPromise = (async () => {
      try {
        const transformers = await import('@xenova/transformers');
        pipeline = transformers.pipeline;
        env = transformers.env;

        // Configure transformers.js
        env.allowLocalModels = false; // Use CDN models for easier deployment
        env.useBrowserCache = false; // Use disk cache in Node.js
        env.backends.onnx.wasm.numThreads = 1;
      } catch {
        console.warn('[LocalEmbeddings] @xenova/transformers not installed, local embeddings disabled');
        console.warn('[LocalEmbeddings] Install with: npm install @xenova/transformers');
      }
    })();
  }
  await transformersLoadPromise;
}

let embedder: any = null;
let crossEncoder: any = null;

/**
 * Initialize BGE-large embedding model
 * Model: BAAI/bge-large-en-v1.5
 * Dimensions: 1024
 * MTEB Score: 63.98 (beats OpenAI text-embedding-3-small!)
 */
async function initEmbedder() {
  await ensureTransformersLoaded();
  if (!pipeline) {
    throw new Error('@xenova/transformers not installed. Run: npm install @xenova/transformers');
  }

  if (!embedder) {
    console.log('[LocalEmbeddings] Loading BGE-large model (first time may take 1-2 min)...');
    const startTime = Date.now();

    embedder = await pipeline(
      'feature-extraction',
      'Xenova/bge-large-en-v1.5',
      { quantized: true } // Use quantized version for faster inference
    );

    console.log(`[LocalEmbeddings] Model loaded in ${Date.now() - startTime}ms`);
  }
  return embedder;
}

/**
 * Initialize cross-encoder re-ranker
 * Model: BAAI/bge-reranker-large (via Xenova) — quantized
 * ~4% NDCG improvement over bge-reranker-base on BEIR benchmarks.
 * Outputs sigmoid-normalised relevance score directly.
 * Falls back to bge-reranker-base if the large model is unavailable.
 */
async function initCrossEncoder() {
  await ensureTransformersLoaded();
  if (!pipeline) {
    throw new Error('@xenova/transformers not installed. Run: npm install @xenova/transformers');
  }

  if (!crossEncoder) {
    const startTime = Date.now();
    const models = ['Xenova/bge-reranker-large', 'Xenova/bge-reranker-base'];
    let lastError: any;
    for (const modelName of models) {
      try {
        console.log(`[CrossEncoder] Loading ${modelName}...`);
        crossEncoder = await pipeline('text-classification', modelName, { quantized: true });
        console.log(`[CrossEncoder] ${modelName} loaded in ${Date.now() - startTime}ms`);
        break;
      } catch (err) {
        lastError = err;
        console.warn(`[CrossEncoder] Could not load ${modelName}, trying next...`);
      }
    }
    if (!crossEncoder) throw lastError;
  }
  return crossEncoder;
}

/**
 * Generate embeddings for multiple texts
 */
export async function embedLocal(texts: string[]): Promise<number[][]> {
  const model = await initEmbedder();

  // Batch process for efficiency
  const results = await Promise.all(
    texts.map(async (text) => {
      const output = await model(text, {
        pooling: 'mean',
        normalize: true,
      });

      // Convert to array
      return Array.from(output.data) as number[];
    })
  );

  return results;
}

/**
 * Generate embedding for a single text
 */
export async function embedSingleLocal(text: string): Promise<number[]> {
  const [embedding] = await embedLocal([text]);
  return embedding;
}

/**
 * Cross-encoder re-ranking
 * Returns scores for each (query, passage) pair
 */
export async function crossEncoderScore(
  query: string,
  passages: string[]
): Promise<number[]> {
  const model = await initCrossEncoder();

  // BGE reranker expects (text, text_pair) — do not concatenate with [SEP]
  const results = await Promise.all(
    passages.map(async (passage) => {
      const output = await model(query, { text_pair: passage });
      // output[0].score is already the sigmoid-normalised relevance score
      return (output[0]?.score ?? 0) as number;
    })
  );

  return results;
}

/**
 * Re-rank results using cross-encoder
 * Much faster than LLM re-ranking (50ms vs 500ms)
 */
export async function rerankWithCrossEncoder<T extends { content: string; score: number }>(
  query: string,
  results: T[],
  topK?: number
): Promise<Array<T & { crossEncoderScore: number; combinedScore: number }>> {
  if (results.length === 0) return [];

  const passages = results.map(r => r.content.slice(0, 1024)); // 1024 chars for better context coverage
  const scores = await crossEncoderScore(query, passages);

  // Combine original score with cross-encoder score
  const reranked = results.map((result, i) => ({
    ...result,
    crossEncoderScore: scores[i],
    // Weighted combination: 40% original score, 60% cross-encoder
    combinedScore: result.score * 0.4 + scores[i] * 0.6,
  }));

  // Sort by combined score
  reranked.sort((a, b) => b.combinedScore - a.combinedScore);

  // Return top K if specified
  return topK ? reranked.slice(0, topK) : reranked;
}

/**
 * Check confidence of top result
 * Returns true if we should use LLM fallback
 */
export function shouldUseLLMFallback(
  rerankedResults: Array<{ crossEncoderScore: number }>
): boolean {
  if (rerankedResults.length === 0) return false;

  const topScore = rerankedResults[0].crossEncoderScore;

  // Use LLM fallback if:
  // 1. Top score is low (< 0.85)
  // 2. OR scores are very close (top 2 within 0.1)
  const shouldFallback = topScore < 0.85 ||
    (rerankedResults.length > 1 &&
     Math.abs(topScore - rerankedResults[1].crossEncoderScore) < 0.1);

  return shouldFallback;
}
