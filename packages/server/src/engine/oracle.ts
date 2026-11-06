/**
 * Oracle Research Mode - Tree-guided hybrid search
 * Like Nia's Oracle feature - SOTA search for complex queries
 */

import { getOpenAIClient } from "./openai-client.js";
import { prisma } from "../db/index.js";
import { embedSingle } from "./embeddings.js";
import { selectOracleScope } from "./oracle-select.js";

interface DocumentTree {
  root: TreeNode;
  depth: number;
  nodeCount: number;
}

interface TreeNode {
  id: string;
  content: string;
  type: "document" | "section" | "chunk";
  children: TreeNode[];
  metadata: Record<string, any>;
  embedding?: number[];
}

const ORACLE_MAX_DOCUMENT_CHUNKS = parseInt(process.env.ORACLE_MAX_DOCUMENT_CHUNKS || "1200", 10);
const ORACLE_SECTION_EMBED_SAMPLE = parseInt(process.env.ORACLE_SECTION_EMBED_SAMPLE || "40", 10);
const ORACLE_LEAF_THRESHOLD = parseFloat(process.env.ORACLE_LEAF_THRESHOLD || "0.28");
const ORACLE_EXPLORE_THRESHOLD = parseFloat(process.env.ORACLE_EXPLORE_THRESHOLD || "0.12");

const ORACLE_SELECT_MAX_SEED_HITS = parseInt(process.env.ORACLE_SELECT_MAX_SEED_HITS || "120", 10);
const ORACLE_SELECT_MAX_DOCUMENTS = parseInt(process.env.ORACLE_SELECT_MAX_DOCUMENTS || "6", 10);
const ORACLE_SELECT_MAX_SECTIONS_PER_DOC = parseInt(process.env.ORACLE_SELECT_MAX_SECTIONS_PER_DOC || "5", 10);
const ORACLE_SELECT_MAX_CANDIDATE_CHUNKS = parseInt(process.env.ORACLE_SELECT_MAX_CANDIDATE_CHUNKS || "1400", 10);

function isNumberArray(value: unknown): value is number[] {
  return Array.isArray(value) && value.every((n) => typeof n === "number" && Number.isFinite(n));
}

function meanEmbedding(embeddings: number[][]): number[] | undefined {
  const first = embeddings.find((e) => e.length > 0);
  if (!first) return undefined;
  const dim = first.length;
  const sum = new Array<number>(dim).fill(0);
  let count = 0;
  for (const vec of embeddings) {
    if (vec.length !== dim) continue;
    count += 1;
    for (let i = 0; i < dim; i += 1) sum[i] += vec[i];
  }
  if (count === 0) return undefined;
  return sum.map((v) => v / count);
}

function safeSectionPath(value: unknown): string {
  const raw = String(value || "").trim();
  return raw.length > 0 ? raw : "Document";
}

export async function buildDocumentTree(
  documentId: string,
  opts: {
    maxChunks?: number;
    sectionPaths?: string[];
    restrictChunkIds?: string[];
  } = {}
): Promise<DocumentTree> {
  const maxChunks = Math.max(50, Math.min(opts.maxChunks ?? ORACLE_MAX_DOCUMENT_CHUNKS, 5000));
  const restrictChunkIds = (opts.restrictChunkIds || []).map(String).filter(Boolean);
  const sectionPaths = (opts.sectionPaths || []).map(safeSectionPath).filter(Boolean);

  const document = await prisma.document.findUnique({ where: { id: documentId } });
  if (!document) {
    throw new Error("Document not found");
  }

  const chunks = await prisma.chunk.findMany({
    where: {
      documentId,
      ...(restrictChunkIds.length > 0 ? { id: { in: restrictChunkIds } } : {}),
      ...(sectionPaths.length > 0 ? { sectionPath: { in: sectionPaths } } : {}),
    },
    orderBy: { chunkIndex: "asc" },
    take: maxChunks,
  });

  const root: TreeNode = {
    id: document.id,
    content: document.title,
    type: "document",
    children: [],
    metadata: (document.metadata || {}) as Record<string, any>,
  };

  const sectionMap = new Map<string, TreeNode>();

  for (const chunk of chunks) {
    const metadata = (chunk.metadata || {}) as Record<string, any>;
    if (metadata?.content_kind === "parent_context") continue;

    const sectionPath = safeSectionPath(metadata?.sectionPath || chunk.sectionPath || "Document");

    if (!sectionMap.has(sectionPath)) {
      const sectionNode: TreeNode = {
        id: `${documentId}::${sectionPath}`,
        content: sectionPath,
        type: "section",
        children: [],
        metadata: { path: sectionPath },
      };
      sectionMap.set(sectionPath, sectionNode);
      root.children.push(sectionNode);
    }

    const section = sectionMap.get(sectionPath)!;
    const embedding = isNumberArray((chunk as any).embedding) ? ((chunk as any).embedding as number[]) : undefined;
    section.children.push({
      id: chunk.id,
      content: chunk.content,
      type: "chunk",
      children: [],
      metadata,
      embedding,
    });
  }

  for (const sectionNode of root.children) {
    const samples: number[][] = [];
    for (const child of sectionNode.children) {
      if (!child.embedding) continue;
      samples.push(child.embedding);
      if (samples.length >= ORACLE_SECTION_EMBED_SAMPLE) break;
    }
    sectionNode.embedding = meanEmbedding(samples);
  }
  root.embedding = meanEmbedding(root.children.map((c) => c.embedding).filter(isNumberArray));

  return {
    root,
    depth: calculateDepth(root),
    nodeCount: countNodes(root),
  };
}

function calculateDepth(node: TreeNode): number {
  if (node.children.length === 0) return 1;
  return 1 + Math.max(...node.children.map(calculateDepth));
}

function countNodes(node: TreeNode): number {
  return 1 + node.children.reduce((sum, child) => sum + countNodes(child), 0);
}

/**
 * Oracle search - Tree-guided retrieval
 */
export async function oracleSearch(params: {
  query: string;
  projectId: string;
  topK?: number;
  maxDepth?: number;
  sourceIds?: string[];
  chunkTypes?: string[];
  metadataFilter?: Record<string, any>;
  oracleSelect?: {
    maxSeedHits?: number;
    maxDocuments?: number;
    maxSectionsPerDoc?: number;
    maxCandidateChunks?: number;
  };
  tree?: {
    maxDocumentChunks?: number;
  };
}): Promise<Array<{ content: string; path: string; relevance: number }>> {
  const {
    query,
    projectId,
    topK = 5,
    maxDepth = 3,
    sourceIds,
    chunkTypes,
    metadataFilter,
    oracleSelect,
    tree,
  } = params;

  console.log(`[Oracle] search: "${query}"`);

  // Step 1: Embed query
  const queryEmbedding = await embedSingle(query);

  // Step 2: Pick scope (docs + sections) using global vector seed hits.
  const scope = queryEmbedding.length > 0
    ? await selectOracleScope({
      projectId,
      queryEmbedding,
      sourceIds,
      chunkTypes,
      metadataFilter,
      maxSeedHits: oracleSelect?.maxSeedHits ?? ORACLE_SELECT_MAX_SEED_HITS,
      maxDocuments: oracleSelect?.maxDocuments ?? ORACLE_SELECT_MAX_DOCUMENTS,
      maxSectionsPerDoc: oracleSelect?.maxSectionsPerDoc ?? ORACLE_SELECT_MAX_SECTIONS_PER_DOC,
      maxCandidateChunks: oracleSelect?.maxCandidateChunks ?? ORACLE_SELECT_MAX_CANDIDATE_CHUNKS,
    })
    : { seed_hits: [], documents: [], candidate_chunk_ids: [] };

  const candidateDocIds = scope.documents.map((d) => d.document_id).filter(Boolean);
  const documents = candidateDocIds.length > 0
    ? await prisma.document.findMany({
      where: { projectId, id: { in: candidateDocIds } },
      include: { _count: { select: { chunks: true } } },
    })
    : await prisma.document.findMany({
      where: { projectId },
      take: 10,
      include: { _count: { select: { chunks: true } } },
    });

  console.log(`[Oracle] Found ${documents.length} documents in project`);

  const results: Array<{ content: string; path: string; relevance: number }> = [];

  // Step 3: For each document, traverse tree (only documents with chunks)
  const sectionPathsByDoc = new Map<string, string[]>();
  for (const entry of scope.documents) {
    sectionPathsByDoc.set(entry.document_id, (entry.section_paths || []).map(safeSectionPath));
  }
  for (const hit of scope.seed_hits) {
    const prev = sectionPathsByDoc.get(hit.document_id) || [];
    prev.push(safeSectionPath(hit.section_path));
    sectionPathsByDoc.set(hit.document_id, prev);
  }

  const documentById = new Map<string, (typeof documents)[number]>(documents.map((doc) => [doc.id, doc]));
  const orderedDocs = candidateDocIds.length > 0
    ? candidateDocIds.map((id) => documentById.get(id)).filter(Boolean) as (typeof documents)
    : documents;

  for (const doc of orderedDocs) {
    if (doc._count.chunks === 0) {
      console.log(`[Oracle] Skipping document ${doc.id} - no chunks`);
      continue;
    }

    console.log(`[Oracle] Building tree for document: ${doc.title} (${doc._count.chunks} chunks)`);

    try {
      const sectionPaths = [...new Set((sectionPathsByDoc.get(doc.id) || []).filter(Boolean))];
      const builtTree = await buildDocumentTree(doc.id, {
        maxChunks: tree?.maxDocumentChunks ?? ORACLE_MAX_DOCUMENT_CHUNKS,
        ...(sectionPaths.length > 0 ? { sectionPaths } : {}),
        ...(scope.candidate_chunk_ids.length > 0 ? { restrictChunkIds: scope.candidate_chunk_ids } : {}),
      });
      console.log(`[Oracle] Tree built: ${builtTree.nodeCount} nodes, depth: ${builtTree.depth}`);

      if (builtTree.nodeCount === 0) {
        continue;
      }

      const traversalResults = await guidedTraversal({
        tree: builtTree,
        query,
        queryEmbedding,
        maxDepth,
        topK,
      });

      console.log(`[Oracle] Traversal found ${traversalResults.length} results for ${doc.title}`);
      results.push(...traversalResults);
    } catch (err: any) {
      console.error(`[Oracle] Error building tree for document ${doc.id}:`, err.message);
    }
  }

  console.log(`[Oracle] Total results before filtering: ${results.length}`);

  // Step 4: Sort by relevance and return top K
  results.sort((a, b) => b.relevance - a.relevance);

  const seen = new Set<string>();
  const deduped = results.filter((r) => {
    const key = `${r.path}::${r.content.slice(0, 64)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  const finalResults = deduped.slice(0, topK);
  console.log(`[Oracle] Returning ${finalResults.length} results`);

  return finalResults;
}

/**
 * Guided tree traversal - intelligently navigate document structure
 */
async function guidedTraversal(params: {
  tree: DocumentTree;
  query: string;
  queryEmbedding: number[];
  maxDepth: number;
  topK: number;
  currentNode?: TreeNode;
  currentDepth?: number;
  path?: string;
}): Promise<Array<{ content: string; path: string; relevance: number }>> {
  const {
    tree,
    query,
    queryEmbedding,
    maxDepth,
    topK,
    currentNode = tree.root,
    currentDepth = 0,
    path = "",
  } = params;

  const results: Array<{ content: string; path: string; relevance: number }> = [];

  // Stop if max depth reached
  if (currentDepth >= maxDepth) {
    return results;
  }

  // Leaf node - calculate relevance
  if (currentNode.type === "chunk") {
    const relevance = currentNode.embedding
      ? cosineSimilarity(queryEmbedding, currentNode.embedding)
      : 0;

    if (relevance > ORACLE_LEAF_THRESHOLD) {
      results.push({
        content: currentNode.content,
        path,
        relevance,
      });
    }

    return results;
  }

  // Internal node - decide which children to explore
  const childScores = await Promise.all(
    currentNode.children.map(async (child) => {
      // Score child based on content relevance
      const score = await scoreNode(child, query, queryEmbedding);
      return { child, score };
    })
  );

  const labelForNode = (node: TreeNode): string => {
    if (node.type === "document") return node.content;
    if (node.type === "section") return String(node.metadata?.path || node.content || "Section");
    const heading =
      node.metadata?.heading_path ||
      node.metadata?.headingPath ||
      node.metadata?.heading ||
      node.metadata?.title ||
      null;
    return heading ? String(heading) : "Chunk";
  };

  // Sort by score and explore top children
  childScores.sort((a, b) => b.score - a.score);
  const baseExplore = currentDepth === 0 ? 5 : 7;
  const dynamicExplore = Math.ceil(Math.sqrt(childScores.length)) + 2;
  const exploreCount = Math.min(
    childScores.length,
    Math.max(2, Math.min(12, Math.max(baseExplore, dynamicExplore)))
  );
  const topChildren = childScores.slice(0, exploreCount);

  for (const { child, score } of topChildren) {
    if (score > ORACLE_EXPLORE_THRESHOLD) {
      const label = labelForNode(child);
      const childPath = path ? `${path} > ${label}` : label;

      const childResults = await guidedTraversal({
        tree,
        query,
        queryEmbedding,
        maxDepth,
        topK,
        currentNode: child,
        currentDepth: currentDepth + 1,
        path: childPath,
      });

      results.push(...childResults);
    }
  }

  return results;
}

/**
 * Score a tree node for relevance
 */
async function scoreNode(
  node: TreeNode,
  query: string,
  queryEmbedding: number[]
): Promise<number> {
  // If node has embedding, use it
  if (node.embedding) {
    return cosineSimilarity(queryEmbedding, node.embedding);
  }

  // Otherwise, use simple keyword matching
  const queryWords = query.toLowerCase().split(/\s+/);
  const nodeWords = node.content.toLowerCase().split(/\s+/);

  const overlap = queryWords.filter((w) => nodeWords.includes(w)).length;
  return overlap / queryWords.length;
}

/**
 * Cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0 || a.length !== b.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Oracle research - Multi-step reasoning
 */
export async function oracleResearch(params: {
  question: string;
  projectId: string;
  maxSteps?: number;
}): Promise<{
  answer: string;
  steps: Array<{
    step: number;
    query: string;
    results: any[];
    reasoning: string;
  }>;
}> {
  const { question, projectId, maxSteps = 5 } = params;

  const steps: any[] = [];
  let currentQuery = question;

  for (let step = 1; step <= maxSteps; step++) {
    console.log(`🔮 Oracle step ${step}: ${currentQuery}`);

    // Search
    const results = await oracleSearch({
      query: currentQuery,
      projectId,
      topK: 5,
    });

    // Reason about results
    const reasoning = await reasonAboutResults(currentQuery, results, question);

    steps.push({
      step,
      query: currentQuery,
      results,
      reasoning: reasoning.thought,
    });

    // If we have enough information, stop
    if (reasoning.hasAnswer) {
      return {
        answer: reasoning.answer ?? "",
        steps,
      };
    }

    // Generate next query
    currentQuery = reasoning.nextQuery || question;
  }

  // Final synthesis
  const finalAnswer = await synthesizeAnswer(question, steps);

  return {
    answer: finalAnswer,
    steps,
  };
}

/**
 * Reason about search results using LLM
 */
async function reasonAboutResults(
  query: string,
  results: any[],
  originalQuestion: string
): Promise<{
  thought: string;
  hasAnswer: boolean;
  answer?: string;
  nextQuery?: string;
}> {
  const prompt = `You are analyzing search results to answer a question.

**Original question:** ${originalQuestion}
**Current query:** ${query}

**Search results:**
${results.map((r, i) => `${i + 1}. ${r.content} (relevance: ${r.relevance.toFixed(2)})`).join("\n")}

Analyze these results:
1. Do they answer the original question?
2. What information is still missing?
3. What should be the next search query?

Return JSON:
{
  "thought": "your analysis",
  "hasAnswer": true or false,
  "answer": "the answer if you have it" or null,
  "nextQuery": "next search query" or null
}`;

  const response = await getOpenAIClient().chat.completions.create({
    model: "gpt-4o",
    max_tokens: 1024,
    temperature: 0.0,
    messages: [{ role: "user", content: prompt }],
    response_format: { type: "json_object" },
  });

  const text = response.choices[0]?.message?.content?.trim();
  if (!text) {
    return {
      thought: "Analysis failed",
      hasAnswer: false,
    };
  }

  const jsonMatch = text.match(/```json\n?([\s\S]*?)\n?```/) || text.match(/\{[\s\S]*\}/);
  const jsonStr = jsonMatch ? (jsonMatch[1] || jsonMatch[0]) : text;

  return JSON.parse(jsonStr);
}

/**
 * Synthesize final answer from all steps
 */
async function synthesizeAnswer(question: string, steps: any[]): Promise<string> {
  const prompt = `Synthesize a final answer from multiple research steps.

**Question:** ${question}

**Research steps:**
${steps.map((s) => `Step ${s.step}: ${s.query}\n  ${s.reasoning}`).join("\n\n")}

Provide a comprehensive answer based on all the information gathered.`;

  const response = await getOpenAIClient().chat.completions.create({
    model: "gpt-4o",
    max_tokens: 2048,
    temperature: 0.0,
    messages: [{ role: "user", content: prompt }],
  });

  return response.choices[0]?.message?.content || "Unable to synthesize answer";
}
