import { randomUUID } from "crypto";
import {
  buildFreshnessMetadata,
  normalizeProfileConfig,
  resolveIngestionPlan,
  type IngestionProfile,
  type ProfileConfig,
  type ResolvedIngestionPlan,
  type StrategyOverride,
} from "./ingestion-profiles.js";

export interface Chunk {
  tempId: string;
  parentTempId?: string;
  content: string;
  metadata: Record<string, any>;
  chunkType: "code" | "documentation" | "api_spec" | "schema" | "config" | "dataset" | "text" | "comment";
  chunkIndex: number;
  role: "parent" | "child" | "standalone";
  searchable: boolean;
  sectionPath?: string;
  headingPath?: string;
}

export interface ChunkingResult {
  chunks: Chunk[];
  plan: ResolvedIngestionPlan;
  stats: {
    totalChunks: number;
    searchableChunks: number;
    parentChunks: number;
    duplicateRate: number;
    duplicateRateScope: "searchable_chunks";
  };
}

type ChunkType = Chunk["chunkType"];

type Section = {
  content: string;
  sectionPath: string;
  headingPath?: string;
  blockType?: string;
  metadata?: Record<string, any>;
};

const CODE_EXTENSIONS = new Set([
  ".ts", ".tsx", ".js", ".jsx", ".py", ".java", ".go", ".rb",
  ".php", ".cs", ".rs", ".swift", ".kt", ".scala", ".c", ".cpp",
  ".h", ".hpp", ".sol", ".vy",
]);

const CONFIG_EXTENSIONS = new Set([
  ".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".xml",
]);

const DATASET_EXTENSIONS = new Set([
  ".csv", ".tsv", ".jsonl", ".ndjson",
]);

export function detectChunkType(filePath?: string, content?: string, sourceType?: string): Chunk["chunkType"] {
  if (sourceType === "dataset") return "dataset";
  if (!filePath) {
    if (content?.includes("```")) return "documentation";
    return "text";
  }

  const ext = "." + filePath.split(".").pop()?.toLowerCase();

  if (CODE_EXTENSIONS.has(ext)) return "code";
  if (DATASET_EXTENSIONS.has(ext)) return "dataset";
  if (CONFIG_EXTENSIONS.has(ext)) return "config";
  if (filePath.includes("schema") || filePath.includes("migration")) return "schema";
  if (filePath.endsWith(".md") || filePath.endsWith(".mdx") || filePath.endsWith(".rst")) return "documentation";
  if (filePath.includes("openapi") || filePath.includes("swagger") || filePath.endsWith(".graphql")) return "api_spec";

  return "text";
}

export function chunkText(
  content: string,
  opts: {
    chunkSize?: number;
    chunkOverlap?: number;
    filePath?: string;
    metadata?: Record<string, any>;
    sourceType?: string;
    ingestionProfile?: IngestionProfile;
    strategyOverride?: StrategyOverride;
    profileConfig?: ProfileConfig;
  } = {}
): ChunkingResult {
  const {
    chunkSize = 1000,
    chunkOverlap = 200,
    filePath,
    metadata = {},
    sourceType,
    ingestionProfile = "auto",
    strategyOverride,
    profileConfig,
  } = opts;

  const chunkType = detectChunkType(filePath, content, sourceType);
  const normalizedConfig = normalizeProfileConfig(profileConfig);
  const plan = resolveIngestionPlan({
    content,
    filePath,
    sourceType,
    metadata,
    ingestionProfile,
    strategyOverride,
    profileConfig: normalizedConfig,
  });

  const baseMetadata = {
    ...metadata,
    ...(filePath ? { filePath } : {}),
    source_type: metadata.source_type || sourceType || metadata.source || metadata.source_kind || null,
    ingestion_profile: plan.profile,
    chunk_strategy: plan.strategy,
    parser: plan.parser,
    parser_confidence: plan.parser_confidence,
    adaptive_used: plan.adaptive_used,
    latency_budget_ms: plan.latency_budget_ms,
    classification_reason: plan.classification.reason,
    classification_signals: plan.classification.document_signals,
    freshness: buildFreshnessMetadata(plan.profile, metadata, normalizedConfig),
  };

  let chunks: Chunk[];
  if (chunkType === "code") {
    chunks = chunkCodeHierarchical(content, {
      chunkSize,
      chunkOverlap,
      chunkType,
      filePath,
      metadata: baseMetadata,
      profileConfig: normalizedConfig,
      plan,
    });
  } else {
    const sections = buildStructuredSections(content, plan.profile, baseMetadata);
    const leafChunks = createLeafChunks(sections, {
      chunkSize,
      chunkOverlap,
      chunkType,
      metadata: baseMetadata,
      plan,
      profileConfig: normalizedConfig,
    });
    chunks = attachParentChunks(leafChunks, {
      chunkType,
      metadata: baseMetadata,
      plan,
      profileConfig: normalizedConfig,
    });
  }

  const duplicateRate = computeDuplicateRate(chunks.filter((chunk) => chunk.searchable));
  return {
    chunks,
    plan,
    stats: {
      totalChunks: chunks.length,
      searchableChunks: chunks.filter((chunk) => chunk.searchable).length,
      parentChunks: chunks.filter((chunk) => chunk.role === "parent").length,
      duplicateRate,
      duplicateRateScope: "searchable_chunks",
    },
  };
}

function buildStructuredSections(
  content: string,
  profile: ResolvedIngestionPlan["profile"],
  metadata: Record<string, any>
): Section[] {
  const lines = normalizeContentForProfile(content, profile).split("\n");
  const sections: Section[] = [];
  const headingStack: string[] = [];
  let buffer: string[] = [];
  let blockType = "text";
  let pageCounter = 0;

  const flush = () => {
    const body = buffer.join("\n").trim();
    if (!body) {
      buffer = [];
      return;
    }
    const sectionPath =
      headingStack.length > 0
        ? headingStack.join(" > ")
        : pageCounter > 0
          ? `Page ${pageCounter}`
          : "Document";
    sections.push({
      content: body,
      sectionPath,
      headingPath: headingStack.length > 0 ? headingStack.join(" > ") : undefined,
      blockType,
      metadata,
    });
    buffer = [];
    blockType = "text";
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/, "");
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    const pageMatch = line.match(/^--- Page (\d+) ---$/i);

    if (pageMatch) {
      flush();
      pageCounter = Number(pageMatch[1]);
      headingStack.length = 0;
      headingStack.push(`Page ${pageCounter}`);
      continue;
    }

    if (headingMatch) {
      flush();
      const level = headingMatch[1].length;
      const heading = headingMatch[2].trim();
      headingStack.splice(level - 1);
      headingStack[level - 1] = heading;
      blockType = "section";
      continue;
    }

    if (/^\|.+\|$/.test(line)) blockType = "table";
    if (/^```/.test(line)) blockType = "code_block";
    buffer.push(line);
  }

  flush();

  if (sections.length === 0) {
    sections.push({
      content: content.trim(),
      sectionPath: "Document",
      metadata,
    });
  }

  return sections;
}

function normalizeContentForProfile(content: string, profile: ResolvedIngestionPlan["profile"]): string {
  if (profile === "pdf_layout") {
    if (/^--- Page \d+ ---$/im.test(content)) {
      return content.replace(/\r\n/g, "\n").replace(/\n{4,}/g, "\n\n");
    }
    const pages = content.split("\f");
    return (pages.length > 1 ? pages : [content])
      .map((page, index) => `--- Page ${index + 1} ---\n${page}`)
      .join("\n\n")
      .replace(/\n{4,}/g, "\n\n");
  }
  return content.replace(/\r\n/g, "\n");
}

function chunkCodeHierarchical(
  content: string,
  opts: {
    chunkSize: number;
    chunkOverlap: number;
    chunkType: ChunkType;
    filePath?: string;
    metadata: Record<string, any>;
    profileConfig: ProfileConfig;
    plan: ResolvedIngestionPlan;
  }
): Chunk[] {
  const { chunkSize, filePath, metadata, profileConfig, plan } = opts;
  const lines = content.split("\n");
  const boundaries = [
    /^(export\s+)?(async\s+)?function\s+/,
    /^(export\s+)?(default\s+)?class\s+/,
    /^(export\s+)?const\s+\w+\s*=\s*(async\s+)?\(/,
    /^(export\s+)?const\s+\w+\s*=\s*\{/,
    /^(export\s+)?interface\s+/,
    /^(export\s+)?type\s+/,
    /^(export\s+)?enum\s+/,
    /^def\s+/,
    /^class\s+/,
    /^func\s+/,
    /^pub\s+(fn|struct|enum|impl)/,
  ];

  const leafChunks: Chunk[] = [];
  let start = 0;
  let current: string[] = [];
  let currentHeading = "File";

  const pushChunk = (endLineExclusive: number) => {
    const chunkContent = current.join("\n").trim();
    if (!chunkContent) return;
    const matchName = current.find((line) => boundaries.some((boundary) => boundary.test(line.trimStart())));
    if (matchName) currentHeading = matchName.trim();
    leafChunks.push({
      tempId: randomUUID(),
      content: chunkContent,
      chunkType: "code",
      chunkIndex: leafChunks.length,
      role: "child",
      searchable: true,
      sectionPath: filePath || "File",
      headingPath: currentHeading,
      metadata: {
        ...metadata,
        filePath,
        startLine: start + 1,
        endLine: endLineExclusive,
        section_path: filePath || "File",
        heading_path: currentHeading,
        block_type: "code_symbol",
        content_kind: "child",
      },
    });
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trimStart();
    const isBoundary = boundaries.some((boundary) => boundary.test(trimmed));

    if (isBoundary && current.length > 0) {
      pushChunk(i);
      current = [line];
      start = i;
      continue;
    }

    current.push(line);

    if (current.join("\n").length > chunkSize * 1.5) {
      pushChunk(i + 1);
      current = [];
      start = i + 1;
    }
  }

  if (current.length > 0) {
    pushChunk(lines.length);
  }

  return attachParentChunks(leafChunks, {
    chunkType: "code",
    metadata,
    plan,
    profileConfig,
  });
}

function createLeafChunks(
  sections: Section[],
  opts: {
    chunkSize: number;
    chunkOverlap: number;
    chunkType: ChunkType;
    metadata: Record<string, any>;
    plan: ResolvedIngestionPlan;
    profileConfig: ProfileConfig;
  }
): Chunk[] {
  const { chunkSize, chunkOverlap, chunkType, metadata, plan, profileConfig } = opts;
  const target = profileConfig.hierarchy?.child_chunk_target || 700;
  const effectiveChunkSize = Math.min(chunkSize, target);
  const chunks: Chunk[] = [];

  for (const section of sections) {
    const fragmentBlocks =
      plan.strategy === "fixed"
        ? splitFixed(section.content, effectiveChunkSize, chunkOverlap)
        : plan.strategy === "semantic"
          ? splitSemanticish(section.content, effectiveChunkSize)
          : splitRecursive(section.content, effectiveChunkSize, chunkOverlap);

    for (const fragment of fragmentBlocks) {
      const trimmed = fragment.trim();
      if (!trimmed) continue;
      chunks.push({
        tempId: randomUUID(),
        content: trimmed,
        chunkType,
        chunkIndex: chunks.length,
        role: "child",
        searchable: true,
        sectionPath: section.sectionPath,
        headingPath: section.headingPath,
        metadata: {
          ...metadata,
          ...section.metadata,
          section_path: section.sectionPath,
          heading_path: section.headingPath,
          block_type: section.blockType || "text",
          content_kind: "child",
        },
      });
    }
  }

  if (chunks.length === 0 && sections.length > 0) {
    chunks.push({
      tempId: randomUUID(),
      content: sections[0].content.trim(),
      chunkType,
      chunkIndex: 0,
      role: "standalone",
      searchable: true,
      sectionPath: sections[0].sectionPath,
      headingPath: sections[0].headingPath,
      metadata: {
        ...metadata,
        section_path: sections[0].sectionPath,
        heading_path: sections[0].headingPath,
        content_kind: "standalone",
      },
    });
  }

  return chunks;
}

function attachParentChunks(
  leafChunks: Chunk[],
  opts: {
    chunkType: ChunkType;
    metadata: Record<string, any>;
    plan: ResolvedIngestionPlan;
    profileConfig: ProfileConfig;
  }
): Chunk[] {
  const { chunkType, metadata, plan, profileConfig } = opts;
  if (leafChunks.length === 0) return [];
  if (plan.strategy !== "hierarchical" && plan.strategy !== "layout_aware") {
    return leafChunks.map((chunk, index) => ({
      ...chunk,
      chunkIndex: index,
      role: "standalone",
      metadata: {
        ...chunk.metadata,
        content_kind: "standalone",
      },
    }));
  }

  const parentTarget = profileConfig.hierarchy?.parent_chunk_target ?? 1800;
  const maxChildren = profileConfig.hierarchy?.max_children_per_parent ?? 4;

  const parents: Chunk[] = [];
  const children: Chunk[] = [];
  let group: Chunk[] = [];
  let groupLength = 0;
  let parentSequence = 0;

  const flushGroup = () => {
    if (group.length === 0) return;
    if (group.length === 1) {
      children.push({
        ...group[0],
        role: "standalone",
        metadata: {
          ...group[0].metadata,
          content_kind: "standalone",
        },
      });
      group = [];
      groupLength = 0;
      return;
    }

    const parentTempId = randomUUID();
    const parentSection = group[0].sectionPath || "Document";
    const parentHeading = group[0].headingPath;
    const parentContent = group
      .map((chunk) => chunk.content)
      .join("\n\n")
      .trim();

    parents.push({
      tempId: parentTempId,
      content: parentContent,
      chunkType,
      chunkIndex: parentSequence++,
      role: "parent",
      searchable: false,
      sectionPath: parentSection,
      headingPath: parentHeading,
      metadata: {
        ...metadata,
        section_path: parentSection,
        heading_path: parentHeading,
        block_type: "parent_context",
        content_kind: "parent_context",
        child_count: group.length,
      },
    });

    for (const chunk of group) {
      children.push({
        ...chunk,
        parentTempId,
        role: "child",
        metadata: {
          ...chunk.metadata,
          content_kind: "child",
        },
      });
    }

    group = [];
    groupLength = 0;
  };

  for (const chunk of leafChunks) {
    const wouldExceed =
      group.length >= maxChildren ||
      groupLength + chunk.content.length > parentTarget ||
      (group.length > 0 && group[0].sectionPath !== chunk.sectionPath);
    if (wouldExceed) flushGroup();
    group.push(chunk);
    groupLength += chunk.content.length;
  }
  flushGroup();

  // Parent chunks must be returned before children so ingest can resolve
  // temp parent ids to persisted DB ids in a single forward pass.
  const all = [...parents, ...children];
  return all.map((chunk, index) => ({ ...chunk, chunkIndex: index }));
}

function codePointBoundaryAtOrAfter(content: string, offset: number): number {
  return (content.codePointAt(offset - 1) ?? 0) > 0xffff ? offset + 1 : offset;
}

function splitFixed(content: string, chunkSize: number, chunkOverlap: number): string[] {
  const chunks: string[] = [];
  let start = 0;
  const overlap = Math.min(chunkOverlap, Math.floor(chunkSize / 3));
  while (start < content.length) {
    const end = codePointBoundaryAtOrAfter(content, Math.min(content.length, start + chunkSize));
    chunks.push(content.slice(start, end));
    if (end >= content.length) break;
    start = codePointBoundaryAtOrAfter(content, Math.max(0, end - overlap));
  }
  return chunks;
}

function splitRecursive(content: string, chunkSize: number, chunkOverlap: number): string[] {
  const blocks = content.split(/\n\n+/).map((block) => block.trim()).filter(Boolean);
  const chunks: string[] = [];
  let current = "";

  const flush = () => {
    if (!current.trim()) return;
    chunks.push(current.trim());
    const overlapWords = current.trim().split(/\s+/).slice(-Math.floor(chunkOverlap / 6));
    current = overlapWords.join(" ");
  };

  for (const block of blocks) {
    if (block.length > chunkSize * 1.35) {
      const sentenceChunks = splitSentences(block, chunkSize);
      for (const sentenceChunk of sentenceChunks) {
        if ((current + "\n\n" + sentenceChunk).trim().length > chunkSize && current.trim()) {
          flush();
        }
        current = current ? `${current}\n\n${sentenceChunk}` : sentenceChunk;
      }
      continue;
    }

    if ((current + "\n\n" + block).trim().length > chunkSize && current.trim()) {
      flush();
    }
    current = current ? `${current}\n\n${block}` : block;
  }

  flush();
  return chunks.length > 0 ? chunks : splitSentences(content, chunkSize);
}

function splitSemanticish(content: string, chunkSize: number): string[] {
  const candidateBlocks = content
    .split(/\n{2,}|(?<=\.)\s+(?=[A-Z])|(?<=:)\n+/)
    .map((block) => block.trim())
    .filter(Boolean);

  const chunks: string[] = [];
  let current = "";
  let lastKeywords = new Set<string>();

  for (const block of candidateBlocks) {
    const keywords = extractKeywords(block);
    const overlap = intersectionSize(lastKeywords, keywords);
    const topicShift = lastKeywords.size > 0 && overlap <= Math.max(1, Math.floor(lastKeywords.size / 6));

    if (topicShift && current.trim()) {
      chunks.push(current.trim());
      current = "";
    }

    if ((current + "\n\n" + block).trim().length > chunkSize && current.trim()) {
      chunks.push(current.trim());
      current = block;
    } else {
      current = current ? `${current}\n\n${block}` : block;
    }
    lastKeywords = keywords;
  }

  if (current.trim()) chunks.push(current.trim());
  return chunks.length > 0 ? chunks : splitRecursive(content, chunkSize, Math.floor(chunkSize / 5));
}

function splitSentences(content: string, chunkSize: number): string[] {
  const sentences = content
    .split(/(?<=[.!?])\s+(?=[A-Z0-9])/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  const chunks: string[] = [];
  let current = "";

  for (const sentence of sentences) {
    if (sentence.length > chunkSize) {
      const wordChunks = splitWordFallback(sentence, chunkSize);
      for (const wordChunk of wordChunks) {
        if ((current + " " + wordChunk).trim().length > chunkSize && current.trim()) {
          chunks.push(current.trim());
          current = wordChunk;
        } else {
          current = current ? `${current} ${wordChunk}` : wordChunk;
        }
      }
      continue;
    }

    if ((current + " " + sentence).trim().length > chunkSize && current.trim()) {
      chunks.push(current.trim());
      current = sentence;
    } else {
      current = current ? `${current} ${sentence}` : sentence;
    }
  }

  if (current.trim()) chunks.push(current.trim());
  return chunks.length > 0 ? chunks : splitWordFallback(content, chunkSize);
}

function splitWordFallback(content: string, chunkSize: number): string[] {
  const words = content.split(/\s+/).filter(Boolean);
  const chunks: string[] = [];
  let current = "";

  for (const word of words) {
    if ((current + " " + word).trim().length > chunkSize && current.trim()) {
      chunks.push(current.trim());
      current = word;
    } else {
      current = current ? `${current} ${word}` : word;
    }
  }

  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

function extractKeywords(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .match(/\b[a-z][a-z0-9_/-]{3,}\b/g)
      ?.filter((token) => ![
        "this", "that", "with", "from", "into", "about", "your", "have",
        "will", "were", "which", "when", "what", "where", "there",
      ].includes(token))
      .slice(0, 18) || []
  );
}

function intersectionSize(left: Set<string>, right: Set<string>): number {
  let size = 0;
  for (const entry of left) {
    if (right.has(entry)) size += 1;
  }
  return size;
}

function computeDuplicateRate(chunks: Chunk[]): number {
  if (chunks.length === 0) return 0;
  const seen = new Set<string>();
  let duplicates = 0;
  for (const chunk of chunks) {
    const normalized = chunk.content.trim().replace(/\s+/g, " ");
    if (seen.has(normalized)) {
      duplicates += 1;
      continue;
    }
    seen.add(normalized);
  }
  return duplicates / chunks.length;
}
