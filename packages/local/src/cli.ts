#!/usr/bin/env node
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { randomUUID, createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { listBrainFileTree, readBrainFile, syncBrainFilesystem, writeAgentBrainFile } from "./filesystem/index.js";
import { SourceStore } from "./sources/store.js";
import { runSourceSync, describeConnectors, newSourceId } from "./sources/sync.js";
import { ensureConnectorsRegistered } from "./connectors/registry.js";
import { getConnector, listConnectorDescriptors, listConnectorTypes } from "./connectors/types.js";
import type { SourceType } from "./sources/types.js";
import { buildCompanyBrain, askBrain, feedAgent, memoryToCitation, isFromSource } from "./brain/company_brain.js";

type Message = { role: string; content: string; timestamp?: string };
type WorkEvent = {
  kind: string;
  summary: string;
  details?: string;
  salience?: "low" | "medium" | "high";
  timestamp?: string;
  filePaths?: string[];
  toolName?: string;
  success?: boolean;
};

type Memory = {
  id: string;
  project: string;
  content: string;
  memory_type: string;
  user_id?: string;
  session_id?: string;
  agent_id?: string;
  task_id?: string;
  importance: number;
  confidence: number;
  metadata: Record<string, unknown>;
  embedding?: number[];
  created_at: string;
  updated_at: string;
  active: boolean;
};

type Project = { id: string; name: string; slug: string; created_at: string };
type Share = {
  id: string;
  project: string;
  session_id: string;
  title: string;
  memory_ids: string[];
  created_at: string;
  expires_at?: string;
};
type ContextSnapshot = {
  hash: string;
  project: string;
  query: string;
  entries: Array<{ kind: string; id: string; hash: string; title: string; content: string; citations?: Array<Record<string, unknown>> }>;
  created_at: string;
};
type StoreData = { projects: Project[]; memories: Memory[]; shares: Share[]; contextSnapshots: ContextSnapshot[] };

const STOP_WORDS = new Set(["the", "and", "for", "that", "with", "this", "from", "into", "user", "agent", "session", "tool", "used", "uses", "using"]);
const LOCAL_WRITE_POLICY_VERSION = "local_memory_write_v2";
const USER_CROSS_SESSION_THRESHOLD = 0.68;
const PROJECT_SCOPE_THRESHOLD = 0.76;
const AGENT_SCOPE_THRESHOLD = 0.74;
const TASK_SCOPE_THRESHOLD = 0.72;
const SESSION_ONLY_THRESHOLD = 0.58;
const DURABLE_MEMORY_TYPES = new Set(["factual", "preference", "relationship", "opinion", "goal", "instruction", "decision", "constraint", "solution", "project_state", "correction", "workflow", "session_summary"]);
const MEMORY_TYPE_WEIGHT: Record<string, number> = {
  decision: 0.78,
  constraint: 0.75,
  preference: 0.72,
  instruction: 0.68,
  workflow: 0.7,
  solution: 0.72,
  correction: 0.66,
  relationship: 0.58,
  opinion: 0.48,
  goal: 0.62,
  project_state: 0.56,
  factual: 0.42,
  semantic: 0.42,
  session_summary: 0.45,
  event: 0.18,
};
const RRF_K = 60;
const LOW_SIGNAL_PATTERNS = [
  /^\s*(ok|okay|thanks|thank you|done|yes|no|hi|hello)\s*[.!?]*\s*$/i,
  /^(read|list|open|show)\s+file$/i,
  /^session_(start|end|idle) captured$/i,
];

const DEFAULT_PORT = Number(process.env.RETAINDB_PORT || process.env.PORT || 3111);
const DEFAULT_PROJECT = process.env.RETAINDB_PROJECT || "default";
const RETAINDB_HOME = resolve(process.env.RETAINDB_HOME || join(homedir(), ".retaindb"));
const STORE_PATH = resolve(process.env.RETAINDB_STORE || join(RETAINDB_HOME, "local-store.json"));
const JOURNAL_PATH = `${STORE_PATH}.journal.jsonl`;
const BENCHMARK_DIR = join(RETAINDB_HOME, "benchmarks");
const EMBEDDING_PROVIDER = process.env.RETAINDB_EMBEDDING_PROVIDER || "hash";
const SKIP_DIRS = new Set([".git", "node_modules", "dist", "build", ".next", "coverage", ".turbo", ".cache", ".vercel", ".wrangler"]);
const CODE_EXTENSIONS = new Set([".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".rs", ".go", ".java", ".cs", ".rb", ".php", ".md", ".mdx", ".json", ".toml", ".yaml", ".yml", ".css", ".scss", ".html", ".sql"]);
let transformerPipeline: Promise<any> | null = null;

function now() {
  return new Date().toISOString();
}

function slugify(value: string) {
  const slug = value.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return slug || "default";
}

function tokenize(text: string) {
  return text.toLowerCase().match(/[a-z0-9_./-]{2,}/g) || [];
}

function uniqueTokens(text: string) {
  return Array.from(new Set(tokenize(text)));
}

function hashString(text: string) {
  return createHash("sha256").update(text).digest("hex");
}

function estimateTokens(text: string) {
  return Math.ceil(text.length / 4);
}

function truncateTokens(text: string, tokenBudget: number) {
  const maxChars = Math.max(0, tokenBudget * 4);
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars).replace(/\s+\S*$/, "")}\n[trimmed to ${tokenBudget} token budget]`;
}

function concepts(text: string, limit = 10) {
  return uniqueTokens(text)
    .filter((token) => token.length > 2 && !STOP_WORDS.has(token))
    .slice(0, limit);
}

function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function signalQuality(text: string) {
  const trimmed = text.trim();
  const normalized = normalizeMemoryText(trimmed);
  if (!trimmed) return 0;
  if (LOW_SIGNAL_PATTERNS.some((pattern) => pattern.test(normalized || trimmed))) return 0.05;
  const tokens = uniqueTokens(normalized || trimmed);
  const hasCodePath = /\b[\w.-]+\/[\w./-]+\b|\b[\w.-]+\.(ts|tsx|js|jsx|py|rs|go|md|json|toml|yaml|yml)\b/i.test(normalized || trimmed);
  const hasDecision = /\b(decided|decision|prefer|preference|use|avoid|because|constraint|fixed|root cause|regression|todo|next|deploy|test|auth|rate limit|schema|migration|standardize|chosen|picked)\b/i.test(normalized || trimmed);
  const hasOutcome = /\b(pass|passed|fail|failed|error|resolved|implemented|created|updated|removed|blocked|deprecated|superseded)\b/i.test(normalized || trimmed);
  const hasStructure = /[:;]|\b(before|after|when|if|then|because|instead|so that)\b/i.test(normalized || trimmed);
  const noisy = /\b(console\.log|stack trace|node_modules|dist\/|build\/|coverage\/)\b/i.test(normalized || trimmed);
  return clamp(0.12 + Math.min(tokens.length, 80) / 120 + (hasCodePath ? 0.18 : 0) + (hasDecision ? 0.25 : 0) + (hasOutcome ? 0.18 : 0) + (hasStructure ? 0.08 : 0) - (noisy ? 0.12 : 0));
}

function normalizeWhitespace(value: string) {
  return String(value || "").trim().replace(/\s+/g, " ");
}

function normalizeMemoryText(text: string) {
  return normalizeWhitespace(text)
    .replace(/^(prompt|message|event|session_start|session_end|tool_result|post_tool_use|pre_tool_use|user asked|agent responded)\s*:\s*/i, "")
    .trim();
}

function normalizeMemoryType(memoryType?: string) {
  const normalized = String(memoryType || "factual").toLowerCase().trim();
  const map: Record<string, string> = {
    factual: "factual",
    semantic: "factual",
    fact: "factual",
    preference: "preference",
    event: "event",
    episodic: "event",
    relationship: "relationship",
    opinion: "opinion",
    goal: "goal",
    instruction: "instruction",
    procedural: "instruction",
    decision: "decision",
    constraint: "constraint",
    solution: "solution",
    outcome: "solution",
    project_state: "project_state",
    state: "project_state",
    correction: "correction",
    failure: "correction",
    fix: "correction",
    workflow: "workflow",
    session_summary: "session_summary",
    summary: "session_summary",
  };
  return map[normalized] || "factual";
}

function buildMemoryNormalizationFields(content: string) {
  const normalized = normalizeWhitespace(content);
  const expanded = normalized.replace(/\bgf\b/gi, "girlfriend").replace(/\bbf\b/gi, "boyfriend");
  const canonical = expanded
    .replace(/\bi'm\b/gi, "the user is")
    .replace(/\bi am\b/gi, "the user is")
    .replace(/\bi\b/gi, "the user")
    .replace(/\bme\b/gi, "the user")
    .replace(/\bmy\b/gi, "the user's")
    .replace(/\bmine\b/gi, "the user's");
  const thirdPersonCanonical = expanded
    .replace(/\bhis\b/gi, "the user's")
    .replace(/\bher\b/gi, "the user's")
    .replace(/\btheir\b/gi, "the user's")
    .replace(/\bhe\b/gi, "the user")
    .replace(/\bshe\b/gi, "the user")
    .replace(/\bthey\b/gi, "the user");
  const firstPersonVariant = canonical
    .replace(/\bthe user's\b/gi, "my")
    .replace(/\bthe user is\b/gi, "i am")
    .replace(/\bthe user\b/gi, "i");
  const variants = Array.from(new Set([normalized, expanded, canonical, thirdPersonCanonical, firstPersonVariant].map((item) => item.toLowerCase()).filter(Boolean)));
  return {
    normalized_content: normalized.toLowerCase(),
    canonical_content: canonical.toLowerCase(),
    search_text: variants.join(" | "),
    search_variants: variants,
    semantic_status: EMBEDDING_PROVIDER === "hash" ? "ready" : "pending",
  };
}

function extractEntityMentions(text: string) {
  const matches = text.match(/\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3}\b/g) || [];
  const tech = text.match(/\b(?:React|Next\.js|TypeScript|JavaScript|Python|Docker|Postgres|pgvector|Redis|Hono|JWT|MCP|Codex|Claude|OpenAI|LangChain|LangGraph|Vercel)\b/gi) || [];
  return Array.from(new Set([...matches, ...tech].map((item) => normalizeWhitespace(item)).filter((item) => item.length > 1))).slice(0, 24);
}

function buildValidatorIssues(input: { content: string; memoryType: string; eventDate?: string | null; entityMentions?: string[] }) {
  const issues: string[] = [];
  const content = normalizeWhitespace(input.content);
  if (content.length < 10) issues.push("too_short");
  if (/^(hi|hello|hey|thanks|thank you|ok|okay|bye)[!.]*$/i.test(content)) issues.push("chatter");
  if (/^(he|she|they|it)\b/i.test(content)) issues.push("unresolved_pronouns");
  if (/\b(the company|that project|this thing|the system|something|stuff|things)\b/i.test(content)) issues.push("vague_reference");
  if (/[.;]\s+\S+/.test(content) || /\b(and|also|plus)\b.+\b(and|also|plus)\b/i.test(content)) issues.push("multi_fact");
  if (content.split(/\s+/).length < 4) issues.push("low_specificity");
  if ((input.entityMentions || []).length === 0 && /\b(react|python|typescript|docker|postgres|redis|auth|jwt)\b/i.test(content)) issues.push("weak_grounding");
  if (normalizeMemoryType(input.memoryType) === "event" && /\b(yesterday|today|last week|last month|next week|soon|recently)\b/i.test(content) && !input.eventDate) issues.push("underspecified_temporal");
  return Array.from(new Set(issues));
}

function calibrateWriteConfidence(input: { confidenceRaw: number; memoryType: string; extractionMethod: string; validatorIssues: string[] }) {
  let calibrated = clamp(input.confidenceRaw, 0, 1);
  const type = normalizeMemoryType(input.memoryType);
  if (["preference", "goal", "instruction", "decision", "constraint", "solution", "correction", "workflow"].includes(type)) calibrated += 0.04;
  if (type === "event") calibrated -= 0.03;
  const method = input.extractionMethod.toLowerCase();
  if (method === "manual") calibrated += 0.08;
  else if (method === "pattern" || method === "bullet") calibrated += 0.06;
  else if (method === "fallback") calibrated -= 0.04;
  for (const issue of input.validatorIssues) {
    if (issue === "too_short") calibrated -= 0.08;
    if (issue === "chatter") calibrated -= 0.2;
    if (issue === "unresolved_pronouns") calibrated -= 0.12;
    if (issue === "vague_reference") calibrated -= 0.1;
    if (issue === "multi_fact") calibrated -= 0.1;
    if (issue === "low_specificity") calibrated -= 0.06;
    if (issue === "weak_grounding") calibrated -= 0.04;
    if (issue === "underspecified_temporal") calibrated -= 0.08;
  }
  return clamp(calibrated, 0, 1);
}

function inferScopeTarget(confidence: number, input: { memoryType: string; userId?: string; sessionId?: string; agentId?: string; taskId?: string; sourceRole?: string; userConfirmed?: boolean; success?: boolean }) {
  const type = normalizeMemoryType(input.memoryType);
  const assistantOnly = input.sourceRole === "assistant" && !input.userConfirmed && !input.success;
  if (assistantOnly && ["decision", "constraint", "solution", "correction"].includes(type)) {
    return confidence >= SESSION_ONLY_THRESHOLD && input.sessionId ? "SESSION" : "DROPPED";
  }
  if (["preference", "goal", "opinion", "factual"].includes(type) && input.userId && confidence >= USER_CROSS_SESSION_THRESHOLD) return "USER";
  if (type === "instruction") {
    if (input.userId && confidence >= USER_CROSS_SESSION_THRESHOLD) return "USER";
    if (input.agentId && confidence >= AGENT_SCOPE_THRESHOLD) return "AGENT";
    if (input.taskId && confidence >= TASK_SCOPE_THRESHOLD) return "TASK";
    if (confidence >= PROJECT_SCOPE_THRESHOLD) return "PROJECT";
  }
  if (type === "workflow") {
    if (input.agentId && confidence >= AGENT_SCOPE_THRESHOLD) return "AGENT";
    if (input.taskId && confidence >= TASK_SCOPE_THRESHOLD) return "TASK";
    if (confidence >= PROJECT_SCOPE_THRESHOLD) return "PROJECT";
  }
  if (["decision", "constraint", "solution", "project_state", "correction", "relationship"].includes(type)) {
    if (input.taskId && confidence >= TASK_SCOPE_THRESHOLD) return "TASK";
    if (confidence >= PROJECT_SCOPE_THRESHOLD) return "PROJECT";
  }
  if (confidence >= SESSION_ONLY_THRESHOLD && input.sessionId) return "SESSION";
  return confidence >= SESSION_ONLY_THRESHOLD ? "PROJECT" : "DROPPED";
}

function inferMemoryType(content: string, fallback = "factual") {
  const text = normalizeMemoryText(content);
  const normalizedFallback = normalizeMemoryType(fallback);
  if (/\b(error|failed|failing|root cause|regression|bug|deprecated|correct(ed|ion)|supersedes?|wrong|broken)\b/i.test(text)) return "correction";
  if (/\b(fixed|resolved|solution|solved|accepted fix)\b/i.test(text)) return "solution";
  if (/\b(prefer|prefers|preference|likes|wants|style|tone)\b/i.test(text)) return "preference";
  if (/\b(must|must not|should|should not|cannot|can't|avoid|required|requirement|constraint|never|always)\b/i.test(text)) return "constraint";
  if (/\b(decided|decision|chose|chosen|picked|standardize|standardized|settled on|use\b|uses\b|switch(ed)? to)\b/i.test(text)) return "decision";
  if (/\b(goal|todo|next step|plan|roadmap|milestone|target|need to|should ship)\b/i.test(text)) return "goal";
  if (/\b(command|workflow|steps?|run|deploy|release|test|build|install|before|after|procedure|playbook)\b/i.test(text)) return "workflow";
  if (/\b(always|when asked|format|respond|write|include|omit)\b/i.test(text)) return "instruction";
  if (/\b(reports to|works with|partner|manager|teammate|customer|vendor|depends on)\b/i.test(text)) return "relationship";
  if (/\bsummary|handoff|session ended\b/i.test(content)) return "session_summary";
  return normalizedFallback;
}

function durableContent(content: string, type: string) {
  const lines = content.split(/\r?\n|(?<=[.!?])\s+(?=[A-Z0-9`"'])/).map((line) => normalizeMemoryText(line)).filter(Boolean);
  if (DURABLE_MEMORY_TYPES.has(type)) {
    const useful = lines.filter((line) => signalQuality(line) >= 0.32).slice(0, 10);
    return useful.length ? useful.join("\n") : lines.slice(0, 6).join("\n");
  }
  return lines.slice(0, 12).join("\n");
}

type ExtractedMemoryCandidate = {
  content: string;
  memory_type: string;
  importance: number;
  confidence: number;
  reason: string;
};

function intentTypes(query: string) {
  const text = query.toLowerCase();
  const types = new Set<string>();
  if (/\b(decid|choice|chosen|why|rationale|standard|use|using)\b/.test(text)) types.add("decision");
  if (/\b(must|should|constraint|requirement|avoid|rule|policy|never|always)\b/.test(text)) types.add("constraint");
  if (/\b(prefer|preference|style|likes|wants)\b/.test(text)) types.add("preference");
  if (/\b(how|workflow|steps?|run|command|process|deploy|release|test|build)\b/.test(text)) types.add("procedural");
  if (/\b(error|bug|fail|failed|fix|regression|root cause|broken)\b/.test(text)) types.add("correction");
  if (/\b(goal|plan|next|todo|roadmap|milestone|target)\b/.test(text)) types.add("goal");
  if (/\b(status|state|current|summary|handoff|where were we)\b/.test(text)) types.add("project_state");
  return types;
}

function extractionCandidates(content: string, fallbackType = "factual") {
  const normalized = redactSecrets(content).trim();
  const rawSegments = normalized
    .split(/\r?\n|(?<=[.!?])\s+(?=(?:[A-Z0-9`"']|We |The |Use |Avoid |Run |Before |After ))/)
    .map((part) => normalizeMemoryText(part))
    .filter(Boolean);
  const candidates: ExtractedMemoryCandidate[] = [];
  const seen = new Set<string>();
  const push = (text: string, reason: string, importanceBoost = 0) => {
    const clean = normalizeMemoryText(text).replace(/^[-*]\s*/, "");
    if (clean.length < 18 || clean.length > 900) return;
    const quality = signalQuality(clean);
    if (quality < 0.26) return;
    const memoryType = inferMemoryType(clean, fallbackType);
    const key = `${memoryType}:${normalizedMemoryKey(clean)}`;
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push({
      content: clean,
      memory_type: memoryType,
      importance: clamp(0.55 + quality * 0.32 + (MEMORY_TYPE_WEIGHT[memoryType] || 0.3) * 0.18 + importanceBoost, 0.25, 0.98),
      confidence: clamp(0.62 + quality * 0.24 + (DURABLE_MEMORY_TYPES.has(memoryType) ? 0.08 : 0), 0.25, 0.97),
      reason,
    });
  };

  for (const segment of rawSegments) {
    const type = inferMemoryType(segment, fallbackType);
    const isActionable = DURABLE_MEMORY_TYPES.has(type) || /\b(decided|must|prefer|failed|fixed|run|next|todo|because|instead)\b/i.test(segment);
    if (isActionable) push(segment, "pattern", type === "correction" || type === "decision" ? 0.06 : 0);
  }

  const bullets = normalized.match(/(?:^|\n)\s*(?:[-*]|\d+[.)])\s+(.+)/g) || [];
  for (const bullet of bullets) push(bullet.replace(/^\s*(?:[-*]|\d+[.)])\s+/, ""), "bullet", 0.03);

  if (candidates.length === 0) push(normalized, "fallback", -0.04);
  return candidates.slice(0, 8);
}

function jaccardSimilarity(a: string, b: string) {
  const left = new Set(uniqueTokens(a).filter((token) => !STOP_WORDS.has(token)));
  const right = new Set(uniqueTokens(b).filter((token) => !STOP_WORDS.has(token)));
  if (left.size === 0 || right.size === 0) return 0;
  let intersection = 0;
  for (const token of left) if (right.has(token)) intersection += 1;
  return intersection / (left.size + right.size - intersection);
}

function hashEmbedding(text: string, dims = 96) {
  const vector = Array.from({ length: dims }, () => 0);
  for (const token of tokenize(text)) {
    const hash = createHash("sha256").update(token).digest();
    const index = hash[0] % dims;
    const sign = hash[1] % 2 === 0 ? 1 : -1;
    vector[index] += sign * (1 + Math.min(token.length, 12) / 12);
  }
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => Number((value / norm).toFixed(6)));
}

async function transformerEmbedding(text: string) {
  if (!transformerPipeline) {
    transformerPipeline = (async () => {
      const dynamicImport = new Function("specifier", "return import(specifier)") as (specifier: string) => Promise<any>;
      const transformers = await dynamicImport("@xenova/transformers");
      return transformers.pipeline("feature-extraction", process.env.RETAINDB_EMBEDDING_MODEL || "Xenova/all-MiniLM-L6-v2");
    })();
  }
  const extractor = await transformerPipeline;
  const output = await extractor(text.slice(0, 8000), { pooling: "mean", normalize: true });
  const raw = Array.from(output.data || output.tolist?.()?.[0] || []) as number[];
  return raw.map((value) => Number(value.toFixed(6)));
}

async function embedText(text: string) {
  if (EMBEDDING_PROVIDER === "local-transformers" || EMBEDDING_PROVIDER === "transformers") {
    try {
      return await transformerEmbedding(text);
    } catch {
      return hashEmbedding(text);
    }
  }
  return hashEmbedding(text);
}

function cosine(a: number[], b: number[]) {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < len; i++) dot += a[i] * b[i];
  return dot;
}

function redactSecrets(text: string) {
  return text
    .replace(/\b(sk-[A-Za-z0-9_-]{20,})\b/g, "[REDACTED_OPENAI_KEY]")
    .replace(/\b(cf[a-z]_[A-Za-z0-9_-]{20,})\b/g, "[REDACTED_CLOUDFLARE_TOKEN]")
    .replace(/\b(xox[baprs]-[A-Za-z0-9-]{10,})\b/g, "[REDACTED_SLACK_TOKEN]")
    .replace(/\b(gh[pousr]_[A-Za-z0-9_]{20,})\b/g, "[REDACTED_GITHUB_TOKEN]")
    .replace(/(["']?(?:api[_-]?key|token|secret|password)["']?\s*[:=]\s*["']?)[^"',\s}]+/gi, "$1[REDACTED]");
}

function redactUnknown<T>(value: T): T {
  if (typeof value === "string") return redactSecrets(value) as T;
  if (Array.isArray(value)) return value.map((item) => redactUnknown(item)) as T;
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value)) {
      out[key] = /api[_-]?key|token|secret|password/i.test(key) ? "[REDACTED]" : redactUnknown(entry);
    }
    return out as T;
  }
  return value;
}

function loadStore(): StoreData {
  mkdirSync(RETAINDB_HOME, { recursive: true });
  if (!existsSync(STORE_PATH)) {
    const created = now();
    return {
      projects: [{ id: "proj_default", name: "default", slug: "default", created_at: created }],
      memories: [],
      shares: [],
      contextSnapshots: [],
    };
  }
  const parsed = JSON.parse(readFileSync(STORE_PATH, "utf8")) as Partial<StoreData>;
  return {
    projects: Array.isArray(parsed.projects) ? parsed.projects : [],
    memories: Array.isArray(parsed.memories) ? parsed.memories : [],
    shares: Array.isArray(parsed.shares) ? parsed.shares : [],
    contextSnapshots: Array.isArray(parsed.contextSnapshots) ? parsed.contextSnapshots : [],
  };
}

function saveStore(data: StoreData) {
  mkdirSync(RETAINDB_HOME, { recursive: true });
  const temp = `${STORE_PATH}.tmp`;
  writeFileSync(temp, `${JSON.stringify(data, null, 2)}\n`, "utf8");
  renameSync(temp, STORE_PATH);
}

function appendJournal(event: string, payload: Record<string, unknown>) {
  mkdirSync(RETAINDB_HOME, { recursive: true });
  appendFileSync(JOURNAL_PATH, `${JSON.stringify({ event, at: now(), ...payload })}\n`, "utf8");
}

function writeBenchmarkReport(report: Record<string, unknown>) {
  mkdirSync(BENCHMARK_DIR, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const path = join(BENCHMARK_DIR, `local-${stamp}.json`);
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  return path;
}

function normalizedMemoryKey(content: string) {
  return content.toLowerCase().replace(/\s+/g, " ").trim();
}

function ageDays(value: string) {
  return Math.max(0, (Date.now() - Date.parse(value)) / 864e5);
}

function topEntries(map: Map<string, number>, limit: number) {
  return [...map.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([name, count]) => ({ name, count }));
}

function isTextCodeFile(path: string) {
  return CODE_EXTENSIONS.has(path.slice(path.lastIndexOf(".")).toLowerCase());
}

function listCodeFiles(root: string, limit = 400): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    if (out.length >= limit) return;
    let entries: any[] = [];
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (out.length >= limit) break;
      if (entry.name.startsWith(".") && entry.name !== ".env.example") continue;
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!SKIP_DIRS.has(entry.name)) walk(full);
      } else if (entry.isFile() && isTextCodeFile(entry.name)) {
        try {
          if (statSync(full).size <= 240_000) out.push(full);
        } catch {}
      }
    }
  };
  walk(resolve(root));
  return out;
}

function extractSymbols(text: string) {
  const symbols: string[] = [];
  const patterns = [
    /^\s*export\s+(?:async\s+)?function\s+([A-Za-z0-9_$]+)/gm,
    /^\s*(?:async\s+)?function\s+([A-Za-z0-9_$]+)/gm,
    /^\s*(?:export\s+)?class\s+([A-Za-z0-9_$]+)/gm,
    /^\s*(?:export\s+)?const\s+([A-Za-z0-9_$]+)\s*=/gm,
    /^\s*(?:def|class)\s+([A-Za-z0-9_]+)/gm,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) symbols.push(match[1]);
  }
  return Array.from(new Set(symbols)).slice(0, 30);
}

function codeMap(root: string, query = "", limit = 80) {
  const q = new Set(uniqueTokens(query));
  return listCodeFiles(root)
    .map((file) => {
      const rel = file.replace(resolve(root), "").replace(/^[/\\]/, "");
      let text = "";
      try {
        text = readFileSync(file, "utf8");
      } catch {}
      const symbols = extractSymbols(text);
      const haystack = `${rel} ${symbols.join(" ")}`.toLowerCase();
      const score = [...q].reduce((sum, token) => sum + (haystack.includes(token) ? 1 : 0), 0);
      return { file: rel, symbols, score, lines: text.split(/\r?\n/).length, hash: hashString(text).slice(0, 16) };
    })
    .sort((a, b) => b.score - a.score || a.file.localeCompare(b.file))
    .slice(0, limit);
}

function packFileChunks(root: string, files: string[], query: string, tokenBudget: number) {
  const qTokens = uniqueTokens(query).filter((token) => !STOP_WORDS.has(token));
  const chunks: Array<{ file: string; content: string; hash: string; tokens: number }> = [];
  for (const input of files.slice(0, 24)) {
    const full = resolve(root, input);
    if (!full.startsWith(resolve(root)) || !existsSync(full)) continue;
    let text = "";
    try {
      if (statSync(full).size > 240_000) continue;
      text = readFileSync(full, "utf8");
    } catch {
      continue;
    }
    const lines = text.split(/\r?\n/);
    const selected = new Set<number>();
    lines.forEach((line, index) => {
      const lower = line.toLowerCase();
      if (/^\s*(export\s+)?(async\s+)?(function|class|const|interface|type)\s+|^\s*(def|class)\s+/i.test(line)) selected.add(index);
      if (qTokens.some((token) => lower.includes(token))) {
        for (let i = Math.max(0, index - 3); i <= Math.min(lines.length - 1, index + 5); i++) selected.add(i);
      }
    });
    const picked = [...selected].sort((a, b) => a - b).slice(0, 140);
    const body = picked.length
      ? picked.map((line) => `${line + 1}: ${lines[line]}`).join("\n")
      : lines.slice(0, 80).map((line, index) => `${index + 1}: ${line}`).join("\n");
    const rel = full.replace(resolve(root), "").replace(/^[/\\]/, "");
    const content = `File: ${rel}\n${truncateTokens(body, Math.max(80, Math.floor(tokenBudget / Math.max(files.length, 1))))}`;
    chunks.push({ file: rel, content, hash: hashString(text), tokens: estimateTokens(content) });
  }
  return chunks;
}

function compressToolOutput(output: string, tokenBudget = 500) {
  const lines = redactSecrets(String(output || "")).split(/\r?\n/);
  const keep = lines.filter((line) =>
    /\b(error|failed|failure|exception|traceback|expected|received|assert|timeout|cannot|denied|not found|stack|at\s+[\w.]+|tests? failed)\b/i.test(line) ||
    /^\s*(FAIL|ERROR|✘|×|Caused by:)/i.test(line)
  );
  const summary = [
    `Original output: ${lines.length} lines, approx ${estimateTokens(output)} tokens.`,
    keep.length ? "Important lines:" : "No obvious errors found. Keeping tail.",
    ...(keep.length ? keep.slice(0, 80) : lines.slice(-40)),
  ].join("\n");
  return truncateTokens(summary, tokenBudget);
}

class LocalMemoryRuntime {
  private data: StoreData;

  constructor() {
    this.data = loadStore();
    this.ensureProject(DEFAULT_PROJECT);
    this.persist();
  }

  persist() {
    saveStore(this.data);
  }

  ensureProject(nameOrSlug: string): Project {
    const slug = slugify(nameOrSlug);
    const existing = this.data.projects.find((project) => project.slug === slug || project.name === nameOrSlug);
    if (existing) return existing;
    const project = { id: `proj_${randomUUID()}`, name: nameOrSlug, slug, created_at: now() };
    this.data.projects.push(project);
    this.persist();
    return project;
  }

  listProjects() {
    return this.data.projects;
  }

  async addMemory(input: {
    project?: string;
    content: string;
    memory_type?: string;
    user_id?: string;
    session_id?: string;
    agent_id?: string;
    task_id?: string;
    importance?: number;
    confidence?: number;
    metadata?: Record<string, unknown>;
  }) {
    const project = this.ensureProject(input.project || DEFAULT_PROJECT).slug;
    const rawText = redactSecrets(input.content).trim();
    const inferredType = inferMemoryType(rawText, input.memory_type || "factual");
    const text = durableContent(rawText, inferredType);
    const quality = signalQuality(text);
    const durable = DURABLE_MEMORY_TYPES.has(inferredType);
    if (quality < (durable ? 0.1 : 0.16)) {
      const timestamp = now();
      return {
        id: `skip_${createHash("sha1").update(`${project}:${timestamp}:${rawText}`).digest("hex").slice(0, 12)}`,
        project,
        content: text || rawText,
        memory_type: "skipped",
        user_id: input.user_id,
        session_id: input.session_id,
        agent_id: input.agent_id,
        task_id: input.task_id,
        importance: 0,
        confidence: 0,
        metadata: { skipped: true, reason: "low_signal", quality },
        created_at: timestamp,
        updated_at: timestamp,
        active: false,
      } satisfies Memory;
    }
    const existingHash = createHash("sha256")
      .update(`${project}\n${input.session_id || ""}\n${input.agent_id || ""}\n${text.toLowerCase()}`)
      .digest("hex");
    const duplicate = this.data.memories.find((memory) => memory.metadata?.hash === existingHash && memory.active);
    if (duplicate) return duplicate;
    const nearDuplicate = this.data.memories.find((memory) =>
      memory.active &&
      memory.project === project &&
      memory.memory_type === inferredType &&
      (!input.session_id || !memory.session_id || memory.session_id === input.session_id) &&
      (!input.agent_id || !memory.agent_id || memory.agent_id === input.agent_id) &&
      jaccardSimilarity(memory.content, text) >= 0.92
    );
    if (nearDuplicate) {
      nearDuplicate.importance = Math.max(nearDuplicate.importance, clamp((input.importance ?? 0.62) + quality * 0.2, 0.1, 0.99));
      nearDuplicate.confidence = Math.max(nearDuplicate.confidence, clamp((input.confidence ?? 0.78) + quality * 0.1, 0.1, 0.98));
      nearDuplicate.metadata = {
        ...nearDuplicate.metadata,
        merged_count: Number(nearDuplicate.metadata?.merged_count || 1) + 1,
        last_merged_at: now(),
      };
      nearDuplicate.updated_at = now();
      this.persist();
      appendJournal("memory.merged", { id: nearDuplicate.id, project, reason: "near_duplicate", similarity: ">=0.92" });
      return nearDuplicate;
    }
    const timestamp = now();
    const memory: Memory = {
      id: `mem_${randomUUID()}`,
      project,
      content: text,
      memory_type: inferredType,
      user_id: input.user_id,
      session_id: input.session_id,
      agent_id: input.agent_id,
      task_id: input.task_id,
      importance: clamp((input.importance ?? 0.62) + quality * 0.28 + (MEMORY_TYPE_WEIGHT[inferredType] || 0.3) * 0.12, 0.1, 0.99),
      confidence: clamp((input.confidence ?? 0.78) + quality * 0.16 + (durable ? 0.04 : 0), 0.1, 0.98),
      metadata: {
        ...redactUnknown(input.metadata || {}),
        concepts: concepts(`${inferredType} ${text}`),
        hash: existingHash,
        quality,
        type_weight: MEMORY_TYPE_WEIGHT[inferredType] || 0.3,
        durable,
        strength: clamp(quality + (input.importance ?? 0.6) / 2 + (durable ? 0.08 : 0)),
        access_count: 0,
        last_accessed_at: undefined,
        original_type: input.memory_type,
      },
      embedding: await embedText(text),
      created_at: timestamp,
      updated_at: timestamp,
      active: true,
    };
    this.data.memories.push(memory);
    this.persist();
    appendJournal("memory.added", { id: memory.id, project, session_id: memory.session_id, memory_type: memory.memory_type });
    return memory;
  }

  async ingestSession(input: {
    project?: string;
    session_id: string;
    user_id?: string;
    agent_id?: string;
    task_id?: string;
    messages?: Message[];
    events?: WorkEvent[];
  }) {
    const memories: Memory[] = [];
    const project = input.project || DEFAULT_PROJECT;
    for (const event of input.events || []) {
      const fallbackType = event.kind === "failure" ? "correction" : event.kind;
      const content = [
        `${event.kind}: ${event.summary}`,
        event.details ? `Details: ${event.details}` : "",
        event.filePaths?.length ? `Files: ${event.filePaths.join(", ")}` : "",
        event.toolName ? `Tool: ${event.toolName}` : "",
      ].filter(Boolean).join("\n");
      const candidates = extractionCandidates(content, fallbackType);
      for (const candidate of candidates) {
        memories.push(await this.addMemory({
          project,
          content: candidate.content,
          memory_type: candidate.memory_type,
          user_id: input.user_id,
          session_id: input.session_id,
          agent_id: input.agent_id,
          task_id: input.task_id,
          importance: event.salience === "high" ? Math.max(0.86, candidate.importance) : event.salience === "low" ? Math.min(0.58, candidate.importance) : candidate.importance,
          confidence: candidate.confidence,
          metadata: {
            source: "agent_event",
            extraction_reason: candidate.reason,
            source_event_kind: event.kind,
            event,
          },
        }));
      }
    }
    for (const message of input.messages || []) {
      if (message.role === "system" || !message.content?.trim()) continue;
      const prefix = message.role === "user" ? "User asked" : message.role === "assistant" ? "Agent responded" : message.role;
      memories.push(await this.addMemory({
        project,
        content: `${prefix}: ${message.content.trim()}`,
        memory_type: message.role === "user" ? "event" : "project_state",
        user_id: input.user_id,
        session_id: input.session_id,
        agent_id: input.agent_id,
        task_id: input.task_id,
        importance: message.role === "user" ? 0.75 : 0.55,
        metadata: { source: "session_message", role: message.role, timestamp: message.timestamp },
      }));
    }
    const stored = memories.filter((memory) => memory.active);
    return { memories_created: stored.length, memories: stored, skipped: memories.length - stored.length };
  }

  async search(input: {
    project?: string;
    query: string;
    user_id?: string;
    session_id?: string;
    agent_id?: string;
    task_id?: string;
    top_k?: number;
    include_inactive?: boolean;
  }) {
    const project = slugify(input.project || DEFAULT_PROJECT);
    const qTokens = uniqueTokens(input.query);
    const qConcepts = concepts(input.query, 12);
    const qIntentTypes = intentTypes(input.query);
    const qVector = await embedText(input.query);
    const topK = Math.min(Math.max(input.top_k || 10, 1), 100);
    const candidates = this.data.memories
      .filter((memory) => memory.project === project)
      .filter((memory) => input.include_inactive || memory.active)
      .filter((memory) => !input.user_id || !memory.user_id || memory.user_id === input.user_id)
      .filter((memory) => !input.session_id || memory.session_id === input.session_id || !input.session_id)
      .filter((memory) => !input.agent_id || memory.agent_id === input.agent_id || !memory.agent_id)
      .filter((memory) => !input.task_id || memory.task_id === input.task_id || !memory.task_id);
    const docs = candidates.map((memory) => ({
      memory,
      tokens: tokenize(`${memory.content} ${memory.memory_type} ${memory.memory_type} ${(Array.isArray(memory.metadata?.concepts) ? memory.metadata.concepts.join(" ") : "")} ${JSON.stringify(memory.metadata)}`),
    }));
    const avgDocLength = docs.reduce((sum, doc) => sum + doc.tokens.length, 0) / Math.max(docs.length, 1);
    const docFreq = new Map<string, number>();
    for (const doc of docs) {
      for (const token of new Set(doc.tokens)) {
        docFreq.set(token, (docFreq.get(token) || 0) + 1);
      }
    }
    const branchScores = docs
      .map((memory) => {
        const termFreq = new Map<string, number>();
        for (const token of memory.tokens) termFreq.set(token, (termFreq.get(token) || 0) + 1);
        const k1 = 1.2;
        const b = 0.75;
        let bm25 = 0;
        for (const token of qTokens) {
          const tf = termFreq.get(token) || 0;
          if (tf === 0) continue;
          const df = docFreq.get(token) || 0;
          const idf = Math.log(1 + (docs.length - df + 0.5) / (df + 0.5));
          const denom = tf + k1 * (1 - b + b * (memory.tokens.length / Math.max(avgDocLength, 1)));
          bm25 += idf * ((tf * (k1 + 1)) / denom);
        }
        const phrase = memory.memory.content.toLowerCase().includes(input.query.toLowerCase()) ? 3 : 0;
        const vector = cosine(qVector, memory.memory.embedding || hashEmbedding(memory.memory.content));
        const memoryConcepts = Array.isArray(memory.memory.metadata.concepts)
          ? memory.memory.metadata.concepts.map(String)
          : concepts(memory.memory.content);
        const conceptOverlap = qConcepts.filter((concept) => memoryConcepts.includes(concept)).length;
        const graph = conceptOverlap + this.relatedConceptBoost(project, qConcepts, memoryConcepts);
        const memoryType = memory.memory.memory_type;
        const typeIntent =
          qIntentTypes.has(memoryType) ? 0.9 :
          memoryType === "semantic" && [...qIntentTypes].some((type) => ["decision", "constraint", "preference"].includes(type)) ? 0.28 :
          qIntentTypes.size === 0 && DURABLE_MEMORY_TYPES.has(memoryType) ? 0.12 :
          0;
        const ageHours = Math.max(1, (Date.now() - Date.parse(memory.memory.created_at)) / 36e5);
        const recency = 1 / Math.sqrt(ageHours);
        const lastAccess = typeof memory.memory.metadata.last_accessed_at === "string" ? Date.parse(memory.memory.metadata.last_accessed_at) : 0;
        const accessAgeHours = lastAccess ? Math.max(1, (Date.now() - lastAccess) / 36e5) : ageHours;
        const accessCount = Number(memory.memory.metadata.access_count || 0);
        const strength = Number(memory.memory.metadata.strength || memory.memory.importance || 0.5);
        const durability = MEMORY_TYPE_WEIGHT[memoryType] || 0;
        const quality = Number(memory.memory.metadata.quality || signalQuality(memory.memory.content));
        const sourceAuthority = memory.memory.metadata?.source_id || memory.memory.metadata?.citation ? 0.08 : 0;
        const decay = clamp(strength / Math.sqrt(ageHours / 24), 0, 1);
        const reinforcement = Math.log1p(accessCount) / 5 + (lastAccess ? 1 / Math.sqrt(accessAgeHours) / 5 : 0);
        return { memory: memory.memory, bm25: bm25 + phrase, vector, graph, recency, decay, durability, reinforcement, typeIntent, quality, sourceAuthority };
      })
      .filter((item) => item.bm25 > 0 || item.vector > 0.05 || item.graph > 0 || item.typeIntent > 0);
    const rank = (key: "bm25" | "vector" | "graph") => {
      const map = new Map<string, number>();
      [...branchScores].sort((a, b) => b[key] - a[key]).forEach((item, index) => {
        if (item[key] > 0) map.set(item.memory.id, index + 1);
      });
      return map;
    };
    const bm25Rank = rank("bm25");
    const vectorRank = rank("vector");
    const graphRank = rank("graph");
    const scored = this.rerank(input.query, branchScores
      .map((item) => {
        const rrf =
          (bm25Rank.has(item.memory.id) ? 1 / (RRF_K + bm25Rank.get(item.memory.id)!) : 0) +
          (vectorRank.has(item.memory.id) ? 1 / (RRF_K + vectorRank.get(item.memory.id)!) : 0) +
          (graphRank.has(item.memory.id) ? 1 / (RRF_K + graphRank.get(item.memory.id)!) : 0);
        const score = rrf * 100 + item.memory.importance + item.memory.confidence + item.recency + item.decay + item.durability + item.reinforcement + item.typeIntent + item.quality * 0.28 + item.sourceAuthority;
        return { ...item, score };
      })
      .sort((a, b) => b.score - a.score))
      .slice(0, topK);
    for (const item of scored) {
      item.memory.metadata.access_count = Number(item.memory.metadata.access_count || 0) + 1;
      item.memory.metadata.last_accessed_at = now();
      item.memory.metadata.strength = clamp(Number(item.memory.metadata.strength || item.memory.importance || 0.5) + 0.03);
      item.memory.updated_at = now();
    }
    if (scored.length) this.persist();
    return scored.map((item) => ({
      id: item.memory.id,
      score: Number(item.score.toFixed(4)),
      content: item.memory.content,
      memory: item.memory,
      metadata: item.memory.metadata,
      source: "local-memory",
      document: item.memory.session_id || item.memory.project,
      type: item.memory.memory_type,
      retrieval_source: "local_rrf",
      scores: {
        bm25: Number(item.bm25.toFixed(4)),
        vector: Number(item.vector.toFixed(4)),
        graph: Number(item.graph.toFixed(4)),
        type_intent: Number(item.typeIntent.toFixed(4)),
        quality: Number(item.quality.toFixed(4)),
      },
    }));
  }

  rerank<T extends { memory: Memory; score: number; bm25: number; vector: number; graph: number; typeIntent?: number; quality?: number }>(query: string, items: T[]) {
    const qTokens = uniqueTokens(query).filter((token) => !STOP_WORDS.has(token));
    const qIntentTypes = intentTypes(query);
    return items
      .map((item) => {
        const content = item.memory.content.toLowerCase();
        const firstHit = qTokens
          .map((token) => content.indexOf(token))
          .filter((index) => index >= 0)
          .sort((a, b) => a - b)[0];
        const proximity = firstHit === undefined ? 0 : 1 / (1 + firstHit / 200);
        const exact = content.includes(query.toLowerCase()) ? 2 : 0;
        const lifecycle = DURABLE_MEMORY_TYPES.has(item.memory.memory_type) ? 0.6 : 0;
        const intent = qIntentTypes.has(item.memory.memory_type) ? 0.7 : 0;
        const evidence = Math.min(1, String(item.memory.metadata?.source_memory_ids || "").split(",").filter(Boolean).length / 8);
        const rerank_score = item.score + exact + proximity + lifecycle + intent + evidence + Number(item.quality || 0) * 0.15;
        return { ...item, score: rerank_score };
      })
      .sort((a, b) => b.score - a.score);
  }

  relatedConceptBoost(project: string, queryConcepts: string[], memoryConcepts: string[]) {
    if (queryConcepts.length === 0 || memoryConcepts.length === 0) return 0;
    const edges = this.graph(project).edges;
    let boost = 0;
    for (const edge of edges) {
      const queryTouches = queryConcepts.includes(edge.from) || queryConcepts.includes(edge.to);
      const memoryTouches = memoryConcepts.includes(edge.from) || memoryConcepts.includes(edge.to);
      if (queryTouches && memoryTouches) boost += Math.min(edge.weight, 5) / 5;
    }
    return boost;
  }

  graph(project?: string) {
    const slug = project ? slugify(project) : undefined;
    const nodes = new Map<string, { id: string; count: number }>();
    const edges = new Map<string, { from: string; to: string; weight: number }>();
    const memories = this.data.memories.filter((memory) => memory.active && (!slug || memory.project === slug));
    for (const memory of memories) {
      const cs = (Array.isArray(memory.metadata.concepts) ? memory.metadata.concepts.map(String) : concepts(memory.content)).slice(0, 8);
      for (const concept of cs) nodes.set(concept, { id: concept, count: (nodes.get(concept)?.count || 0) + 1 });
      for (let i = 0; i < cs.length; i++) {
        for (let j = i + 1; j < cs.length; j++) {
          const [from, to] = [cs[i], cs[j]].sort();
          const key = `${from}\0${to}`;
          const existing = edges.get(key);
          edges.set(key, { from, to, weight: (existing?.weight || 0) + 1 });
        }
      }
    }
    return {
      nodes: [...nodes.values()].sort((a, b) => b.count - a.count).slice(0, 100),
      edges: [...edges.values()].sort((a, b) => b.weight - a.weight).slice(0, 300),
    };
  }

  session(sessionId: string, project?: string, limit = 30) {
    const slug = slugify(project || DEFAULT_PROJECT);
    return this.data.memories
      .filter((memory) => memory.project === slug && memory.session_id === sessionId && memory.active)
      .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
      .slice(0, Math.min(Math.max(limit, 1), 200));
  }

  profile(userId: string, project?: string, limit = 100) {
    const slug = slugify(project || DEFAULT_PROJECT);
    return this.data.memories
      .filter((memory) => memory.project === slug && (!memory.user_id || memory.user_id === userId) && memory.active)
      .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
      .slice(0, Math.min(Math.max(limit, 1), 200));
  }

  delete(memoryId: string) {
    const memory = this.data.memories.find((item) => item.id === memoryId);
    if (!memory) return false;
    memory.active = false;
    memory.updated_at = now();
    this.persist();
    appendJournal("memory.deleted", { id: memoryId, project: memory.project });
    return true;
  }

  createShare(input: { project?: string; session_id: string; title?: string; expiry_days?: number }) {
    const project = slugify(input.project || DEFAULT_PROJECT);
    const memories = this.session(input.session_id, project, 100);
    const created = now();
    const share: Share = {
      id: `share_${randomUUID()}`,
      project,
      session_id: input.session_id,
      title: input.title || `RetainDB handoff: ${input.session_id}`,
      memory_ids: memories.map((memory) => memory.id),
      created_at: created,
      expires_at: input.expiry_days ? new Date(Date.now() + input.expiry_days * 864e5).toISOString() : undefined,
    };
    this.data.shares.push(share);
    this.persist();
    appendJournal("share.created", { id: share.id, project, session_id: share.session_id, memory_count: share.memory_ids.length });
    return { ...share, memories, share_id: share.id };
  }

  resumeShare(shareId: string, newSessionId?: string) {
    const share = this.data.shares.find((item) => item.id === shareId);
    if (!share) return null;
    const memories = this.data.memories.filter((memory) => share.memory_ids.includes(memory.id));
    return {
      share,
      session_id: newSessionId || `resumed-${Date.now()}`,
      context: memories.map((memory) => `- ${memory.content}`).join("\n"),
      memories,
    };
  }

  stats() {
    return {
      projects: this.data.projects.length,
      memories: this.data.memories.filter((memory) => memory.active).length,
      sessions: new Set(this.data.memories.map((memory) => memory.session_id).filter(Boolean)).size,
      shares: this.data.shares.length,
      store_path: STORE_PATH,
      journal_path: JOURNAL_PATH,
      embedding_provider: EMBEDDING_PROVIDER,
    };
  }

  inspect(project?: string) {
    const slug = project ? slugify(project) : undefined;
    const memories = this.data.memories.filter((memory) => !slug || memory.project === slug);
    const active = memories.filter((memory) => memory.active);
    const inactive = memories.filter((memory) => !memory.active);
    const typeCounts = new Map<string, number>();
    const sourceCounts = new Map<string, number>();
    const conceptCounts = new Map<string, number>();
    const duplicateGroups = new Map<string, Memory[]>();

    for (const memory of active) {
      typeCounts.set(memory.memory_type, (typeCounts.get(memory.memory_type) || 0) + 1);
      const source = String(memory.metadata?.source || memory.metadata?.source_type || "manual");
      sourceCounts.set(source, (sourceCounts.get(source) || 0) + 1);
      const cs = Array.isArray(memory.metadata?.concepts) ? memory.metadata.concepts.map(String) : concepts(memory.content);
      for (const concept of cs.slice(0, 8)) conceptCounts.set(concept, (conceptCounts.get(concept) || 0) + 1);
      const key = normalizedMemoryKey(memory.content);
      duplicateGroups.set(key, [...(duplicateGroups.get(key) || []), memory]);
    }

    const weakMemories = active
      .filter((memory) => !DURABLE_MEMORY_TYPES.has(memory.memory_type))
      .filter((memory) => Number(memory.metadata?.quality || 0) < 0.35)
      .sort((a, b) => Number(a.metadata?.quality || 0) - Number(b.metadata?.quality || 0))
      .slice(0, 12);

    const staleMemories = active
      .filter((memory) => !DURABLE_MEMORY_TYPES.has(memory.memory_type))
      .filter((memory) => ageDays(memory.created_at) > 7)
      .filter((memory) => Number(memory.metadata?.access_count || 0) === 0)
      .filter((memory) => Number(memory.metadata?.strength || memory.importance || 0) < 0.55)
      .sort((a, b) => ageDays(b.created_at) - ageDays(a.created_at))
      .slice(0, 12);

    const duplicateCandidates = [...duplicateGroups.values()]
      .filter((group) => group.length > 1)
      .sort((a, b) => b.length - a.length)
      .slice(0, 8)
      .map((group) => ({
        count: group.length,
        ids: group.map((memory) => memory.id),
        sample: group[0]?.content.slice(0, 220) || "",
      }));

    const reusable = active
      .filter((memory) => DURABLE_MEMORY_TYPES.has(memory.memory_type))
      .sort((a, b) => b.importance + b.confidence - (a.importance + a.confidence))
      .slice(0, 10)
      .map((memory) => ({
        id: memory.id,
        type: memory.memory_type,
        content: memory.content,
        importance: memory.importance,
        confidence: memory.confidence,
        access_count: Number(memory.metadata?.access_count || 0),
      }));

    const topReused = active
      .filter((memory) => Number(memory.metadata?.access_count || 0) > 0)
      .sort((a, b) => Number(b.metadata?.access_count || 0) - Number(a.metadata?.access_count || 0))
      .slice(0, 10)
      .map((memory) => ({
        id: memory.id,
        type: memory.memory_type,
        content: memory.content,
        access_count: Number(memory.metadata?.access_count || 0),
        last_accessed_at: memory.metadata?.last_accessed_at || null,
      }));

    const sessions = new Set(active.map((memory) => memory.session_id).filter(Boolean));
    const projects = new Set(active.map((memory) => memory.project));
    const durableCount = active.filter((memory) => DURABLE_MEMORY_TYPES.has(memory.memory_type)).length;
    const recalledCount = active.filter((memory) => Number(memory.metadata?.access_count || 0) > 0).length;
    const avgQuality = active.length
      ? active.reduce((sum, memory) => sum + Number(memory.metadata?.quality || signalQuality(memory.content)), 0) / active.length
      : 0;
    const durableRatio = active.length ? durableCount / active.length : 0;
    const weakRatio = active.length ? weakMemories.length / active.length : 0;
    const staleRatio = active.length ? staleMemories.length / active.length : 0;
    const duplicateRatio = active.length ? duplicateCandidates.reduce((sum, group) => sum + group.count, 0) / active.length : 0;
    const recallRatio = active.length ? recalledCount / active.length : 0;

    const score = Math.round(clamp(
      45 +
      avgQuality * 18 +
      durableRatio * 18 +
      Math.min(recallRatio, 0.35) * 18 +
      Math.min(sessions.size / 8, 1) * 8 -
      weakRatio * 22 -
      staleRatio * 16 -
      duplicateRatio * 12,
      active.length ? 1 : 0,
      100,
    ));

    const recommendations: string[] = [];
    if (active.length === 0) recommendations.push("Run `retaindb connect all --install`, use your agent normally, then rerun `retaindb inspect`.");
    if (durableRatio < 0.25 && active.length >= 8) recommendations.push("Run `retaindb consolidate` to promote raw events into semantic, procedural, correction, and session-summary memory.");
    if (weakMemories.length > 0) recommendations.push("Low-signal memories are present; consolidation can decay old weak observations and keep recall clean.");
    if (staleMemories.length > 0) recommendations.push("Stale unrecalled memories are building up; run `retaindb consolidate` or delete irrelevant memories from the viewer.");
    if (duplicateCandidates.length > 0) recommendations.push("Duplicate memory groups found; run `retaindb consolidate` to deactivate repeated entries.");
    if (topReused.length === 0 && active.length >= 5) recommendations.push("No memories have been reused yet; ask your agent to call RetainDB context tools before coding tasks.");
    if (conceptCounts.size < 5 && active.length >= 10) recommendations.push("Memory concepts are narrow; ingest broader project docs or connect sources to improve context coverage.");
    if (recommendations.length === 0) recommendations.push("Memory hygiene looks healthy. Keep using context packs and run `retaindb inspect` after heavier sessions.");

    return {
      project: slug || "all",
      score,
      summary: {
        active_memories: active.length,
        inactive_memories: inactive.length,
        durable_memories: durableCount,
        sessions: sessions.size,
        projects: projects.size,
        recalled_memories: recalledCount,
        average_quality: Number(avgQuality.toFixed(3)),
        durable_ratio: Number(durableRatio.toFixed(3)),
        recall_ratio: Number(recallRatio.toFixed(3)),
      },
      counts: {
        by_type: Object.fromEntries([...typeCounts.entries()].sort((a, b) => b[1] - a[1])),
        by_source: Object.fromEntries([...sourceCounts.entries()].sort((a, b) => b[1] - a[1])),
      },
      top_concepts: topEntries(conceptCounts, 15),
      reusable,
      top_reused: topReused,
      risks: {
        weak_memories: weakMemories.map((memory) => ({
          id: memory.id,
          type: memory.memory_type,
          quality: Number(memory.metadata?.quality || 0),
          content: memory.content.slice(0, 220),
        })),
        stale_memories: staleMemories.map((memory) => ({
          id: memory.id,
          type: memory.memory_type,
          age_days: Number(ageDays(memory.created_at).toFixed(1)),
          strength: Number(memory.metadata?.strength || memory.importance || 0),
          content: memory.content.slice(0, 220),
        })),
        duplicate_candidates: duplicateCandidates,
      },
      recommendations,
    };
  }

  async reembed(project?: string) {
    const slug = project ? slugify(project) : undefined;
    let updated = 0;
    for (const memory of this.data.memories.filter((item) => item.active && (!slug || item.project === slug))) {
      memory.embedding = await embedText(memory.content);
      memory.metadata = { ...memory.metadata, embedding_provider: EMBEDDING_PROVIDER };
      memory.updated_at = now();
      updated += 1;
    }
    this.persist();
    appendJournal("embeddings.refreshed", { project: slug || "all", updated, provider: EMBEDDING_PROVIDER });
    return { updated, provider: EMBEDDING_PROVIDER };
  }

  async contextPack(input: {
    project?: string;
    query: string;
    cwd?: string;
    files?: string[];
    tool_output?: string;
    previous_context_hash?: string;
    token_budget?: number;
    include_memory?: boolean;
    include_company_brain?: boolean;
  }) {
    const project = slugify(input.project || DEFAULT_PROJECT);
    const budget = Math.min(Math.max(Number(input.token_budget || 1600), 300), 8000);
    const cwd = resolve(input.cwd || process.cwd());
    const entries: ContextSnapshot["entries"] = [];
    let remaining = budget;

    const memoryResults = input.include_memory === false ? [] : await this.search({ project, query: input.query, top_k: 8 });
    const memoryBlock = memoryResults.map((result) => `- ${result.content}`).join("\n");
    if (memoryBlock) {
      const content = `Relevant memory:\n${truncateTokens(memoryBlock, Math.min(remaining, Math.floor(budget * 0.35)))}`;
      entries.push({ kind: "memory", id: "memory", hash: hashString(content), title: "Relevant memory", content });
      remaining -= estimateTokens(content);
    }

    if (input.include_company_brain !== false) {
      const sourceHits = memoryResults
        .map((r: any) => ({
          ...r.memory,
          source_id: r.memory.metadata?.source_id,
          source_type: r.memory.metadata?.source_type,
          source_title: r.memory.metadata?.source_title,
          external_id: r.memory.metadata?.external_id,
          url: r.memory.metadata?.url,
          citation: r.memory.metadata?.citation,
        }))
        .filter((m: any) => isFromSource(m));
      if (sourceHits.length > 0) {
        const brain = buildCompanyBrain({
          project,
          memories: sourceHits,
          maxTokens: Math.min(remaining, Math.floor(budget * 0.4)),
        });
        if (brain.sources.length > 0) {
          const header = `Company brain (${brain.total_sources} sources, ${brain.total_memories} memos):`;
          const content = `${header}\n${truncateTokens(brain.text, Math.max(0, remaining - 20))}`;
          entries.push({ kind: "company_brain", id: "company_brain", hash: hashString(content), title: "Company brain", content, citations: brain.citations as unknown as Array<Record<string, unknown>> });
          remaining -= estimateTokens(content);
        }
      }
    }

    if (input.tool_output) {
      const content = `Compressed tool output:\n${compressToolOutput(input.tool_output, Math.min(remaining, Math.floor(budget * 0.25)))}`;
      entries.push({ kind: "tool_output", id: "tool_output", hash: hashString(content), title: "Compressed tool output", content });
      remaining -= estimateTokens(content);
    }

    const mapped = codeMap(cwd, input.query, 12);
    const selectedFiles = Array.from(new Set([...(input.files || []), ...mapped.filter((item) => item.score > 0).slice(0, 6).map((item) => item.file)]));
    const chunks = packFileChunks(cwd, selectedFiles, input.query, Math.max(120, remaining));
    for (const chunk of chunks) {
      if (remaining <= 80) break;
      const content = truncateTokens(chunk.content, Math.min(chunk.tokens, remaining));
      entries.push({ kind: "file_chunk", id: chunk.file, hash: chunk.hash, title: chunk.file, content });
      remaining -= estimateTokens(content);
    }

    const mapContent = mapped.slice(0, 30).map((item) => `- ${item.file}${item.symbols.length ? `: ${item.symbols.slice(0, 8).join(", ")}` : ""}`).join("\n");
    if (mapContent && remaining > 100) {
      const content = `Code map:\n${truncateTokens(mapContent, Math.min(remaining, 300))}`;
      entries.push({ kind: "code_map", id: "code_map", hash: hashString(content), title: "Code map", content });
      remaining -= estimateTokens(content);
    }

    const prior = input.previous_context_hash ? this.data.contextSnapshots.find((item) => item.hash === input.previous_context_hash) : undefined;
    const priorHashes = new Map((prior?.entries || []).map((entry) => [`${entry.kind}:${entry.id}`, entry.hash]));
    const changed = entries.filter((entry) => priorHashes.get(`${entry.kind}:${entry.id}`) !== entry.hash);
    const removed = prior ? prior.entries.filter((entry) => !entries.some((next) => next.kind === entry.kind && next.id === entry.id)).map((entry) => ({ kind: entry.kind, id: entry.id, title: entry.title })) : [];
    const context = entries.map((entry) => entry.content).join("\n\n---\n\n");
    const deltaContext = changed.map((entry) => entry.content).join("\n\n---\n\n");
    const snapshot: ContextSnapshot = {
      hash: hashString(context),
      project,
      query: input.query,
      entries,
      created_at: now(),
    };
    this.data.contextSnapshots = [snapshot, ...this.data.contextSnapshots.filter((item) => item.hash !== snapshot.hash)].slice(0, 80);
    this.persist();
    appendJournal("context.packed", { project, hash: snapshot.hash, previous_context_hash: input.previous_context_hash, entries: entries.length, changed: changed.length, budget });
    return {
      context,
      delta_context: prior ? deltaContext || "No meaningful context changes since the previous pack." : context,
      context_hash: snapshot.hash,
      previous_context_hash: input.previous_context_hash || null,
      token_budget: budget,
      estimated_tokens: estimateTokens(context),
      estimated_delta_tokens: estimateTokens(prior ? deltaContext : context),
      compression_ratio: Number((estimateTokens(context) / Math.max(1, entries.reduce((sum, entry) => sum + estimateTokens(entry.content), 0))).toFixed(4)),
      changed: changed.map((entry) => ({ kind: entry.kind, id: entry.id, title: entry.title, hash: entry.hash })),
      removed,
      entries: entries.map((entry) => ({
        kind: entry.kind,
        id: entry.id,
        title: entry.title,
        hash: entry.hash,
        ...(entry.citations && entry.citations.length > 0
          ? { citations: entry.citations, citation_count: entry.citations.length }
          : {}),
      })),
    };
  }

  snapshot(project?: string) {
    const slug = project ? slugify(project) : undefined;
    const memories = this.data.memories
      .filter((memory) => !slug || memory.project === slug)
      .filter((memory) => memory.active)
      .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
    const sessions = Array.from(new Set(memories.map((memory) => memory.session_id).filter(Boolean))).map((sessionId) => {
      const sessionMemories = memories.filter((memory) => memory.session_id === sessionId);
      return {
        id: sessionId,
        project: sessionMemories[0]?.project || slug || DEFAULT_PROJECT,
        memory_count: sessionMemories.length,
        last_seen: sessionMemories[0]?.created_at,
        summary: sessionMemories.slice(0, 5).map((memory) => memory.content).join("\n"),
      };
    });
    const type_counts = memories.reduce<Record<string, number>>((acc, memory) => {
      acc[memory.memory_type] = (acc[memory.memory_type] || 0) + 1;
      return acc;
    }, {});
    return { stats: this.stats(), projects: this.data.projects, memories: memories.slice(0, 200), sessions, type_counts, graph: this.graph(slug) };
  }

  replay(sessionId: string, project?: string) {
    const memories = this.session(sessionId, project, 500).reverse();
    let previousTime: number | null = null;
    return {
      session_id: sessionId,
      events: memories.map((memory, index) => ({
        ...(() => {
          const currentTime = Date.parse(memory.created_at);
          const delta = previousTime === null ? 0 : Math.max(0, currentTime - previousTime);
          previousTime = currentTime;
          const event = memory.metadata?.event && typeof memory.metadata.event === "object" ? memory.metadata.event as Record<string, unknown> : {};
          return {
            index,
            id: memory.id,
            type: memory.memory_type,
            timestamp: memory.created_at,
            elapsed_ms: delta,
            title: memory.content.split("\n")[0]?.slice(0, 140) || memory.memory_type,
            content: memory.content,
            tool: event.toolName || event.tool_name || memory.metadata?.toolName || null,
            files: Array.isArray(event.filePaths) ? event.filePaths : [],
            metadata: memory.metadata,
          };
        })(),
      })),
    };
  }

  async consolidate(project?: string) {
    const slug = project ? slugify(project) : undefined;
    let duplicates_removed = 0;
    let summaries_created = 0;
    let decayed_memories = 0;
    const seen = new Map<string, Memory>();
    for (const memory of this.data.memories.filter((item) => item.active && (!slug || item.project === slug))) {
      const key = createHash("sha1").update(`${memory.project}:${memory.content.toLowerCase().replace(/\s+/g, " ").trim()}`).digest("hex");
      const prior = seen.get(key);
      if (prior) {
        memory.active = false;
        memory.updated_at = now();
        duplicates_removed += 1;
        appendJournal("memory.superseded", { id: memory.id, project: memory.project, reason: "duplicate", prior_id: prior.id });
      } else {
        seen.set(key, memory);
      }
    }
    const sessions = new Set(this.data.memories.filter((memory) => memory.active && memory.session_id && (!slug || memory.project === slug)).map((memory) => `${memory.project}\0${memory.session_id}`));
    for (const sessionKey of sessions) {
      const [sessionProject, sessionId] = sessionKey.split("\0");
      const memories = this.session(sessionId, sessionProject, 50).filter((memory) => memory.memory_type !== "session_summary");
      if (memories.length < 4) continue;
      const content = `Session ${sessionId} consolidated summary:\n${memories.slice(0, 12).map((memory) => `- ${memory.content.split("\n")[0]}`).join("\n")}`;
      const before = this.data.memories.length;
      await this.addMemory({
        project: sessionProject,
        session_id: sessionId,
        memory_type: "session_summary",
        content,
        importance: 0.85,
        metadata: { source: "consolidation", source_memory_ids: memories.map((memory) => memory.id) },
      });
      if (this.data.memories.length > before) summaries_created += 1;
      const decisions = memories.filter((memory) => /decision|constraint|fix|resolved|prefer|use /i.test(memory.content)).slice(0, 8);
      if (decisions.length >= 2) {
        await this.addMemory({
          project: sessionProject,
          session_id: sessionId,
          memory_type: "semantic",
          content: `Stable project knowledge from ${sessionId}:\n${decisions.map((memory) => `- ${memory.content.split("\n")[0]}`).join("\n")}`,
          importance: 0.9,
          metadata: { source: "semantic_consolidation", source_memory_ids: decisions.map((memory) => memory.id) },
        });
      }
      const workflows = memories.filter((memory) => /workflow|steps?|command|run|test|deploy|build|fix/i.test(memory.content)).slice(0, 8);
      if (workflows.length >= 2) {
        await this.addMemory({
          project: sessionProject,
          session_id: sessionId,
          memory_type: "procedural",
          content: `Useful workflow from ${sessionId}:\n${workflows.map((memory) => `- ${memory.content.split("\n")[0]}`).join("\n")}`,
          importance: 0.82,
          metadata: { source: "procedural_consolidation", source_memory_ids: workflows.map((memory) => memory.id) },
        });
      }
    }
    for (const memory of this.data.memories.filter((item) => item.active && (!slug || item.project === slug))) {
      const ageDays = Math.max(0, (Date.now() - Date.parse(memory.created_at)) / 864e5);
      const strength = Number(memory.metadata.strength || memory.importance || 0.5);
      const accessCount = Number(memory.metadata.access_count || 0);
      const durable = DURABLE_MEMORY_TYPES.has(memory.memory_type);
      if (!durable && ageDays > 14 && accessCount === 0 && strength < 0.45) {
        memory.active = false;
        memory.updated_at = now();
        memory.metadata.decayed_at = memory.updated_at;
        decayed_memories += 1;
        appendJournal("memory.decayed", { id: memory.id, project: memory.project, age_days: Number(ageDays.toFixed(2)), strength });
      }
    }
    this.persist();
    appendJournal("consolidation.completed", { project: slug || "all", duplicates_removed, summaries_created, decayed_memories });
    return { duplicates_removed, summaries_created, decayed_memories };
  }
}

function json(c: any, payload: Record<string, unknown>, status = 200) {
  return c.json({ success: status < 400, ...payload, trace_id: randomUUID() }, status);
}

function requestLogger() {
  return async (c: any, next: any) => {
    const ignoredPaths = new Set([
      '/',
    ]);

    if (ignoredPaths.has(c.req.path) || c.req.path.endsWith('/health')) {
      await next();
      return;
    }

    const start = Date.now();
    const statusCode = c.res.status;
    const reqId = c.get('trace_id') || randomUUID();
    const method = c.req.method;
    const path = c.req.path;
    const route = c.req.routePath;
    const rawUrl = c.req.raw.url;
    const query = rawUrl.includes('?')
      ? rawUrl.split('?')[1]
      : '';
    const queryParams = Object.fromEntries(
      new URL(rawUrl).searchParams.entries()
    );
    const headers = {
      host: c.req.header('host'),
      origin: c.req.header('origin'),
      referer: c.req.header('referer'),
      'x-forwarded-for': c.req.header('x-forwarded-for'),
      'x-real-ip': c.req.header('x-real-ip'),
      'x-forwarded-proto': c.req.header('x-forwarded-proto'),
      'x-forwarded-host': c.req.header('x-forwarded-host'),
      'x-forwarded-port': c.req.header('x-forwarded-port'),
      'user-agent': c.req.header('user-agent'),
      'content-type': c.req.header('content-type'),
      'content-length': c.req.header('content-length'),
      accept: c.req.header('accept'),
      'accept-language': c.req.header('accept-language'),
      'accept-encoding': c.req.header('accept-encoding'),
      'cache-control': c.req.header('cache-control'),
      'if-none-match': c.req.header('if-none-match'),
      'if-modified-since': c.req.header('if-modified-since'),
      'x-request-id': c.req.header('x-request-id'),
      'x-correlation-id': c.req.header('x-correlation-id'),
    };

    await next();

    const log = {
      timestamp: new Date().toISOString(),
      durationMs: Date.now() - start,
      statusCode,
      request: {
        id: reqId,
        method,
        path,
        route,
        url: rawUrl,
        query,
        queryParams,
        headers
      },
    };

    console.log(JSON.stringify(log));
  };
}

function createApp(runtime: LocalMemoryRuntime) {
  ensureConnectorsRegistered();
  const app = new Hono();

  app.use('*', requestLogger());

  const sources = new SourceStore(RETAINDB_HOME);
  app.get("/", (c) => json(c, {
    name: "RetainDB Local",
    version: "0.1.0",
    message: "Persistent memory for coding agents. Runs on your machine.",
    endpoints: {
      health: "GET /health",
      context: "POST /v1/context/query",
      memory_write: "POST /v1/memory",
      memory_search: "POST /v1/memory/search",
      context_pack: "POST /v1/context/pack",
      context_delta: "POST /v1/context/delta",
      inspect: "GET /retaindb/inspect",
      filesystem: "GET /v1/filesystem",
      filesystem_sync: "POST /v1/filesystem/sync",
      filesystem_write: "POST /v1/filesystem/write",
      memory_ingest: "POST /v1/memory/ingest/session",
      agent_event: "POST /v1/agent-events",
      sources: "GET /v1/sources",
      source_create: "POST /v1/sources",
      source_get: "GET /v1/sources/:id",
      source_patch: "PATCH /v1/sources/:id",
      source_delete: "DELETE /v1/sources/:id",
      source_sync: "POST /v1/sources/:id/sync",
      source_descriptors: "GET /v1/sources/connectors",
      company_brain: "GET /v1/company-brain",
      company_brain_ask: "POST /v1/company-brain/ask",
      company_brain_feed: "POST /v1/company-brain/feed",
    },
  }));
  app.get("/health", (c) => json(c, { status: "ok", local: true, stats: runtime.stats() }));
  app.get("/retaindb/health", (c) => json(c, { status: "ok", local: true, stats: runtime.stats() }));
  app.get("/retaindb/snapshot", (c) => json(c, runtime.snapshot(c.req.query("project"))));
  app.get("/retaindb/inspect", (c) => json(c, runtime.inspect(c.req.query("project"))));
  app.get("/retaindb/graph", (c) => json(c, runtime.graph(c.req.query("project"))));
  app.get("/retaindb/replay/:sessionId", (c) => json(c, runtime.replay(c.req.param("sessionId"), c.req.query("project"))));
  app.post("/retaindb/consolidate", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    return json(c, await runtime.consolidate(body.project));
  });
  app.get("/v1/projects", (c) => json(c, { projects: runtime.listProjects() }));
  app.post("/v1/projects", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    const project = runtime.ensureProject(body.name || body.slug || DEFAULT_PROJECT);
    return json(c, project);
  });
  app.get("/v1/sources/connectors", (c) => json(c, { connectors: listConnectorDescriptors() }));
  app.get("/v1/sources", (c) => {
    const project = c.req.query("project");
    return json(c, { sources: sources.list(project) });
  });
  app.get("/v1/sources/:id", (c) => {
    const src = sources.get(c.req.param("id"));
    if (!src) return json(c, { error: "not_found" }, 404);
    return json(c, src as unknown as Record<string, unknown>);
  });
  app.post("/v1/sources", async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as any;
    const type = body.type as SourceType;
    if (!type) return json(c, { error: "type_required" }, 400);
    if (!listConnectorTypes().includes(type)) {
      return json(c, { error: `unknown_source_type: ${type}` }, 400);
    }
    const provider = getConnector(type);
    const validation = provider?.validateConfig(body.config || {});
    if (validation && !validation.ok) {
      return json(c, { error: "invalid_config", detail: validation.error }, 400);
    }
    const source = sources.create({
      type,
      name: String(body.name || `${type}-${newSourceId().slice(4, 10)}`),
      project: String(body.project || DEFAULT_PROJECT),
      config: body.config || {},
    });
    runtime.ensureProject(source.project);
    return json(c, source as unknown as Record<string, unknown>, 201);
  });
  app.patch("/v1/sources/:id", async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as any;
    const updated = sources.update(c.req.param("id"), {
      ...(body.name !== undefined ? { name: String(body.name) } : {}),
      ...(body.config !== undefined ? { config: body.config } : {}),
      ...(body.status !== undefined ? { status: body.status } : {}),
    });
    if (!updated) return json(c, { error: "not_found" }, 404);
    if (body.config) {
      const provider = getConnector(updated.type);
      const validation = provider?.validateConfig(updated.config);
      if (validation && !validation.ok) {
        sources.update(updated.id, updated);
        return json(c, { error: "invalid_config", detail: validation.error }, 400);
      }
    }
    return json(c, updated as unknown as Record<string, unknown>);
  });
  app.delete("/v1/sources/:id", (c) => {
    const id = c.req.param("id");
    const ok = sources.delete(id);
    if (!ok) return json(c, { error: "not_found" }, 404);
    return json(c, { deleted: true, id });
  });
  app.post("/v1/sources/:id/sync", async (c) => {
    const id = c.req.param("id");
    const source = sources.get(id);
    if (!source) return json(c, { error: "not_found" }, 404);
    sources.update(id, { status: "syncing", last_error: undefined });
    try {
      const result = await runSourceSync({
        source,
        ingest: (input) =>
          runtime.addMemory({
            project: source.project,
            content: input.content,
            memory_type: input.memory_type,
            importance: input.importance,
            confidence: input.confidence,
            agent_id: input.agent_id,
            session_id: input.session_id,
            metadata: input.metadata,
          }),
      });
      const summary = {
        documents_indexed: result.documents_indexed,
        memories_created: result.memories_created,
        errors: result.errors.length,
        duration_ms: result.duration_ms,
      };
      sources.update(id, {
        status: result.errors.length > 0 && result.memories_created === 0 ? "error" : "connected",
        last_synced_at: new Date().toISOString(),
        last_sync_status: result.errors.length > 0 ? (result.memories_created > 0 ? "partial" : "error") : "ok",
        last_sync_summary: summary,
        last_error: result.errors.length > 0 ? result.errors.slice(0, 3).join(" | ") : undefined,
      });
      return json(c, { source_id: id, result, citations: result.citations });
    } catch (err: any) {
      sources.update(id, { status: "error", last_error: err?.message || String(err) });
      return json(c, { error: "sync_failed", detail: err?.message || String(err) }, 500);
    }
  });
  app.get("/v1/company-brain", (c) => {
    const project = slugify(c.req.query("project") || DEFAULT_PROJECT);
    const maxTokens = Number(c.req.query("maxTokens") || 8000);
    const memories = (runtime as any).data.memories
      .filter((m: any) => m.project === project && m.active)
      .map((m: any) => ({ ...m, source_id: m.metadata?.source_id, source_type: m.metadata?.source_type, source_title: m.metadata?.source_title, external_id: m.metadata?.external_id, url: m.metadata?.url, citation: m.metadata?.citation }));
    const brain = buildCompanyBrain({ project, memories, maxTokens });
    return json(c, brain as unknown as Record<string, unknown>);
  });
  app.post("/v1/company-brain/ask", async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as any;
    const project = slugify(body.project || DEFAULT_PROJECT);
    if (!body.query) return json(c, { error: "query_required" }, 400);
    const topK = Math.min(Math.max(Number(body.top_k || 12), 1), 60);
    const maxTokens = Math.min(Math.max(Number(body.max_tokens || 2400), 200), 8000);
    const includeAgent = body.include_agent_memories !== false;
    const result = await askBrain({
      project,
      query: String(body.query),
      top_k: topK,
      maxTokens,
      includeAgentMemories: includeAgent,
      search: async (input) => {
        const results = await runtime.search(input);
        return results.map((r: any) => ({
          memory: { ...r.memory, source_id: r.memory.metadata?.source_id, source_type: r.memory.metadata?.source_type, source_title: r.memory.metadata?.source_title, external_id: r.memory.metadata?.external_id, url: r.memory.metadata?.url, citation: r.memory.metadata?.citation },
          score: r.score,
        }));
      },
    });
    return json(c, result as unknown as Record<string, unknown>);
  });
  app.post("/v1/company-brain/feed", async (c) => {
    const body = (await c.req.json().catch(() => ({}))) as any;
    const project = slugify(body.project || DEFAULT_PROJECT);
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const includeAgent = body.include_agent_memories !== false;
    const maxCtx = Math.min(Math.max(Number(body.max_context_tokens || 2400), 200), 8000);
    let ask;
    if (body.query) {
      ask = await askBrain({
        project,
        query: String(body.query),
        top_k: Number(body.top_k || 12),
        maxTokens: maxCtx,
        includeAgentMemories: includeAgent,
        search: async (input) => {
          const results = await runtime.search(input);
          return results.map((r: any) => ({
            memory: { ...r.memory, source_id: r.memory.metadata?.source_id, source_type: r.memory.metadata?.source_type, source_title: r.memory.metadata?.source_title, external_id: r.memory.metadata?.external_id, url: r.memory.metadata?.url, citation: r.memory.metadata?.citation },
            score: r.score,
          }));
        },
      });
    }
    const out = feedAgent({ project, messages, query: body.query, maxContextTokens: maxCtx, includeAgentMemories: includeAgent, ask });
    return json(c, out as unknown as Record<string, unknown>);
  });
  app.post("/v1/memory", async (c) => {
    const body = await c.req.json();
    const memory = await runtime.addMemory(body);
    return json(c, { memory_id: memory.id, id: memory.id, stored: true, memory });
  });
  app.post("/v1/memory/bulk", async (c) => {
    const body = await c.req.json();
    const memories = Array.isArray(body.memories) ? body.memories : [];
    const stored = await Promise.all(memories.map((memory: any) => runtime.addMemory({ ...memory, project: body.project || memory.project })));
    return json(c, { memories_created: stored.length, memories: stored });
  });
  app.post("/v1/memory/search", async (c) => {
    const body = await c.req.json();
    const results = await runtime.search(body);
    return json(c, { results, memories: results.map((result) => result.memory), count: results.length });
  });
  app.post("/v1/context/query", async (c) => {
    const body = await c.req.json();
    const results = await runtime.search(body);
    return json(c, {
      results,
      memories: body.include_memories === false ? [] : results.map((result) => result.memory),
      context: results.map((result) => `- ${result.content}`).join("\n"),
      meta: {
        query: body.query,
        total: results.length,
        latency_ms: 0,
        cache_hit: false,
        tokens_used: 0,
        context_hash: createHash("sha256").update(results.map((result) => result.content).join("\n")).digest("hex"),
      },
    });
  });
  app.post("/v1/context/pack", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    return json(c, await runtime.contextPack(body));
  });
  app.post("/v1/context/delta", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    return json(c, await runtime.contextPack({ ...body, previous_context_hash: body.previous_context_hash || body.context_hash }));
  });
  app.post("/v1/context/compress-output", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    const compressed = compressToolOutput(String(body.output || body.tool_output || ""), Number(body.token_budget || 500));
    return json(c, {
      compressed,
      estimated_tokens: estimateTokens(compressed),
      original_estimated_tokens: estimateTokens(String(body.output || body.tool_output || "")),
      compression_ratio: Number((estimateTokens(compressed) / Math.max(1, estimateTokens(String(body.output || body.tool_output || "")))).toFixed(4)),
    });
  });
  app.post("/v1/context/code-map", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    return json(c, { files: codeMap(String(body.cwd || process.cwd()), String(body.query || ""), Number(body.limit || 80)) });
  });
  app.get("/v1/filesystem", (c) => {
    try {
      const project = c.req.query("project") || DEFAULT_PROJECT;
      const cwd = c.req.query("cwd") || process.cwd();
      const path = c.req.query("path");
      syncBrainFilesystem({ cwd, project, snapshot: runtime.snapshot(project) });
      if (path) return json(c, { file: readBrainFile({ cwd, path, includeContents: c.req.query("includeContents") !== "false" }) });
      return json(c, listBrainFileTree(cwd, c.req.query("includeContents") === "true", Number(c.req.query("limit") || 250)));
    } catch (error) {
      return json(c, { error: { code: "FILESYSTEM_ERROR", message: error instanceof Error ? error.message : String(error) } }, 400);
    }
  });
  app.get("/v1/context/files", (c) => {
    try {
      const project = c.req.query("project") || DEFAULT_PROJECT;
      const cwd = c.req.query("cwd") || process.cwd();
      const path = c.req.query("path");
      syncBrainFilesystem({ cwd, project, snapshot: runtime.snapshot(project) });
      if (path) return json(c, { file: readBrainFile({ cwd, path, includeContents: c.req.query("includeContents") !== "false" }) });
      return json(c, listBrainFileTree(cwd, c.req.query("includeContents") === "true", Number(c.req.query("limit") || 250)));
    } catch (error) {
      return json(c, { error: { code: "FILESYSTEM_ERROR", message: error instanceof Error ? error.message : String(error) } }, 400);
    }
  });
  app.post("/v1/filesystem/sync", async (c) => {
    try {
      const body = await c.req.json().catch(() => ({} as any));
      const project = String(body.project || DEFAULT_PROJECT);
      const result = syncBrainFilesystem({ cwd: body.cwd || process.cwd(), project, snapshot: runtime.snapshot(project) });
      return json(c, result);
    } catch (error) {
      return json(c, { error: { code: "FILESYSTEM_SYNC_ERROR", message: error instanceof Error ? error.message : String(error) } }, 400);
    }
  });
  app.post("/v1/filesystem/write", async (c) => {
    try {
      const body = await c.req.json().catch(() => ({} as any));
      const written = writeAgentBrainFile({ ...body, cwd: body.cwd || process.cwd(), project: body.project || DEFAULT_PROJECT });
      const memory = await runtime.addMemory({
        project: String(body.project || DEFAULT_PROJECT),
        content: written.memoryContent,
        memory_type: body.kind === "handoff" ? "session_summary" : body.kind || "project_state",
        session_id: body.sessionId || body.session_id,
        agent_id: body.agentId || body.agent_id,
        task_id: body.taskId || body.task_id,
        importance: body.kind === "handoff" || body.kind === "decision" ? 0.92 : 0.72,
        metadata: { source: "local_brain_file", path: written.path, files: body.files || [] },
      });
      syncBrainFilesystem({ cwd: body.cwd || process.cwd(), project: String(body.project || DEFAULT_PROJECT), snapshot: runtime.snapshot(body.project || DEFAULT_PROJECT) });
      return json(c, { written, memory_id: memory.id, stored: memory.active });
    } catch (error) {
      return json(c, { error: { code: "FILESYSTEM_WRITE_ERROR", message: error instanceof Error ? error.message : String(error) } }, 400);
    }
  });
  app.post("/v1/memory/ingest/session", async (c) => {
    const body = await c.req.json();
    const result = await runtime.ingestSession(body);
    return json(c, {
      memories_created: result.memories_created,
      skipped: result.skipped,
      relations_created: 0,
      memories_invalidated: 0,
      memories: result.memories,
    });
  });
  app.post("/v1/agent-events", async (c) => {
    const body = await c.req.json();
    const result = await runtime.ingestSession({
      project: body.project,
      session_id: body.session_id || `session-${Date.now()}`,
      user_id: body.user_id,
      agent_id: body.agent_id,
      task_id: body.task_id,
      events: [body.event || body],
      messages: body.messages || [],
    });
    return json(c, { stored: true, memories_created: result.memories_created, skipped: result.skipped, memories: result.memories });
  });
  app.post("/retaindb/observe", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    const hookType = String(body.hookType || body.hook_type || "observation");
    const data = body.data && typeof body.data === "object" ? body.data as Record<string, unknown> : {};
    const summary =
      typeof data.prompt === "string" ? `Prompt: ${data.prompt}` :
      typeof data.tool_name === "string" ? `${data.tool_name}: ${String(data.tool_output || data.tool_response || "").slice(0, 2000)}` :
      typeof data.message === "string" ? data.message :
      `${hookType} captured`;
    const result = await runtime.ingestSession({
      project: String(body.project || DEFAULT_PROJECT),
      session_id: String(body.sessionId || body.session_id || `session-${Date.now()}`),
      agent_id: String(body.agentId || body.agent_id || "agent"),
      events: [{
        kind: hookType,
        summary,
        details: JSON.stringify(data).slice(0, 12000),
        timestamp: String(body.timestamp || now()),
        toolName: typeof data.tool_name === "string" ? data.tool_name : undefined,
      }],
      messages: [],
    });
    return json(c, { stored: true, memories_created: result.memories_created, skipped: result.skipped });
  });
  app.post("/retaindb/session/start", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    const sessionId = String(body.sessionId || body.session_id || `session-${Date.now()}`);
    const project = String(body.project || DEFAULT_PROJECT);
    await runtime.ingestSession({
      project,
      session_id: sessionId,
      agent_id: "agent",
      events: [{ kind: "session_start", summary: `Session started in ${String(body.cwd || project)}`, timestamp: now() }],
    });
    const context = (await runtime.search({ project, query: "recent project decisions constraints workflow", top_k: 8 }))
      .map((result) => `- ${result.content}`)
      .join("\n");
    return json(c, { sessionId, context });
  });
  app.post("/retaindb/session/end", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    const sessionId = String(body.sessionId || body.session_id || "unknown");
    const result = await runtime.ingestSession({
      project: String(body.project || DEFAULT_PROJECT),
      session_id: sessionId,
      agent_id: "agent",
      events: [{ kind: "session_end", summary: `Session ended: ${sessionId}`, timestamp: now() }],
    });
    return json(c, { stored: true, memories_created: result.memories_created, skipped: result.skipped });
  });
  app.post("/retaindb/summarize", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    const sessionId = String(body.sessionId || body.session_id || "unknown");
    const memories = runtime.session(sessionId, String(body.project || DEFAULT_PROJECT), 20);
    if (memories.length > 0) {
      await runtime.addMemory({
        project: String(body.project || DEFAULT_PROJECT),
        session_id: sessionId,
        memory_type: "session_summary",
        content: `Session ${sessionId} summary:\n${memories.map((memory) => `- ${memory.content}`).join("\n")}`,
        metadata: { source: "session_summary", memory_count: memories.length },
      });
    }
    return json(c, { summarized: true, memories_seen: memories.length });
  });
  app.post("/retaindb/smart-search", async (c) => {
    const body = await c.req.json().catch(() => ({} as any));
    const results = await runtime.search({
      project: String(body.project || DEFAULT_PROJECT),
      query: String(body.query || body.q || ""),
      top_k: Number(body.limit || body.top_k || 10),
    });
    return json(c, { results, memories: results.map((result) => result.memory) });
  });
  app.get("/v1/memory/session/:sessionId", (c) => {
    const memories = runtime.session(c.req.param("sessionId"), c.req.query("project"), Number(c.req.query("limit") || 30));
    return json(c, { memories, count: memories.length });
  });
  app.get("/v1/memory/profile/:userId", (c) => {
    const memories = runtime.profile(c.req.param("userId"), c.req.query("project"), Number(c.req.query("limit") || 100));
    return json(c, { memories, count: memories.length });
  });
  app.get("/v1/memories", (c) => {
    const memories = runtime.profile(c.req.query("user_id") || "local-user", c.req.query("project"), Number(c.req.query("limit") || 100));
    return json(c, { memories });
  });
  app.delete("/v1/memory/:memoryId", (c) => {
    const deleted = runtime.delete(c.req.param("memoryId"));
    return json(c, { deleted: deleted ? c.req.param("memoryId") : null });
  });
  app.post("/v1/context/share", async (c) => {
    const body = await c.req.json();
    const share = runtime.createShare(body);
    return json(c, share);
  });
  app.post("/v1/context/resume", async (c) => {
    const body = await c.req.json();
    const result = runtime.resumeShare(body.share_id, body.new_session_id);
    if (!result) return json(c, { error: { code: "NOT_FOUND", message: "Share not found" } }, 404);
    return json(c, result);
  });
  app.notFound((c) => json(c, { error: { code: "NOT_FOUND", message: "Route not found" } }, 404));
  return app;
}

function viewerHtml(apiPort: number) {
  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>RetainDB Local</title>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f7f7f4; color: #191a1c; }
    header { padding: 24px 28px 16px; border-bottom: 1px solid #ddd9d0; background: #ffffff; }
    h1 { margin: 0; font-size: 24px; letter-spacing: 0; }
    .sub { margin-top: 6px; color: #62646a; font-size: 14px; }
    main { display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 86px); }
    aside { border-right: 1px solid #ddd9d0; padding: 18px; background: #fbfbf8; }
    section { padding: 18px; }
    .stats { display: grid; gap: 10px; }
    .stat { border: 1px solid #ddd9d0; background: #fff; border-radius: 8px; padding: 12px; }
    .stat strong { display: block; font-size: 22px; }
    .toolbar { display: flex; gap: 8px; margin-bottom: 14px; }
    input { flex: 1; min-width: 0; border: 1px solid #c9c6be; border-radius: 6px; padding: 9px 10px; font: inherit; }
    button { border: 1px solid #191a1c; background: #191a1c; color: #fff; border-radius: 6px; padding: 9px 12px; font: inherit; cursor: pointer; }
    .grid { display: grid; gap: 10px; }
    .item { border: 1px solid #ddd9d0; border-radius: 8px; background: #fff; padding: 12px; }
    .item[data-session] { cursor: pointer; }
    .meta { color: #62646a; font-size: 12px; margin-bottom: 6px; display: flex; gap: 8px; flex-wrap: wrap; }
    .timeline { display: grid; gap: 8px; }
    .event { border-left: 3px solid #191a1c; padding: 8px 10px; background: #fbfbf8; border-radius: 0 6px 6px 0; }
    .event.active { background: #eef6ff; border-left-color: #2457c5; }
    .graphbox { height: 240px; border: 1px solid #ddd9d0; border-radius: 8px; background: #fff; overflow: hidden; }
    .graphbox svg { width: 100%; height: 100%; display: block; }
    .node { cursor: pointer; }
    .node text { font-size: 11px; fill: #191a1c; pointer-events: none; }
    .edge { stroke: #c9c6be; stroke-width: 1.25; }
    pre { white-space: pre-wrap; margin: 0; font: inherit; line-height: 1.45; }
    @media (max-width: 800px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid #ddd9d0; } }
  </style>
</head>
<body>
  <header>
    <h1>RetainDB Local</h1>
    <div class="sub">Live memory browser for the local runtime on :${apiPort}</div>
  </header>
  <main>
    <aside>
      <div class="stats" id="stats"></div>
      <h3>Sessions</h3>
      <div class="grid" id="sessions"></div>
      <h3>Graph</h3>
      <div class="grid" id="graph"></div>
    </aside>
    <section>
      <div class="toolbar">
        <input id="query" placeholder="Search memories" />
        <button id="search">Search</button>
        <button id="refresh">Refresh</button>
      </div>
      <div class="grid" id="replay" style="margin-bottom:14px;"></div>
      <div class="grid" id="memories"></div>
    </section>
  </main>
  <script>
    const api = location.protocol + '//' + location.hostname + ':${apiPort}';
    const statsEl = document.getElementById('stats');
    const memoriesEl = document.getElementById('memories');
    const sessionsEl = document.getElementById('sessions');
    const graphEl = document.getElementById('graph');
    const replayEl = document.getElementById('replay');
    const queryEl = document.getElementById('query');
    function esc(s) { return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    function renderStats(stats, typeCounts) {
      statsEl.innerHTML = ['memories','sessions','projects','shares'].map(k => '<div class="stat"><strong>' + esc(stats[k]) + '</strong><span>' + k + '</span></div>').join('') +
        '<div class="stat"><span>' + esc(Object.entries(typeCounts || {}).map(([k,v]) => k + ':' + v).join(' · ') || 'no types yet') + '</span></div>';
    }
    function renderMemories(memories) {
      memoriesEl.innerHTML = (memories || []).map(m => '<div class="item"><div class="meta"><span>' + esc(m.memory_type || m.type) + '</span><span>' + esc(m.project) + '</span><span>' + esc(m.session_id || '') + '</span><span>' + esc(m.created_at || '') + '</span></div><pre>' + esc(m.content) + '</pre></div>').join('') || '<div class="item">No memories yet.</div>';
    }
    function renderSessions(sessions) {
      sessionsEl.innerHTML = (sessions || []).slice(0, 20).map(s => '<div class="item" data-session="' + esc(s.id) + '"><div class="meta"><span>' + esc(s.memory_count) + ' memories</span><span>' + esc(s.last_seen || '') + '</span></div><pre>' + esc(s.id) + '</pre></div>').join('') || '<div class="item">No sessions yet.</div>';
      for (const node of sessionsEl.querySelectorAll('[data-session]')) node.onclick = () => replay(node.getAttribute('data-session'));
    }
    function renderGraph(graph) {
      const nodes = ((graph && graph.nodes) || []).slice(0, 24);
      const edges = ((graph && graph.edges) || []).filter(e => nodes.some(n => n.id === e.from) && nodes.some(n => n.id === e.to)).slice(0, 60);
      if (!nodes.length) {
        graphEl.innerHTML = '<div class="item">No graph yet.</div>';
        return;
      }
      const w = 260, h = 240, cx = w / 2, cy = h / 2;
      const maxCount = Math.max(...nodes.map(n => n.count || 1), 1);
      const positioned = nodes.map((n, i) => {
        const angle = (Math.PI * 2 * i) / nodes.length;
        const radius = 46 + Math.min(62, (nodes.length * 4));
        return { ...n, x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius, r: 5 + 10 * ((n.count || 1) / maxCount) };
      });
      const byId = Object.fromEntries(positioned.map(n => [n.id, n]));
      const edgeSvg = edges.map(e => byId[e.from] && byId[e.to] ? '<line class="edge" x1="' + byId[e.from].x + '" y1="' + byId[e.from].y + '" x2="' + byId[e.to].x + '" y2="' + byId[e.to].y + '" />' : '').join('');
      const nodeSvg = positioned.map(n => '<g class="node" data-concept="' + esc(n.id) + '"><circle cx="' + n.x + '" cy="' + n.y + '" r="' + n.r + '" fill="#eef6ff" stroke="#2457c5"></circle><text x="' + (n.x + n.r + 3) + '" y="' + (n.y + 4) + '">' + esc(n.id.slice(0, 18)) + '</text></g>').join('');
      graphEl.innerHTML = '<div class="graphbox"><svg viewBox="0 0 ' + w + ' ' + h + '">' + edgeSvg + nodeSvg + '</svg></div><div class="item"><div class="meta"><span>' + esc(nodes.length) + ' nodes</span><span>' + esc(edges.length) + ' edges</span></div><pre>' + esc(nodes.slice(0, 10).map(n => n.id + ':' + n.count).join('\\n')) + '</pre></div>';
      for (const node of graphEl.querySelectorAll('[data-concept]')) node.onclick = () => { queryEl.value = node.getAttribute('data-concept') || ''; search(); };
    }
    async function replay(sessionId) {
      const res = await fetch(api + '/retaindb/replay/' + encodeURIComponent(sessionId));
      const body = await res.json();
      const events = body.events || [];
      replayEl.innerHTML = '<div class="item"><div class="meta"><span>Replay</span><span>' + esc(sessionId) + '</span><span>' + esc(events.length) + ' events</span><button id="stepReplay" style="margin-left:auto">Step</button></div><div class="timeline">' + events.map((e, i) => '<div class="event" data-step="' + i + '"><div class="meta"><span>#' + esc(i + 1) + '</span><span>' + esc(e.type) + '</span><span>+' + esc(e.elapsed_ms || 0) + 'ms</span><span>' + esc(e.tool || '') + '</span></div><pre>' + esc(e.title) + '</pre></div>').join('') + '</div></div>';
      let index = -1;
      const step = () => {
        const nodes = Array.from(replayEl.querySelectorAll('.event'));
        nodes.forEach(n => n.classList.remove('active'));
        index = Math.min(index + 1, nodes.length - 1);
        if (nodes[index]) nodes[index].classList.add('active');
      };
      document.getElementById('stepReplay').onclick = step;
    }
    async function refresh() {
      const res = await fetch(api + '/retaindb/snapshot');
      const snap = await res.json();
      renderStats(snap.stats || {}, snap.type_counts || {});
      renderMemories(snap.memories || []);
      renderSessions(snap.sessions || []);
      renderGraph(snap.graph || {});
    }
    async function search() {
      const query = queryEl.value.trim();
      if (!query) return refresh();
      const res = await fetch(api + '/v1/memory/search', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ query, top_k: 50 }) });
      const body = await res.json();
      renderMemories((body.results || []).map(r => r.memory || r));
    }
    document.getElementById('refresh').onclick = refresh;
    document.getElementById('search').onclick = search;
    queryEl.addEventListener('keydown', e => { if (e.key === 'Enter') search(); });
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>`;
}

function createViewerApp(apiPort: number) {
  const app = new Hono();
  app.use('*', requestLogger());
  app.get("/", (c) => c.html(viewerHtml(apiPort)));
  app.get("/health", (c) => json(c, { status: "ok", local: true }));
  app.get("/viewer/health", (c) => json(c, { status: "ok", local: true }));
  return app;
}

function arg(name: string) {
  const prefix = `--${name}=`;
  for (let i = 2; i < process.argv.length; i++) {
    const a = process.argv[i];
    if (a === `--${name}`) {
      const next = process.argv[i + 1];
      if (next !== undefined && !next.startsWith("--")) return next;
      return "true";
    }
    if (a.startsWith(prefix)) return a.slice(prefix.length);
  }
  return undefined;
}

function mcpEnv(baseUrl: string, project: string) {
  return { RETAINDB_BASE_URL: baseUrl, RETAINDB_PROJECT: project };
}

function connectConfig(target: string, baseUrl: string, project: string) {
  const env = mcpEnv(baseUrl, project);
  const hookCommand = ["npx", "-y", "@retaindb/local", "hook"];
  const mcpCommand = ["npx", "-y", "@retaindb/local", "mcp"];
  if (target === "codex") {
    return [
      "[mcp_servers.retaindb]",
      'command = "npx"',
      'args = ["-y", "@retaindb/local", "mcp"]',
      "",
      "[mcp_servers.retaindb.env]",
      `RETAINDB_BASE_URL = "${baseUrl}"`,
      `RETAINDB_PROJECT = "${project}"`,
      "",
      "# Optional hook commands for Codex user hooks:",
      `# ${hookCommand.join(" ")} --kind=tool_result --agent=codex`,
    ].join("\n");
  }
  if (target === "opencode") {
    return JSON.stringify({
      mcp: { retaindb: { type: "local", command: mcpCommand, env, enabled: true } },
      plugin: ["./.retaindb/opencode/retaindb-capture.ts"],
      hooks: { postToolUse: hookCommand.concat(["--kind=tool_result", "--agent=opencode"]) },
    }, null, 2);
  }
  return JSON.stringify({
    mcpServers: { retaindb: { command: "npx", args: ["-y", "@retaindb/local", "mcp"], env } },
    hooks: {
      UserPromptSubmit: hookCommand.concat(["--kind=prompt", "--agent=claude-code"]),
      PostToolUse: hookCommand.concat(["--kind=tool_result", "--agent=claude-code"]),
      Stop: hookCommand.concat(["--kind=session_end", "--agent=claude-code"]),
    },
  }, null, 2);
}

function opencodePluginSource(baseUrl: string, project: string) {
  return `import type { Plugin } from "@opencode-ai/plugin";

const API = process.env.RETAINDB_BASE_URL || "${baseUrl}";
const PROJECT = process.env.RETAINDB_PROJECT || "${project}";

async function observe(sessionId: string, hookType: string, data: Record<string, unknown>) {
  try {
    await fetch(\`\${API.replace(/\\/+$/, "")}/retaindb/observe\`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hookType,
        sessionId,
        project: PROJECT,
        cwd: data.cwd || process.cwd(),
        timestamp: new Date().toISOString(),
        data,
      }),
      signal: AbortSignal.timeout(3000),
    });
  } catch {}
}

export const RetainDBCapturePlugin: Plugin = async (ctx) => {
  let activeSessionId: string | null = null;
  const projectPath = ctx.worktree || ctx.project?.id || process.cwd();
  return {
    event: async ({ event }) => {
      const type = event.type;
      const props = (event as any).properties || {};
      const sid = props.sessionID || props.session_id || activeSessionId || "opencode-session";
      if (type === "session.created") {
        activeSessionId = props.info?.id || props.sessionID || sid;
        await observe(activeSessionId, "session_start", { cwd: projectPath, info: props.info || {} });
      }
      if (type === "session.status" && props.status?.type === "idle") {
        await observe(sid, "session_idle", { cwd: projectPath, status: props.status });
      }
      if (type === "session.deleted") {
        await observe(sid, "session_end", { cwd: projectPath });
        if (sid === activeSessionId) activeSessionId = null;
      }
      if (type === "session.diff") {
        await observe(sid, "file_edit", { cwd: projectPath, diff: props.diff || [] });
      }
      if (type === "message.part.updated" && props.part?.type === "tool") {
        await observe(sid, props.part.state?.status === "error" ? "post_tool_failure" : "post_tool_use", {
          cwd: projectPath,
          tool_name: props.part.tool,
          state: props.part.state,
        });
      }
    },
    "chat.message": async (input, output) => {
      const sid = input.sessionID || activeSessionId || "opencode-session";
      const text = (output.parts || []).filter((part: any) => part.type === "text").map((part: any) => part.text || "").join("\\n");
      await observe(sid, "prompt_submit", { cwd: projectPath, prompt: text.slice(0, 8000), agent: input.agent || null });
    },
  };
};
`;
}

function writeConnectSnippets(target = "all") {
  const baseUrl = process.env.RETAINDB_BASE_URL || `http://localhost:${DEFAULT_PORT}`;
  const project = process.env.RETAINDB_PROJECT || DEFAULT_PROJECT;
  const targets = target === "all" ? ["codex", "claude-code", "opencode"] : [target];
  const outputRoot = process.env.INIT_CWD || process.cwd();
  const dir = join(outputRoot, ".retaindb", "agent-bridge");
  mkdirSync(dir, { recursive: true });
  for (const item of targets) {
    const ext = item === "codex" ? "toml" : "json";
    const dest = join(dir, `${item}.${ext}`);
    writeFileSync(dest, `${connectConfig(item, baseUrl, project)}\n`, "utf8");
    console.log(`Wrote ${item}: ${dest}`);
    if (item === "opencode") {
      const pluginDir = join(outputRoot, ".retaindb", "opencode");
      mkdirSync(pluginDir, { recursive: true });
      const pluginDest = join(pluginDir, "retaindb-capture.ts");
      writeFileSync(pluginDest, opencodePluginSource(baseUrl, project), "utf8");
      console.log(`Wrote opencode plugin: ${pluginDest}`);
    }
  }
  console.log("");
  console.log("Merge these snippets into the agent config. Start RetainDB Local with `npx -y @retaindb/local` first.");
}

function backupPath(path: string) {
  return `${path}.retaindb-backup-${Date.now()}`;
}

function readJsonFile(path: string) {
  if (!existsSync(path)) return {};
  try {
    return JSON.parse(readFileSync(path, "utf8")) as Record<string, any>;
  } catch {
    return {};
  }
}

function writeJsonFile(path: string, value: unknown) {
  mkdirSync(dirname(path), { recursive: true });
  if (existsSync(path)) writeFileSync(backupPath(path), readFileSync(path));
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function stripTomlBlock(toml: string) {
  const lines = toml.split(/\r?\n/);
  const out: string[] = [];
  let skipping = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "[mcp_servers.retaindb]" || trimmed === "[mcp_servers.retaindb.env]") {
      skipping = true;
      continue;
    }
    if (skipping && trimmed.startsWith("[") && trimmed !== "[mcp_servers.retaindb.env]") skipping = false;
    if (!skipping) out.push(line);
  }
  return out.join("\n").trimEnd();
}

function installCodexConfig(baseUrl: string, project: string) {
  const dir = join(homedir(), ".codex");
  const path = join(dir, "config.toml");
  mkdirSync(dir, { recursive: true });
  const current = existsSync(path) ? readFileSync(path, "utf8") : "";
  if (current) writeFileSync(backupPath(path), current, "utf8");
  const block = connectConfig("codex", baseUrl, project);
  writeFileSync(path, `${stripTomlBlock(current)}\n\n${block}\n`, "utf8");
  return path;
}

function installClaudeConfig(baseUrl: string, project: string) {
  const path = join(homedir(), ".claude.json");
  const config = readJsonFile(path);
  config.mcpServers = {
    ...(config.mcpServers || {}),
    retaindb: { command: "npx", args: ["-y", "@retaindb/local", "mcp"], env: { ...mcpEnv(baseUrl, project), RETAINDB_AUTO_CONTEXT: "true", RETAINDB_TOKEN_BUDGET: "1200", RETAINDB_COMPRESS_TOOL_OUTPUT: "true" } },
  };
  writeJsonFile(path, config);
  const settingsPath = join(homedir(), ".claude", "settings.json");
  const settings = readJsonFile(settingsPath);
  settings.hooks = {
    ...(settings.hooks || {}),
    UserPromptSubmit: [{ hooks: [{ type: "command", command: "npx -y @retaindb/local hook --kind=prompt --agent=claude-code", env: { RETAINDB_AUTO_CONTEXT: "true", RETAINDB_TOKEN_BUDGET: "1200" } }] }],
    PreToolUse: [{ hooks: [{ type: "command", command: "npx -y @retaindb/local hook --kind=pre_tool_use --agent=claude-code", env: { RETAINDB_AUTO_CONTEXT: "true", RETAINDB_TOKEN_BUDGET: "1200" } }] }],
    PostToolUse: [{ hooks: [{ type: "command", command: "npx -y @retaindb/local hook --kind=tool_result --agent=claude-code", env: { RETAINDB_COMPRESS_TOOL_OUTPUT: "true", RETAINDB_TOKEN_BUDGET: "1200" } }] }],
    PreCompact: [{ hooks: [{ type: "command", command: "npx -y @retaindb/local hook --kind=pre_compact --agent=claude-code", env: { RETAINDB_AUTO_CONTEXT: "true", RETAINDB_TOKEN_BUDGET: "1200" } }] }],
    Stop: [{ hooks: [{ type: "command", command: "npx -y @retaindb/local hook --kind=session_end --agent=claude-code" }] }],
  };
  writeJsonFile(settingsPath, settings);
  return `${path}, ${settingsPath}`;
}

function findOpenCodeConfigPath(): string {
  const xdg = process.env.XDG_CONFIG_HOME;
  if (xdg) return join(xdg, "opencode", "opencode.json");
  if (process.platform === "win32" && process.env.APPDATA) {
    return join(process.env.APPDATA, "opencode", "opencode.json");
  }
  return join(homedir(), ".config", "opencode", "opencode.json");
}

function installOpenCodeConfig(baseUrl: string, project: string): string {
  // First write the plugin file
  const outputRoot = process.env.INIT_CWD || process.cwd();
  const pluginDir = join(outputRoot, ".retaindb", "opencode");
  mkdirSync(pluginDir, { recursive: true });
  const pluginDest = join(pluginDir, "retaindb-capture.ts");
  writeFileSync(pluginDest, opencodePluginSource(baseUrl, project), "utf8");
  // Now register the plugin in opencode.json
  const configPath = findOpenCodeConfigPath();
  mkdirSync(dirname(configPath), { recursive: true });
  const current = readJsonFile(configPath);
  const plugins = (current.plugins || []) as unknown[];
  const pluginRef = pluginDest;
  if (!plugins.some((p) => String(p).includes("retaindb-capture"))) {
    plugins.push(pluginRef);
  }
  current.plugins = plugins;
  writeJsonFile(configPath, current);
  return `${configPath} (+ ${pluginDest})`;
}

async function runMcp() {
  const [{ McpServer }, { StdioServerTransport }, zod] = await Promise.all([
    import("@modelcontextprotocol/sdk/server/mcp.js"),
    import("@modelcontextprotocol/sdk/server/stdio.js"),
    import("zod"),
  ]);
  const z = zod.z;
  const baseUrl = (process.env.RETAINDB_BASE_URL || `http://localhost:${DEFAULT_PORT}`).replace(/\/+$/, "");
  const project = process.env.RETAINDB_PROJECT || DEFAULT_PROJECT;
  const server = new McpServer({ name: "retaindb-local", version: "0.1.0" });
  const post = async (path: string, body: Record<string, unknown>) => {
    const res = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project, ...body }),
    });
    return res.json();
  };
  const get = async (path: string) => {
    const res = await fetch(`${baseUrl}${path}`);
    return res.json();
  };
  const out = (value: unknown) => ({ content: [{ type: "text" as const, text: typeof value === "string" ? value : JSON.stringify(value, null, 2) }] });

  server.tool("context", "Retrieve packed RetainDB Local context for the current task.", {
    query: z.string(),
    user_id: z.string().optional(),
    session_id: z.string().optional(),
    top_k: z.number().optional(),
  }, async (args: any) => out(await post("/v1/context/query", args)));

  server.tool("context_pack", "Build a small token-budgeted context pack from memory, relevant files, code map, and compressed tool output.", {
    query: z.string(),
    cwd: z.string().optional(),
    files: z.array(z.string()).optional(),
    tool_output: z.string().optional(),
    previous_context_hash: z.string().optional(),
    token_budget: z.number().optional(),
    include_memory: z.boolean().optional(),
  }, async (args: any) => out(await post("/v1/context/pack", args)));

  server.tool("context_delta", "Return only what changed since a previous RetainDB context pack.", {
    query: z.string(),
    previous_context_hash: z.string(),
    cwd: z.string().optional(),
    files: z.array(z.string()).optional(),
    tool_output: z.string().optional(),
    token_budget: z.number().optional(),
  }, async (args: any) => out(await post("/v1/context/delta", args)));

  server.tool("compress_output", "Compress terminal, test, build, or tool output while keeping failures and stack traces.", {
    output: z.string(),
    token_budget: z.number().optional(),
  }, async (args: any) => out(await post("/v1/context/compress-output", args)));

  server.tool("code_map", "Build a compact map of relevant files and symbols for a coding task.", {
    query: z.string().optional(),
    cwd: z.string().optional(),
    limit: z.number().optional(),
  }, async (args: any) => out(await post("/v1/context/code-map", args)));

  server.tool("files_sync", "Generate the repo-local .retaindb/files company brain for multi-agent handoff.", {
    cwd: z.string().optional(),
  }, async (args: any) => out(await post("/v1/filesystem/sync", args)));

  server.tool("files_list", "List RetainDB Local brain files under .retaindb/files.", {
    cwd: z.string().optional(),
    include_contents: z.boolean().optional(),
    limit: z.number().optional(),
  }, async (args: any) => {
    const params = new URLSearchParams({
      project,
      ...(args.cwd ? { cwd: args.cwd } : {}),
      includeContents: args.include_contents ? "true" : "false",
      limit: String(args.limit || 250),
    });
    return out(await get(`/v1/filesystem?${params.toString()}`));
  });

  server.tool("files_read", "Read a RetainDB Local brain file such as /README.md or /memories/decisions.md.", {
    path: z.string(),
    cwd: z.string().optional(),
  }, async (args: any) => {
    const params = new URLSearchParams({
      project,
      path: args.path,
      includeContents: "true",
      ...(args.cwd ? { cwd: args.cwd } : {}),
    });
    return out(await get(`/v1/filesystem?${params.toString()}`));
  });

  server.tool("files_write", "Write an append-only agent note or handoff into .retaindb/files/inbox and memory.", {
    content: z.string(),
    title: z.string().optional(),
    kind: z.enum(["note", "handoff", "decision", "task", "file_edit", "failure"]).optional(),
    agent_id: z.string().optional(),
    to_agent_id: z.string().optional(),
    session_id: z.string().optional(),
    task_id: z.string().optional(),
    files: z.array(z.string()).optional(),
    cwd: z.string().optional(),
  }, async (args: any) => out(await post("/v1/filesystem/write", {
    ...args,
    agentId: args.agent_id,
    toAgentId: args.to_agent_id,
    sessionId: args.session_id,
    taskId: args.task_id,
  })));

  server.tool("remember", "Save a durable memory to RetainDB Local.", {
    content: z.string(),
    memory_type: z.string().optional(),
    user_id: z.string().optional(),
    session_id: z.string().optional(),
    importance: z.number().optional(),
  }, async (args: any) => out(await post("/v1/memory", args)));

  server.tool("recall", "Search RetainDB Local memories.", {
    query: z.string(),
    user_id: z.string().optional(),
    session_id: z.string().optional(),
    top_k: z.number().optional(),
  }, async (args: any) => out(await post("/v1/memory/search", args)));

  server.tool("session_history", "List memories for a RetainDB Local session.", {
    session_id: z.string(),
    limit: z.number().optional(),
  }, async (args: any) => out(await get(`/v1/memory/session/${encodeURIComponent(args.session_id)}?project=${encodeURIComponent(project)}&limit=${encodeURIComponent(String(args.limit || 30))}`)));

  server.tool("handoff", "Create a session handoff from RetainDB Local memories.", {
    session_id: z.string(),
    title: z.string().optional(),
    expiry_days: z.number().optional(),
  }, async (args: any) => out(await post("/v1/context/share", args)));

  server.tool("forget", "Delete or deactivate a RetainDB Local memory by ID.", {
    memory_id: z.string(),
  }, async (args: any) => {
    const res = await fetch(`${baseUrl}/v1/memory/${encodeURIComponent(args.memory_id)}`, { method: "DELETE" });
    return out(await res.json());
  });

  await server.connect(new StdioServerTransport());
  console.error(`[retaindb-local] MCP running on stdio (${baseUrl}, project: ${project})`);
}

function installUserConfigs(target = "all") {
  const baseUrl = process.env.RETAINDB_BASE_URL || `http://localhost:${DEFAULT_PORT}`;
  const project = process.env.RETAINDB_PROJECT || DEFAULT_PROJECT;
  const targets = target === "all" ? ["codex", "claude-code", "opencode"] : [target];
  const installed: string[] = [];
  for (const item of targets) {
    if (item === "codex") installed.push(installCodexConfig(baseUrl, project));
    if (item === "claude-code") installed.push(installClaudeConfig(baseUrl, project));
    if (item === "opencode") installed.push(installOpenCodeConfig(baseUrl, project));
  }
  console.log(JSON.stringify({ installed, note: "Backups were written next to changed config files." }, null, 2));
}

async function runHook() {
  const baseUrl = process.env.RETAINDB_BASE_URL || `http://localhost:${DEFAULT_PORT}`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const apiKey = process.env.RETAINDB_API_KEY;
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  const kind = arg("kind") || "tool_result";
  const agentId = arg("agent") || process.env.RETAINDB_AGENT_ID || "agent";
  const sessionId = arg("session") || process.env.RETAINDB_SESSION_ID || `session-${new Date().toISOString().slice(0, 10)}`;
  const summaryArg = arg("summary");
  let stdin = "";
  if (!process.stdin.isTTY) {
    for await (const chunk of process.stdin) stdin += chunk;
  }
  let data: any = {};
  try {
    data = stdin.trim() ? JSON.parse(stdin) : {};
  } catch {
    data = { raw: stdin.slice(0, 12000) };
  }
  const project = process.env.RETAINDB_PROJECT || data.project || data.cwd || DEFAULT_PROJECT;
  const autoContext = process.env.RETAINDB_AUTO_CONTEXT !== "false" && ["prompt", "pre_tool_use", "pre_compact"].includes(kind);
  if (autoContext) {
    try {
      const res = await fetch(`${baseUrl}/v1/context/pack`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          project,
          query: summaryArg || data.prompt || data.message || data.tool_input || data.tool_name || kind,
          cwd: data.cwd || process.cwd(),
          files: data.files || data.filePaths || data.paths || [],
          previous_context_hash: data.previous_context_hash || data.context_hash,
          token_budget: Number(process.env.RETAINDB_TOKEN_BUDGET || 1200),
        }),
      });
      if (res.ok) {
        const body = await res.json();
        const text = body.delta_context || body.context;
        if (text) process.stdout.write(`\nRetainDB compact context:\n${text}\n`);
      }
    } catch {}
  }
  if (process.env.RETAINDB_COMPRESS_TOOL_OUTPUT !== "false" && (data.tool_output || data.tool_response)) {
    try {
      const raw = String(data.tool_output || data.tool_response);
      if (raw.length > 1200) {
        const res = await fetch(`${baseUrl}/v1/context/compress-output`, {
          method: "POST",
          headers,
          body: JSON.stringify({ output: raw, token_budget: Number(process.env.RETAINDB_TOKEN_BUDGET || 1200) }),
        });
        if (res.ok) {
          const body = await res.json();
          data.tool_output = body.compressed || raw.slice(0, 4000);
          data.tool_response = data.tool_output;
          data.output_compressed = true;
          data.raw_output_hash = createHash("sha256").update(raw).digest("hex");
        }
      }
    } catch {}
  }
  const summary = summaryArg || stdin.slice(0, 4000).trim() || `${agentId} ${kind}`;
  await fetch(`${baseUrl}/v1/agent-events`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      project,
      session_id: sessionId,
      agent_id: agentId,
      event: { kind, summary, details: JSON.stringify(data).slice(0, 12000), timestamp: now() },
    }),
  }).catch(() => {});
}

async function runDemo() {
  ensureConnectorsRegistered();
  const runtime = new LocalMemoryRuntime();
  const project = "demo";
  const baseUrl = process.env.RETAINDB_BASE_URL || `http://localhost:${DEFAULT_PORT}`;

  console.log("RetainDB Local Demo\n");

  // 1. Seed a couple of agent memories so the brain has context.
  await runtime.ingestSession({
    project,
    session_id: "demo-session-1",
    user_id: "demo-user",
    agent_id: "demo",
    events: [
      { kind: "decision", summary: "RetainDB uses jose middleware for JWT auth in src/middleware/auth.ts", salience: "high" },
      { kind: "constraint", summary: "Prefer edge-compatible auth libraries over jsonwebtoken", salience: "medium" },
      { kind: "outcome", summary: "Rate limiting should reuse existing Hono middleware instead of adding Redis", salience: "high" },
    ],
  });
  console.log(`  1/4 seeded ${3} demo memories\n`);

  // 2. Add a web source (example.com, no auth needed).
  const sources = new SourceStore(RETAINDB_HOME);
  const web = sources.create({
    type: "web",
    name: "demo-web",
    project,
    config: { url: "https://example.com/" },
  });
  console.log(`  2/4 created source: ${web.id} (web → https://example.com/)\n`);

  // 3. Sync the web source so the brain has external content.
  const result = await runSourceSync({
    source: web,
    ingest: (input) =>
      runtime.addMemory({
        project,
        content: input.content,
        memory_type: input.memory_type,
        importance: input.importance ?? 0.7,
        confidence: input.confidence ?? 0.8,
        metadata: {
          source_id: web.id,
          source_type: "web",
          source_title: "Example Domain",
          external_id: input.metadata?.external_id,
          citation: input.metadata?.citation,
        },
      }),
    onProgress: (p) => {
      if (p.stage !== "done") process.stderr.write(`    ${p.stage} ${p.current}/${p.total} ${p.message}\n`);
    },
  });
  sources.update(web.id, {
    status: result.errors.length > 0 && result.memories_created === 0 ? "error" : "connected",
    last_synced_at: new Date().toISOString(),
    last_sync_status: result.errors.length > 0 ? (result.memories_created > 0 ? "partial" : "error") : "ok",
    last_sync_summary: { documents_indexed: result.documents_indexed, memories_created: result.memories_created, errors: result.errors.length, duration_ms: result.duration_ms },
    last_error: result.errors.length > 0 ? result.errors.slice(0, 3).join(" | ") : undefined,
  });
  console.log(`  3/4 synced web source: ${result.documents_indexed} docs, ${result.memories_created} memories\n`);

  // 4. Ask the brain a question, then show the feedAgent output.
  const allMemories = (runtime as any).data.memories
    .filter((m: any) => m.project === project && m.active)
    .map((m: any) => ({
      ...m,
      source_id: m.metadata?.source_id,
      source_type: m.metadata?.source_type,
      source_title: m.metadata?.source_title,
    }));
  const brain = buildCompanyBrain({ project, memories: allMemories, maxTokens: 4000 });
  const ask = await askBrain({
    project,
    query: "what is an example domain used for",
    top_k: 8,
    maxTokens: 2000,
    includeAgentMemories: true,
    search: async (input) => {
      const hits = await runtime.search(input);
      return hits.map((r: any) => ({
        memory: {
          ...r.memory,
          source_id: r.memory.metadata?.source_id,
          source_type: r.memory.metadata?.source_type,
          source_title: r.memory.metadata?.source_title,
        },
        score: r.score,
      }));
    },
  });
  const feed = feedAgent({
    project,
    query: "what is an example domain used for",
    messages: [{ role: "user", content: "What does https://example.com/ say about example domains?" }],
    maxContextTokens: 2000,
    includeAgentMemories: true,
    ask,
  });

  console.log("  4/4 company brain dump (grouped by source):\n");
  console.log(brain.text.slice(0, 1200) + (brain.text.length > 1200 ? "\n  … (truncated)\n" : "\n"));

  console.log("  Ask result:\n");
  console.log(`    query: ${ask.query}`);
  console.log(`    hits: ${ask.hits}`);
  console.log(`    citations: ${ask.citations.length}\n`);

  console.log("  FeedAgent system prompt (first 16 lines):\n");
  const lines = feed.system_prompt.split("\n");
  console.log(lines.slice(0, 16).map((l) => `    ${l}`).join("\n"));
  if (lines.length > 16) console.log("    … (truncated)");
  console.log(`\n    → ${feed.citations.length} citations available`);
  console.log(`    → ${feed.messages.length} messages (system + user)`);
  console.log(`\n  ✦ Demo complete. Run \`retaindb start\` to keep the server running,` +
    ` then add real sources with \`retaindb add github foo/bar --sync\`.`);
}

function percentile(values: number[], p: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
}

async function runBenchmark() {
  const runtime = new LocalMemoryRuntime();
  const project = "benchmark";
  const fixtures = [
    { q: "jose jwt auth middleware", expect: "jose middleware" },
    { q: "n+1 query dashboard batching", expect: "N+1 query" },
    { q: "rate limiting Hono middleware", expect: "Hono middleware" },
    { q: "edge-compatible auth library", expect: "edge-compatible" },
  ];
  await runtime.ingestSession({
    project,
    session_id: "bench-session-1",
    agent_id: "benchmark",
    events: [
      { kind: "decision", summary: "Auth uses jose middleware in src/middleware/auth.ts because it is edge-compatible", salience: "high" },
      { kind: "fix", summary: "N+1 query was fixed by batching project lookups before rendering dashboard rows", salience: "high" },
      { kind: "workflow", summary: "Rate limiting should reuse existing Hono middleware and avoid adding Redis locally", salience: "high" },
    ],
  });
  const latencies: number[] = [];
  let hits = 0;
  for (const fixture of fixtures) {
    const started = performance.now();
    const results = await runtime.search({ project, query: fixture.q, top_k: 5 });
    latencies.push(performance.now() - started);
    if (results.some((result) => result.content.toLowerCase().includes(fixture.expect.toLowerCase()))) hits += 1;
  }
  const report = {
    queries: fixtures.length,
    top5_hit_rate: hits / fixtures.length,
    p50_ms: Number(percentile(latencies, 50).toFixed(2)),
    p90_ms: Number(percentile(latencies, 90).toFixed(2)),
    p99_ms: Number(percentile(latencies, 99).toFixed(2)),
    max_ms: Number(Math.max(...latencies).toFixed(2)),
    embedding_provider: EMBEDDING_PROVIDER,
    retrieval: "bm25+vector+graph+rrf+rerank",
    corpus: "retaindb-local-smoke-v1",
    created_at: now(),
  };
  const report_path = writeBenchmarkReport(report);
  appendJournal("benchmark.completed", { report_path, ...report });
  Object.assign(report, { report_path });
  console.log(JSON.stringify(report, null, 2));
}

async function installEmbeddings() {
  const previous = process.env.RETAINDB_EMBEDDING_PROVIDER;
  process.env.RETAINDB_EMBEDDING_PROVIDER = "local-transformers";
  const started = performance.now();
  try {
    const vector = await transformerEmbedding("RetainDB local embedding warmup");
    const result = {
      ok: true,
      provider: "local-transformers",
      model: process.env.RETAINDB_EMBEDDING_MODEL || "Xenova/all-MiniLM-L6-v2",
      dimensions: vector.length,
      warmup_ms: Number((performance.now() - started).toFixed(2)),
      next: "Set RETAINDB_EMBEDDING_PROVIDER=local-transformers and run `retaindb reembed`.",
    };
    appendJournal("embeddings.installed", result);
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    const result = {
      ok: false,
      provider: "local-transformers",
      error: error instanceof Error ? error.message : String(error),
      fallback: "hash",
      next: "Run `pnpm rebuild onnxruntime-node` or keep the default hash embeddings.",
    };
    appendJournal("embeddings.install_failed", result);
    console.log(JSON.stringify(result, null, 2));
  } finally {
    if (previous === undefined) delete process.env.RETAINDB_EMBEDDING_PROVIDER;
    else process.env.RETAINDB_EMBEDDING_PROVIDER = previous;
  }
}

function* jsonlFiles(root: string): Generator<string> {
  if (!existsSync(root)) return;
  const stat = statSync(root);
  if (stat.isFile()) {
    if (root.endsWith(".jsonl")) yield root;
    return;
  }
  if (!stat.isDirectory()) return;
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const full = join(root, entry.name);
    if (entry.isDirectory()) yield* jsonlFiles(full);
    else if (entry.isFile() && entry.name.endsWith(".jsonl")) yield full;
  }
}

function textFromJsonlEntry(entry: Record<string, unknown>) {
  const message = entry.message && typeof entry.message === "object" ? entry.message as Record<string, unknown> : undefined;
  const content = message?.content ?? entry.content ?? entry.text ?? entry.prompt;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (typeof part === "string") return part;
      if (part && typeof part === "object") {
        const record = part as Record<string, unknown>;
        return typeof record.text === "string" ? record.text : "";
      }
      return "";
    }).filter(Boolean).join("\n");
  }
  return "";
}

async function importJsonl(pathArg?: string) {
  const root = resolve(pathArg || join(homedir(), ".claude", "projects"));
  const runtime = new LocalMemoryRuntime();
  let files = 0;
  let memories = 0;
  for (const file of jsonlFiles(root)) {
    files += 1;
    const sessionId = `jsonl-${createHash("sha1").update(file).digest("hex").slice(0, 12)}`;
    const lines = readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean);
    const messages: Message[] = [];
    const events: WorkEvent[] = [];
    for (const line of lines) {
      try {
        const entry = JSON.parse(line) as Record<string, unknown>;
        const role = String((entry.message as any)?.role || entry.role || entry.type || "event");
        const text = redactSecrets(textFromJsonlEntry(entry)).trim();
        if (text) messages.push({ role, content: text, timestamp: String(entry.timestamp || now()) });
        const toolName = String(entry.tool_name || entry.toolName || (entry.tool_use as any)?.name || "");
        if (toolName) {
          events.push({
            kind: "tool_result",
            summary: `${toolName} used during imported session`,
            details: JSON.stringify(redactUnknown(entry)).slice(0, 12000),
            toolName,
            timestamp: String(entry.timestamp || now()),
          });
        }
      } catch {
        // Skip malformed transcript lines.
      }
    }
    const result = await runtime.ingestSession({
      project: DEFAULT_PROJECT,
      session_id: sessionId,
      agent_id: "import-jsonl",
      messages,
      events,
    });
    memories += result.memories_created;
  }
  console.log(JSON.stringify({ imported: true, root, files, memories, store: STORE_PATH }, null, 2));
}

async function runSourcesCommand() {
  ensureConnectorsRegistered();
  const runtime = new LocalMemoryRuntime();
  const store = new SourceStore(RETAINDB_HOME);
  const subcommand = process.argv[3] || "list";
  const project = arg("project") || process.env.RETAINDB_PROJECT || DEFAULT_PROJECT;

  if (subcommand === "list") {
    console.log(JSON.stringify({ sources: store.list(project) }, null, 2));
    return;
  }
  if (subcommand === "get") {
    const id = process.argv[4] || arg("id");
    if (!id) throw new Error("Usage: retaindb sources get <id>");
    const s = store.get(id);
    if (!s) throw new Error(`Source ${id} not found`);
    console.log(JSON.stringify(s, null, 2));
    return;
  }
  if (subcommand === "delete") {
    const id = process.argv[4] || arg("id");
    if (!id) throw new Error("Usage: retaindb sources delete <id>");
    const ok = store.delete(id);
    console.log(JSON.stringify({ deleted: ok, id }, null, 2));
    return;
  }
  if (subcommand === "sync") {
    const id = process.argv[4] || arg("id");
    if (!id) throw new Error("Usage: retaindb sources sync <id>");
    const source = store.get(id);
    if (!source) throw new Error(`Source ${id} not found`);
    store.update(id, { status: "syncing", last_error: undefined });
    const t0 = Date.now();
    const result = await runSourceSync({
      source,
      ingest: (input) =>
        runtime.addMemory({
          project: source.project,
          content: input.content,
          memory_type: input.memory_type,
          importance: input.importance,
          confidence: input.confidence,
          agent_id: input.agent_id,
          session_id: input.session_id,
          metadata: input.metadata,
        }),
      onProgress: (p) => {
        if (p.stage === "extracting" || p.stage === "indexing") {
          process.stderr.write(`  ${p.stage} ${p.current}/${p.total} ${p.message}\n`);
        } else if (p.stage !== "done") {
          process.stderr.write(`  ${p.stage} ${p.message}\n`);
        }
      },
    });
    store.update(id, {
      status: result.errors.length > 0 && result.memories_created === 0 ? "error" : "connected",
      last_synced_at: new Date().toISOString(),
      last_sync_status: result.errors.length > 0 ? (result.memories_created > 0 ? "partial" : "error") : "ok",
      last_sync_summary: {
        documents_indexed: result.documents_indexed,
        memories_created: result.memories_created,
        errors: result.errors.length,
        duration_ms: result.duration_ms,
      },
      last_error: result.errors.length > 0 ? result.errors.slice(0, 3).join(" | ") : undefined,
    });
    console.log(JSON.stringify({ source_id: id, result, duration_ms: Date.now() - t0, citations: result.citations.slice(0, 10) }, null, 2));
    return;
  }
  throw new Error("Usage: retaindb sources list|get|delete|sync");
}

function formatInspectReport(report: ReturnType<LocalMemoryRuntime["inspect"]>) {
  const lines: string[] = [];
  const summary = report.summary;
  lines.push(`RetainDB memory inspection (${report.project})`);
  lines.push("");
  lines.push(`Score: ${report.score}/100`);
  lines.push(`Active memories: ${summary.active_memories}`);
  lines.push(`Durable memories: ${summary.durable_memories} (${Math.round(summary.durable_ratio * 100)}%)`);
  lines.push(`Sessions: ${summary.sessions}`);
  lines.push(`Recalled memories: ${summary.recalled_memories} (${Math.round(summary.recall_ratio * 100)}%)`);
  lines.push(`Average quality: ${summary.average_quality}`);

  const byType = Object.entries(report.counts.by_type);
  if (byType.length > 0) {
    lines.push("");
    lines.push("Types:");
    for (const [type, count] of byType.slice(0, 10)) lines.push(`  ${type}: ${count}`);
  }

  if (report.top_concepts.length > 0) {
    lines.push("");
    lines.push("Top concepts:");
    lines.push(`  ${report.top_concepts.slice(0, 12).map((item) => `${item.name}(${item.count})`).join(", ")}`);
  }

  if (report.reusable.length > 0) {
    lines.push("");
    lines.push("Reusable memory:");
    for (const memory of report.reusable.slice(0, 5)) {
      lines.push(`  [${memory.type}] ${memory.content.split("\n")[0].slice(0, 130)}`);
    }
  }

  const weakCount = report.risks.weak_memories.length;
  const staleCount = report.risks.stale_memories.length;
  const duplicateCount = report.risks.duplicate_candidates.length;
  if (weakCount || staleCount || duplicateCount) {
    lines.push("");
    lines.push("Risks:");
    if (weakCount) lines.push(`  Weak low-signal memories: ${weakCount}`);
    if (staleCount) lines.push(`  Stale unrecalled memories: ${staleCount}`);
    if (duplicateCount) lines.push(`  Duplicate groups: ${duplicateCount}`);
  }

  lines.push("");
  lines.push("Next actions:");
  for (const item of report.recommendations) lines.push(`  - ${item}`);
  return lines.join("\n");
}

function runInspectCommand() {
  const runtime = new LocalMemoryRuntime();
  const project = arg("project") || process.env.RETAINDB_PROJECT;
  const report = runtime.inspect(project);
  if (process.argv.includes("--json") || process.argv.includes("--format=json")) {
    console.log(JSON.stringify(report, null, 2));
    return;
  }
  console.log(formatInspectReport(report));
}

async function runBrainCommand() {
  ensureConnectorsRegistered();
  const runtime = new LocalMemoryRuntime();
  const subcommand = process.argv[3] || "show";
  const project = arg("project") || process.env.RETAINDB_PROJECT || DEFAULT_PROJECT;

  if (subcommand === "show") {
    const project2 = slugify(project);
    const memories = (runtime as any).data.memories
      .filter((m: any) => m.project === project2 && m.active)
      .map((m: any) => ({
        ...m,
        source_id: m.metadata?.source_id,
        source_type: m.metadata?.source_type,
        source_title: m.metadata?.source_title,
        external_id: m.metadata?.external_id,
        url: m.metadata?.url,
        citation: m.metadata?.citation,
      }));
    const brain = buildCompanyBrain({ project: project2, memories, maxTokens: Number(arg("maxTokens") || 8000) });
    if (arg("format") === "text") {
      console.log(brain.text);
    } else {
      console.log(JSON.stringify(brain, null, 2));
    }
    return;
  }
  if (subcommand === "ask") {
    const query = process.argv.slice(4).join(" ") || arg("query");
    if (!query) throw new Error("Usage: retaindb brain ask <query>");
    const topK = Number(arg("top-k") || arg("topK") || 12);
    const maxTokens = Number(arg("max-tokens") || 2400);
    const includeAgent = arg("no-agent") !== "true";
    const project2 = slugify(project);
    const result = await askBrain({
      project: project2,
      query,
      top_k: topK,
      maxTokens,
      includeAgentMemories: includeAgent,
      search: async (input) => {
        const results = await runtime.search(input);
        return results.map((r: any) => ({
          memory: {
            ...r.memory,
            source_id: r.memory.metadata?.source_id,
            source_type: r.memory.metadata?.source_type,
            source_title: r.memory.metadata?.source_title,
            external_id: r.memory.metadata?.external_id,
            url: r.memory.metadata?.url,
            citation: r.memory.metadata?.citation,
          },
          score: r.score,
        }));
      },
    });
    if (arg("format") === "text") {
      console.log(result.context);
    } else {
      console.log(JSON.stringify(result, null, 2));
    }
    return;
  }
  if (subcommand === "feed") {
    const query = arg("query");
    const maxCtx = Number(arg("max-context-tokens") || 2400);
    const includeAgent = arg("no-agent") !== "true";
    const project2 = slugify(project);
    const messagesJson = arg("messages") || "[]";
    let messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [];
    try { messages = JSON.parse(messagesJson); } catch { /* ignore */ }
    if (messages.length === 0) {
      const stdin = await readStdin();
      if (stdin) {
        try { messages = JSON.parse(stdin); } catch { /* ignore */ }
      }
    }
    if (messages.length === 0) {
      throw new Error("Usage: retaindb brain feed --query <q> --messages '[{...}]'  (or pipe JSON to stdin)");
    }
    let ask;
    if (query) {
      ask = await askBrain({
        project: project2,
        query,
        top_k: Number(arg("top-k") || 12),
        maxTokens: maxCtx,
        includeAgentMemories: includeAgent,
        search: async (input) => {
          const results = await runtime.search(input);
          return results.map((r: any) => ({
            memory: {
              ...r.memory,
              source_id: r.memory.metadata?.source_id,
              source_type: r.memory.metadata?.source_type,
              source_title: r.memory.metadata?.source_title,
              external_id: r.memory.metadata?.external_id,
              url: r.memory.metadata?.url,
              citation: r.memory.metadata?.citation,
            },
            score: r.score,
          }));
        },
      });
    }
    const out = feedAgent({ project: project2, messages, query, maxContextTokens: maxCtx, includeAgentMemories: includeAgent, ask });
    if (arg("format") === "messages") {
      console.log(JSON.stringify(out.messages, null, 2));
    } else if (arg("format") === "system") {
      console.log(out.system_prompt);
    } else {
      console.log(JSON.stringify(out, null, 2));
    }
    return;
  }
  throw new Error("Usage: retaindb brain show|ask|feed");
}

async function readStdin(): Promise<string> {
  if (process.stdin.isTTY) return "";
  let s = "";
  for await (const chunk of process.stdin) s += chunk;
  return s.trim();
}

function flagName(camel: string): string {
  return "--" + camel.replace(/[A-Z]/g, (m) => "-" + m.toLowerCase());
}

function parseFlag(flag: string): string | string[] | number | boolean | undefined {
  for (let i = 3; i < process.argv.length; i++) {
    const a = process.argv[i];
    if (a === flag) {
      const next = process.argv[i + 1];
      if (next === undefined || next.startsWith("--")) return true;
      return next;
    }
    if (a.startsWith(flag + "=")) {
      return a.slice(flag.length + 1);
    }
  }
  return undefined;
}

function parseStringArrayFlag(flag: string): string[] | undefined {
  const raw = parseFlag(flag);
  if (raw === undefined) return undefined;
  if (Array.isArray(raw)) return raw as string[];
  if (typeof raw === "boolean") return [];
  return String(raw).split(",").map((s) => s.trim()).filter(Boolean);
}

function parseNumberFlag(flag: string): number | undefined {
  const raw = parseFlag(flag);
  if (raw === undefined) return undefined;
  if (typeof raw === "boolean") return undefined;
  const n = Number(raw);
  return isNaN(n) ? undefined : n;
}

function parseBooleanFlag(flag: string): boolean | undefined {
  const raw = parseFlag(flag);
  if (raw === undefined) return undefined;
  if (typeof raw === "boolean") return raw;
  if (raw === "true" || raw === "1" || raw === "yes") return true;
  if (raw === "false" || raw === "0" || raw === "no") return false;
  return undefined;
}

async function prompt(question: string, opts: { silent?: boolean } = {}): Promise<string> {
  if (!process.stdin.isTTY) {
    throw new Error(`Missing required value: ${question}\n(Re-run interactively, pass it as a flag, or pipe into stdin)`);
  }
  process.stdout.write(question);
  return await new Promise((resolve) => {
    let s = "";
    const onData = (chunk: Buffer) => {
      s += chunk.toString("utf8");
      if (s.endsWith("\n")) {
        process.stdin.removeListener("data", onData);
        process.stdin.pause();
        resolve(s.replace(/\r?\n$/, ""));
      }
    };
    process.stdin.resume();
    process.stdin.on("data", onData);
  });
}

function maskSecret(value: string): string {
  if (!value) return "";
  if (value.length <= 4) return "****";
  return "****" + value.slice(-4);
}

function printConnectorsList(): void {
  const descs = listConnectorDescriptors();
  console.log("Available connectors:");
  for (const d of descs) {
    console.log(`  ${d.type.padEnd(12)} ${d.requiresAuth ? "[auth]" : "[no-auth]"}  ${d.description}`);
  }
  console.log("\nRun `retaindb describe <type>` for the full config schema and an example.");
}

function printSchemaHuman(s: ReturnType<typeof getConnector> extends infer C ? (C extends { schema: () => infer S } ? S : never) : never): void {
  console.log(`${s.type} — ${s.summary}`);
  console.log(`auth: ${s.requiresAuth ? "required" : "not required"}`);
  if (s.positionalHint) console.log(`usage:  retaindb add ${s.type} ${s.positionalHint} [--flag=value ...]`);
  console.log("\nFields:");
  for (const f of s.fields) {
    const req = f.required ? "required" : `optional${f.default !== undefined ? ` (default: ${String(f.default)})` : ""}`;
    const sec = f.secret ? " [secret]" : "";
    const cli = f.cliFlag ? ` [--${f.cliFlag}]` : (f.positional ? ` (positional: ${f.positional})` : "");
    console.log(`  ${f.name.padEnd(18)} ${f.type.padEnd(10)} ${req}${sec}${cli}`);
    console.log(`    ${f.description}`);
  }
  console.log("\nExample:");
  console.log("  " + JSON.stringify(s.example, null, 2).split("\n").join("\n  "));
}

function runDescribeCommand(): void {
  ensureConnectorsRegistered();
  const type = process.argv[3];
  if (!type || type === "--help" || type === "-h" || type === "all") {
    printConnectorsList();
    return;
  }
  const provider = getConnector(type as any);
  if (!provider) {
    console.error(`Unknown connector: ${type}\nRun \`retaindb describe\` for a list.`);
    process.exit(1);
  }
  printSchemaHuman(provider.schema());
}

async function runAddCommand(): Promise<void> {
  ensureConnectorsRegistered();
  const type = process.argv[3];
  if (!type || type === "--help" || type === "-h") {
    printConnectorsList();
    console.log("\nUsage: retaindb add <type> [args...] [--sync] [--name=...] [--project=...]");
    return;
  }
  const provider = getConnector(type as any);
  if (!provider) {
    console.error(`Unknown connector: ${type}\nRun \`retaindb describe\` for a list.`);
    process.exit(1);
  }
  const schema = provider.schema();
  const positional = schema.fields.filter((f) => f.positional).sort((a, b) => {
    if (!a.positional) return 1;
    if (!b.positional) return -1;
    return schema.fields.indexOf(a) - schema.fields.indexOf(b);
  });
  const config: Record<string, unknown> = {};
  // Pull positional values from process.argv (after the connector type)
  const positionalValues: string[] = [];
  for (let i = 4; i < process.argv.length; i++) {
    const a = process.argv[i];
    if (a.startsWith("--")) break;
    positionalValues.push(a);
  }
  // github's owner/repo can be supplied as "owner/repo" (combined positional)
  if (type === "github" && positionalValues.length === 1 && positionalValues[0].includes("/")) {
    const [owner, ...rest] = positionalValues[0].split("/");
    config["owner"] = owner;
    config["repo"] = rest.join("/");
  } else {
    for (let i = 0; i < positional.length; i++) {
      const f = positional[i];
      const v = positionalValues[i];
      if (v) {
        config[f.name] = f.type === "string[]" ? v.split(",").map((s) => s.trim()).filter(Boolean) : v;
      }
    }
  }
  // Pull named flags
  for (const f of schema.fields) {
    if (config[f.name] !== undefined) continue;
    const flag = "--" + (f.cliFlag || f.name);
    if (f.type === "string[]") {
      const v = parseStringArrayFlag(flag);
      if (v !== undefined) config[f.name] = v;
    } else if (f.type === "number") {
      const v = parseNumberFlag(flag);
      if (v !== undefined) config[f.name] = v;
    } else if (f.type === "boolean") {
      const v = parseBooleanFlag(flag);
      if (v !== undefined) config[f.name] = v;
    } else {
      const v = parseFlag(flag);
      if (v !== undefined && typeof v !== "boolean") config[f.name] = v;
    }
  }
  // Apply defaults
  for (const f of schema.fields) {
    if (config[f.name] === undefined && f.default !== undefined) config[f.name] = f.default;
  }
  // Prompt for missing required fields (TTY only)
  for (const f of schema.fields) {
    if (config[f.name] === undefined && f.required) {
      const value = await prompt(`${f.name} (${f.description}): `, { silent: f.secret });
      if (!value) {
        console.error(`\n  ✗ ${f.name} is required — aborting.`);
        process.exit(2);
      }
      if (f.type === "number") {
        const n = Number(value);
        if (isNaN(n)) {
          console.error(`\n  ✗ ${f.name} must be a number, got: ${value}`);
          process.exit(2);
        }
        config[f.name] = n;
      } else if (f.type === "string[]") {
        config[f.name] = value.split(",").map((s) => s.trim()).filter(Boolean);
      } else {
        config[f.name] = value;
      }
    }
  }
  // Re-validate via the connector
  const validation = provider.validateConfig(config);
  if (!validation.ok) {
    console.error(`  ✗ ${validation.error}`);
    console.error(`\n  hint: run \`retaindb describe ${type}\` for the full schema.`);
    process.exit(2);
  }
  // Echo the (secret-redacted) config so the user can confirm
  console.log("  ✓ config valid");
  const redacted: Record<string, unknown> = {};
  for (const f of schema.fields) {
    const v = config[f.name];
    if (v === undefined) continue;
    redacted[f.name] = f.secret && typeof v === "string" ? maskSecret(v) : v;
  }
  console.log("  config:", JSON.stringify(redacted));
  // POST to the local server
  const baseUrl = process.env.RETAINDB_BASE_URL || `http://localhost:${DEFAULT_PORT}`;
  const project = arg("project") || process.env.RETAINDB_PROJECT || DEFAULT_PROJECT;
  const explicitName = arg("name");
  const name = explicitName || `${type}-${Date.now().toString(36).slice(-6)}`;
  const res = await fetch(`${baseUrl}/v1/sources`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, name, project, config }),
  });
  const body = await res.json();
  if (!res.ok) {
    console.error(`  ✗ server returned ${res.status}: ${JSON.stringify(body)}`);
    process.exit(1);
  }
  const id = body.id;
  console.log(`  ✓ source created: ${id} (${name})`);
  if (process.argv.includes("--sync") || process.argv.includes("--sync-now")) {
    console.log("  ↳ syncing now…");
    const syncRes = await fetch(`${baseUrl}/v1/sources/${id}/sync`, { method: "POST" });
    const syncBody = await syncRes.json();
    if (!syncRes.ok) {
      console.error(`  ✗ sync failed: ${JSON.stringify(syncBody)}`);
      process.exit(1);
    }
    const r = syncBody.result || {};
    console.log(`  ✓ sync done: docs=${r.documents_indexed} memories=${r.memories_created} errors=${(r.errors || []).length} duration=${r.duration_ms}ms`);
  } else {
    console.log(`  ↳ run \`retaindb sources sync ${id}\` (or re-run with --sync) to ingest.`);
  }
}

function runHelpCommand(): void {
  console.log(`RetainDB Local — persistent memory for coding agents.

Usage:
  retaindb <command> [args]

Core:
  start [--port=N]                          Start the local server + viewer (default).
  mcp                                      Run the local server's MCP bridge on stdio.
  status | doctor | inspect | benchmark | demo
                                           Inspect health, stats, and smoke tests.
  files sync|list|read|write               Repo-local .retaindb/files brain.

Sources (connectors -> memory):
  connectors                                List registered connectors (text).
  describe <type|all>                       Print the full config schema + example.
  add <type> [args] [--sync]                Create a source. Positional + flags per type.
  sources list|get|delete|sync              Mirror of the HTTP routes.

Agent context:
  brain show [--project=...] [--format=text]   Dump the whole company brain, grouped by source.
  brain ask <query> [--top-k=N] [--max-tokens=N]   Search the brain; returns context + citations.
  brain feed --query=... --messages=...      LLM-ready system prompt + message list.

  connect [all|codex|claude-code|opencode]   Write agent-bridge snippets (unchanged).

Memory:
  inspect [--project=...] [--json]           Score memory quality and show cleanup actions.
  memory write ...                            Alias for POST /v1/memory.

Help:
  help                                      This screen.
  describe <type>                           Schema + example for one connector.
  <command> --help                          Per-command help (where available).

Environment:
  RETAINDB_HOME       Local data directory (default: ~/.retaindb).
  RETAINDB_BASE_URL   Server URL for CLI calls (default: http://localhost:<port>).
  RETAINDB_PORT       Server port (default: 3111).
  RETAINDB_PROJECT    Default project name (default: "default").
  RETAINDB_KEY        API key for the @retaindb/sdk client (local accepts any non-empty).
`);
}

async function runFilesCommand() {
  const runtime = new LocalMemoryRuntime();
  const subcommand = process.argv[3] || "sync";
  const project = arg("project") || process.env.RETAINDB_PROJECT || DEFAULT_PROJECT;
  const cwd = arg("cwd") || process.cwd();
  if (subcommand === "sync") {
    console.log(JSON.stringify(syncBrainFilesystem({ cwd, project, snapshot: runtime.snapshot(project) }), null, 2));
    return;
  }
  if (subcommand === "list") {
    syncBrainFilesystem({ cwd, project, snapshot: runtime.snapshot(project) });
    console.log(JSON.stringify(listBrainFileTree(cwd, process.argv.includes("--contents")), null, 2));
    return;
  }
  if (subcommand === "read") {
    const path = process.argv[4] || arg("path") || "/README.md";
    syncBrainFilesystem({ cwd, project, snapshot: runtime.snapshot(project) });
    const file = readBrainFile({ cwd, path, includeContents: true });
    process.stdout.write(file.content || "");
    return;
  }
  if (subcommand === "write") {
    let stdin = "";
    if (!process.stdin.isTTY) {
      for await (const chunk of process.stdin) stdin += chunk;
    }
    const content = arg("content") || stdin.trim();
    const kind = arg("kind") || "note";
    const written = writeAgentBrainFile({
      cwd,
      project,
      content,
      title: arg("title"),
      kind: kind as any,
      agentId: arg("agent") || process.env.RETAINDB_AGENT_ID || "agent",
      toAgentId: arg("to"),
      sessionId: arg("session") || process.env.RETAINDB_SESSION_ID,
      taskId: arg("task"),
      files: (arg("files") || "").split(",").map((item) => item.trim()).filter(Boolean),
    });
    const memory = await runtime.addMemory({
      project,
      content: written.memoryContent,
      memory_type: kind === "handoff" ? "session_summary" : kind,
      agent_id: arg("agent") || process.env.RETAINDB_AGENT_ID || "agent",
      session_id: arg("session") || process.env.RETAINDB_SESSION_ID,
      task_id: arg("task"),
      importance: kind === "handoff" || kind === "decision" ? 0.92 : 0.72,
      metadata: { source: "local_brain_file", path: written.path },
    });
    syncBrainFilesystem({ cwd, project, snapshot: runtime.snapshot(project) });
    console.log(JSON.stringify({ written, memory_id: memory.id, stored: memory.active }, null, 2));
    return;
  }
  throw new Error("Usage: retaindb files sync|list|read|write");
}

function startServer() {
  const runtime = new LocalMemoryRuntime();
  const app = createApp(runtime);
  const viewerPort = Number(process.env.RETAINDB_VIEWER_PORT || DEFAULT_PORT + 2);
  const viewer = createViewerApp(DEFAULT_PORT);
  serve({ fetch: app.fetch, port: DEFAULT_PORT });
  serve({ fetch: viewer.fetch, port: viewerPort });
  setInterval(() => {
    runtime.consolidate().catch(() => {});
  }, 60 * 60 * 1000).unref?.();
  console.log(`RetainDB Local running on http://localhost:${DEFAULT_PORT}`);
  console.log(`Viewer running on http://localhost:${viewerPort}`);
  console.log(`Store: ${STORE_PATH}`);
  console.log("MCP env: RETAINDB_BASE_URL=http://localhost:" + DEFAULT_PORT);
}

async function main() {
  const command = process.argv[2] || "start";
  if (command === "start" || command === "serve") return startServer();
  if (command === "mcp") return runMcp();
  if (command === "demo") return runDemo();
  if (command === "benchmark") return runBenchmark();
  if (command === "install-embeddings") return installEmbeddings();
  if (command === "connect") {
    const target = process.argv.slice(3).find((item) => !item.startsWith("--")) || "all";
    if (process.argv.includes("--install")) return installUserConfigs(target);
    return writeConnectSnippets(target);
  }
  if (command === "hook") return runHook();
  if (command === "files") return runFilesCommand();
  if (command === "sources") return runSourcesCommand();
  if (command === "brain") return runBrainCommand();
  if (command === "inspect") return runInspectCommand();
  if (command === "connectors") {
    ensureConnectorsRegistered();
    if (process.argv.includes("--json") || process.argv.includes("--format=json")) {
      console.log(JSON.stringify(listConnectorDescriptors(), null, 2));
    } else {
      printConnectorsList();
    }
    return;
  }
  if (command === "describe") return runDescribeCommand();
  if (command === "add") return runAddCommand();
  if (command === "help" || command === "--help" || command === "-h") return runHelpCommand();
  if (command === "import-jsonl") return importJsonl(process.argv[3]);
  if (command === "consolidate") {
    const runtime = new LocalMemoryRuntime();
    console.log(JSON.stringify(await runtime.consolidate(process.env.RETAINDB_PROJECT), null, 2));
    return;
  }
  if (command === "reembed") {
    const runtime = new LocalMemoryRuntime();
    console.log(JSON.stringify(await runtime.reembed(process.env.RETAINDB_PROJECT), null, 2));
    return;
  }
  if (command === "status") {
    const runtime = new LocalMemoryRuntime();
    console.log(JSON.stringify({ ok: true, ...runtime.snapshot(), api: `http://localhost:${DEFAULT_PORT}`, viewer: `http://localhost:${Number(process.env.RETAINDB_VIEWER_PORT || DEFAULT_PORT + 2)}` }, null, 2));
    return;
  }
  if (command === "doctor") {
    const runtime = new LocalMemoryRuntime();
    console.log(JSON.stringify({ ok: true, stats: runtime.stats(), port: DEFAULT_PORT }, null, 2));
    return;
  }
  console.log("RetainDB Local");
  console.log("");
  console.log("Usage:");
  console.log("  retaindb                 Start local memory server");
  console.log("  retaindb mcp             Run the bundled MCP bridge against local memory");
  console.log("  retaindb demo            Seed and search demo memories");
  console.log("  retaindb inspect         Score memory quality and show cleanup actions");
  console.log("  retaindb benchmark       Run a small local recall/latency benchmark");
  console.log("  retaindb install-embeddings  Warm local transformer embeddings and model cache");
  console.log("  retaindb connect all     Write Codex/Claude Code/OpenCode snippets");
  console.log("  retaindb connect all --install  Merge Codex/Claude Code user configs with backups");
  console.log("  retaindb hook            Capture a hook payload from stdin");
  console.log("  retaindb files sync      Generate .retaindb/files for multi-agent context");
  console.log("  retaindb files read /README.md");
  console.log("  retaindb files write --kind=handoff --agent=planner --to=builder");
  console.log("  retaindb import-jsonl    Import Claude-style JSONL transcripts");
  console.log("  retaindb consolidate     Deduplicate and summarize local sessions");
  console.log("  retaindb reembed         Refresh vectors with the configured local embedding provider");
  console.log("  retaindb status          Print memory/session stats");
  console.log("  retaindb doctor          Print local runtime status");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
