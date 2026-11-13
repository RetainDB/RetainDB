import OpenAI from "openai";
import { z } from "zod";
import type { MemorySourceRole, MemoryType, ExtractedMemory } from "./types.js";

// ── Lazy clients ──────────────────────────────────────────────────────────────

function getOpenAI(): OpenAI | null {
  const apiKey = process.env.OPENAI_API_KEY || "";
  if (!apiKey) return null;
  return new OpenAI({
    apiKey,
    ...(process.env.OPENAI_BASE_URL ? { baseURL: process.env.OPENAI_BASE_URL } : {}),
  });
}

type ChatParams = Parameters<OpenAI["chat"]["completions"]["create"]>[0];

/**
 * OpenAI-compatible providers accept routing fields (`reasoning_effort`,
 * `reasoning_format`) that the SDK types narrow or omit. This helper is the
 * single place where that widening happens.
 */
function providerChatParams(params: Record<string, unknown>): ChatParams {
  return params as unknown as ChatParams;
}

function shouldRetryWithoutStructuredOutput(err: any): boolean {
  const message = String(err?.message || "");
  return err?.status === 400 && /Failed to generate JSON|Failed to validate JSON|tool_use_failed|json_validate_failed/i.test(message);
}

function getInferenceModel(modelOverride?: string): string {
  return modelOverride
    || process.env.INFERENCE_MODEL
    || process.env.EXTRACTOR_MODEL
    || process.env.OPENAI_MODEL
    || "gpt-5.4-mini";
}

function getMaxOutputTokensParam(model: string, maxTokens: number) {
  return /^gpt-5/i.test(model)
    ? { max_completion_tokens: maxTokens }
    : { max_tokens: maxTokens };
}

function getProviderSpecificModelParams(model: string) {
  if (/^qwen\/qwen3-/i.test(model)) {
    return {
      reasoning_format: "hidden",
      reasoning_effort: "none",
    };
  }
  return {};
}

// ── Zod schema ────────────────────────────────────────────────────────────────

const MEMORY_TYPES = [
  "factual", "preference", "event", "relationship", "opinion", "goal", "instruction",
  "decision", "constraint", "solution", "project_state", "correction", "workflow",
] as const;
const MEMORY_TYPE_MAP: Record<string, typeof MEMORY_TYPES[number]> = {
  fact: "factual",
  facts: "factual",
  factual: "factual",
  information: "factual",
  info: "factual",
  preference: "preference",
  preferences: "preference",
  event: "event",
  episode: "event",
  episodic: "event",
  relationship: "relationship",
  opinion: "opinion",
  goal: "goal",
  instruction: "instruction",
  decision: "decision",
  constraint: "constraint",
  solution: "solution",
  project_state: "project_state",
  "project state": "project_state",
  correction: "correction",
  workflow: "workflow",
};

const InferredMemorySchema = z.object({
  content: z.string().min(8).max(400),
  memoryType: z.enum(MEMORY_TYPES),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().max(150).optional(),
  entities: z.array(z.string().max(80)).max(8).default([]),
});


// ── Prompt ────────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a high-precision session-state memory inference system.

Your task: infer only durable, future-useful context from a single message plus its surrounding context.

Infer memories that are likely to matter later, such as:
- durable user facts and preferences
- project decisions and accepted constraints
- stable project or task state
- accepted solutions or workflows
- corrections that clearly replace stale state

Rules:
- Be conservative.
- Never infer greetings, filler, or generic chat actions.
- Never turn assistant output into USER profile memory.
- If the message source is assistant, only infer project/task/agent/session memories that are strongly grounded by context.
- Do not restate obvious transcript text unless it is truly durable.
- Prefer standalone, unambiguous statements with no pronouns.

Output rules:
- Return at most 4 memories
- Minimum confidence to include: 0.70
- reasoning: one short sentence (debug only)
- Return JSON: { "memories": [...] }`;

// ── Input sanitization ────────────────────────────────────────────────────────

const MAX_MESSAGE_LEN = 2500;
const MAX_CONTEXT_LEN = 1000;

/**
 * Truncate and sanitize input before embedding in the prompt.
 * Does NOT escape the content (the model receives it as data, not instructions),
 * but hard-limits length to bound both cost and prompt-injection surface area.
 */
function sanitizeInput(value: string, maxLen: number): string {
  return value.slice(0, maxLen).trim();
}

// ── Model routing ─────────────────────────────────────────────────────────────

async function runWithOpenAI(message: string, context: string, model: string): Promise<string | null> {
  const client = getOpenAI();
  if (!client) return null;

  const userContent = [
    context ? `## Context\n${context}` : null,
    `## Message\n${message}`,
  ].filter(Boolean).join("\n\n");
  const structuredParams = providerChatParams({
    model,
    ...getMaxOutputTokensParam(model, 512),
    temperature: 0.1,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userContent },
    ],
    ...getProviderSpecificModelParams(model),
    response_format: { type: "json_object" as const },
  });
  const fallbackParams = providerChatParams({
    model,
    ...getMaxOutputTokensParam(model, 512),
    temperature: 0.1,
    messages: [
      { role: "system", content: `${SYSTEM_PROMPT}\n\nReturn ONLY valid JSON. Do not use markdown fences or extra commentary.` },
      { role: "user", content: `${userContent}\n\nReturn ONLY a valid JSON object with a top-level "memories" array.` },
    ],
    ...getProviderSpecificModelParams(model),
  });

  let response;
  try {
    response = await client.chat.completions.create(structuredParams);
  } catch (err: any) {
    if (!shouldRetryWithoutStructuredOutput(err)) throw err;
    response = await client.chat.completions.create(fallbackParams);
  }
  return response.choices[0]?.message?.content?.trim() ?? null;
}

// ── Response parsing ──────────────────────────────────────────────────────────

function parseInferenceResponse(text: string): unknown[] {
  let parsed: unknown;
  try {
    const jsonText = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/, "").trim();
    parsed = JSON.parse(jsonText);
  } catch {
    // Try to extract a JSON object/array from surrounding text if the model wrapped it
    const jsonMatch = text.match(/\{[\s\S]*\}/) ?? text.match(/\[[\s\S]*\]/);
    if (!jsonMatch) return [];
    try { parsed = JSON.parse(jsonMatch[0]); } catch { return []; }
  }

  // Extract the raw items array regardless of wrapper shape
  let items: unknown[];
  if (Array.isArray(parsed)) {
    items = parsed;
  } else if (parsed && typeof parsed === "object" && Array.isArray((parsed as any).memories)) {
    items = (parsed as any).memories;
  } else {
    return [];
  }

  // Validate items individually — drop bad ones instead of failing the whole batch
  const normalizeMemoryType = (raw: unknown): typeof MEMORY_TYPES[number] => {
    const key = String(raw || "factual").toLowerCase().trim();
    return MEMORY_TYPE_MAP[key] || "factual";
  };
  const normalizeConfidence = (raw: unknown): number => {
    if (typeof raw === "number" && Number.isFinite(raw)) return raw;
    if (typeof raw === "string") {
      const parsedValue = Number.parseFloat(raw);
      if (Number.isFinite(parsedValue)) return parsedValue;
    }
    return 0.8;
  };
  const normalizeEntities = (raw: unknown): string[] => {
    if (Array.isArray(raw)) {
      return raw
        .filter((value): value is string => typeof value === "string")
        .map((value) => value.trim())
        .filter(Boolean);
    }
    if (typeof raw === "string" && raw.trim()) {
      return [raw.trim()];
    }
    return [];
  };

  const normalizedItems = items.map((item: any) => ({
    content: item?.content ?? item?.memory ?? item?.text ?? item?.description ?? item?.fact ?? item?.statement,
    memoryType: normalizeMemoryType(item?.memoryType ?? item?.memory_type ?? item?.type ?? item?.category),
    confidence: normalizeConfidence(item?.confidence ?? item?.score),
    reasoning: item?.reasoning,
    entities: normalizeEntities(
      item?.entities
        ?? item?.entityMentions
        ?? item?.entity_mentions
        ?? item?.people
    ),
  }));

  const valid: z.infer<typeof InferredMemorySchema>[] = [];
  for (const item of normalizedItems) {
    const r = InferredMemorySchema.safeParse(item);
    if (r.success) {
      valid.push(r.data);
    }
  }
  return valid;
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function extractImplicitMemories(
  message: string,
  context: string = "",
  options?: {
    model?: string;
    minConfidence?: number;
    sourceRole?: MemorySourceRole;
  },
): Promise<ExtractedMemory[]> {
  if (!message.trim() || message.trim().length < 8) return [];

  const safeMessage = sanitizeInput(message, MAX_MESSAGE_LEN);
  const safeContext = sanitizeInput(context, MAX_CONTEXT_LEN);
  const minConfidence = typeof options?.minConfidence === "number" ? options.minConfidence : 0.70;

  try {
    let text: string | null = null;
    const sourceRole = options?.sourceRole || "user";
    const effectiveContext = [
      `## Message Source\n${sourceRole}`,
      safeContext ? `## Context\n${safeContext}` : "",
    ].filter(Boolean).join("\n\n");

    const openai = getOpenAI();
    if (!openai) {
      return [];
    }
    text = await runWithOpenAI(safeMessage, effectiveContext, getInferenceModel(options?.model));

    if (!text) return [];

    const rawMemories = parseInferenceResponse(text);
    if (rawMemories.length === 0) return [];

    return (rawMemories as z.infer<typeof InferredMemorySchema>[])
      .filter(m => m.confidence >= minConfidence)
      .slice(0, 4)
      .map(m => ({
        content: m.content,
        memoryType: m.memoryType as MemoryType,
        entityMentions: m.entities,
        eventDate: null,
        confidence: m.confidence,
        reasoning: m.reasoning,
        inferred: true,
        sourceRole,
      }));

  } catch (err: any) {
    console.error("[inference] extractImplicitMemories failed:", err?.message || err);
    return [];
  }
}

export async function extractImplicitMemoriesBatch(
  messages: Array<{ content: string; context?: string }>,
  options?: { model?: string; minConfidence?: number },
): Promise<ExtractedMemory[]> {
  const CONCURRENCY = 5;
  const results: ExtractedMemory[] = [];
  const queue = [...messages];
  const inFlight = new Set<Promise<void>>();

  const runOne = async (msg: { content: string; context?: string }) => {
    const mems = await extractImplicitMemories(msg.content, msg.context ?? "", options);
    results.push(...mems);
  };

  while (queue.length > 0 || inFlight.size > 0) {
    while (queue.length > 0 && inFlight.size < CONCURRENCY) {
      const msg = queue.shift()!;
      const p = runOne(msg).finally(() => inFlight.delete(p));
      inFlight.add(p);
    }
    if (inFlight.size > 0) await Promise.race(inFlight);
  }

  return deduplicateMemories(results);
}

// ── Dedup ─────────────────────────────────────────────────────────────────────

function deduplicateMemories(memories: ExtractedMemory[]): ExtractedMemory[] {
  const seen = new Map<string, ExtractedMemory>();
  for (const memory of memories) {
    const key = normalizeDedupeKey(memory.memoryType, memory.content);
    const existing = seen.get(key);
    if (!existing || memory.confidence > existing.confidence) {
      seen.set(key, memory);
    }
  }
  return Array.from(seen.values());
}

const DEDUPE_STOP = new Set([
  "the","a","an","is","are","was","were","be","been","being","have","has","had",
  "do","does","did","will","would","could","should","may","might","shall","can",
  "need","dare","ought","to","of","in","for","on","with","at","by","from","as",
  "into","through","about","against","between","i","my","me",
]);

function normalizeDedupeKey(type: string, content: string): string {
  const words = content
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .split(/\s+/)
    .filter(w => w.length > 2 && !DEDUPE_STOP.has(w))
    .sort()
    .slice(0, 6);
  return `${type}:${words.join(":")}`;
}
