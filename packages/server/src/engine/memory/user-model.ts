import { db } from "../../db/index.js";

type MemoryRecord = {
  id: string;
  content: string;
  memoryType: string;
  confidence: number;
  importance: number;
  entityMentions: string[];
  createdAt: Date;
  updatedAt: Date;
  lastAccessedAt: Date | null;
  metadata: unknown;
};

export interface UserModelProfile {
  name: string | null;
  role: string | null;
  preferences: string[];
  current_goals: string[];
  working_style: string | null;
  frequent_entities: string[];
  trust_level: number;
}

export interface SynthesizedUserModel {
  user_id: string;
  profile: UserModelProfile;
  memory_count: number;
  last_updated: string | null;
  evidence: {
    supporting_memory_ids: string[];
    coverage_score: number;
  };
}

export interface UserGap {
  topic: string;
  question: string;
  priority: "high" | "medium" | "low";
  reason: string;
}

export interface UserGapResult {
  user_id: string;
  gaps: UserGap[];
  coverage_score: number;
  known_topics: string[];
  missing_topics: string[];
}

const GENERIC_ENTITIES = new Set([
  "user",
  "assistant",
  "project",
  "task",
  "memory",
  "session",
]);

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function cleanSentence(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function dedupeStrings(values: string[]): string[] {
  return Array.from(new Set(values.map((value) => cleanSentence(value)).filter(Boolean)));
}

function byStrength(memories: MemoryRecord[]): MemoryRecord[] {
  return [...memories].sort((a, b) => {
    const scoreA = a.importance * 0.55 + a.confidence * 0.45;
    const scoreB = b.importance * 0.55 + b.confidence * 0.45;
    return scoreB - scoreA || b.updatedAt.getTime() - a.updatedAt.getTime();
  });
}

function extractName(memories: MemoryRecord[]): string | null {
  const candidates = byStrength(memories);
  for (const memory of candidates) {
    const content = memory.content;
    const patterns = [
      /\bmy name is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b/i,
      /\bi am ([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b/i,
      /\buser name[: ]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b/i,
    ];
    for (const pattern of patterns) {
      const match = content.match(pattern);
      if (match?.[1]) return cleanSentence(match[1]);
    }
  }
  return null;
}

function extractRole(memories: MemoryRecord[]): string | null {
  const candidates = byStrength(memories);
  for (const memory of candidates) {
    const content = memory.content;
    const patterns = [
      /\b(?:i am|i'm|user is|works as|role is)\s+(?:an?\s+)?([^.,;]{4,80})/i,
      /\b(?:senior|staff|principal|lead|junior)\s+[^.,;]{2,60}/i,
    ];
    for (const pattern of patterns) {
      const match = content.match(pattern);
      const value = match?.[1] ?? match?.[0];
      if (value) {
        const cleaned = cleanSentence(value.replace(/\b(user|person)\b/gi, "").trim());
        if (cleaned.length >= 4 && cleaned.length <= 80) return cleaned;
      }
    }
  }
  return null;
}

function collectPreferences(memories: MemoryRecord[]): string[] {
  return dedupeStrings(
    byStrength(memories)
      .filter((memory) => ["preference", "instruction", "opinion"].includes(memory.memoryType))
      .map((memory) => memory.content)
  ).slice(0, 6);
}

function collectGoals(memories: MemoryRecord[]): string[] {
  return dedupeStrings(
    byStrength(memories)
      .filter((memory) => ["goal", "project_state", "workflow", "decision"].includes(memory.memoryType))
      .map((memory) => memory.content)
  ).slice(0, 6);
}

function deriveWorkingStyle(preferences: string[], instructions: string[], entities: string[]): string | null {
  const joined = [...preferences, ...instructions].join(" ").toLowerCase();
  const descriptors: string[] = [];
  if (/\b(iterative|step by step|incremental)\b/.test(joined)) descriptors.push("iterative");
  if (/\b(concise|brief|short)\b/.test(joined)) descriptors.push("concise");
  if (/\b(bullet|bulleted|list)\b/.test(joined)) descriptors.push("structured");
  if (/\b(code|typescript|example)\b/.test(joined)) descriptors.push("code-oriented");
  if (/\b(no boilerplate|minimal boilerplate)\b/.test(joined)) descriptors.push("anti-boilerplate");
  if (descriptors.length === 0 && entities.length > 0) descriptors.push(`focused on ${entities.slice(0, 2).join(" and ")}`);
  return descriptors.length > 0 ? descriptors.join(", ") : null;
}

function collectFrequentEntities(memories: MemoryRecord[]): string[] {
  const counts = new Map<string, number>();
  for (const memory of memories) {
    for (const rawEntity of memory.entityMentions || []) {
      const entity = cleanSentence(rawEntity);
      if (!entity) continue;
      if (GENERIC_ENTITIES.has(entity.toLowerCase())) continue;
      counts.set(entity, (counts.get(entity) || 0) + 1);
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 8)
    .map(([entity]) => entity);
}

function computeCoverage(memories: MemoryRecord[], profile: Omit<UserModelProfile, "trust_level">): number {
  const dimensions = [
    profile.name,
    profile.role,
    profile.preferences.length > 0 ? "preferences" : null,
    profile.current_goals.length > 0 ? "goals" : null,
    profile.working_style,
    profile.frequent_entities.length > 0 ? "entities" : null,
  ];
  const fieldCoverage = dimensions.filter(Boolean).length / dimensions.length;
  const evidenceCoverage = clamp01(memories.length / 12);
  const recallCoverage =
    memories.length === 0
      ? 0
      : memories.filter((memory) => memory.lastAccessedAt !== null).length / memories.length;
  return Number((fieldCoverage * 0.55 + evidenceCoverage * 0.3 + recallCoverage * 0.15).toFixed(2));
}

function computeTrustLevel(memories: MemoryRecord[], coverageScore: number): number {
  if (memories.length === 0) return 0;
  const weightedConfidence =
    memories.reduce((sum, memory) => sum + clamp01(memory.confidence) * (0.5 + clamp01(memory.importance) * 0.5), 0) /
    memories.length;
  const recencyFactor =
    memories.filter((memory) => Date.now() - memory.updatedAt.getTime() < 1000 * 60 * 60 * 24 * 30).length / memories.length;
  return Number((weightedConfidence * 0.55 + coverageScore * 0.3 + recencyFactor * 0.15).toFixed(2));
}

export async function loadUserModelMemories(params: {
  userId: string;
  projectId: string;
}): Promise<MemoryRecord[]> {
  const { userId, projectId } = params;
  return db.memory.findMany({
    where: {
      userId,
      projectId,
      isActive: true,
      scope: "USER",
      OR: [
        { validUntil: null },
        { validUntil: { gt: new Date() } },
      ],
    },
    select: {
      id: true,
      content: true,
      memoryType: true,
      confidence: true,
      importance: true,
      entityMentions: true,
      createdAt: true,
      updatedAt: true,
      lastAccessedAt: true,
      metadata: true,
    },
    orderBy: [
      { importance: "desc" },
      { updatedAt: "desc" },
    ],
    take: 200,
  });
}

export async function synthesizeUserModel(params: {
  userId: string;
  projectId: string;
}): Promise<SynthesizedUserModel> {
  const memories = await loadUserModelMemories(params);
  const preferences = collectPreferences(memories);
  const goals = collectGoals(memories);
  const frequentEntities = collectFrequentEntities(memories);
  const instructions = memories
    .filter((memory) => memory.memoryType === "instruction")
    .map((memory) => memory.content);

  const profileBase = {
    name: extractName(memories),
    role: extractRole(memories),
    preferences,
    current_goals: goals,
    working_style: deriveWorkingStyle(preferences, instructions, frequentEntities),
    frequent_entities: frequentEntities,
  };

  const coverageScore = computeCoverage(memories, profileBase);
  const trustLevel = computeTrustLevel(memories, coverageScore);
  const supportingIds = byStrength(
    memories.filter((memory) => {
      if (preferences.includes(memory.content)) return true;
      if (goals.includes(memory.content)) return true;
      if ((profileBase.name && memory.content.includes(profileBase.name)) || (profileBase.role && memory.content.includes(profileBase.role))) {
        return true;
      }
      return memory.entityMentions.some((entity) => frequentEntities.includes(entity));
    })
  )
    .slice(0, 12)
    .map((memory) => memory.id);

  return {
    user_id: params.userId,
    profile: {
      ...profileBase,
      trust_level: trustLevel,
    },
    memory_count: memories.length,
    last_updated: memories[0]?.updatedAt?.toISOString() ?? null,
    evidence: {
      supporting_memory_ids: supportingIds,
      coverage_score: coverageScore,
    },
  };
}

function scoreGapPriority(topic: string, context: string): "high" | "medium" | "low" {
  const lc = context.toLowerCase();
  if ((topic === "plan" || topic === "payment_method") && /\b(billing|invoice|payment|subscription|plan)\b/.test(lc)) return "high";
  if ((topic === "output_format" || topic === "working_style") && /\b(explain|answer|format|example|respond)\b/.test(lc)) return "high";
  if (topic === "current_goal" && /\b(build|ship|fix|refactor|working on)\b/.test(lc)) return "high";
  if (topic === "role" || topic === "name") return "medium";
  return "low";
}

export async function detectUserGaps(params: {
  userId: string;
  projectId: string;
  context: string;
  limit?: number;
}): Promise<UserGapResult> {
  const model = await synthesizeUserModel({ userId: params.userId, projectId: params.projectId });
  const context = cleanSentence(params.context || "");
  const lc = context.toLowerCase();
  const gaps: UserGap[] = [];
  const knownTopics: string[] = [];
  const missingTopics: string[] = [];

  if (model.profile.name) knownTopics.push("name");
  else missingTopics.push("name");

  if (model.profile.role) knownTopics.push("role");
  else missingTopics.push("role");

  if (model.profile.preferences.length > 0) knownTopics.push("output_format");
  else missingTopics.push("output_format");

  if (model.profile.current_goals.length > 0) knownTopics.push("current_goal");
  else missingTopics.push("current_goal");

  if (model.profile.working_style) knownTopics.push("working_style");
  else missingTopics.push("working_style");

  const gapCandidates: Array<Omit<UserGap, "priority"> & { topic: string }> = [];

  if (!model.profile.current_goals.length) {
    gapCandidates.push({
      topic: "current_goal",
      question: "What are you trying to accomplish right now?",
      reason: "No active goal is grounded in memory, which makes future-session continuity weaker.",
    });
  }

  if (!model.profile.preferences.length) {
    gapCandidates.push({
      topic: "output_format",
      question: "Do you prefer concise bullets, detailed prose, or code-first answers?",
      reason: "The system lacks a stable answer-format preference for this user.",
    });
  }

  if (!model.profile.working_style) {
    gapCandidates.push({
      topic: "working_style",
      question: "Do you prefer iterative back-and-forth, or should I give you the full answer in one pass?",
      reason: "Working style is not yet grounded, so follow-up behavior is less personalized.",
    });
  }

  if (!model.profile.role && /\b(team|engineer|developer|support|billing|customer)\b/.test(lc)) {
    gapCandidates.push({
      topic: "role",
      question: "What role are you working in so I can tailor the level of detail correctly?",
      reason: "Role is missing and the current topic suggests the right level of abstraction matters.",
    });
  }

  if (/\b(billing|invoice|subscription|price|payment|plan)\b/.test(lc)) {
    if (!model.profile.frequent_entities.some((entity) => /\b(plan|billing|stripe|subscription)\b/i.test(entity))) {
      gapCandidates.push({
        topic: "plan",
        question: "Which plan or subscription tier are you on?",
        reason: "Billing answers usually depend on the user's plan, and that is not grounded yet.",
      });
      gapCandidates.push({
        topic: "payment_method",
        question: "Which payment method do you use for billing?",
        reason: "Payment-method context is often needed for precise billing help.",
      });
    }
  }

  const ranked = gapCandidates
    .map((gap) => ({
      ...gap,
      priority: scoreGapPriority(gap.topic, context),
    }))
    .sort((a, b) => {
      const order = { high: 0, medium: 1, low: 2 };
      return order[a.priority] - order[b.priority] || a.topic.localeCompare(b.topic);
    });

  const limit = Math.max(1, Math.min(params.limit ?? 5, 10));

  return {
    user_id: params.userId,
    gaps: ranked.slice(0, limit),
    coverage_score: model.evidence.coverage_score,
    known_topics: dedupeStrings(knownTopics),
    missing_topics: dedupeStrings([...missingTopics, ...ranked.map((gap) => gap.topic)]),
  };
}
