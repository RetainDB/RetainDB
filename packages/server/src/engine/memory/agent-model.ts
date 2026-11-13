/**
 * Agent self-model — synthesizes what an agent "knows about itself" from
 * AGENT-scoped memories: persona traits, persistent instructions, goals,
 * and working style derived from stored observations.
 *
 * Gives agents persistent self-knowledge that survives across sessions.
 */

import { db } from "../../db/index.js";

type AgentMemoryRecord = {
  id: string;
  content: string;
  memoryType: string;
  confidence: number;
  importance: number;
  entityMentions: string[];
  createdAt: Date;
  updatedAt: Date;
};

export interface AgentSelfModel {
  agent_id: string;
  persona: string | null;
  persistent_instructions: string[];
  capabilities: string[];
  working_style: string | null;
  goals: string[];
  memory_count: number;
  last_updated: string | null;
  coverage_score: number;
}

function byStrength(memories: AgentMemoryRecord[]): AgentMemoryRecord[] {
  return [...memories].sort(
    (a, b) =>
      b.importance * 0.55 + b.confidence * 0.45 -
      (a.importance * 0.55 + a.confidence * 0.45) ||
      b.updatedAt.getTime() - a.updatedAt.getTime()
  );
}

function cleanSentence(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

function extractPersona(memories: AgentMemoryRecord[]): string | null {
  const candidates = byStrength(memories);
  for (const m of candidates) {
    const patterns = [
      /\b(?:i am|you are|this agent is|my role is)\s+([^.,;]{6,120})/i,
      /\bpersona[: ]+([^.,;]{6,120})/i,
    ];
    for (const p of patterns) {
      const match = m.content.match(p);
      if (match?.[1]) return cleanSentence(match[1]);
    }
  }
  return null;
}

function collectInstructions(memories: AgentMemoryRecord[]): string[] {
  return byStrength(memories)
    .filter((m) => ["instruction", "preference"].includes(m.memoryType))
    .map((m) => cleanSentence(m.content))
    .filter(Boolean)
    .slice(0, 8);
}

function collectCapabilities(memories: AgentMemoryRecord[]): string[] {
  const caps: string[] = [];
  const markers = ["can ", "able to ", "capable of ", "supports ", "handles "];
  for (const m of byStrength(memories)) {
    const lc = m.content.toLowerCase();
    if (markers.some((mk) => lc.includes(mk))) {
      caps.push(cleanSentence(m.content));
      if (caps.length >= 6) break;
    }
  }
  return caps;
}

function deriveWorkingStyle(memories: AgentMemoryRecord[]): string | null {
  const joined = memories.map((m) => m.content).join(" ").toLowerCase();
  const descriptors: string[] = [];
  if (/\b(iterative|step.?by.?step|incremental)\b/.test(joined)) descriptors.push("iterative");
  if (/\b(concise|brief|terse)\b/.test(joined)) descriptors.push("concise");
  if (/\b(structured|bullet|outline)\b/.test(joined)) descriptors.push("structured");
  if (/\b(code.first|code oriented|technical)\b/.test(joined)) descriptors.push("code-first");
  if (/\b(proactive|anticipate|suggest)\b/.test(joined)) descriptors.push("proactive");
  return descriptors.length > 0 ? descriptors.join(", ") : null;
}

function collectGoals(memories: AgentMemoryRecord[]): string[] {
  return byStrength(memories)
    .filter((m) => ["goal", "decision", "workflow"].includes(m.memoryType))
    .map((m) => cleanSentence(m.content))
    .slice(0, 4);
}

function computeCoverage(model: AgentSelfModel): number {
  const dims = [
    model.persona,
    model.persistent_instructions.length > 0 ? "instructions" : null,
    model.capabilities.length > 0 ? "capabilities" : null,
    model.working_style,
    model.goals.length > 0 ? "goals" : null,
  ];
  const filled = dims.filter(Boolean).length;
  const evidenceFactor = Math.min(model.memory_count / 8, 1);
  return Number(((filled / dims.length) * 0.65 + evidenceFactor * 0.35).toFixed(2));
}

export async function loadAgentMemories(params: {
  agentId: string;
  projectId: string;
}): Promise<AgentMemoryRecord[]> {
  return db.memory.findMany({
    where: {
      agentId: params.agentId,
      projectId: params.projectId,
      scope: "AGENT",
      isActive: true,
      OR: [{ validUntil: null }, { validUntil: { gt: new Date() } }],
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
    },
    orderBy: [{ importance: "desc" }, { updatedAt: "desc" }],
    take: 100,
  });
}

export async function synthesizeAgentModel(params: {
  agentId: string;
  projectId: string;
}): Promise<AgentSelfModel> {
  const memories = await loadAgentMemories(params);

  const persona = extractPersona(memories);
  const persistent_instructions = collectInstructions(memories);
  const capabilities = collectCapabilities(memories);
  const working_style = deriveWorkingStyle(memories);
  const goals = collectGoals(memories);

  const partial: AgentSelfModel = {
    agent_id: params.agentId,
    persona,
    persistent_instructions,
    capabilities,
    working_style,
    goals,
    memory_count: memories.length,
    last_updated: memories[0]?.updatedAt?.toISOString() ?? null,
    coverage_score: 0,
  };

  partial.coverage_score = computeCoverage(partial);
  return partial;
}

/**
 * Ingest raw identity text (e.g. SOUL.md) as AGENT-scoped memories.
 * Each paragraph becomes a candidate memory after deduplication.
 */
export async function seedAgentIdentity(params: {
  agentId: string;
  projectId: string;
  orgId: string;
  content: string;
  source?: string;
}): Promise<{ seeded: number }> {
  const { writeMemoryCanonical } = await import("./write.js");

  const paragraphs = params.content
    .split(/\n{2,}/)
    .map((p) => p.replace(/\s+/g, " ").trim())
    .filter((p) => p.length >= 20 && p.length <= 800);

  let seeded = 0;
  for (const paragraph of paragraphs.slice(0, 30)) {
    try {
      const result = await writeMemoryCanonical({
        projectId: params.projectId,
        orgId: params.orgId,
        agentId: params.agentId,
        content: paragraph,
        memoryType: "instruction",
        confidenceRaw: 0.85,
        importance: 0.75,
        entityMentions: [],
        writeSource: params.source ?? "soul_md",
        writeMode: "source_extract",
        extractionMethod: "soul_md_seed",
        scopeHint: "AGENT",
        metadata: { seeded_from: params.source ?? "soul_md", agent_id: params.agentId },
      });
      if (result.outcome === "created") seeded++;
    } catch {
      // non-fatal
    }
  }

  return { seeded };
}
