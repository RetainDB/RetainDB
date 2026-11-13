/**
 * SOTA Memory System Types
 * Supports temporal reasoning, relational versioning, and knowledge graphs
 */

export type MemoryType =
  | "factual"       // Objective facts: "User's name is John"
  | "preference"    // User preferences: "User prefers dark mode"
  | "event"         // Events: "User attended conference on Jan 15"
  | "relationship"  // Relationships: "Alex reports to Maria"
  | "opinion"       // Opinions: "User thinks React is great"
  | "goal"          // Goals: "User wants to learn Python"
  | "instruction"   // Instructions: "Always use formal tone"
  | "decision"      // Durable decisions: "The project standardizes on Bun"
  | "constraint"    // Constraints: "Deployment must stay on AWS Lambda"
  | "solution"      // Accepted fixes or solutions
  | "project_state" // Ongoing project state worth carrying forward
  | "correction"    // Correction that supersedes stale memory
  | "workflow";     // Reusable workflow or agent habit

export type RelationType =
  | "updates"       // New memory supersedes old (State mutation)
  | "extends"       // Adds detail to existing memory (Refinement)
  | "derives"       // Inferred from other memories (Inference)
  | "contradicts"   // Conflicts with another memory
  | "supports";     // Provides evidence for another memory

export type MemoryScopeTarget =
  | "USER"
  | "SESSION"
  | "PROJECT"
  | "AGENT"
  | "TASK"
  | "DOCUMENT"
  | "DROPPED";

export type PromotionMode = "session_state_v1" | "user_specific_legacy";

export type MemorySourceRole = "user" | "assistant" | "event" | "document" | "system";

export type SessionWorkEventKind =
  | "decision"
  | "constraint"
  | "outcome"
  | "failure"
  | "task_update"
  | "file_edit"
  | "tool_result";

export type SessionWorkEventSalience = "low" | "medium" | "high";

export interface SessionWorkEvent {
  kind: SessionWorkEventKind;
  summary: string;
  details?: string;
  salience?: SessionWorkEventSalience;
  timestamp?: string;
  filePaths?: string[];
  toolName?: string;
  success?: boolean;
}

export interface ExtractedMemory {
  content: string;
  memoryType: MemoryType;
  entityMentions: string[];
  eventDate: Date | null;
  /** How precisely the eventDate was determined */
  temporalPrecision?: "exact" | "inferred_day" | "inferred_month" | "unknown";
  confidence: number; // 0-1
  reasoning?: string;
  /** Short verbatim fragment from source that grounded this memory */
  sourceSpan?: string;
  inferred?: boolean;
  sourceRole?: MemorySourceRole;
  userConfirmed?: boolean;
  supportingEvent?: boolean;
  sourceEventKind?: SessionWorkEventKind;
}

export interface MemoryExtractionOptions {
  enablePattern?: boolean;
  enableInference?: boolean;
  minConfidence?: number;
  maxMemories?: number;
  async?: boolean;
  tieredEscalation?: boolean;
  escalationModel?: string;
}

export interface MemoryRelationship {
  toMemoryId: string;
  relationType: RelationType;
  confidence: number;
  reasoning: string;
}

export interface TemporalFilter {
  hasTemporalConstraint: boolean;
  relative?: "today" | "yesterday" | "last_week" | "last_month" | "last_year";
  absoluteDate?: Date;
  dateRange?: {
    start: Date;
    end: Date;
  };
}

export interface MemorySearchParams {
  query: string;
  questionDate: Date;
  userId?: string;
  projectId: string;
  orgId?: string;
  sessionId?: string;
  agentId?: string;
  taskId?: string;
  topK?: number;
  includeInactive?: boolean;
  temporalFilter?: TemporalFilter;
  memoryTypes?: MemoryType[];
  scopes?: MemoryScopeTarget[];
  namespace?: string;
  tags?: string[];
  /** Fast mode skips graph traversal and chunk injection for sub-50ms latency */
  fastMode?: boolean;
  /** Preserve full-quality retrieval even if latency guardrails would normally degrade the pipeline */
  disableLatencyDegradation?: boolean;
  diagnosticsCollector?: (diagnostics: MemorySearchDiagnostics) => void;
}

export interface MemorySearchDiagnostics {
  cache_ms: number;
  embed_ms: number;
  vector_ms: number;
  lexical_ms: number;
  merge_ms: number;
  total_ms: number;
  cache_hit: boolean;
  cache_hit_type: "none" | "simple" | "semantic";
  fast_mode: boolean;
}

export interface MemorySearchResult {
  memory: {
    id: string;
    content: string;
    memoryType: MemoryType;
    entityMentions: string[];
    confidence: number;
    version: number;
    scope?: string;
    scopeTarget?: MemoryScopeTarget;
    userId?: string | null;
    sessionId?: string | null;
    agentId?: string | null;
    taskId?: string | null;
    metadata: Record<string, unknown>;
    temporal: {
      documentDate: Date | null;
      eventDate: Date | null;
      validFrom: Date | null;
      validUntil: Date | null;
    };
  };
  chunk?: {
    id: string;
    content: string;
    metadata: Record<string, any>;
  };
  similarity: number;
  relations?: Array<{
    memoryId: string;
    relationType: RelationType;
    content: string;
  }>;
}

export interface ExtractionContext {
  sessionId: string;
  userId: string;
  projectId: string;
  orgId?: string;
  agentId?: string;
  taskId?: string;
  promotionMode?: PromotionMode;
  currentRole?: MemorySourceRole;
  documentDate: Date;
  previousMessages?: string[];
  entityContext?: Map<string, string>; // pronoun -> name mapping
}

export interface KnowledgeVersion {
  memoryId: string;
  version: number;
  content: string;
  validFrom: Date;
  validUntil: Date | null;
  supersededBy: string | null;
  changeType: "initial" | "update" | "correction" | "refinement";
}
