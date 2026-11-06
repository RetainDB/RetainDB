import { getOpenAIClient } from "./openai-client.js";
import { prisma } from "../db/index.js";
import { embedSingle } from "./embeddings.js";
import { writeMemoryCanonical } from "./memory/write.js";

// ═══════════════════════════════════════════════════════════════
// ENTITY EXTRACTION (runs during ingestion)
// ═══════════════════════════════════════════════════════════════

interface ExtractedEntity {
  name: string;
  type: string; // function, class, module, concept, api_endpoint, config, service, etc.
  description: string;
}

interface ExtractedRelation {
  from: string;
  fromType: string;
  to: string;
  toType: string;
  relation: string; // imports, calls, extends, depends_on, etc.
}

export async function extractEntities(
  projectId: string,
  content: string,
  chunkType: string,
  metadata: Record<string, any> = {},
  chunkId?: string
): Promise<{ entities: number; relations: number }> {
  // Skip small chunks
  if (content.length < 100) return { entities: 0, relations: 0 };

  const isCode = ["code", "function", "class"].includes(chunkType);

  const prompt = isCode
    ? `Analyze this code and extract entities and relationships.

Entities: functions, classes, interfaces, types, modules, variables, constants, API endpoints, services.
Relations: imports, exports, calls, implements, extends, depends_on, references, part_of.

Code:
\`\`\`
${content.slice(0, 3000)}
\`\`\`

Respond with JSON only:
{
  "entities": [{"name": "...", "type": "function|class|interface|module|constant|api_endpoint|service", "description": "one line"}],
  "relations": [{"from": "name", "fromType": "type", "to": "name", "toType": "type", "relation": "imports|calls|extends|implements|depends_on|references|part_of"}]
}`
    : `Analyze this text and extract key entities (concepts, people, tools, services, APIs, technologies) and their relationships.

Text:
${content.slice(0, 3000)}

Respond with JSON only:
{
  "entities": [{"name": "...", "type": "concept|tool|service|api|technology|person|organization", "description": "one line"}],
  "relations": [{"from": "name", "fromType": "type", "to": "name", "toType": "type", "relation": "references|depends_on|related_to|part_of|supersedes"}]
}`;

  try {
    const res = await getOpenAIClient().chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      max_tokens: 1000,
      response_format: { type: "json_object" },
    });

    const text = res.choices[0]?.message?.content?.trim() || "{}";
    const parsed = JSON.parse(text);

    const extractedEntities: ExtractedEntity[] = parsed.entities || [];
    const extractedRelations: ExtractedRelation[] = parsed.relations || [];

    let entityCount = 0;
    let relationCount = 0;

    // Upsert entities
    const entityMap = new Map<string, string>(); // name:type -> id

    for (const ent of extractedEntities.slice(0, 20)) {
      if (!ent.name || !ent.type) continue;

      const embedding = await embedSingle(`${ent.type}: ${ent.name} - ${ent.description || ""}`);
      const embeddingStr = `[${embedding.join(",")}]`;

      try {
        await prisma.$queryRaw`
          INSERT INTO "entities" (
            id, "projectId", name, "entityType", description, "sourceChunkId", embedding,
            "createdAt", "updatedAt"
          )
          VALUES (
            gen_random_uuid(), ${projectId}, ${ent.name}, ${ent.type}, ${ent.description || ""},
            ${chunkId || null}, ${embeddingStr}::vector, NOW(), NOW()
          )
          ON CONFLICT ("projectId", name, "entityType")
          DO UPDATE SET
            description = EXCLUDED.description,
            "sourceChunkId" = EXCLUDED."sourceChunkId",
            embedding = EXCLUDED.embedding,
            "updatedAt" = NOW()
        `;
        entityMap.set(`${ent.name}:${ent.type}`, ent.name);
        entityCount++;
      } catch (err: any) {
        console.error("[Extractor] Error upserting entity:", err.message);
      }
    }

    // Upsert relations
    for (const rel of extractedRelations.slice(0, 30)) {
      if (!rel.from || !rel.to || !rel.relation) continue;

      const fromId = entityMap.get(`${rel.from}:${rel.fromType}`);
      const toId = entityMap.get(`${rel.to}:${rel.toType}`);

      if (!fromId || !toId) continue;

      // Validate relation type
      const validRelations = [
        "imports", "exports", "calls", "implements", "extends",
        "references", "depends_on", "related_to", "part_of",
        "contradicts", "supersedes",
      ];
      if (!validRelations.includes(rel.relation)) continue;

      await prisma.entityRelation.upsert({
        where: {
          fromEntityId_toEntityId_relationType: {
            fromEntityId: fromId,
            toEntityId: toId,
            relationType: rel.relation,
          },
        },
        update: {
          metadata: { autoExtracted: true },
        },
        create: {
          projectId,
          fromEntityId: fromId,
          toEntityId: toId,
          relationType: rel.relation,
          metadata: { autoExtracted: true },
        },
      });

      relationCount++;
    }

    return { entities: entityCount, relations: relationCount };
  } catch {
    return { entities: 0, relations: 0 };
  }
}

// ═══════════════════════════════════════════════════════════════
// MEMORY EXTRACTION (runs on conversation messages)
// ═══════════════════════════════════════════════════════════════

interface ExtractedMemory {
  content: string;
  type: "factual" | "episodic" | "semantic" | "procedural";
  importance: number;
}

/**
 * Analyzes a conversation message (or batch) and extracts facts worth remembering.
 * Call this after adding messages to a conversation.
 */
export async function extractMemories(
  projectId: string,
  messages: { role: string; content: string }[],
  opts?: { userId?: string; sessionId?: string; agentId?: string }
): Promise<{ memoriesCreated: number }> {
  if (messages.length === 0) return { memoriesCreated: 0 };

  const conversation = messages
    .map((m) => `[${m.role}]: ${m.content}`)
    .join("\n\n");

  // Skip very short conversations
  if (conversation.length < 50) return { memoriesCreated: 0 };

  const prompt = `Analyze this conversation and extract important facts, preferences, decisions, or knowledge worth remembering for future interactions.

Rules:
- Only extract truly useful information (not greetings, acknowledgments, etc.)
- Each memory should be a standalone fact
- Set importance 0-1 (1 = critical preference/decision, 0.3 = minor detail)
- Type: factual (facts/preferences), episodic (what happened), semantic (general knowledge), procedural (how to do something)
- If nothing worth remembering, return empty array

Conversation:
${conversation.slice(0, 4000)}

Respond with JSON only:
{
  "memories": [
    {"content": "standalone fact", "type": "factual|episodic|semantic|procedural", "importance": 0.5}
  ]
}`;

  try {
    const res = await getOpenAIClient().chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      max_tokens: 500,
      response_format: { type: "json_object" },
    });

    const text = res.choices[0]?.message?.content?.trim() || "{}";
    const parsed = JSON.parse(text);
    const extracted: ExtractedMemory[] = parsed.memories || [];

    let created = 0;

    for (const mem of extracted.slice(0, 10)) {
      if (!mem.content || mem.content.length < 10) continue;

      const validTypes = ["factual", "episodic", "semantic", "procedural"];
      const memType = validTypes.includes(mem.type) ? mem.type : "factual";
      const importance = Math.max(0, Math.min(1, mem.importance || 0.5));

      try {
        const writeResult = await writeMemoryCanonical({
          projectId,
          userId: opts?.userId,
          sessionId: opts?.sessionId,
          agentId: opts?.agentId,
          content: mem.content,
          memoryType: memType,
          importance,
          confidenceRaw: importance,
          metadata: { autoExtracted: true },
          writeSource: "engine.extractor.legacy",
          writeMode: "session_extract",
          extractionMethod: "legacy_llm",
          publishPendingOverlay: false,
          sessionRetentionDays: 14,
        });
        if (writeResult.outcome === "created") created++;
      } catch (err: any) {
        console.error("[Extractor] Error creating memory:", err.message);
      }
    }

    return { memoriesCreated: created };
  } catch {
    return { memoriesCreated: 0 };
  }
}
