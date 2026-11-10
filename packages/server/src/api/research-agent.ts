/**
 * Research Agent API - OpenAI-powered tree-guided research
 * Streams real-time events as the agent navigates the document tree
 */

import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { z } from "zod";
import { zValidator } from "@hono/zod-validator";
import OpenAI from "openai";
import { getOpenAIClient } from "../engine/openai-client.js";
import { prisma } from "../db/index.js";
import { retrieve } from "../engine/retriever.js";
import { buildDocumentTree } from "../engine/oracle.js";
import { getUserMemory, updateUserMemory, rememberFact, saveResearchSession, getRecentResearch, searchMemory } from "../engine/agent-memory.js";
import { browseWeb, deepResearch } from "../engine/browser-agent.js";
import type { AuthContext } from "../middleware/auth.js";
import { resolveProjectReference, getEffectiveOrgId } from "./helpers.js";

type Variables = { auth: AuthContext };

export const researchAgentRoutes = new Hono<{ Variables: Variables }>();

const researchSchema = z.object({
  project: z.string().optional(), // specific project ID, "all" for multi-project, or omit
  query: z.string().min(1).max(2000),
  model: z.string().optional().default("gpt-4o"),
  maxSteps: z.number().optional().default(8),
  enableWebBrowse: z.boolean().optional().default(true),
  mode: z.enum(["research", "onboarding", "contradiction"]).optional().default("research"),
  conversationHistory: z.array(z.object({
    role: z.enum(["user", "assistant"]),
    content: z.string(),
  })).optional().default([]),
});

// Tool definitions for the OpenAI agent
const AGENT_TOOLS: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "search_documents",
      description:
        "Search the project knowledge base using hybrid semantic + keyword search. Returns relevant document chunks with their content and metadata.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The search query to find relevant documents",
          },
          topK: {
            type: "number",
            description: "Number of results to return (1-10)",
            default: 5,
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_document_tree",
      description:
        "Get the hierarchical tree structure of a specific document, showing sections and subsections. Use this to understand a document's structure before diving deeper.",
      parameters: {
        type: "object",
        properties: {
          documentId: {
            type: "string",
            description: "The document ID to get the tree for",
          },
        },
        required: ["documentId"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_document_section",
      description:
        "Get the full content of a specific section within a document. Use after get_document_tree to read a particular section in detail.",
      parameters: {
        type: "object",
        properties: {
          documentId: {
            type: "string",
            description: "The document ID",
          },
          sectionPath: {
            type: "string",
            description: "The section path (from document tree)",
          },
        },
        required: ["documentId", "sectionPath"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_project_documents",
      description:
        "List all documents in the project with their titles and metadata. Use this to discover what's available before searching.",
      parameters: {
        type: "object",
        properties: {
          limit: {
            type: "number",
            description: "Max documents to return",
            default: 20,
          },
        },
      },
    },
  },
  // Web browsing with Playwright
  {
    type: "function",
    function: {
      name: "browse_web",
      description:
        "Browse a web page and extract its content. Use this to get fresh information from the internet, read documentation, or explore websites. Supports clicking, scrolling, and extracting text, links, or images.",
      parameters: {
        type: "object",
        properties: {
          url: {
            type: "string",
            description: "The URL to browse",
          },
          action: {
            type: "string",
            enum: ["click", "type", "scroll", "screenshot", "extract"],
            description: "Action to perform on the page",
          },
          selector: {
            type: "string",
            description: "CSS selector for click/type actions",
          },
          text: {
            type: "string",
            description: "Text to type (for type action)",
          },
          extractWhat: {
            type: "string",
            enum: ["text", "links", "images", "all"],
            description: "What to extract from the page",
            default: "text",
          },
          maxWaitMs: {
            type: "number",
            description: "Max wait time in milliseconds",
            default: 30000,
          },
        },
        required: ["url"],
      },
    },
  },
  // Remember something about the user or research
  {
    type: "function",
    function: {
      name: "remember",
      description:
        "Store important information in memory for future reference. Use this to remember user preferences, key findings from research, or anything important you'd like to recall later. The agent can recall this information in future conversations.",
      parameters: {
        type: "object",
        properties: {
          fact: {
            type: "string",
            description: "The fact or information to remember",
          },
          category: {
            type: "string",
            enum: ["preference", "interest", "knowledge", "note"],
            description: "Category of the memory",
            default: "note",
          },
        },
        required: ["fact"],
      },
    },
  },
  // Recall past research and memories
  {
    type: "function",
    function: {
      name: "recall",
      description:
        "Search and retrieve past research sessions and stored memories. Use this to find previous research on similar topics, recall user preferences, or build on past work.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query to find relevant memories or past research",
          },
          type: {
            type: "string",
            enum: ["research", "memory", "all"],
            description: "Type of recall",
            default: "all",
          },
        },
        required: ["query"],
      },
    },
  },
  // Get recent research history
  {
    type: "function",
    function: {
      name: "get_research_history",
      description:
        "Get the user's recent research history. Use this to understand what the user has been researching lately and build on previous work.",
      parameters: {
        type: "object",
        properties: {
          limit: {
            type: "number",
            description: "Number of recent researches to return",
            default: 5,
          },
        },
      },
    },
  },
  // Deep research - autonomous multi-source investigation
  {
    type: "function",
    function: {
      name: "deep_research",
      description:
        "Conduct deep autonomous research on a topic by searching the web and gathering information from multiple sources. Use this for complex questions that require gathering information from many websites. This is more thorough than browse_web but takes longer.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Research topic or question",
          },
          maxSources: {
            type: "number",
            description: "Maximum number of sources to investigate",
            default: 5,
          },
        },
        required: ["query"],
      },
    },
  },
];

// Execute a tool call
async function executeTool(
  name: string,
  args: Record<string, any>,
  projectId: string,
  userId: string,
  orgId?: string
): Promise<any> {
  switch (name) {
    case "search_documents": {
      // Multi-project: if no projectId or "all", search across all org projects
      let allResults: any[] = [];
      const topK = Math.min(args.topK || 5, 10);

      if (!projectId || projectId === "all") {
        // Fetch all projects for this org and search each
        const orgProjects = orgId
          ? await prisma.project.findMany({ where: { orgId }, select: { id: true, name: true }, take: 10 })
          : [];
        const searches = await Promise.allSettled(
          orgProjects.map(p => retrieve({ query: args.query, projectId: p.id, topK: 3 }))
        );
        for (let i = 0; i < searches.length; i++) {
          const s = searches[i];
          if (s.status === "fulfilled") {
            const r = s.value.results || [];
            allResults.push(...r.map((x: any) => ({ ...x, _projectName: orgProjects[i].name, _projectId: orgProjects[i].id })));
          }
        }
        // Sort by score descending, take top results
        allResults.sort((a, b) => (b.score || 0) - (a.score || 0));
        allResults = allResults.slice(0, topK * 2);
      } else {
        const response = await retrieve({ query: args.query, projectId, topK });
        allResults = (response.results || []).map((r: any) => ({ ...r, _projectId: projectId }));
      }

      const chunkIds = allResults.map((r: any) => r.id).filter(Boolean);
      let docIdMap: Map<string, string> = new Map();
      if (chunkIds.length > 0) {
        try {
          const chunks = await prisma.chunk.findMany({
            where: { id: { in: chunkIds } },
            select: { id: true, documentId: true },
          });
          chunks.forEach((c) => docIdMap.set(c.id, c.documentId));
        } catch {}
      }
      return {
        results: allResults.map((r: any) => ({
          documentId: docIdMap.get(r.id) || r.metadata?.documentId,
          documentTitle: r.documentTitle || r.metadata?.documentTitle || "Unknown",
          projectName: r._projectName,
          content: r.content?.substring(0, 500) + (r.content?.length > 500 ? "..." : ""),
          relevance: r.score,
          sectionPath: r.metadata?.sectionPath,
        })),
        count: allResults.length,
      };
    }

    case "get_document_tree": {
      try {
        const tree = await buildDocumentTree(args.documentId);
        // Return a simplified tree for the agent
        const simplify = (node: any, depth = 0): any => {
          if (depth > 3) return { id: node.id, content: node.content.substring(0, 80) };
          return {
            id: node.id,
            content: node.content.substring(0, 150),
            type: node.type,
            childCount: node.children?.length || 0,
            children: node.children?.slice(0, 8).map((c: any) => simplify(c, depth + 1)),
          };
        };
        return {
          documentId: args.documentId,
          tree: simplify(tree.root),
          nodeCount: tree.nodeCount,
          depth: tree.depth,
        };
      } catch (err: any) {
        return { error: err.message };
      }
    }

    case "get_document_section": {
      const chunks = await prisma.chunk.findMany({
        where: {
          documentId: args.documentId,
          OR: [
            { sectionPath: args.sectionPath },
            { sectionPath: { contains: args.sectionPath } },
          ],
        },
        orderBy: { chunkIndex: "asc" },
        take: 10,
      });
      return {
        sectionPath: args.sectionPath,
        chunks: chunks.map((c) => ({
          id: c.id,
          content: c.content,
          chunkIndex: c.chunkIndex,
        })),
        totalChunks: chunks.length,
      };
    }

    case "list_project_documents": {
      const docs = await prisma.document.findMany({
        where: { projectId },
        select: {
          id: true,
          title: true,
          webUrl: true,
          createdAt: true,
          chunks: { select: { id: true }, take: 1 },
        },
        orderBy: { createdAt: "desc" },
        take: Math.min(args.limit || 20, 50),
      });
      // Get chunk counts separately
      const chunkCounts = await Promise.all(
        docs.map((d) => prisma.chunk.count({ where: { documentId: d.id } }))
      );
      return {
        documents: docs.map((d, i) => ({
          id: d.id,
          title: d.title,
          url: d.webUrl,
          chunkCount: chunkCounts[i],
          indexedAt: d.createdAt,
        })),
        total: docs.length,
      };
    }

    case "browse_web": {
      try {
        const result = await browseWeb({
          url: args.url,
          action: args.action,
          selector: args.selector,
          text: args.text,
          extractWhat: args.extractWhat,
          maxWaitMs: args.maxWaitMs,
        });
        return result;
      } catch (error: any) {
        return { error: error.message || "Web browsing failed" };
      }
    }

    case "deep_research": {
      try {
        const result = await deepResearch(args.query, args.maxSources || 5);
        return result;
      } catch (error: any) {
        return { error: error.message || "Deep research failed" };
      }
    }

    case "remember": {
      try {
        await rememberFact(userId, orgId, args.fact, args.category || "note");
        return { success: true, message: "Remembered: " + args.fact.substring(0, 100) };
      } catch (error: any) {
        return { error: error.message || "Failed to remember" };
      }
    }

    case "recall": {
      try {
        const searchResults = await searchMemory(userId, orgId, args.query);
        return {
          research: searchResults.relevantResearch.slice(0, 5),
          notes: searchResults.matchingNotes.slice(0, 10),
        };
      } catch (error: any) {
        return { error: error.message || "Failed to recall" };
      }
    }

    case "get_research_history": {
      try {
        const history = await getRecentResearch(userId, orgId, args.limit || 5);
        return {
          sessions: history.map((h) => ({
            id: h.id,
            query: h.query,
            answer: h.answer.substring(0, 500) + (h.answer.length > 500 ? "..." : ""),
            sources: h.sources,
            createdAt: h.createdAt,
          })),
          count: history.length,
        };
      } catch (error: any) {
        return { error: error.message || "Failed to get history" };
      }
    }

    default:
      return { error: `Unknown tool: ${name}` };
  }
}

// POST /v1/agent/research
researchAgentRoutes.post(
  "/v1/agent/research",
  zValidator("json", researchSchema),
  async (c) => {
    const auth = c.get("auth");
    const body = c.req.valid("json");

    // Only admins may override org context via X-Organization-Id.
    const orgId = getEffectiveOrgId(c.req.header("X-Organization-Id") || c.req.header("X-RetainDB-Org-Id"), auth);

    let projectId: string = "";
    let projectName = "All Projects";

    if (body.project && body.project !== "all") {
      const project = await resolveProjectReference(orgId, body.project, auth.isAdmin);
      if (project) {
        projectId = project.id;
        projectName = project.name;
      }
    } else if (!body.project || body.project === "all") {
      projectId = "all"; // signals multi-project search in executeTool
    }

    return streamSSE(c, async (stream) => {
      const send = async (type: string, data: Record<string, any>) => {
        await stream.writeSSE({
          data: JSON.stringify({ type, ...data }),
          event: type,
        });
      };

      try {
        await send("start", {
          projectId,
          projectName,
          query: body.query,
          timestamp: new Date().toISOString(),
        });

        // Get user memory for personalized context
        const userMemory = await getUserMemory(auth.userId ?? "", orgId);
        const recentResearch = await getRecentResearch(auth.userId ?? "", orgId, 3);

        const mode = body.mode || "research";
        const isMultiProject = projectId === "all";

        // ── System prompt by mode ────────────────────────────────────────
        let systemPrompt = "";

        if (mode === "onboarding") {
          systemPrompt = `You are an expert technical onboarding assistant. Your job is to read through all relevant documentation and produce a comprehensive, structured onboarding guide for a new team member.

## Instructions
1. Use search_documents to find ALL relevant documents (architecture, setup, processes, APIs, team norms)
2. Use get_document_tree and get_document_section to read key documents in depth
3. Organize your output as a clear onboarding guide with numbered sections
4. Include: overview, key concepts, setup steps, important files/systems, team conventions, FAQs
5. Use ## headings and - bullet points. Be thorough — this is someone's first day.

Project scope: ${projectName}`;
        } else if (mode === "contradiction") {
          systemPrompt = `You are a documentation auditor. Your job is to find contradictions, inconsistencies, and conflicts in the project's documentation.

## Instructions
1. Search for the same topics across multiple documents using search_documents
2. Look for: conflicting facts, outdated information, contradictory instructions, version mismatches
3. For each contradiction found, cite BOTH sources with their document names
4. Format contradictions as:
   **⚠️ Contradiction found:** [topic]
   - **Document A** says: [quote]
   - **Document B** says: [quote]
   - **Recommendation:** [which to trust or how to reconcile]
5. If no contradictions found, say so clearly and list what you checked.

Project scope: ${projectName}`;
        } else {
          systemPrompt = `You are an expert research agent with access to ${isMultiProject ? "ALL projects in the organization" : `the "${projectName}" knowledge base`} and web browsing.

## Your Capabilities
1. **search_documents** — semantic search across ${isMultiProject ? "all projects simultaneously" : "the knowledge base"}. Results include the project name each doc belongs to.
2. **get_document_tree** — explore a document's structure
3. **get_document_section** — read a specific section in depth
4. **browse_web** — fetch live web pages for current information
5. **deep_research** — multi-source web investigation
6. **remember** / **recall** — persist and retrieve facts

## Citation Rules — IMPORTANT
When you make a factual claim from a document, add an inline citation like [^1] immediately after it.
At the end of your answer, add a ## References section listing each [^1], [^2]... with document title and section.

## Guidelines
- Search the knowledge base FIRST, then supplement with web browsing if needed
- For questions about current prices, events, or external info — use browse_web
- Be thorough: search from multiple angles before concluding
- User has interacted ${userMemory.interactionCount} times — build on their context`;
        }

        if (recentResearch.length > 0) {
          systemPrompt += `\n\n## Recent Research History\n${recentResearch.map((r, i) => `${i + 1}. "${r.query}" — ${r.answer.substring(0, 200)}...`).join('\n')}`;
        }

        // ── Build messages (with conversation history for follow-ups) ────
        const conversationHistory = body.conversationHistory || [];
        const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
          { role: "system", content: systemPrompt },
          // Inject prior conversation turns so agent has context
          ...conversationHistory.map(m => ({ role: m.role as "user" | "assistant", content: m.content })),
          { role: "user", content: body.query },
        ];

        let stepCount = 0;
        const maxSteps = body.maxSteps;
        const sources: Array<{ documentId: string; title: string; path?: string }> = [];

        // Agentic loop
        while (stepCount < maxSteps) {
          stepCount++;

          await send("thinking", {
            step: stepCount,
            message: stepCount === 1 ? "Analyzing your question and planning web research..." : "Continuing research...",
          });

          const completion = await getOpenAIClient().chat.completions.create({
            model: body.model,
            messages,
            tools: AGENT_TOOLS,
            tool_choice: stepCount < maxSteps ? "auto" : "none",
            temperature: 0,
            max_tokens: 2048,
          });

          const choice = completion.choices[0];
          const assistantMessage = choice.message;
          messages.push(assistantMessage);

          // If the model chose to respond without tools, we're done
          if (choice.finish_reason === "stop" || !assistantMessage.tool_calls?.length) {
            const answer = assistantMessage.content || "";
            await send("answer", {
              content: answer,
              sources: sources,
              steps: stepCount,
            });
            break;
          }

          // Process tool calls
          const toolResults: OpenAI.Chat.Completions.ChatCompletionToolMessageParam[] = [];

          for (const toolCall of assistantMessage.tool_calls) {
            const toolName = toolCall.function.name;
            let toolArgs: Record<string, any> = {};
            try {
              toolArgs = JSON.parse(toolCall.function.arguments);
            } catch {}

            await send("tool_call", {
              step: stepCount,
              tool: toolName,
              args: toolArgs,
              callId: toolCall.id,
            });

            const result = await executeTool(toolName, toolArgs, projectId || "", auth.userId ?? "", orgId);

            // Track sources
            if (toolName === "search_documents" && result.results) {
              for (const r of result.results) {
                if (r.documentId && !sources.find((s) => s.documentId === r.documentId)) {
                  sources.push({
                    documentId: r.documentId,
                    title: r.documentTitle,
                    path: r.sectionPath,
                  });
                }
              }
            }

            await send("tool_result", {
              step: stepCount,
              tool: toolName,
              callId: toolCall.id,
              result: result,
              resultSummary:
                toolName === "search_documents"
                  ? `Found ${result.count || 0} relevant chunks`
                  : toolName === "get_document_tree"
                  ? `Tree has ${result.nodeCount || 0} nodes`
                  : toolName === "list_project_documents"
                  ? `Found ${result.total || 0} documents`
                  : "Done",
            });

            toolResults.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: JSON.stringify(result),
            });
          }

          messages.push(...toolResults);
        }

        // If we hit max steps without finishing, force a synthesis
        if (stepCount >= maxSteps) {
          await send("thinking", { step: stepCount, message: "Synthesizing findings..." });

          const finalCompletion = await getOpenAIClient().chat.completions.create({
            model: body.model,
            messages: [
              ...messages,
              {
                role: "user",
                content:
                  "Based on everything you've found so far, provide a comprehensive final answer to the original question.",
              },
            ],
            temperature: 0,
            max_tokens: 2048,
          });

          await send("answer", {
            content:
              finalCompletion.choices[0]?.message?.content || "Unable to synthesize answer.",
            sources,
            steps: stepCount,
          });
        }

        await send("done", { timestamp: new Date().toISOString() });
      } catch (err: any) {
        console.error("[ResearchAgent] Error:", err);
        await send("error", { message: err.message || "Research failed" });
      }
    });
  }
);

// ─── Proactive Insights ───────────────────────────────────────────────────
// GET /v1/agent/insights — surfaces gaps, stale docs, hot topics
researchAgentRoutes.get("/v1/agent/insights", async (c) => {
  const auth = c.get("auth");
  const orgId = getEffectiveOrgId(c.req.header("X-Organization-Id") || c.req.header("X-RetainDB-Org-Id"), auth);
  const userId = auth.userId || orgId || "anon";

  const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
  const ninetyDaysAgo = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000);

  // Run all queries in parallel
  const [projects, staleDocs, recentResearch, recentMemory] = await Promise.all([
    prisma.project.findMany({ where: { orgId }, select: { id: true, name: true }, take: 20 }),
    // Documents not updated in 30+ days
    prisma.document.findMany({
      where: { project: { orgId }, updatedAt: { lt: thirtyDaysAgo } },
      select: { id: true, title: true, updatedAt: true, project: { select: { name: true } } },
      orderBy: { updatedAt: "asc" },
      take: 10,
    }),
    // Recent research sessions to find hot topics
    getRecentResearch(userId, orgId, 20),
    // Recent memory notes
    getUserMemory(userId, orgId),
  ]);

  // Find stale docs > 90 days
  const veryStale = staleDocs.filter(d => d.updatedAt < ninetyDaysAgo);

  // Extract hot topics from recent research queries
  const queryWords = recentResearch
    .flatMap(r => r.query.toLowerCase().split(/\s+/).filter(w => w.length > 4))
    .reduce((acc: Record<string, number>, w) => { acc[w] = (acc[w] || 0) + 1; return acc; }, {});
  const hotTopics = Object.entries(queryWords)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
    .map(([word, count]) => ({ word, count }));

  // Docs that have been searched but may need updating
  const searchedTopics = recentResearch.slice(0, 10).map(r => r.query);

  // Insights array
  const insights: Array<{ type: string; title: string; description: string; action?: string; severity: "high" | "medium" | "low" }> = [];

  if (veryStale.length > 0) {
    insights.push({
      type: "stale_docs",
      title: `${veryStale.length} docs haven't been updated in 90+ days`,
      description: veryStale.slice(0, 3).map(d => `"${d.title}" (${d.project?.name ?? ""})`).join(", ") + (veryStale.length > 3 ? ` and ${veryStale.length - 3} more` : ""),
      action: "Review and update these documents",
      severity: "high",
    });
  }

  if (staleDocs.length > veryStale.length) {
    const mildStale = staleDocs.filter(d => d.updatedAt >= ninetyDaysAgo);
    if (mildStale.length > 0) {
      insights.push({
        type: "stale_docs_mild",
        title: `${mildStale.length} docs not updated in 30+ days`,
        description: mildStale.slice(0, 3).map(d => `"${d.title}" (${d.project?.name ?? ""})`).join(", "),
        action: "Consider reviewing for accuracy",
        severity: "medium",
      });
    }
  }

  if (hotTopics.length > 0) {
    insights.push({
      type: "hot_topics",
      title: `You've been researching "${hotTopics[0].word}" frequently`,
      description: `Top searched topics: ${hotTopics.map(t => `"${t.word}" (${t.count}x)`).join(", ")}. Consider creating dedicated docs for frequently queried topics.`,
      action: "Create documentation for these topics",
      severity: "medium",
    });
  }

  if (recentResearch.length > 0) {
    // Find queries where the answer was short (might indicate gaps)
    const possibleGaps = recentResearch.filter(r => r.answer.length < 200).slice(0, 3);
    if (possibleGaps.length > 0) {
      insights.push({
        type: "knowledge_gaps",
        title: `${possibleGaps.length} recent queries may have weak answers`,
        description: `Questions like "${possibleGaps[0].query}" returned short answers — your knowledge base might be missing coverage here.`,
        action: "Add documentation for these topics",
        severity: "medium",
      });
    }
  }

  if (projects.length > 0) {
    // Check if any project has no recent documents
    const projectDocCounts = await Promise.all(
      projects.map(p => prisma.document.count({ where: { projectId: p.id } }))
    );
    const emptyProjects = projects.filter((_, i) => projectDocCounts[i] === 0);
    if (emptyProjects.length > 0) {
      insights.push({
        type: "empty_projects",
        title: `${emptyProjects.length} project(s) have no indexed documents`,
        description: emptyProjects.map(p => `"${p.name}"`).join(", "),
        action: "Add data sources to these projects",
        severity: "low",
      });
    }
  }

  return c.json({
    insights,
    summary: {
      totalProjects: projects.length,
      staleDocCount: staleDocs.length,
      recentResearchCount: recentResearch.length,
      hotTopics: hotTopics.slice(0, 3),
    },
  });
});

// ─── Browse Agent (web-only, no project required) ─────────────────────────

const browseSchema = z.object({
  query: z.string().min(1).max(2000),
  url: z.string().optional(),
  model: z.string().optional().default("gpt-4o"),
  maxSteps: z.number().optional().default(6),
  mode: z.string().optional().default("research"),
  schema: z.string().optional(), // comma-separated field names for structured extraction
});

// Only web + memory tools — no doc search
const BROWSE_TOOLS = AGENT_TOOLS.filter((t) =>
  ["browse_web", "deep_research", "remember", "recall", "get_research_history"].includes(
    t.function.name
  )
);

// GET /v1/agent/browse/context — returns user's persistent memory + recent research
researchAgentRoutes.get("/v1/agent/browse/context", async (c) => {
  const auth = c.get("auth");
  const orgId = getEffectiveOrgId(c.req.header("X-Organization-Id") || c.req.header("X-RetainDB-Org-Id"), auth);
  const userId = auth.userId || orgId || "anon";

  const [userMemory, recentResearch] = await Promise.all([
    getUserMemory(userId, orgId),
    getRecentResearch(userId, orgId, 10),
  ]);

  let parsedNotes: Array<{ fact: string; category: string; timestamp: string }> = [];
  if (userMemory.selfNotes) {
    try {
      const parsed = JSON.parse(userMemory.selfNotes);
      if (Array.isArray(parsed)) parsedNotes = parsed;
    } catch {}
  }

  return c.json({
    notes: parsedNotes.slice(-20).reverse(), // Most recent first
    preferences: userMemory.preferences,
    interactionCount: userMemory.interactionCount,
    recentResearch: recentResearch.map(r => ({
      id: r.id,
      query: r.query,
      answer: r.answer.substring(0, 300),
      createdAt: r.createdAt,
    })),
  });
});

// POST /v1/agent/browse
researchAgentRoutes.post(
  "/v1/agent/browse",
  zValidator("json", browseSchema),
  async (c) => {
    const auth = c.get("auth");
    const body = c.req.valid("json");
    const orgId = getEffectiveOrgId(c.req.header("X-Organization-Id") || c.req.header("X-RetainDB-Org-Id"), auth);
    const userId = auth.userId || orgId || "anon";

    return streamSSE(c, async (stream) => {
      const send = async (type: string, data: Record<string, any>) => {
        await stream.writeSSE({
          data: JSON.stringify({ type, ...data }),
          event: type,
        });
      };

      try {
        await send("start", {
          query: body.query,
          url: body.url,
          timestamp: new Date().toISOString(),
        });

        const userMemory = await getUserMemory(userId, orgId);
        const recentResearch = await getRecentResearch(userId, orgId, 3);

        const ddgSearchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(body.query)}`;

        let systemPrompt = `You are an elite web research agent. You ALWAYS browse real web pages and provide comprehensive, accurate answers backed by real sources.

## CRITICAL RULES — follow these exactly:
1. **ALWAYS start by searching** — use browse_web on "${ddgSearchUrl}" to find relevant pages${body.url ? ` OR start directly at the provided URL: ${body.url}` : ""}
2. **Browse at least 3-5 real pages** before writing your final answer — never answer from just 1 page
3. **NEVER say "I cannot access" or "technical difficulties"** — if a page fails, immediately try another URL from search results
4. **Extract specific facts, data, company names, examples** — not generic advice
5. **Always format your response in clean markdown** with headers (##), bullet lists (- item), and bold (**text**) for key terms
6. **Structure your answer** with: an intro sentence, specific findings from real pages, and a summary

## Your Tools:
- **browse_web(url)** — visits any URL and extracts its full text content. Use this constantly.
- **deep_research(query)** — runs autonomous multi-page research on a topic. Use for broad discovery.
- **remember(fact, category)** — save important findings for future sessions
- **recall(query)** — retrieve past research

## Research Strategy:
1. Browse DuckDuckGo search results for "${body.query}"
2. Click into the top 3-5 most relevant links from those results
3. Extract specific names, examples, data points, and insights
4. If a page blocks you (paywalled, JS-heavy), skip it immediately and try the next result
5. Synthesize all findings into a clear, detailed answer

## Output Format:
- Use ## for main sections
- Use - for bullet points
- Bold (**term**) key company names, tools, or concepts
- End with a ## Sources section listing the pages you visited
- Write at least 300 words in your final answer`;

        // Mode-specific instructions
        const mode = body.mode || "research";
        const schema = body.schema;
        if (mode === "lead_gen" && schema) {
          const fields = schema.split(",").map((f: string) => f.trim());
          systemPrompt += `

## LEAD GEN MODE — CRITICAL INSTRUCTIONS:
Your goal is to find specific leads and return them as a STRUCTURED MARKDOWN TABLE.
Required fields for each lead: ${fields.join(", ")}

ALWAYS end your response with a markdown table like this:
| ${fields.join(" | ")} |
| ${fields.map(() => "---").join(" | ")} |
| value1 | value2 | ... |

Find at least 5-10 real leads with actual company names, real websites, and verifiable contact info.
Do not make up data — only include what you actually found on real web pages.`;
        } else if (mode === "price_monitor" && schema) {
          const fields = schema.split(",").map((f: string) => f.trim());
          systemPrompt += `

## PRICE MONITOR MODE — CRITICAL INSTRUCTIONS:
Extract pricing data as a STRUCTURED MARKDOWN TABLE.
Required fields: ${fields.join(", ")}

ALWAYS end your response with a markdown table:
| ${fields.join(" | ")} |
| ${fields.map(() => "---").join(" | ")} |
| value | value | ... |

Get exact prices from actual pricing pages. Include both monthly and annual if available.`;
        } else if (mode === "competitive" && schema) {
          const fields = schema.split(",").map((f: string) => f.trim());
          systemPrompt += `

## COMPETITIVE ANALYSIS MODE — CRITICAL INSTRUCTIONS:
Build a side-by-side competitive comparison as a STRUCTURED MARKDOWN TABLE.
Required fields: ${fields.join(", ")}

ALWAYS end your response with a markdown table:
| ${fields.join(" | ")} |
| ${fields.map(() => "---").join(" | ")} |
| value | value | ... |

Research each competitor's website, pricing page, and feature docs. Be specific and accurate.`;
        }

        // Inject persistent memory context
        if (userMemory?.selfNotes) {
          let parsedNotes: any[] = [];
          try { parsedNotes = JSON.parse(userMemory.selfNotes); } catch {}
          if (Array.isArray(parsedNotes) && parsedNotes.length > 0) {
            const recentNotes = parsedNotes.slice(-10).map((n: any) => `- [${n.category}] ${n.fact}`).join("\n");
            systemPrompt += `\n\n## Your Persistent Memory (from past sessions)\n${recentNotes}`;
          }
        }
        if (recentResearch.length > 0) {
          systemPrompt += `\n\n## Your Recent Research\n${recentResearch
            .map((r, i) => `${i + 1}. "${r.query}" — ${r.answer.substring(0, 200)}...`)
            .join("\n")}`;
        }

        const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
          { role: "system", content: systemPrompt },
          {
            role: "user",
            content: `Research this thoroughly using real web browsing: ${body.query}${body.url ? `\n\nStarting URL: ${body.url}` : `\n\nStart by searching DuckDuckGo: ${ddgSearchUrl}`}`,
          },
        ];

        let stepCount = 0;
        const maxSteps = body.maxSteps || 12;
        const sources: Array<{ url: string; title: string }> = [];
        let toolCallsMade = 0;

        while (stepCount < maxSteps) {
          stepCount++;

          await send("thinking", {
            step: stepCount,
            message: stepCount === 1
              ? "Planning research strategy..."
              : toolCallsMade < 3
              ? "Browsing more sources..."
              : "Synthesizing findings...",
          });

          // Force tool use for the first 4 steps
          const forceTools = toolCallsMade < 4;

          const completion = await getOpenAIClient().chat.completions.create({
            model: body.model,
            messages,
            tools: BROWSE_TOOLS,
            tool_choice: forceTools ? "required" : "auto",
            temperature: 0.1,
            max_tokens: 4096,
          });

          const choice = completion.choices[0];
          const assistantMessage = choice.message;
          messages.push(assistantMessage);

          if (choice.finish_reason === "stop" || !assistantMessage.tool_calls?.length) {
            // If the model tries to answer without browsing enough, force it to browse more
            if (toolCallsMade < 3 && stepCount < maxSteps) {
              messages.push({
                role: "user",
                content: `You haven't browsed enough sources yet (only ${toolCallsMade} so far). Please browse at least 3 real web pages before answering. Start with: browse_web("${ddgSearchUrl}")`,
              });
              continue;
            }
            const answer = assistantMessage.content || "";
            await send("answer", { content: answer, sources, steps: stepCount });
            // Save to persistent memory (fire-and-forget)
            saveResearchSession(userId, orgId, undefined, {
              query: body.query,
              answer,
              sources,
              steps: stepCount,
              durationMs: 0,
            }).catch(() => {});
            break;
          }

          const toolResults: OpenAI.Chat.Completions.ChatCompletionToolMessageParam[] = [];

          for (const toolCall of assistantMessage.tool_calls) {
            const toolName = toolCall.function.name;
            let toolArgs: Record<string, any> = {};
            try { toolArgs = JSON.parse(toolCall.function.arguments); } catch {}

            toolCallsMade++;
            await send("tool_call", {
              step: stepCount,
              tool: toolName,
              args: toolArgs,
              callId: toolCall.id,
            });

            const result = await executeTool(toolName, toolArgs, "", userId, orgId);

            // Track visited pages
            if (toolName === "browse_web" && toolArgs.url) {
              const existing = sources.find((s) => s.url === toolArgs.url);
              if (!existing) sources.push({ url: toolArgs.url, title: result.title || toolArgs.url });
            }
            if (toolName === "deep_research" && result.findings) {
              for (const f of result.findings) {
                if (f.url && !sources.find((s) => s.url === f.url)) {
                  sources.push({ url: f.url, title: f.title || f.url });
                }
              }
            }

            const resultSummary =
              toolName === "browse_web"
                ? `Extracted content from ${toolArgs.url}`
                : toolName === "deep_research"
                ? `Gathered ${result.findings?.length || 0} sources`
                : toolName === "remember"
                ? `Saved: ${toolArgs.fact?.substring(0, 60)}...`
                : toolName === "recall"
                ? `Found ${result.notes?.length || 0} relevant memories`
                : "Done";

            await send("tool_result", {
              step: stepCount,
              tool: toolName,
              callId: toolCall.id,
              result,
              resultSummary,
            });

            toolResults.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: JSON.stringify(result),
            });
          }

          messages.push(...toolResults);
        }

        await send("done", { timestamp: new Date().toISOString() });
      } catch (err: any) {
        console.error("[BrowseAgent] Error:", err);
        await send("error", { message: err.message || "Browse failed" });
      }
    });
  }
);
