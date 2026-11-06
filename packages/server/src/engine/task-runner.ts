import { getOpenAIClient } from "./openai-client.js";
import { prisma } from "../db/index.js";
import {
  newAntiDetectPage,
  navigatePage,
  clickBestMatch,
  fillForm,
  infiniteScroll,
  bypassCookieBanner,
  getPageLinks,
  takeScreenshot,
  getPageHTML,
} from "./browser-agent.js";
import { extractWithSchema, diffContent } from "./page-extractor.js";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface PlannedStep {
  type: "navigate" | "click" | "type" | "scroll" | "extract" | "search" | "done";
  description: string;
  params: Record<string, any>;
}

export interface SSEStream {
  writeSSE(event: { data: string; event?: string }): Promise<void>;
}

export interface RunOptions {
  maxSteps?: number;
  antiDetect?: boolean;
  outputSchema?: Record<string, string>;
  orgId: string;
  userId: string;
}

// ─── Cost tracking ─────────────────────────────────────────────────────────────

function estimateCost(inputTokens: number, outputTokens: number, model: string): number {
  const rates: Record<string, { in: number; out: number }> = {
    "gpt-4o": { in: 0.000005, out: 0.000015 },
    "gpt-4o-mini": { in: 0.00000015, out: 0.0000006 },
  };
  const r = rates[model] || rates["gpt-4o-mini"];
  return inputTokens * r.in + outputTokens * r.out;
}

// ─── Planner ─────────────────────────────────────────────────────────────────

async function planSteps(goal: string, memory: string): Promise<PlannedStep[]> {
  const resp = await getOpenAIClient().chat.completions.create({
    model: "gpt-4o",
    temperature: 0.2,
    max_tokens: 1500,
    messages: [
      {
        role: "system",
        content: `You are an expert web research planner. Given a goal, output a JSON array of steps to accomplish it using a web browser.

Step types:
- navigate: { url: string } — go to a URL
- search: { query: string } — search DuckDuckGo for a query (use this to find URLs first)
- click: { description: string } — click an element described in natural language
- type: { description: string, text: string } — type text into a field
- scroll: {} — scroll down to load more content
- extract: { description: string } — extract data from current page
- done: {} — task complete, synthesize results

Rules:
1. ALWAYS start with a search or navigate step to find relevant pages
2. Plan 4-10 steps for most tasks
3. Include multiple navigate/extract steps to gather thorough data
4. End with a done step

Return ONLY a JSON array of steps.`,
      },
      {
        role: "user",
        content: `Goal: ${goal}${memory ? `\n\nContext from memory:\n${memory}` : ""}

Create a step-by-step plan:`,
      },
    ],
  });

  const raw = resp.choices[0]?.message?.content?.trim() || "[]";
  try {
    const cleaned = raw.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");
    return JSON.parse(cleaned);
  } catch {
    return [
      { type: "search", description: "Search for information", params: { query: goal } },
      { type: "extract", description: "Extract relevant data", params: { description: goal } },
      { type: "done", description: "Complete", params: {} },
    ];
  }
}

// ─── Synthesizer ───────────────────────────────────────────────────────────────

async function synthesizeResult(
  goal: string,
  collectedData: Array<{ url: string; text: string; extracted: Record<string, any> }>,
  outputSchema: Record<string, string>
): Promise<any> {
  const dataStr = collectedData
    .map((d, i) => `--- Source ${i + 1}: ${d.url} ---\n${d.text.substring(0, 3000)}`)
    .join("\n\n");

  const hasSchema = Object.keys(outputSchema).length > 0;

  if (hasSchema) {
    return extractWithSchema(dataStr, outputSchema, "gpt-4o");
  }

  const resp = await getOpenAIClient().chat.completions.create({
    model: "gpt-4o",
    temperature: 0.3,
    max_tokens: 2000,
    messages: [
      {
        role: "system",
        content: `You synthesize research data into clear, structured answers. Be comprehensive and specific.`,
      },
      {
        role: "user",
        content: `Goal: ${goal}

Data collected:
${dataStr}

Synthesize a comprehensive answer to the goal. Include specific facts, numbers, and sources. Use markdown formatting.`,
      },
    ],
  });

  return resp.choices[0]?.message?.content || "No result generated.";
}

// ─── Page cache ────────────────────────────────────────────────────────────────

async function getCachedPage(orgId: string, url: string) {
  const cached = await prisma.pageCache.findUnique({ where: { orgId_url: { orgId, url } } });
  if (!cached) return null;
  // Fresh for 1 hour
  const ageMs = Date.now() - cached.lastFetched.getTime();
  if (ageMs > 3600000) return null;
  return cached;
}

async function cachePage(
  orgId: string,
  url: string,
  title: string,
  content: string,
  links: string[]
) {
  const crypto = await import("crypto");
  const contentHash = crypto.createHash("md5").update(content).digest("hex");

  const existing = await prisma.pageCache.findUnique({ where: { orgId_url: { orgId, url } } });

  if (existing) {
    const diff =
      existing.contentHash !== contentHash
        ? diffContent(existing.content, content)
        : { changed: false, added: [], removed: [] };

    await prisma.pageCache.update({
      where: { orgId_url: { orgId, url } },
      data: { contentHash, title, content, links, lastFetched: new Date() },
    });

    return { diff, isNew: false };
  }

  await prisma.pageCache.create({
    data: { orgId, url, contentHash, title: title || "", content, links, lastFetched: new Date() },
  });
  return { diff: { changed: false, added: [], removed: [] }, isNew: true };
}

// ─── Main loop ─────────────────────────────────────────────────────────────────

export async function runAgentTask(
  runId: string,
  goal: string,
  options: RunOptions,
  stream: SSEStream
): Promise<void> {
  const maxSteps = options.maxSteps || 12;
  const outputSchema = options.outputSchema || {};
  const { orgId, userId } = options;

  let page: any = null;
  let stepIndex = 0;
  let tokensUsed = 0;
  let costUsd = 0;
  let pagesVisited = 0;
  const collectedData: Array<{ url: string; text: string; extracted: Record<string, any> }> = [];
  let currentUrl = "";
  let currentText = "";
  let hitlPaused = false;

  const sendEvent = async (type: string, data: any) => {
    await stream.writeSSE({ event: type, data: JSON.stringify(data) });
  };

  try {
    // Update run status to PLANNING
    await prisma.agentRun.update({ where: { id: runId }, data: { status: "PLANNING" } });

    // Generate plan
    await sendEvent("status", { message: "Generating plan..." });
    const steps = await planSteps(goal, "");

    await prisma.agentRun.update({
      where: { id: runId },
      data: { status: "RUNNING", plan: steps as any },
    });

    await sendEvent("plan", { steps });

    // Launch browser
    if (options.antiDetect !== false) {
      page = await newAntiDetectPage();
    } else {
      const { chromium } = await import("playwright");
      const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
      page = await browser.newPage();
    }

    // Execute steps
    for (const step of steps) {
      if (stepIndex >= maxSteps) break;
      if (hitlPaused) break;

      const stepStart = Date.now();
      stepIndex++;

      await sendEvent("step_start", {
        index: stepIndex,
        description: step.description,
        stepType: step.type,
      });

      // Save step start
      const dbStep = await prisma.agentRunStep.create({
        data: {
          runId,
          stepIndex,
          type: step.type,
          description: step.description,
          input: step.params as any,
        },
      });

      let stepOutput: Record<string, any> = {};
      let screenshot: string | null = null;
      let confidence = 1.0;
      let error: string | null = null;

      try {
        if (step.type === "search") {
          const query = step.params.query || goal;
          const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
          const result = await navigatePage(page, searchUrl);
          currentUrl = result.finalUrl;
          currentText = result.text;
          pagesVisited++;

          // Cache it
          await cachePage(orgId, currentUrl, result.title, currentText, []);

          stepOutput = { url: currentUrl, resultsFound: currentText.length > 100 };

          // Extract links from search results
          const links = await getPageLinks(page);
          stepOutput.links = links.slice(0, 10);

          screenshot = await takeScreenshot(page);
          await sendEvent("screenshot", { data: screenshot, url: currentUrl });
        } else if (step.type === "navigate") {
          const url = step.params.url;
          if (!url) throw new Error("No URL provided for navigate step");

          // Check cache
          const cached = await getCachedPage(orgId, url);
          if (cached) {
            currentUrl = url;
            currentText = cached.content;
            stepOutput = { url, fromCache: true, title: cached.title };
            await sendEvent("step_done", { index: stepIndex, confidence: 1.0, summary: `Loaded ${url} (cached)` });
          } else {
            const result = await navigatePage(page, url);
            currentUrl = result.finalUrl;
            currentText = result.text;
            pagesVisited++;

            const { diff } = await cachePage(orgId, currentUrl, result.title, currentText, []);

            if (diff.changed) {
              await sendEvent("page_diff", { url: currentUrl, changes: [...diff.added.slice(0, 5), ...diff.removed.slice(0, 5).map((l) => `- ${l}`)] });
            }

            stepOutput = { url: currentUrl, title: result.title };
            screenshot = await takeScreenshot(page);
            await sendEvent("screenshot", { data: screenshot, url: currentUrl });
          }
        } else if (step.type === "click") {
          const desc = step.params.description || step.params.selector || "";
          const success = await clickBestMatch(page, desc);
          if (!success) {
            confidence = 0.3;
            error = `Could not find element matching "${desc}"`;
          } else {
            currentText = await page.evaluate(() => document.body?.innerText?.substring(0, 60000) || "");
            currentUrl = page.url();
            pagesVisited++;
            screenshot = await takeScreenshot(page);
            await sendEvent("screenshot", { data: screenshot, url: currentUrl });
          }
          stepOutput = { success, description: desc };
        } else if (step.type === "type") {
          await fillForm(page, { [step.params.description || "input"]: step.params.text || "" });
          stepOutput = { field: step.params.description, text: step.params.text };
        } else if (step.type === "scroll") {
          await infiniteScroll(page, 3);
          currentText = await page.evaluate(() => document.body?.innerText?.substring(0, 60000) || "");
          stepOutput = { scrolled: true };
        } else if (step.type === "extract") {
          // Get full HTML for extraction
          const html = await getPageHTML(page);
          const extracted = await extractWithSchema(html, outputSchema, "gpt-4o-mini");

          if (Object.keys(extracted).length > 0) {
            collectedData.push({ url: currentUrl, text: currentText, extracted });
            await sendEvent("extracted", { data: extracted });
            stepOutput = { extracted };
          } else {
            // Fall back to text collection
            collectedData.push({ url: currentUrl, text: currentText, extracted: {} });
            stepOutput = { textLength: currentText.length };
          }
          confidence = Object.keys(extracted).length > 0 ? 0.9 : 0.5;
        } else if (step.type === "done") {
          // Synthesize final result
          if (collectedData.length === 0) {
            collectedData.push({ url: currentUrl, text: currentText, extracted: {} });
          }

          await sendEvent("status", { message: "Synthesizing results..." });
          const result = await synthesizeResult(goal, collectedData, outputSchema);

          // Mark run complete
          await prisma.agentRun.update({
            where: { id: runId },
            data: {
              status: "DONE",
              result: typeof result === "string" ? { answer: result } : result,
              totalSteps: stepIndex,
              pagesVisited,
              tokensUsed,
              costUsd,
              completedAt: new Date(),
            },
          });

          const sources = collectedData.map((d) => ({ url: d.url }));
          await sendEvent("answer", { result, sources, totalSteps: stepIndex, costUsd });

          // Update step
          await prisma.agentRunStep.update({
            where: { id: dbStep.id },
            data: { output: { result: "done" }, durationMs: Date.now() - stepStart },
          });

          return; // Done!
        }
      } catch (stepError: any) {
        error = stepError.message;
        confidence = 0;
      }

      const durationMs = Date.now() - stepStart;

      // Update step in DB
      await prisma.agentRunStep.update({
        where: { id: dbStep.id },
        data: {
          output: stepOutput as any,
          screenshot,
          confidence,
          durationMs,
          error,
        },
      });

      await sendEvent("step_done", {
        index: stepIndex,
        confidence,
        summary: step.description,
        error,
      });

      // Update run progress
      await prisma.agentRun.update({
        where: { id: runId },
        data: { totalSteps: stepIndex, pagesVisited },
      });
    }

    // If we exhausted steps without hitting "done", synthesize anyway
    if (collectedData.length === 0 && currentText) {
      collectedData.push({ url: currentUrl, text: currentText, extracted: {} });
    }

    await sendEvent("status", { message: "Synthesizing results..." });
    const result = await synthesizeResult(goal, collectedData, outputSchema);

    await prisma.agentRun.update({
      where: { id: runId },
      data: {
        status: "DONE",
        result: typeof result === "string" ? { answer: result } : result,
        totalSteps: stepIndex,
        pagesVisited,
        tokensUsed,
        costUsd,
        completedAt: new Date(),
      },
    });

    const sources = collectedData.map((d) => ({ url: d.url }));
    await sendEvent("answer", { result, sources, totalSteps: stepIndex, costUsd });
  } catch (err: any) {
    await prisma.agentRun.update({
      where: { id: runId },
      data: { status: "FAILED", errorMsg: err.message, completedAt: new Date() },
    });
    await sendEvent("error", { message: err.message });
  } finally {
    if (page) {
      try {
        const context = page.context?.();
        await page.close();
        if (context) await context.close();
      } catch {}
    }
  }
}
