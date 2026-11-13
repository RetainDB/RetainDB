import { getOpenAIClient } from "./openai-client.js";

/**
 * Extract structured data from HTML matching a given schema.
 * schema: { field_name: "string" | "number" | "boolean" | "string[]" }
 */
export async function extractWithSchema(
  html: string,
  schema: Record<string, string>,
  model = "gpt-4o-mini"
): Promise<Record<string, any>> {
  if (Object.keys(schema).length === 0) return {};

  const truncated = html.substring(0, 40000);
  const schemaDesc = Object.entries(schema)
    .map(([k, v]) => `  "${k}": ${v}`)
    .join(",\n");

  const resp = await getOpenAIClient().chat.completions.create({
    model,
    temperature: 0,
    max_tokens: 2000,
    messages: [
      {
        role: "system",
        content: `You are a data extraction engine. Extract structured data from HTML/text and return ONLY valid JSON.`,
      },
      {
        role: "user",
        content: `Extract the following fields from the page content and return as JSON:
{
${schemaDesc}
}

If a field is not found, use null.
For string[] fields, return an array of strings.

PAGE CONTENT:
${truncated}

Return ONLY the JSON object, no markdown, no explanation.`,
      },
    ],
  });

  const raw = resp.choices[0]?.message?.content?.trim() || "{}";
  try {
    // Strip markdown code fences if present
    const cleaned = raw.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");
    return JSON.parse(cleaned);
  } catch {
    return {};
  }
}

/**
 * Extract all tables from HTML as arrays of objects.
 */
export function extractTables(html: string): Array<Record<string, string>[]> {
  const tables: Array<Record<string, string>[]> = [];
  const tableRegex = /<table[^>]*>([\s\S]*?)<\/table>/gi;
  let tableMatch;

  while ((tableMatch = tableRegex.exec(html)) !== null) {
    const tableHtml = tableMatch[1];
    const headers: string[] = [];
    const rows: Record<string, string>[] = [];

    // Extract headers
    const headerRegex = /<th[^>]*>([\s\S]*?)<\/th>/gi;
    let headerMatch;
    while ((headerMatch = headerRegex.exec(tableHtml)) !== null) {
      headers.push(stripHtml(headerMatch[1]));
    }

    if (headers.length === 0) continue;

    // Extract rows
    const rowRegex = /<tr[^>]*>([\s\S]*?)<\/tr>/gi;
    let rowMatch;
    while ((rowMatch = rowRegex.exec(tableHtml)) !== null) {
      const cells: string[] = [];
      const cellRegex = /<td[^>]*>([\s\S]*?)<\/td>/gi;
      let cellMatch;
      while ((cellMatch = cellRegex.exec(rowMatch[1])) !== null) {
        cells.push(stripHtml(cellMatch[1]));
      }
      if (cells.length === headers.length) {
        const row: Record<string, string> = {};
        headers.forEach((h, i) => (row[h] = cells[i]));
        rows.push(row);
      }
    }

    if (rows.length > 0) tables.push(rows);
  }

  return tables;
}

/**
 * Detect pricing information from a page.
 */
export async function detectPricing(
  html: string
): Promise<Array<{ plan: string; price: string; features: string[] }>> {
  const schema = {
    plans: "array of { plan: string, price: string, features: string[] }",
  };

  const truncated = html.substring(0, 40000);
  const resp = await getOpenAIClient().chat.completions.create({
    model: "gpt-4o-mini",
    temperature: 0,
    max_tokens: 2000,
    messages: [
      {
        role: "system",
        content: `Extract pricing plans from the page. Return JSON array of plans.`,
      },
      {
        role: "user",
        content: `Find all pricing plans/tiers on this page and return as JSON:
[{ "plan": "plan name", "price": "price string", "features": ["feature1", "feature2"] }]

PAGE:
${truncated}

Return ONLY the JSON array.`,
      },
    ],
  });

  const raw = resp.choices[0]?.message?.content?.trim() || "[]";
  try {
    const cleaned = raw.replace(/^```(?:json)?\n?/, "").replace(/\n?```$/, "");
    return JSON.parse(cleaned);
  } catch {
    return [];
  }
}

/**
 * Diff two text contents and return a summary of changes.
 */
export function diffContent(
  oldContent: string,
  newContent: string
): { added: string[]; removed: string[]; changed: boolean } {
  if (oldContent === newContent) return { added: [], removed: [], changed: false };

  const oldLines = oldContent.split("\n").map((l) => l.trim()).filter(Boolean);
  const newLines = newContent.split("\n").map((l) => l.trim()).filter(Boolean);

  const oldSet = new Set(oldLines);
  const newSet = new Set(newLines);

  const added = newLines.filter((l) => !oldSet.has(l)).slice(0, 20);
  const removed = oldLines.filter((l) => !newSet.has(l)).slice(0, 20);

  return { added, removed, changed: added.length > 0 || removed.length > 0 };
}

function stripHtml(html: string): string {
  return html
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
