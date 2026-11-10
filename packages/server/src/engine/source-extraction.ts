/**
 * Post-sync source-level extraction.
 *
 * After all pages/files for a source are indexed, this runs an LLM pass to
 * generate a rich source profile: purpose, brand identity, design tokens,
 * key topics, target audience, content structure — stored as a searchable
 * `__source_profile__` document.
 *
 * Usage (called from connectors after sync):
 *   import { generateSourceProfile } from "../engine/source-extraction.js";
 *   await generateSourceProfile(sourceId, projectId, { sourceType: "web", rootUrl });
 */

import { getOpenAIClient } from "./openai-client.js";
import { prisma } from "../db/index.js"; // used for document queries
import { ingestDocument } from "./ingest.js";

export interface SourceProfileOptions {
  sourceType: string;
  rootUrl?: string;
}

interface AggregatedDesign {
  colors: string[];
  fonts: string[];
  themeColors: string[];
}

function aggregateDesign(documents: { metadata: any }[]): AggregatedDesign {
  const colorFreq = new Map<string, number>();
  const fontFreq = new Map<string, number>();
  const themeColors = new Set<string>();

  for (const doc of documents) {
    const meta = doc.metadata as any;
    if (!meta) continue;

    // Colors aggregated from page-level metadata (set by html-structure.ts)
    const colors: string[] = meta.design?.colors || [];
    for (const c of colors) {
      colorFreq.set(c, (colorFreq.get(c) || 0) + 1);
    }

    const fonts: string[] = meta.design?.fonts || [];
    for (const f of fonts) {
      fontFreq.set(f, (fontFreq.get(f) || 0) + 1);
    }

    if (meta.design?.themeColor) {
      themeColors.add(meta.design.themeColor);
    }
  }

  // Sort by frequency — most common across pages = brand colors
  const sortedColors = [...colorFreq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([c]) => c);

  const sortedFonts = [...fontFreq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([f]) => f);

  return {
    colors: sortedColors,
    fonts: sortedFonts,
    themeColors: [...themeColors],
  };
}

function sampleDocumentContent(documents: { title: string; content: string; metadata: any }[]): string {
  // Take up to 6 documents — prioritize shorter ones (landing pages, about pages)
  // that have good signal-to-noise ratio
  const sorted = [...documents].sort((a, b) => a.content.length - b.content.length);
  const sample = sorted.slice(0, 6);

  return sample
    .map((doc, i) => `--- Page ${i + 1}: ${doc.title} ---\n${doc.content.slice(0, 1200)}`)
    .join("\n\n");
}

export async function generateSourceProfile(
  sourceId: string,
  projectId: string,
  opts: SourceProfileOptions,
): Promise<void> {
  try {
    // Fetch all documents for this source (lightweight — title + metadata + first 2k of content)
    const documents = await prisma.document.findMany({
      where: { sourceId, deletedAt: null },
      select: { id: true, title: true, content: true, metadata: true, webUrl: true },
    });

    if (documents.length === 0) return;

    const design = aggregateDesign(documents);
    const sampleContent = sampleDocumentContent(documents as any);

    const isWeb = opts.sourceType === "web" || opts.sourceType === "url";
    const isRepo = opts.sourceType === "github" || opts.sourceType === "gitlab";
    const isPdf = opts.sourceType === "pdf";
    const isDataset = opts.sourceType === "dataset";

    let prompt: string;

    if (isDataset) {
      prompt = `You are analyzing a dataset that has been indexed. Based on the schema and sample content below, extract a comprehensive profile.

Files indexed: ${documents.length}

Schema and sample:
${sampleContent}

Respond with JSON only:
{
  "name": "dataset name",
  "summary": "2-3 sentence description of what this dataset contains, its coverage, and potential use cases",
  "type": "one of: tabular, time_series, text_corpus, graph, image_metadata, log_data, survey, other",
  "domain": "the subject area or industry (e.g. healthcare, finance, NLP, computer vision)",
  "mainTopics": ["topic or concept this dataset covers"],
  "keyColumns": ["important column names and what they represent"],
  "scale": "description of size (rows/columns)",
  "targetAudience": "who would use this dataset (e.g. ML researchers, analysts)",
  "useCases": ["use case 1", "use case 2"],
  "limitations": ["known limitation or caveat"]
}`;
    } else if (isWeb) {
      prompt = `You are analyzing a website that has been indexed. Based on the page content below, extract a comprehensive profile.

Website: ${opts.rootUrl || ""}
Pages indexed: ${documents.length}

Sample page content:
${sampleContent}

Respond with JSON only:
{
  "name": "website/brand name",
  "tagline": "short tagline if present",
  "summary": "2-3 sentence description of what this website/product is and does",
  "type": "one of: marketing_site, documentation, blog, ecommerce, saas_product, landing_page, portfolio, news, forum, other",
  "targetAudience": "who this is built for",
  "mainTopics": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "keyFeatures": ["feature or offering 1", "feature 2", "feature 3"],
  "industry": "the industry or vertical",
  "language": "primary language (e.g. English)",
  "visualStyle": "brief description of the visual/design aesthetic (e.g. 'minimal dark theme with yellow accents', 'clean white SaaS', 'editorial serif typography')",
  "contentSections": ["section or page names found"],
  "callsToAction": ["main CTA text found on site"],
  "technologies": ["tech stack hints if detectable, e.g. Next.js, Stripe, etc."]
}`;
    } else if (isRepo) {
      prompt = `You are analyzing a code repository that has been indexed. Based on the file content below, extract a comprehensive profile.

Repository: ${opts.rootUrl || ""}
Files indexed: ${documents.length}

Sample file content:
${sampleContent}

Respond with JSON only:
{
  "name": "repository/project name",
  "summary": "2-3 sentence description of what this project does",
  "type": "one of: library, framework, application, cli_tool, api_service, documentation, example_project, other",
  "primaryLanguage": "main programming language",
  "languages": ["list of languages used"],
  "frameworks": ["frameworks and major dependencies"],
  "mainTopics": ["key technical topics or problem domains"],
  "architecture": "brief description of the architecture or structure",
  "targetAudience": "who would use this project",
  "keyModules": ["main modules, packages, or components"],
  "hasTests": true,
  "hasDocs": true
}`;
    } else if (isPdf) {
      prompt = `You are analyzing a PDF document that has been indexed. Based on the content below, extract a comprehensive profile.

Pages indexed: ${documents.length}

Sample content:
${sampleContent}

Respond with JSON only:
{
  "title": "document title",
  "summary": "2-3 sentence description of what this document covers",
  "type": "one of: research_paper, report, manual, guide, specification, legal, presentation, other",
  "mainTopics": ["topic1", "topic2", "topic3"],
  "keyPoints": ["key finding or point 1", "key point 2", "key point 3"],
  "targetAudience": "intended reader",
  "language": "primary language",
  "estimatedLength": "short/medium/long"
}`;
    } else {
      prompt = `You are analyzing indexed content from source type: ${opts.sourceType}. Based on the content below, extract a comprehensive profile.

Items indexed: ${documents.length}

Sample content:
${sampleContent}

Respond with JSON only:
{
  "name": "source name",
  "summary": "2-3 sentence description of what this source contains",
  "mainTopics": ["topic1", "topic2", "topic3"],
  "contentType": "description of the type of content",
  "targetAudience": "who this content is for",
  "language": "primary language"
}`;
    }

    const res = await getOpenAIClient().chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      max_tokens: 800,
      response_format: { type: "json_object" },
    });

    const text = res.choices[0]?.message?.content?.trim() || "{}";
    let profile: Record<string, any> = {};
    try {
      profile = JSON.parse(text);
    } catch {
      console.warn("[SourceExtraction] Failed to parse LLM response");
      return;
    }

    // Merge in aggregated design tokens
    if (isWeb) {
      profile.design = {
        colors: design.colors,
        fonts: design.fonts,
        themeColors: design.themeColors,
      };
    }

    profile.pageCount = documents.length;
    profile.sourceType = opts.sourceType;
    if (opts.rootUrl) profile.rootUrl = opts.rootUrl;

    // Build the profile document content — human-readable, fully searchable
    const profileContent = buildProfileContent(profile, isWeb, isRepo, isPdf, isDataset);

    // Store as a special synthetic document — externalId is stable so it upserts on re-sync
    await ingestDocument({
      sourceId,
      projectId,
      externalId: "__source_profile__",
      title: `Source Profile: ${profile.name || profile.title || opts.rootUrl || sourceId}`,
      content: profileContent,
      metadata: {
        source_type: opts.sourceType,
        is_source_profile: true,
        profile,
      },
      sourceType: opts.sourceType,
      ingestionProfile: "plain_text",
      skipEntityExtraction: true,
    });

    console.log(`[SourceExtraction] Profile generated for source ${sourceId} (${documents.length} docs)`);
  } catch (err: any) {
    // Non-fatal — log and continue
    console.warn(`[SourceExtraction] Failed for source ${sourceId}:`, err?.message ?? err);
  }
}

function buildProfileContent(
  profile: Record<string, any>,
  isWeb: boolean,
  isRepo: boolean,
  isPdf: boolean,
  isDataset = false,
): string {
  const lines: string[] = [];

  // Header
  const name = profile.name || profile.title || "Unknown";
  lines.push(`# ${name}`);
  if (profile.tagline) lines.push(`*${profile.tagline}*`);
  lines.push("");

  // Summary
  if (profile.summary) {
    lines.push("## Summary");
    lines.push(profile.summary);
    lines.push("");
  }

  // Type + audience
  const typeLine = [profile.type || profile.contentType, profile.industry].filter(Boolean).join(" · ");
  if (typeLine) lines.push(`**Type:** ${typeLine}`);
  if (profile.targetAudience) lines.push(`**Audience:** ${profile.targetAudience}`);
  if (profile.language || profile.primaryLanguage) lines.push(`**Language:** ${profile.language || profile.primaryLanguage}`);
  if (profile.pageCount) lines.push(`**Pages/Files indexed:** ${profile.pageCount}`);
  if (profile.rootUrl) lines.push(`**URL:** ${profile.rootUrl}`);
  lines.push("");

  // Main topics
  if (profile.mainTopics?.length) {
    lines.push("## Main Topics");
    lines.push(profile.mainTopics.join(", "));
    lines.push("");
  }

  // Key features / key points
  const features = profile.keyFeatures || profile.keyPoints;
  if (features?.length) {
    lines.push(isRepo ? "## Key Modules / Features" : isPdf ? "## Key Points" : "## Key Features");
    for (const f of features) lines.push(`- ${f}`);
    lines.push("");
  }

  // Web-specific sections
  if (isWeb) {
    if (profile.contentSections?.length) {
      lines.push("## Content Sections");
      lines.push(profile.contentSections.join(", "));
      lines.push("");
    }

    if (profile.callsToAction?.length) {
      lines.push("## Calls to Action");
      lines.push(profile.callsToAction.join(" · "));
      lines.push("");
    }

    if (profile.design?.colors?.length || profile.design?.fonts?.length) {
      lines.push("## Design Identity");
      if (profile.visualStyle) lines.push(`Style: ${profile.visualStyle}`);
      if (profile.design.colors?.length) lines.push(`Colors: ${profile.design.colors.join(", ")}`);
      if (profile.design.fonts?.length) lines.push(`Fonts: ${profile.design.fonts.join(", ")}`);
      if (profile.design.themeColors?.length) lines.push(`Theme: ${profile.design.themeColors.join(", ")}`);
      lines.push("");
    }

    if (profile.technologies?.length) {
      lines.push("## Technologies");
      lines.push(profile.technologies.join(", "));
      lines.push("");
    }
  }

  // Repo-specific sections
  if (isRepo) {
    if (profile.languages?.length) {
      lines.push("## Languages & Frameworks");
      lines.push([...profile.languages, ...(profile.frameworks || [])].join(", "));
      lines.push("");
    }
    if (profile.architecture) {
      lines.push("## Architecture");
      lines.push(profile.architecture);
      lines.push("");
    }
    if (profile.keyModules?.length) {
      lines.push("## Key Modules");
      lines.push(profile.keyModules.join(", "));
      lines.push("");
    }
  }

  // Dataset-specific sections
  if (isDataset) {
    if (profile.domain) lines.push(`**Domain:** ${profile.domain}`);
    if (profile.scale) lines.push(`**Scale:** ${profile.scale}`);
    lines.push("");

    if (profile.keyColumns?.length) {
      lines.push("## Key Columns");
      for (const c of profile.keyColumns) lines.push(`- ${c}`);
      lines.push("");
    }

    if (profile.useCases?.length) {
      lines.push("## Use Cases");
      for (const u of profile.useCases) lines.push(`- ${u}`);
      lines.push("");
    }

    if (profile.limitations?.length) {
      lines.push("## Limitations");
      for (const l of profile.limitations) lines.push(`- ${l}`);
      lines.push("");
    }
  }

  return lines.join("\n").trim();
}
