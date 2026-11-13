/**
 * Autosubscribe - Auto-index package dependencies
 * Parses package.json/requirements.txt and automatically indexes documentation
 * Like Nia's autosubscribe feature
 */

import { Octokit } from "@octokit/rest";
import { db } from "../db/index.js";
import { syncNpmPackage } from "../connectors/npm_package.js";
import { syncPyPIPackage } from "../connectors/pypi_package.js";

interface PackageDependency {
  name: string;
  version: string;
  ecosystem: "npm" | "pypi" | "maven" | "cargo";
}

/**
 * Parse package.json and extract dependencies
 */
export async function parsePackageJson(content: string): Promise<PackageDependency[]> {
  try {
    const pkg = JSON.parse(content);
    const deps: PackageDependency[] = [];

    // Regular dependencies
    if (pkg.dependencies) {
      for (const [name, version] of Object.entries(pkg.dependencies)) {
        deps.push({
          name,
          version: String(version),
          ecosystem: "npm",
        });
      }
    }

    // Dev dependencies
    if (pkg.devDependencies) {
      for (const [name, version] of Object.entries(pkg.devDependencies)) {
        deps.push({
          name,
          version: String(version),
          ecosystem: "npm",
        });
      }
    }

    return deps;
  } catch (error) {
    console.error("Failed to parse package.json:", error);
    return [];
  }
}

/**
 * Parse requirements.txt and extract Python dependencies
 */
export function parseRequirementsTxt(content: string): PackageDependency[] {
  const deps: PackageDependency[] = [];
  const lines = content.split("\n");

  for (const line of lines) {
    const trimmed = line.trim();

    // Skip comments and empty lines
    if (!trimmed || trimmed.startsWith("#")) continue;

    // Parse "package==version" or "package>=version"
    const match = trimmed.match(/^([a-zA-Z0-9_-]+)([><=!~]+)?(.+)?$/);
    if (match) {
      deps.push({
        name: match[1],
        version: match[3] || "latest",
        ecosystem: "pypi",
      });
    }
  }

  return deps;
}

/**
 * Parse Cargo.toml (Rust)
 */
export function parseCargoToml(content: string): PackageDependency[] {
  const deps: PackageDependency[] = [];

  // Simple parser for [dependencies] section
  const depsMatch = content.match(/\[dependencies\]([\s\S]*?)(\[|$)/);
  if (!depsMatch) return deps;

  const depsSection = depsMatch[1];
  const lines = depsSection.split("\n");

  for (const line of lines) {
    const match = line.match(/^([a-zA-Z0-9_-]+)\s*=\s*"([^"]+)"/);
    if (match) {
      deps.push({
        name: match[1],
        version: match[2],
        ecosystem: "cargo",
      });
    }
  }

  return deps;
}

/**
 * Auto-detect and parse dependency file
 */
export function parseDependencyFile(
  filename: string,
  content: string
): Promise<PackageDependency[]> {
  return parseDependencyFileAsync(filename, content);
}

async function parseDependencyFileAsync(
  filename: string,
  content: string
): Promise<PackageDependency[]> {
  if (filename === "package.json") {
    return parsePackageJson(content);
  } else if (filename === "requirements.txt" || filename.endsWith(".txt")) {
    return parseRequirementsTxt(content);
  } else if (filename === "Cargo.toml") {
    return parseCargoToml(content);
  }

  return [];
}

/**
 * Fetch dependency file from GitHub repo
 */
export async function fetchDependencyFile(params: {
  owner: string;
  repo: string;
  branch?: string;
  githubToken?: string;
}): Promise<{ filename: string; content: string } | null> {
  const { owner, repo, branch = "main", githubToken } = params;

  const octokit = new Octokit({
    auth: githubToken || process.env.GITHUB_TOKEN,
  });

  // Try common dependency files
  const files = [
    "package.json",
    "requirements.txt",
    "Cargo.toml",
    "pom.xml", // Maven
    "build.gradle", // Gradle
  ];

  for (const filename of files) {
    try {
      const { data } = await octokit.repos.getContent({
        owner,
        repo,
        path: filename,
        ref: branch,
      });

      if ("content" in data && data.content) {
        const content = Buffer.from(data.content, "base64").toString("utf-8");
        return { filename, content };
      }
    } catch (error) {
      // File doesn't exist, try next
      continue;
    }
  }

  return null;
}

/**
 * Resolve documentation URL for a package
 */
export async function resolveDocsUrl(dep: PackageDependency): Promise<string | null> {
  switch (dep.ecosystem) {
    case "npm":
      // Try npmjs.com first
      try {
        const response = await fetch(`https://registry.npmjs.org/${dep.name}`);
        const data = await response.json();

        // Try homepage, repository, or default to npmjs
        if (data.homepage) return data.homepage;
        if (data.repository?.url) {
          const url = data.repository.url
            .replace("git+", "")
            .replace(".git", "")
            .replace("git://", "https://");
          return url;
        }

        return `https://www.npmjs.com/package/${dep.name}`;
      } catch (error) {
        return `https://www.npmjs.com/package/${dep.name}`;
      }

    case "pypi":
      // PyPI packages
      try {
        const response = await fetch(`https://pypi.org/pypi/${dep.name}/json`);
        const data = await response.json();

        if (data.info.home_page) return data.info.home_page;
        if (data.info.project_urls?.Documentation) {
          return data.info.project_urls.Documentation;
        }

        return `https://pypi.org/project/${dep.name}/`;
      } catch (error) {
        return `https://pypi.org/project/${dep.name}/`;
      }

    case "cargo":
      // Rust crates
      return `https://docs.rs/${dep.name}`;

    default:
      return null;
  }
}

/**
 * Main autosubscribe function
 * Automatically index all dependencies from a project
 */
export async function autosubscribe(params: {
  projectId: string;
  orgId: string;
  source: {
    type: "github" | "local";
    owner?: string;
    repo?: string;
    branch?: string;
    filePath?: string; // For local files
  };
  indexLimit?: number; // Max packages to index (to avoid blowing up costs)
}): Promise<{
  discovered: number;
  indexed: number;
  skipped: number;
  errors: string[];
}> {
  const { projectId, orgId, source, indexLimit = 50 } = params;

  const result = {
    discovered: 0,
    indexed: 0,
    skipped: 0,
    errors: [] as string[],
  };

  try {
    // Step 1: Fetch dependency file
    let depFile: { filename: string; content: string } | null = null;

    if (source.type === "github" && source.owner && source.repo) {
      depFile = await fetchDependencyFile({
        owner: source.owner,
        repo: source.repo,
        branch: source.branch,
      });
    } else if (source.type === "local" && source.filePath) {
      const fs = await import("fs");
      const content = fs.readFileSync(source.filePath, "utf-8");
      const filename = source.filePath.split("/").pop() || "package.json";
      depFile = { filename, content };
    }

    if (!depFile) {
      result.errors.push("No dependency file found");
      return result;
    }

    // Step 2: Parse dependencies
    const dependencies = await parseDependencyFile(depFile.filename, depFile.content);
    result.discovered = dependencies.length;

    console.log(`📦 Discovered ${dependencies.length} dependencies`);

    // Step 3: Index each dependency (up to limit)
    const toIndex = dependencies.slice(0, indexLimit);

    for (const dep of toIndex) {
      try {
        // Check if already indexed
        const existing = await db.package.findFirst({
          where: {
            orgId,
            ecosystem: dep.ecosystem,
            name: dep.name,
          },
        });

        if (existing) {
          console.log(`⏭️  Skipping ${dep.name} (already indexed)`);
          result.skipped++;
          continue;
        }

        // Resolve docs URL
        const docsUrl = await resolveDocsUrl(dep);
        if (!docsUrl) {
          result.errors.push(`No docs URL for ${dep.name}`);
          continue;
        }

        console.log(`📚 Indexing ${dep.name} from ${docsUrl}`);

        // Create package record
        await db.package.create({
          data: {
            orgId,
            name: dep.name,
            ecosystem: dep.ecosystem,
            version: dep.version,
            registryUrl: docsUrl,
            autoSync: true,
          },
        });

        // Index the package docs
        const packageSourceId = await ensurePackageSource(projectId, orgId, dep);
        if (dep.ecosystem === "npm") {
          await syncNpmPackage(packageSourceId, projectId, { packageName: dep.name });
        } else if (dep.ecosystem === "pypi") {
          await syncPyPIPackage(packageSourceId, projectId, { packageName: dep.name });
        }

        result.indexed++;
        console.log(`✅ Indexed ${dep.name}`);

        // Small delay to avoid rate limits
        await new Promise((resolve) => setTimeout(resolve, 1000));
      } catch (error) {
        result.errors.push(`Failed to index ${dep.name}: ${error}`);
        console.error(`❌ Failed to index ${dep.name}:`, error);
      }
    }

    if (dependencies.length > indexLimit) {
      result.errors.push(
        `Only indexed first ${indexLimit} of ${dependencies.length} packages (limit reached)`
      );
    }

    return result;
  } catch (error) {
    result.errors.push(`Autosubscribe failed: ${error}`);
    return result;
  }
}

/**
 * Auto-sync: Re-index packages when they're updated
 */
export async function ensurePackageSource(
  projectId: string,
  orgId: string,
  pkg: { name: string; ecosystem: string }
): Promise<string> {
  const lookup = { projectId, orgId, name: pkg.name, type: pkg.ecosystem, deletedAt: null } as const;
  const existing = await db.source.findFirst({ where: lookup, select: { id: true } });
  if (existing) return existing.id;
  try {
    const created = await db.source.create({
      data: { projectId, orgId, name: pkg.name, type: pkg.ecosystem, status: "READY" },
      select: { id: true },
    });
    return created.id;
  } catch (error) {
    // Concurrent syncs lose the unique-index race; adopt the winner's row.
    if ((error as { code?: string }).code !== "P2002") throw error;
    const winner = await db.source.findFirstOrThrow({ where: lookup, select: { id: true } });
    return winner.id;
  }
}

export async function autoSyncPackages(orgId: string): Promise<void> {
  const packages = await db.package.findMany({
    where: {
      orgId,
      autoSync: true,
    },
  });

  console.log(`🔄 Auto-syncing ${packages.length} packages...`);

  for (const pkg of packages) {
    try {
      // Check if package has been updated (compare lastIndexedAt with registry)
      const shouldUpdate = await checkPackageUpdated(pkg);

      if (shouldUpdate) {
        console.log(`📦 Re-indexing ${pkg.name}...`);

        if (!pkg.projectId) {
          console.warn(`⏭️  Skipping ${pkg.name} (no project context)`);
          continue;
        }
        const packageSourceId = await ensurePackageSource(pkg.projectId, orgId, pkg);
        if (pkg.ecosystem === "npm") {
          await syncNpmPackage(packageSourceId, pkg.projectId, { packageName: pkg.name });
        } else if (pkg.ecosystem === "pypi") {
          await syncPyPIPackage(packageSourceId, pkg.projectId, { packageName: pkg.name });
        }

        await db.package.update({
          where: { id: pkg.id },
          data: { lastIndexedAt: new Date() },
        });
      }
    } catch (error) {
      console.error(`Failed to sync ${pkg.name}:`, error);
    }
  }

  console.log("✅ Auto-sync complete");
}

/**
 * Check if package has been updated since last index
 */
async function checkPackageUpdated(pkg: any): Promise<boolean> {
  if (!pkg.lastIndexedAt) return true;

  const daysSinceIndex = Math.floor(
    (Date.now() - pkg.lastIndexedAt.getTime()) / (1000 * 60 * 60 * 24)
  );

  // Re-index if more than 7 days old
  return daysSinceIndex > 7;
}
