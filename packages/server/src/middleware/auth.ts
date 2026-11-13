/**
 * OSS Auth middleware
 *
 * If RETAINDB_API_KEY is set: every request must send
 *   Authorization: Bearer ***
 * A key in the query string is deliberately not accepted, because URLs are
 * logged by proxies and browsers.
 *
 * If RETAINDB_API_KEY is NOT set: open access — no key required.
 * This is fine for local dev or a private network deployment.
 */
import { timingSafeEqual } from "node:crypto";
import type { Context, Next } from "hono";

export interface AuthContext {
  orgId: string;   // always "default" in OSS (single-tenant)
  userId?: string;
  authType: "api_key" | "open";
  // OSS: these are always true/undefined — kept for source compatibility
  isAdmin?: boolean;
  scopes?: string[];
  keyId?: string;
  actorId?: string;
}

function extractToken(raw: string | undefined): string {
  if (!raw) return "";
  const trimmed = raw.trim();
  return trimmed.match(/^Bearer\s+(.+)$/i)?.[1]?.trim() ?? trimmed;
}

export function verifyApiKey(candidate: string, configured: string): boolean {
  const left = Buffer.from(candidate);
  const right = Buffer.from(configured);
  if (left.length !== right.length || left.length === 0) return false;
  return timingSafeEqual(left, right);
}

export async function authMiddleware(c: Context, next: Next) {
  const configuredKey = process.env.RETAINDB_API_KEY;

  // No key configured → open access
  if (!configuredKey) {
    c.set("auth", { orgId: "default", authType: "open" } satisfies AuthContext);
    await next();
    return;
  }

  const token = extractToken(c.req.header("Authorization"));

  if (!verifyApiKey(token, configuredKey)) {
    return c.json({ error: "Unauthorized. Set Authorization: Bearer <RETAINDB_API_KEY>." }, 401);
  }

  c.set("auth", { orgId: "default", authType: "api_key" } satisfies AuthContext);
  await next();
}
