import { prisma } from "../db/index.js";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export const DEFAULT_PROJECT_REF = "default";

function slugifyRef(input: string): string {
  return input.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

export async function resolveProjectReference(
  orgId: string,
  projectRef: string,
  isAdmin?: boolean
) {
  const ref = (projectRef || "").trim();
  if (!ref) return null;

  const isUuid = UUID_RE.test(ref);
  const normalizedSlug = slugifyRef(ref);
  const normalizedRef = ref.toLowerCase();

  if (isAdmin) {
    if (isUuid) {
      const byId = await prisma.project.findFirst({ where: { id: ref } });
      if (byId) return byId;
    }
    const exact = await prisma.project.findFirst({
      where: { OR: [{ name: ref }, { slug: ref }] },
    });
    if (exact) return exact;

    const normalized = await prisma.project.findFirst({
      where: {
        OR: [
          { slug: normalizedRef },
          { slug: normalizedSlug },
          { name: { equals: ref, mode: "insensitive" } },
        ],
      },
    });
    return normalized;
  }

  if (isUuid) {
    const byId = await prisma.project.findFirst({
      where: { id: ref, orgId },
    });
    if (byId) return byId;
  }

  const exact = await prisma.project.findFirst({
    where: { orgId, OR: [{ name: ref }, { slug: ref }] },
  });
  if (exact) return exact;

  return prisma.project.findFirst({
    where: {
      orgId,
      OR: [
        { slug: normalizedRef },
        { slug: normalizedSlug },
        { name: { equals: ref, mode: "insensitive" } },
      ],
    },
  });
}

export async function ensureProject(
  orgId: string,
  projectRef?: string,
  isAdmin?: boolean
) {
  const ref = (projectRef || "").trim() || DEFAULT_PROJECT_REF;
  const resolved = await resolveProjectReference(orgId, ref, isAdmin);
  if (resolved) return resolved;

  // Auto-create the project — never 404 on a missing project
  const slug = slugifyRef(ref);
  try {
    return await prisma.project.create({
      data: {
        orgId,
        name: ref,
        slug: slug || DEFAULT_PROJECT_REF,
      },
    });
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "P2002") {
      const concurrent = await resolveProjectReference(orgId, ref, isAdmin);
      if (concurrent) return concurrent;
    }
    throw error;
  }
}

export function getEffectiveOrgId(
  requestedOrgId: string | null | undefined,
  auth: { orgId: string; isAdmin?: boolean }
): string {
  if (requestedOrgId && auth.isAdmin) return requestedOrgId;
  return auth.orgId;
}
