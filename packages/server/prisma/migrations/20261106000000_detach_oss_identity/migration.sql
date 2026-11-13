-- OSS scopes are plain organization identifiers, not cloud identity records.
-- Keep project/source data and their project-level relationships intact.
ALTER TABLE "projects" DROP CONSTRAINT IF EXISTS "projects_orgId_fkey";
ALTER TABLE "sources" DROP CONSTRAINT IF EXISTS "sources_orgId_fkey";
