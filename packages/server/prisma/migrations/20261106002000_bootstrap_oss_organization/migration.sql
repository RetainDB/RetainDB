BEGIN;

ALTER TABLE "organizations" ALTER COLUMN "ownerId" DROP NOT NULL;
ALTER TABLE "organizations" ADD CONSTRAINT "organizations_owner_required_except_oss"
  CHECK ("ownerId" IS NOT NULL OR "id" = 'default');

INSERT INTO "organizations" ("id", "name", "slug", "ownerId", "updatedAt")
VALUES ('default', 'Default', 'default', NULL, NOW())
ON CONFLICT ("id") DO NOTHING;

ALTER TABLE "projects" ADD CONSTRAINT "projects_orgId_fkey"
  FOREIGN KEY ("orgId") REFERENCES "organizations"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "sources" ADD CONSTRAINT "sources_orgId_fkey"
  FOREIGN KEY ("orgId") REFERENCES "organizations"("id") ON DELETE CASCADE ON UPDATE CASCADE;

COMMIT;
