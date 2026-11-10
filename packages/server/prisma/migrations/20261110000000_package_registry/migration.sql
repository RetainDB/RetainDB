CREATE TABLE IF NOT EXISTS "packages" (
  "id" TEXT NOT NULL,
  "orgId" TEXT NOT NULL DEFAULT 'default',
  "ecosystem" TEXT NOT NULL,
  "name" TEXT NOT NULL,
  "version" TEXT NOT NULL DEFAULT '',
  "registryUrl" TEXT,
  "autoSync" BOOLEAN NOT NULL DEFAULT false,
  "lastIndexedAt" TIMESTAMP(3),
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "packages_pkey" PRIMARY KEY ("id")
);

ALTER TABLE "packages" ADD COLUMN IF NOT EXISTS "projectId" TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS "packages_orgId_ecosystem_name_key" ON "packages"("orgId", "ecosystem", "name");
CREATE INDEX IF NOT EXISTS "packages_orgId_idx" ON "packages"("orgId");
