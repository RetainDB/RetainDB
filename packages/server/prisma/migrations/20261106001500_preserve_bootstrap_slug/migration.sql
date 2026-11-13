BEGIN;

LOCK TABLE "organizations" IN SHARE ROW EXCLUSIVE MODE;

DO $$
DECLARE
  bootstrap_slug TEXT := 'default-oss-1';
  suffix INTEGER := 1;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM "organizations" WHERE "id" = 'default')
    AND EXISTS (SELECT 1 FROM "organizations" WHERE "slug" = 'default') THEN
    WHILE EXISTS (SELECT 1 FROM "organizations" WHERE "slug" = bootstrap_slug) LOOP
      suffix := suffix + 1;
      bootstrap_slug := 'default-oss-' || suffix;
    END LOOP;
    ALTER TABLE "organizations" ALTER COLUMN "ownerId" DROP NOT NULL;
    INSERT INTO "organizations" ("id", "name", "slug", "ownerId", "updatedAt")
    VALUES ('default', 'Default', bootstrap_slug, NULL, NOW());
  END IF;
END
$$;

COMMIT;
