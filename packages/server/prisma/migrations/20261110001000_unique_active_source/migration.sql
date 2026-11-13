-- A project can hold at most one active source per connector name and type.
CREATE UNIQUE INDEX IF NOT EXISTS "sources_projectId_name_type_active_key"
  ON "sources" ("projectId", "name", "type")
  WHERE "deletedAt" IS NULL;
