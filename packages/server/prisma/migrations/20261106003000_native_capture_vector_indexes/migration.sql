CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
  ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw_idx
  ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS entities_embedding_hnsw_idx
  ON entities USING hnsw (embedding vector_cosine_ops);

DROP INDEX IF EXISTS chunks_embedding_idx;
DROP INDEX IF EXISTS memories_embedding_idx;
DROP INDEX IF EXISTS entities_embedding_idx;

CREATE INDEX IF NOT EXISTS messages_native_capture_pending_idx
  ON messages ("createdAt")
  WHERE metadata->>'source' = 'native_agent_hook'
    AND metadata->>'native_capture_state' IN ('pending', 'processing');
