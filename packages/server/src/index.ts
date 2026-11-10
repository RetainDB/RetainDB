import "./env.js";
import { serve } from "@hono/node-server";
import { createNodeApp } from "./api/app.js";
import { api } from "./api/routes.js";
import { startScheduler } from "./engine/scheduler.js";
import { startEmbeddingWorker } from "./engine/workers/embedding-worker.js";
import { recoverInterruptedJobs } from "./engine/sync-queue.js";
import { startNativeCaptureWorker } from "./engine/native-capture.js";

process.on("uncaughtException", (err) => {
  console.error("[RetainDB] Uncaught Exception:", err.message);
});

process.on("unhandledRejection", (reason) => {
  console.error("[RetainDB] Unhandled Rejection:", reason);
});

const app = createNodeApp({ routeApp: api });
const port = Number(process.env.PORT || 3000);

// Pre-load local embedding model on startup to avoid cold-start latency
async function preloadEmbeddings() {
  const mode = process.env.EMBEDDING_MODE || (process.env.OPENAI_API_KEY ? "openai" : "local");
  if (mode === "local" || mode === "hybrid") {
    try {
      console.log("[Startup] Pre-loading local embedding model (nomic-embed-text)...");
      const { embedSingleLocal } = await import("./engine/embeddings-local.js");
      await embedSingleLocal("warmup");
      console.log("[Startup] ✓ Local embedding model ready");
    } catch (err) {
      console.warn("[Startup] Local model pre-load failed (will load on first request):", err);
    }
  }
}

serve({ fetch: app.fetch, port }, async () => {
  console.log(`\n🧠 RetainDB running on http://localhost:${port}\n`);

  if (process.send) process.send("ready");
  startNativeCaptureWorker();

  // Start async embedding worker
  startEmbeddingWorker().catch((err) =>
    console.error("[Startup] Failed to start embedding worker:", err)
  );

  // Pre-load embedding model (fire-and-forget)
  preloadEmbeddings().catch(() => {});

  // Recover any jobs interrupted by a previous crash
  recoverInterruptedJobs().catch((err) =>
    console.warn("[Startup] Job recovery error:", err)
  );

  // Start the scheduled sync runner
  if (process.env.DISABLE_SCHEDULER !== "true") {
    startScheduler();
  }
});

export default app;
