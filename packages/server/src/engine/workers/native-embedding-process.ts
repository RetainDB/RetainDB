import { embed } from "../embeddings.js";

process.on("message", async (message: unknown) => {
  if (!message || typeof message !== "object" || !("id" in message) || !("texts" in message)) return;
  const { id, texts } = message;
  if (typeof id !== "string" || !Array.isArray(texts) || !texts.every(text => typeof text === "string")) return;
  try {
    const vectors = await embed(texts);
    process.send?.({ id, vectors });
  } catch (error) {
    process.send?.({ id, error: error instanceof Error ? error.message : "Native embedding failed" });
  }
});

process.once("disconnect", () => process.exit(0));
