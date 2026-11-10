import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";

export async function waitForServerReady({ startedAt, health, logs, pause = delay }) {
  for (let attempt = 0; attempt < 120; attempt++) {
    const response = await health();
    if (response?.ok) {
      const output = await logs(startedAt);
      assert.ok(!output.includes("Local model pre-load failed"), "Local embedding model failed to initialize");
      if (output.includes("Local embedding model ready")) return;
    }
    await pause(2000);
  }
  throw new Error("Compose API and local embedding model did not become ready");
}

export async function waitForNativeCapture(read, { now = Date.now, pause = delay } = {}) {
  const deadline = now() + 150_000;
  let captured;
  while (now() < deadline) {
    captured = await read();
    if (captured.status === "indexed") break;
    await pause(Math.max(0, Math.min(500, deadline - now())));
  }
  return captured;
}
