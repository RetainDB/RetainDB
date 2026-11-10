import assert from "node:assert/strict";
import { test } from "node:test";
import { waitForNativeCapture, waitForServerReady } from "../../../scripts/server-compose-readiness.mjs";

test("Compose readiness ignores a previous process's ready log after restart", async () => {
  const startedAt = "2026-01-02T03:04:05.000000000Z";
  let currentReady = false;
  let pauses = 0;
  const generations: Array<string | undefined> = [];
  await waitForServerReady({
    startedAt,
    health: async () => ({ ok: true }),
    logs: async (since?: string) => {
      generations.push(since);
      if (since !== startedAt || currentReady) return "Local embedding model ready";
      return "Pre-loading local embedding model";
    },
    pause: async () => { pauses++; currentReady = true; },
  });
  assert.equal(pauses, 1);
  assert.deepEqual(generations, [startedAt, startedAt]);
});

test("Compose readiness reports a current initialization failure despite older ready logs", async () => {
  const startedAt = "2026-01-02T03:04:05.000000000Z";
  await assert.rejects(waitForServerReady({
    startedAt,
    health: async () => ({ ok: true }),
    logs: async (since?: string) => since === startedAt
      ? "Local model pre-load failed"
      : "Local embedding model ready\nLocal model pre-load failed",
    pause: async () => {},
  }), /Local embedding model failed to initialize/);
});

test("Compose capture waits beyond one minute for a valid production embedding attempt", async () => {
  let elapsed = 0;
  const captured = await waitForNativeCapture(async () => ({
    status: elapsed >= 90_000 ? "indexed" : "processing",
  }), {
    now: () => elapsed,
    pause: async (milliseconds: number) => { elapsed += milliseconds; },
  });
  assert.equal(captured.status, "indexed");
  assert.equal(elapsed, 90_000);
});

test("Compose capture still terminates when indexing never completes", async () => {
  let elapsed = 0;
  const captured = await waitForNativeCapture(async () => ({ status: "processing" }), {
    now: () => elapsed,
    pause: async (milliseconds: number) => { elapsed += milliseconds; },
  });
  assert.equal(captured.status, "processing");
  assert.equal(elapsed, 150_000);
});
