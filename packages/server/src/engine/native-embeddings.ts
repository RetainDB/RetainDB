import { fork, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";

interface PendingEmbedding {
  id: string;
  resolve: (vectors: number[][]) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
  expected: number;
  failure?: Error;
}

let worker: ChildProcess | undefined;
let pending: PendingEmbedding | undefined;

function loaderArguments(): string[] {
  const args: string[] = [];
  for (let index = 0; index < process.execArgv.length; index++) {
    if (["--require", "--import", "--loader"].includes(process.execArgv[index])) {
      args.push(process.execArgv[index], process.execArgv[++index]);
    }
  }
  return args;
}

function rejectPending(error: Error): void {
  const request = pending;
  pending = undefined;
  if (!request) return;
  clearTimeout(request.timer);
  request.reject(request.failure || error);
}

function startWorker(): ChildProcess {
  if (worker) return worker;
  const source = import.meta.url.endsWith(".ts");
  const child = fork(new URL(source ? "./workers/native-embedding-process.ts" : "./engine/workers/native-embedding-process.js", import.meta.url), [], {
    execArgv: source ? loaderArguments() : [],
    stdio: ["ignore", "ignore", "ignore", "ipc"],
    env: { ...process.env, TSX_DISABLE_CACHE: "1" },
  });
  worker = child;
  child.unref();
  child.on("message", (message: unknown) => {
    const request = pending;
    if (!request || !message || typeof message !== "object" || !("id" in message) || message.id !== request.id || request.failure) return;
    if ("error" in message && typeof message.error === "string") {
      rejectPending(new Error(message.error));
      return;
    }
    const vectors = "vectors" in message ? message.vectors : null;
    if (!Array.isArray(vectors) || vectors.length !== request.expected || !vectors.every(row =>
      Array.isArray(row) && row.length > 0 && row.every(value => typeof value === "number" && Number.isFinite(value)))) {
      rejectPending(new Error("Native embedding process returned invalid vectors"));
      return;
    }
    clearTimeout(request.timer);
    pending = undefined;
    request.resolve(vectors);
  });
  child.on("error", error => {
    if (worker === child && pending) pending.failure = error;
    child.kill("SIGKILL");
  });
  child.on("close", (code, signal) => {
    if (worker !== child) return;
    worker = undefined;
    rejectPending(new Error(`Native embedding process exited before completing the request (code ${code}, signal ${signal})`));
  });
  child.channel?.unref();
  return child;
}

export function embedNativeCapture(texts: string[], remainingMs = 120_000): Promise<number[][]> {
  if (!texts.length) return Promise.resolve([]);
  if (pending) return Promise.reject(new Error("Native embedding process already has an active request"));
  const child = startWorker();
  const configured = Number(process.env.NATIVE_CAPTURE_EMBEDDING_TIMEOUT_MS || 120_000);
  const timeout = Math.max(1, Math.min(remainingMs, Number.isFinite(configured) ? Math.min(120_000, Math.max(100, configured)) : 120_000));
  return new Promise((resolve, reject) => {
    const id = randomUUID();
    const timer = setTimeout(() => {
      if (pending?.id !== id) return;
      pending.failure = new Error(`Native embedding attempt timed out after ${timeout}ms`);
      child.kill("SIGKILL");
    }, timeout);
    pending = { id, resolve, reject, timer, expected: texts.length };
    child.send({ id, texts }, error => {
      if (error && pending?.id === id) {
        pending.failure = error;
        child.kill("SIGKILL");
      }
    });
  });
}

export async function stopNativeEmbeddingProcess(): Promise<void> {
  const child = worker;
  if (!child) return;
  if (pending) pending.failure = new Error("Native embedding process stopped");
  await new Promise<void>(resolve => {
    child.once("close", () => resolve());
    child.kill("SIGKILL");
  });
}

process.once("exit", () => { worker?.kill("SIGKILL"); });
