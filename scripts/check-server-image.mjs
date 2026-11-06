import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";

const image = process.argv[2];
if (!image) throw new Error("Usage: node scripts/check-server-image.mjs <image>");

const engine = process.env.CONTAINER_ENGINE || "docker";
const timeout = 120_000;

function runCheck(args) {
  const name = `retaindb-runtime-check-${randomUUID()}`;
  try {
    const result = spawnSync(engine, [
      "run", "--name", name, "--network", "none", "--read-only", "--entrypoint", "node", image,
      ...args,
    ], { encoding: "utf8", timeout, killSignal: "SIGKILL" });
    if (result.error) throw result.error;
    assert.equal(result.status, 0, result.stderr || result.stdout);
    process.stdout.write(result.stdout);
  } finally {
    const cleanup = spawnSync(engine, ["rm", "--force", name], {
      encoding: "utf8", timeout: 30_000, killSignal: "SIGKILL",
    });
    if (cleanup.error) throw cleanup.error;
    assert.ok(cleanup.status === 0 || /no such (container|object)/i.test(cleanup.stderr), cleanup.stderr);
  }
}

runCheck([
  "--input-type=module", "--eval", `
    import assert from "node:assert/strict";
    import { createRequire } from "node:module";
    import { access } from "node:fs/promises";
    assert.equal(process.cwd(), "/app", "Preserve the runtime root for relative storage paths");
    const require = createRequire("/app/packages/server/package.json");
    const cli = require("prisma/package.json");
    const client = require("@prisma/client/package.json");
    const { PrismaClient } = require("@prisma/client");
    assert.equal(cli.version, client.version);
    await access("packages/server/dist/index.js");
    await access("packages/server/dist/engine/workers/native-embedding-process.js");
    await access("packages/server/prisma/schema.prisma");
    await access("packages/server/prisma/migrations");
    const db = new PrismaClient();
    assert.equal(typeof db.project.findMany, "function");
    await db.$disconnect();
    console.log(JSON.stringify({ prisma: cli.version, client: client.version }));
  `,
]);

runCheck([
  "--input-type=module", "--eval", `
    import { spawnSync } from "node:child_process";
    const result = spawnSync(process.execPath, ["packages/server/node_modules/prisma/build/index.js", "version"], {
      stdio: "inherit", env: { ...process.env, CHECKPOINT_DISABLE: "1" }, timeout: 90_000, killSignal: "SIGKILL",
    });
    if (result.error) throw result.error;
    process.exitCode = result.status ?? 1;
  `,
]);
