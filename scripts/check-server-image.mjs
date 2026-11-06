import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

const image = process.argv[2];
if (!image) throw new Error("Usage: node scripts/check-server-image.mjs <image>");

const engine = process.env.CONTAINER_ENGINE || "docker";
const result = spawnSync(engine, [
  "run", "--rm", "--network", "none", "--read-only", "--entrypoint", "node", image,
  "--input-type=module", "--eval", `
    import assert from "node:assert/strict";
    import { createRequire } from "node:module";
    import { access } from "node:fs/promises";
    const require = createRequire(import.meta.url);
    const cli = require("prisma/package.json");
    const client = require("@prisma/client/package.json");
    const { PrismaClient } = require("@prisma/client");
    assert.equal(cli.version, client.version);
    await access("dist/index.js");
    await access("prisma/schema.prisma");
    await access("prisma/migrations");
    const db = new PrismaClient();
    assert.equal(typeof db.project.findMany, "function");
    await db.$disconnect();
    console.log(JSON.stringify({ prisma: cli.version, client: client.version }));
  `,
], { encoding: "utf8" });

if (result.error) throw result.error;
assert.equal(result.status, 0, result.stderr || result.stdout);
process.stdout.write(result.stdout);

const version = spawnSync(engine, [
  "run", "--rm", "--network", "none", "--read-only", "--env", "CHECKPOINT_DISABLE=1",
  "--entrypoint", "node", image,
  "node_modules/prisma/build/index.js", "version",
], { encoding: "utf8" });

if (version.error) throw version.error;
assert.equal(version.status, 0, version.stderr || version.stdout);
process.stdout.write(version.stdout);
