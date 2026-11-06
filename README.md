<p align="center">
  <img src="https://retaindb.com/retaindb-mark.svg" alt="RetainDB" height="64" />
</p>

<h1 align="center">RetainDB</h1>

<p align="center">
  Memory infrastructure for AI agents, from local coding-agent recall to hosted product memory.
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@retaindb/sdk"><img src="https://img.shields.io/npm/v/@retaindb/sdk?label=%40retaindb%2Fsdk" alt="@retaindb/sdk on npm" /></a>
  <a href="https://www.npmjs.com/package/@retaindb/mcp"><img src="https://img.shields.io/npm/v/@retaindb/mcp?label=%40retaindb%2Fmcp" alt="@retaindb/mcp on npm" /></a>
  <img src="https://img.shields.io/badge/local-first-success" alt="Local first" />
  <img src="https://img.shields.io/badge/license-Apache%202.0%20%2F%20BSL%201.1-blue" alt="License" />
</p>

RetainDB gives agents durable memory: decisions, preferences, workflows, corrections, project facts, and session handoffs that survive across conversations.

It ships as two related products:

- **RetainDB Local**: persistent memory for coding agents. Runs on your machine, no cloud account required.
- **RetainDB Server / Cloud**: memory API infrastructure for apps, teams, SDKs, auth, connectors, dashboards, and managed deployments.

## Why RetainDB

Agents are most useful when they remember the right things and forget the noise. RetainDB is built around memory quality, not just vector storage:

- **Capture** prompts, tool events, file edits, failures, session summaries, and explicit memories.
- **Filter** low-signal noise before it pollutes recall.
- **Structure** memories into semantic facts, procedures, corrections, preferences, and events.
- **Retrieve** with lexical search, vector similarity, graph signals, RRF fusion, and reranking.
- **Reinforce** memories that get reused, so important knowledge gets stronger over time.
- **Handoff** context between sessions, agents, and workflows.
- **Route** normal coding context so agents stop rereading whole files, logs, and sessions every turn.
- **Compress** repeated context with deltas, hard token budgets, and tool-output cleanup.

## Quick Start

### RetainDB Local

Start local memory in one command:

```bash
npx -y @retaindb/local
```

RetainDB Local starts an API on `http://localhost:3111` and a viewer on `http://localhost:3113`.

Or use docker compose to run RetainDB in container:

```bash
git clone https://github.com/retaindb/retaindb
```

Set the correct directory path for `RETAINDB_HOME` map in `docker-compose-local.yml`.
Run docker compose:

```bash
cd retaindb
docker compose -f docker-compose-local.yml up -d
```
To see the container logs:

```bash
docker compose -f docker-compose-local.yml logs -f -n 10
```

RetainDB Local inside container starts an API on `http://localhost:3111` and a viewer on `http://localhost:3113`.

Run the demo:

```bash
npx -y @retaindb/local demo
```

Wire local memory into Codex, Claude Code, and OpenCode:

```bash
npx -y @retaindb/local connect all
```

Run the MCP bridge:

```bash
RETAINDB_BASE_URL=http://localhost:3111 npx -y @retaindb/local mcp
```

Local mode uses an atomic disk snapshot plus append-only journal under `~/.retaindb/`. It does not require Postgres, Redis, Kafka, Qdrant, Cloudflare, or API keys.

### Self-Hosted Server

Use this path when you want the full API server with Postgres and pgvector.

```bash
git clone https://github.com/retaindb/retaindb
cd retaindb
docker compose up
```

Server runs on `http://localhost:3000`.

To verify the packaged Prisma client and migration CLI without network access or a running database:

```bash
docker build -t retaindb-runtime-check -f packages/server/Dockerfile .
node scripts/check-server-image.mjs retaindb-runtime-check
```

### Node + Postgres

```bash
git clone https://github.com/retaindb/retaindb
cd retaindb
cp .env.example packages/server/.env
pnpm install
pnpm --filter @retaindb/server run db:push
pnpm dev:server
```

Postgres must have the `pgvector` extension enabled.

## RetainDB Local

RetainDB Local is an agent-native memory runtime for coding workflows.

What it includes:

- One-process local runtime on `:3111`
- Built-in viewer on `:3113`
- MCP tools for `context`, `remember`, `recall`, `handoff`, `session_history`, and `forget`
- Auto-capture hooks for coding-agent lifecycle events
- Codex, Claude Code, and OpenCode setup snippets
- Session replay API and step-through replay viewer
- Clickable concept graph
- BM25 + vector + graph retrieval with RRF and reranking
- Low-signal capture filtering
- Semantic, procedural, correction, and summary memory typing
- Recall reinforcement with access counts, last-access timestamps, and memory strength
- Consolidation with duplicate cleanup, session rollups, and stale weak-memory decay
- Optional local transformer embeddings
- Local benchmark reports in `~/.retaindb/benchmarks/`
- Token-budgeted context packs for files, memory, code maps, and tool output
- Delta compression so agents receive only what changed since the last context pack
- Tool-output compression that keeps errors, failing tests, and stack traces while dropping noise
- Code maps that show relevant files and symbols without dumping the whole repo

Useful commands:

```bash
retaindb                  # start local memory
retaindb demo             # seed and search demo memories
retaindb benchmark        # run recall/latency benchmark
retaindb connect all      # write agent config snippets
retaindb connect all --install
retaindb import-jsonl     # import Claude-style JSONL transcripts
retaindb consolidate      # dedupe and roll up memories
retaindb reembed          # refresh embeddings
retaindb doctor           # print local status
```

Optional local transformer embeddings:

```bash
retaindb install-embeddings
RETAINDB_EMBEDDING_PROVIDER=local-transformers retaindb reembed
```

If the local native embedding runtime is unavailable, RetainDB falls back to hash-vector embeddings so local memory keeps working.

### Context Router and Token Reduction

RetainDB Local can also act as a local context router for coding agents. Instead of sending huge files, repeated terminal logs, or the same project summary every turn, agents can ask RetainDB for a compact context pack.

```bash
curl -X POST http://localhost:3111/v1/context/pack \
  -H "Content-Type: application/json" \
  -d '{
    "query": "fix auth middleware tests",
    "cwd": ".",
    "files": ["src/auth.ts", "tests/auth.test.ts"],
    "token_budget": 1200
  }'
```

The response includes:

- `context`: packed memory, relevant file chunks, code map, and compressed tool output
- `context_hash`: stable hash for the pack
- `delta_context`: only the changed parts when `previous_context_hash` is supplied
- `estimated_tokens` and `estimated_delta_tokens`
- `changed` and `removed` entries for delta-aware agents

Useful endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/context/pack` | Build a token-budgeted context pack |
| `POST` | `/v1/context/delta` | Return only context changes since a previous pack |
| `POST` | `/v1/context/compress-output` | Compress test/build/tool output |
| `POST` | `/v1/context/code-map` | Build a compact file and symbol map |

MCP tools expose the same flow as `context_pack`, `context_delta`, `compress_output`, and `code_map`.

## Platform Features

RetainDB Server and Cloud are the API/product infrastructure layer around the same memory model.

Core platform features:

- **Structured memory extraction** from conversations, sessions, and explicit writes
- **Semantic search + BM25 + reranking** for context retrieval
- **Memory graph** with relationships like `updates`, `contradicts`, `supports`, `extends`, and `derives`
- **Temporal validity** with `validFrom` and `validUntil` so stale facts can be superseded instead of silently reused
- **Versioned memories** for corrections and evolving user/project state
- **Project and user scoping** for multi-agent and product use cases
- **Connectors** for GitHub, docs sites, PDFs, Notion, Confluence, Slack, Discord, package docs, and more
- **Framework adapters** for Vercel AI SDK, LangChain, and LangGraph
- **MCP server** for agent clients that speak tools
- **Self-hosted Postgres + pgvector path** for teams that want to own the stack
- **Cloud track** for hosted auth, dashboards, SDK defaults, lifecycle email, usage controls, and managed reliability

RetainDB Local optimizes for coding-agent memory on one machine. RetainDB Server and Cloud optimize for product memory across users, apps, teams, and integrations.

## SDK

```bash
npm install @retaindb/sdk
```

### Local

```ts
import { RetainDBContext } from "@retaindb/sdk";

const db = new RetainDBContext({
  baseUrl: "http://localhost:3111",
  project: "my-agent",
});
```

### Cloud or Self-Hosted API

```ts
import { RetainDBContext } from "@retaindb/sdk";

const db = new RetainDBContext({
  apiKey: process.env.RETAINDB_API_KEY,
  baseUrl: "https://api.retaindb.com",
  project: "my-agent",
});
```

Store a memory:

```ts
await db.addMemory({
  project: "my-agent",
  user_id: "user_123",
  memory_type: "preference",
  content: "User prefers concise answers with concrete next steps.",
});
```

Retrieve context before an LLM call:

```ts
const { context, memories } = await db.query({
  project: "my-agent",
  user_id: "user_123",
  query: "What should I remember about this user?",
  include_memories: true,
});
```

Ingest a session:

```ts
await db.ingestSession({
  project: "my-agent",
  session_id: "session_abc",
  user_id: "user_123",
  messages: [
    { role: "user", content: "I'm building a SaaS in Next.js, Prisma, and Postgres." },
    { role: "assistant", content: "Got it. I will remember that stack." },
  ],
});
```

### Vercel AI SDK

```ts
import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";
import { RetainDBContext } from "@retaindb/sdk";

const db = new RetainDBContext({
  apiKey: process.env.RETAINDB_API_KEY,
  baseUrl: process.env.RETAINDB_BASE_URL,
  project: "my-agent",
});

const { context } = await db.query({
  user_id,
  query: userMessage,
  include_memories: true,
});

const result = await streamText({
  model: openai("gpt-4o"),
  system: `You are a helpful assistant.\n\nRelevant memory:\n${context}`,
  messages,
});
```

### LangChain and LangGraph

RetainDB also ships adapters for LangChain and LangGraph workflows, so memory can be retrieved before a chain/graph step and written back after the agent learns something useful.

## MCP

Start the MCP server through the Local package:

```bash
RETAINDB_BASE_URL=http://localhost:3111 npx -y @retaindb/local mcp
```

Example MCP config:

```json
{
  "mcpServers": {
    "retaindb": {
      "command": "npx",
      "args": ["-y", "@retaindb/local", "mcp"],
      "env": {
        "RETAINDB_BASE_URL": "http://localhost:3111",
        "RETAINDB_PROJECT": "my-agent"
      }
    }
  }
}
```

Core tools:

| Tool | Purpose |
| --- | --- |
| `context` | Retrieve packed context for the current task |
| `remember` | Save a durable memory |
| `recall` | Search memory |
| `handoff` | Share session context |
| `session_history` | Inspect prior session memory |
| `forget` | Delete or deactivate memory |

## REST API

Local runs on `:3111`. The self-hosted server runs on `:3000` by default.

If `RETAINDB_API_KEY` is set, protected deployments require `Authorization: Bearer <RETAINDB_API_KEY>`. Local development can run without a key.

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/memory` | Store a memory |
| `POST` | `/v1/memory/bulk` | Store multiple memories |
| `POST` | `/v1/memory/search` | Search memories |
| `POST` | `/v1/context/query` | Retrieve packed context |
| `POST` | `/v1/context/pack` | Build a token-budgeted coding context pack |
| `POST` | `/v1/context/delta` | Return changed context since a prior pack |
| `POST` | `/v1/context/compress-output` | Compress noisy tool output |
| `POST` | `/v1/context/code-map` | Return relevant files and symbols |
| `POST` | `/v1/memory/ingest/session` | Ingest messages and work events |
| `GET` | `/v1/memory/session/:sessionId` | List session memories |
| `GET` | `/v1/memory/profile/:userId` | List profile memories |
| `DELETE` | `/v1/memory/:id` | Delete or deactivate memory |
| `GET` | `/v1/projects` | List projects |
| `POST` | `/v1/projects/:id/sources` | Connect a knowledge source |
| `GET` | `/v1/projects/:id/sources` | List project sources |
| `POST` | `/v1/sources/:id/sync` | Trigger source sync |
| `GET` | `/v1/sources/:id` | Inspect source status |

Local-only runtime endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/retaindb/health` | Local health |
| `GET` | `/retaindb/snapshot` | Viewer snapshot |
| `GET` | `/retaindb/graph` | Concept graph |
| `GET` | `/retaindb/replay/:sessionId` | Replay session events |
| `POST` | `/retaindb/observe` | Hook capture endpoint |
| `POST` | `/retaindb/consolidate` | Run consolidation |

## Memory Model

RetainDB stores more than raw snippets. A memory can represent:

```ts
type MemoryType =
  | "factual"
  | "preference"
  | "semantic"
  | "procedural"
  | "decision"
  | "constraint"
  | "instruction"
  | "goal"
  | "event"
  | "correction"
  | "session_summary"
  | "project_state";

type RelationType =
  | "updates"
  | "extends"
  | "contradicts"
  | "supports"
  | "derives";
```

Memories include confidence, importance, metadata, session and user scope, timestamps, and optional embeddings. Server and cloud deployments also support relationship and temporal metadata for richer memory graphs.

Examples:

- `preference`: "User prefers concise responses with concrete next steps."
- `decision`: "The project standardizes on Postgres and pgvector for self-hosted search."
- `constraint`: "Local mode must work without cloud API keys."
- `correction`: "The old webhook path is deprecated; use `/v1/agent-events`."
- `procedural`: "Before release, run typecheck, build, benchmark, and smoke replay."

## Connectors

The self-hosted server can index external knowledge sources into agent context:

| Connector | Type |
| --- | --- |
| GitHub | Repos, code, issues |
| Web / Sitemap | Docs sites and pages |
| PDF | Local or remote documents |
| Notion | Pages and databases |
| Confluence | Spaces and pages |
| Slack | Channel history |
| Discord | Server history |
| arXiv | Papers |
| npm / PyPI | Package docs |
| HuggingFace | Model cards |
| Plain text | Inline content |

## Configuration

| Env var | Default | Description |
| --- | --- | --- |
| `RETAINDB_BASE_URL` | `http://localhost:3111` for local MCP | RetainDB API base URL |
| `RETAINDB_PROJECT` | `default` | Default local project |
| `RETAINDB_HOME` | `~/.retaindb` | Local runtime data directory |
| `RETAINDB_STORE` | `~/.retaindb/local-store.json` | Local snapshot path |
| `RETAINDB_EMBEDDING_PROVIDER` | `hash` | `hash` or `local-transformers` |
| `DATABASE_URL` | unset | Postgres connection string for server |
| `RETAINDB_API_KEY` | unset | Optional API auth key |
| `OPENAI_API_KEY` | unset | Enables OpenAI-backed server embeddings |
| `EMBEDDING_MODE` | `auto` | Server embedding mode |
| `EXTRACTION_MODEL` | `gpt-4o-mini` | Server extraction model |
| `PORT` | `3000` | Server HTTP port |
| `DISABLE_SCHEDULER` | `false` | Disable server background sync |

## Local vs Cloud

| Use case | Choose |
| --- | --- |
| Coding-agent memory on your machine | RetainDB Local |
| Self-hosted memory API with Postgres | RetainDB Server |
| Production teams, auth, hosted reliability, dashboard, billing, managed infra | RetainDB Cloud |

RetainDB Cloud adds managed infrastructure, hosted auth, team workflows, production reliability, and cloud product features. RetainDB Local stays useful without cloud sync or an account.

Cloud adds:

- Hosted API keys, auth, and team access controls
- Managed database, vector search, and scaling
- Dashboard workflows for projects, keys, usage, and memory inspection
- Lifecycle email and product update campaigns
- Usage controls, alerts, and operational guardrails
- Higher-level product integrations that do not belong in the local-only runtime

Self-hosted notes:

- The OSS server is single-tenant by default.
- If `RETAINDB_API_KEY` is set, it acts as a shared deployment key.
- Local development can run open on localhost.
- Cloud-style organization isolation belongs in RetainDB Cloud.

## Packages

| Package | npm | Description |
| --- | --- | --- |
| `packages/local` | `@retaindb/local` | Local coding-agent memory runtime |
| `packages/sdk` | `@retaindb/sdk` | TypeScript SDK |
| `packages/mcp` | `@retaindb/mcp` | MCP server |
| `packages/server` | not published | Self-hostable API server |

## Development

```bash
pnpm install
pnpm local:demo
pnpm --filter @retaindb/local typecheck
pnpm --filter @retaindb/mcp typecheck
pnpm --filter @retaindb/sdk typecheck
```

Server development:

```bash
cp .env.example packages/server/.env
pnpm --filter @retaindb/server run db:push
pnpm dev:server
```

## Contributing

PRs are welcome. For larger changes, open an issue first with the problem, proposed behavior, and test plan.

## License

- `packages/local`: Apache 2.0
- `packages/sdk`: Apache 2.0
- `packages/mcp`: Apache 2.0
- `packages/server`: [Business Source License 1.1](./LICENSE-BSL)

Self-hosting is free. Building a hosted service on top of the server requires a commercial license. Contact `alex@retaindb.com`.
