/**
 * Minimal structural types for the optional `redis` dependency.
 *
 * Redis is not declared in `package.json`: the cache layer treats it as an
 * optional extra and falls back to the in-memory cache when it is absent.
 * Declaring only what this codebase uses keeps `tsc` honest without pulling in
 * a dependency the runtime does not require.
 */
declare module "redis" {
  export interface RedisClient {
    isOpen: boolean;
    isReady: boolean;
    on(event: string, listener: (payload: unknown) => void): void;
    connect(): Promise<void>;
    get(key: string): Promise<string | null>;
    set(key: string, value: string, options?: Record<string, unknown>): Promise<unknown>;
  }

  export function createClient(options: Record<string, unknown>): RedisClient;
}
