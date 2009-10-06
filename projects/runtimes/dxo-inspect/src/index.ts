/**
 * Placeholder protocol surface for U1 (`studio-run-smoke`).
 * No VMZ dependency. Append-only run store lands in a later slice.
 */

export const INSPECT_PROTOCOL = 'dxo-inspect' as const;
export const INSPECT_PROTOCOL_VERSION = 0 as const;

export type ScalarMetric = {
    name: string;
    value: number;
    step?: number;
    wallTimeMs?: number;
};

export type RunEvent =
    | { type: 'run/start'; runId: string; wallTimeMs: number; meta?: Record<string, unknown> }
    | { type: 'run/end'; runId: string; wallTimeMs: number; status: 'ok' | 'cancelled' | 'error' }
    | { type: 'metric/scalar'; runId: string; metric: ScalarMetric }
    | { type: 'artifact/ref'; runId: string; name: string; kind: string; uri: string }
    | { type: 'log'; runId: string; level: 'info' | 'warn' | 'error'; message: string };

export type ProfileSpan = {
    name: string;
    category: 'op' | 'kernel' | 'transfer' | 'readback' | 'other';
    startMs: number;
    endMs: number;
    meta?: Record<string, unknown>;
};

export type ProfileTrace = {
    format: 'dxo-profile';
    version: 0;
    spans: ProfileSpan[];
};

/** Package identity for smoke / tooling. */
export function inspectVersion(): string {
    return `${INSPECT_PROTOCOL}@${INSPECT_PROTOCOL_VERSION}`;
}
