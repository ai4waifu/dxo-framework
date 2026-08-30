/**
 * Structured DXO error / diagnostic wire (Living 15).
 * Native napi errors may arrive as `DXO_DIAGNOSTIC:{json}` in `Error.message`.
 */

export type DxoSeverity = 'error' | 'warning' | 'advice';

export type DxoDiagnostic = {
    code: string;
    severity: DxoSeverity;
    args?: Record<string, string | number | boolean>;
    details?: Record<string, unknown>;
    backend?: string;
    operation?: string;
    /** Display convenience only — not a machine contract. */
    message?: string;
};

export type DxoErrorInit = {
    code: string;
    severity?: DxoSeverity;
    message?: string;
    args?: Record<string, string | number | boolean>;
    details?: Record<string, unknown>;
    backend?: string;
    operation?: string;
    cause?: unknown;
};

const DIAGNOSTIC_PREFIX = 'DXO_DIAGNOSTIC:';

function isSeverity(v: unknown): v is DxoSeverity {
    return v === 'error' || v === 'warning' || v === 'advice';
}

function parseArgs(raw: unknown): Record<string, string | number | boolean> | undefined {
    if (!raw || typeof raw !== 'object') return undefined;
    const out: Record<string, string | number | boolean> = {};
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
        if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
            out[k] = v;
        } else if (v != null) {
            out[k] = String(v);
        }
    }
    return Object.keys(out).length ? out : undefined;
}

/** Structured error with stable `code` / `args` / `details` (localization is separate). */
export class DxoError extends Error {
    readonly code: string;
    readonly severity: DxoSeverity;
    readonly args?: Record<string, string | number | boolean>;
    readonly details?: Record<string, unknown>;
    readonly backend?: string;
    readonly operation?: string;

    constructor(init: DxoErrorInit) {
        super(init.message ?? init.code, init.cause !== undefined ? { cause: init.cause } : undefined);
        this.name = 'DxoError';
        this.code = init.code;
        this.severity = init.severity ?? 'error';
        this.args = init.args;
        this.details = init.details;
        this.backend = init.backend;
        this.operation = init.operation;
    }

    toDiagnostic(): DxoDiagnostic {
        return {
            code: this.code,
            severity: this.severity,
            args: this.args,
            details: this.details,
            backend: this.backend,
            operation: this.operation,
            message: this.message,
        };
    }

    /** Parse napi / unknown throws into `DxoError`. */
    static fromUnknown(err: unknown): DxoError {
        if (err instanceof DxoError) return err;
        const message = err instanceof Error ? err.message : String(err);
        if (message.startsWith(DIAGNOSTIC_PREFIX)) {
            try {
                const raw = JSON.parse(message.slice(DIAGNOSTIC_PREFIX.length)) as Record<string, unknown>;
                const code = typeof raw.code === 'string' ? raw.code : 'DXO_UNKNOWN';
                const severity = isSeverity(raw.severity) ? raw.severity : 'error';
                const msg = typeof raw.message === 'string' ? raw.message : message;
                return new DxoError({
                    code,
                    severity,
                    message: msg,
                    args: parseArgs(raw.args),
                    details: raw.details && typeof raw.details === 'object' ? (raw.details as Record<string, unknown>) : undefined,
                    backend: typeof raw.backend === 'string' ? raw.backend : undefined,
                    operation: typeof raw.operation === 'string' ? raw.operation : undefined,
                    cause: err,
                });
            } catch {
                // fall through
            }
        }
        return new DxoError({
            code: 'DXO_UNKNOWN',
            message,
            severity: 'error',
            cause: err,
        });
    }
}

/** Re-throw unknown as `DxoError` (for wrapping native calls). */
export function rethrowAsDxoError(err: unknown): never {
    throw DxoError.fromUnknown(err);
}

export function wrapNative<T>(fn: () => T): T {
    try {
        return fn();
    } catch (err) {
        rethrowAsDxoError(err);
    }
}
