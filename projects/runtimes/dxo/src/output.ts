import type { CliError } from './errors.js';

export type GlobalFlags = {
    json?: boolean;
    quiet?: boolean;
    verbose?: boolean;
    color?: boolean;
};

export function writeResult(flags: GlobalFlags, value: unknown, text: () => string): void {
    if (flags.json) {
        process.stdout.write(`${JSON.stringify(value)}\n`);
        return;
    }
    if (!flags.quiet) {
        process.stdout.write(`${text()}\n`);
    }
}

export function writeError(flags: GlobalFlags, err: unknown): void {
    const payload = formatError(err, Boolean(flags.verbose));
    if (flags.json) {
        process.stderr.write(`${JSON.stringify(payload)}\n`);
        return;
    }
    process.stderr.write(`${payload.message}\n`);
    if (flags.verbose && payload.stack) {
        process.stderr.write(`${payload.stack}\n`);
    }
}

export function formatError(
    err: unknown,
    verbose: boolean,
): { code: string; message: string; details?: unknown; stack?: string; cause?: string } {
    if (err && typeof err === 'object' && 'code' in err && 'exitCode' in err) {
        const e = err as CliError;
        return {
            code: e.code,
            message: e.message,
            details: e.details,
            stack: verbose ? e.stack : undefined,
            cause: verbose && e.cause instanceof Error ? e.cause.message : undefined,
        };
    }
    if (err instanceof Error) {
        return {
            code: 'RUNTIME_ERROR',
            message: err.message,
            stack: verbose ? err.stack : undefined,
        };
    }
    return { code: 'RUNTIME_ERROR', message: String(err) };
}
