/** Stable CLI exit codes (Living `13`). */
export const EXIT = {
    OK: 0,
    FAIL: 1,
    USAGE: 2,
    ENV: 78,
    CANCEL: 130,
} as const;

export type ExitCode = (typeof EXIT)[keyof typeof EXIT];

export class CliError extends Error {
    readonly code: string;
    readonly exitCode: ExitCode;
    readonly details?: unknown;

    constructor(code: string, message: string, exitCode: ExitCode, details?: unknown, options?: { cause?: unknown }) {
        super(message, options);
        this.name = 'CliError';
        this.code = code;
        this.exitCode = exitCode;
        this.details = details;
    }
}

export function unsupportedCommand(name: string): CliError {
    return new CliError('UNSUPPORTED_COMMAND', `command not supported yet: ${name}`, EXIT.USAGE, { command: name });
}
