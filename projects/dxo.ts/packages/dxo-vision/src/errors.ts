/** Structured vision error with stable `code` for callers / verify. */
export class VisionError extends Error {
    readonly code: string;

    constructor(code: string, message: string) {
        super(message);
        this.name = 'VisionError';
        this.code = code;
    }
}

export function unsupported(op: string, detail?: string): never {
    const suffix = detail ? `: ${detail}` : '';
    throw new VisionError('UNSUPPORTED', `@dxo/vision ${op} is not implemented yet${suffix}`);
}
