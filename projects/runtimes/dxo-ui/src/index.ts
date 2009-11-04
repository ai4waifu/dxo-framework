import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';

export type ImageInputSpec = {
    kind: 'image';
    sources?: Array<'upload' | 'camera'>;
};

export type LabelsOutputSpec = {
    kind: 'labels';
    limit?: number;
};

export type ModelAppRunContext = {
    signal: AbortSignal;
    progress?: (update: { stage: string }) => void;
};

export type ModelAppDefinition = {
    title: string;
    input: ImageInputSpec;
    output: LabelsOutputSpec;
    run: (
        input: { bytes: Uint8Array; mime: string },
        ctx: ModelAppRunContext,
    ) => AsyncIterable<unknown> | Promise<unknown>;
};

export type ModelAppServeHandle = {
    host: string;
    port: number;
    url: string;
    close: () => Promise<void>;
};

export type ModelApp = {
    readonly title: string;
    readonly input: ImageInputSpec;
    readonly output: LabelsOutputSpec;
    serve(options?: { host?: string; port?: number }): Promise<ModelAppServeHandle>;
};

export function image(options: Partial<ImageInputSpec> = {}): ImageInputSpec {
    return { kind: 'image', sources: options.sources ?? ['upload'] };
}

export function labels(options: Partial<LabelsOutputSpec> = {}): LabelsOutputSpec {
    return { kind: 'labels', limit: options.limit ?? 5 };
}

function json(res: ServerResponse, status: number, body: unknown): void {
    const text = JSON.stringify(body);
    res.writeHead(status, {
        'content-type': 'application/json; charset=utf-8',
        'content-length': Buffer.byteLength(text),
        'access-control-allow-origin': '*',
    });
    res.end(text);
}

async function readBody(req: IncomingMessage): Promise<Buffer> {
    const chunks: Buffer[] = [];
    for await (const chunk of req) {
        chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
    }
    return Buffer.concat(chunks);
}

function parseMultipart(buffer: Buffer, boundary: string): Map<string, Buffer> {
    const parts = new Map<string, Buffer>();
    const delim = Buffer.from(`--${boundary}`);
    let start = buffer.indexOf(delim);
    while (start >= 0) {
        const next = buffer.indexOf(delim, start + delim.length);
        if (next < 0) break;
        const slice = buffer.subarray(start + delim.length, next);
        const headerEnd = slice.indexOf('\r\n\r\n');
        if (headerEnd >= 0) {
            const headers = slice.subarray(0, headerEnd).toString('utf8');
            const nameMatch = /name="([^"]+)"/.exec(headers);
            const body = slice.subarray(headerEnd + 4, slice.length - 2);
            if (nameMatch) parts.set(nameMatch[1]!, body);
        }
        start = next;
    }
    return parts;
}

function abortError(): Error {
    const err = new Error('aborted');
    err.name = 'AbortError';
    return err;
}

async function collectRunResult(
    result: AsyncIterable<unknown> | Promise<unknown>,
    signal: AbortSignal,
): Promise<unknown> {
    if (result && typeof (result as AsyncIterable<unknown>)[Symbol.asyncIterator] === 'function') {
        const iter = (result as AsyncIterable<unknown>)[Symbol.asyncIterator]();
        let last: unknown;
        while (true) {
            const step = await iter.next();
            if (step.done) return step.value ?? last;
            if (signal.aborted) throw abortError();
            last = step.value;
        }
    }
    return result;
}

export function defineModelApp(def: ModelAppDefinition): ModelApp {
    const title = def.title || 'untitled';
    return {
        title,
        input: def.input,
        output: def.output,
        async serve(options = {}) {
            const host = options.host ?? '127.0.0.1';
            const port = options.port ?? 7860;

            const server: Server = createServer((req, res) => {
                void handle(req, res).catch((err) => {
                    const message = err instanceof Error ? err.message : String(err);
                    json(res, 500, { error: message });
                });
            });

            async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
                const url = req.url ?? '/';
                if (req.method === 'GET' && url === '/health') {
                    json(res, 200, { ok: true, title });
                    return;
                }
                if (req.method === 'POST' && url === '/run') {
                    const ctype = req.headers['content-type'] ?? '';
                    const controller = new AbortController();
                    req.on('aborted', () => controller.abort());

                    let bytes: Uint8Array;
                    let mime = 'application/octet-stream';
                    if (ctype.startsWith('multipart/form-data')) {
                        const boundary = /boundary=(.+)$/.exec(ctype)?.[1]?.trim();
                        if (!boundary) {
                            json(res, 400, { error: 'missing multipart boundary' });
                            return;
                        }
                        const raw = await readBody(req);
                        const parts = parseMultipart(raw, boundary);
                        const image = parts.get('image');
                        if (!image) {
                            json(res, 400, { error: 'missing image field' });
                            return;
                        }
                        bytes = new Uint8Array(image);
                        mime = 'image/*';
                    } else {
                        const raw = await readBody(req);
                        bytes = new Uint8Array(raw);
                    }

                    const out = await collectRunResult(
                        def.run({ bytes, mime }, { signal: controller.signal }),
                        controller.signal,
                    );
                    json(res, 200, { title, output: out });
                    return;
                }
                json(res, 404, { error: 'not_found' });
            }

            await new Promise<void>((resolve, reject) => {
                server.once('error', reject);
                server.listen(port, host, () => resolve());
            });

            const addr = server.address();
            if (!addr || typeof addr === 'string') {
                throw new Error('failed to bind model app server');
            }

            return {
                host,
                port: addr.port,
                url: `http://${host}:${addr.port}`,
                close: () =>
                    new Promise((resolve, reject) => {
                        server.close((err) => (err ? reject(err) : resolve()));
                    }),
            };
        },
    };
}

export function uiVersion(): string {
    return 'dxo-ui@0.1.0-preview';
}
