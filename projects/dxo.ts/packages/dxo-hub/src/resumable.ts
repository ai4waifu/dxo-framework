/**
 * Resumable asset downloader (design contract 43 §4).
 *
 * Pure-TypeScript mirror of the DXO native downloader semantics so callers
 * get identical behaviour before the napi binding lands:
 *
 * - cache layout `<cacheDir>/<id>/<filename>` with `.part` + `.state.json`
 *   side files and a `wx`-created `.write.lock` for mutual exclusion;
 * - HTTP Range resume guarded by `If-Range` (ETag / Last-Modified);
 * - a server that ignores `Range` restarts the download from zero;
 * - `416` finalizes when the local part is already complete, otherwise
 *   resets; `429` honours `Retry-After` (capped); `5xx` and stream cuts
 *   back off exponentially;
 * - mirror rotation after `maxAttemptsPerMirror` transient failures, but
 *   trusted progress is only destroyed when there is another mirror;
 * - the asset is published (atomic rename) only after the full-file
 *   sha256 matches; a corrupt published copy is re-downloaded;
 * - cancellation keeps `.part` + `.state.json` so the next run resumes.
 */

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, open, readFile, rename, rm, stat, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';

/** One download origin for an asset. */
export type DownloadSource = {
    url: string;
    label?: string;
    headers?: Record<string, string>;
};

/** What the caller requires one asset to be. */
export type AssetSpec = {
    id: string;
    filename: string;
    expectedBytes: number;
    sha256: string;
    sources: DownloadSource[];
};

/** Downloader policy; defaults match the Rust implementation. */
export type DownloadConfig = {
    cacheDir: string;
    maxAttempts?: number;
    maxAttemptsPerMirror?: number;
    backoffInitialMs?: number;
    backoffMaxMs?: number;
    retryAfterCapMs?: number;
    requestTimeoutMs?: number;
    streamIdleTimeoutMs?: number;
    maxRedirects?: number;
    staleLockAfterMs?: number;
};

/** Result of one successful `fetch`. */
export type DownloadReport = {
    path: string;
    fromCache: boolean;
    resumedFrom: number;
    bytesTransferred: number;
    attempts: number;
    sha256: string;
};

export type DownloadOptions = {
    signal?: AbortSignal;
    onProgress?: (downloaded: number, total: number) => void;
};

/** On-disk resume record; field names mirror the Rust `ResumeState`. */
type ResumeState = {
    version: 1;
    id: string;
    sha256: string;
    expected_bytes: number;
    source_url: string;
    source_index: number;
    committed_bytes: number;
    etag?: string;
    last_modified?: string;
};

const DEFAULTS = {
    maxAttempts: 8,
    maxAttemptsPerMirror: 3,
    backoffInitialMs: 500,
    backoffMaxMs: 30_000,
    retryAfterCapMs: 120_000,
    requestTimeoutMs: 30_000,
    streamIdleTimeoutMs: 60_000,
    staleLockAfterMs: 3_600_000,
};

/** Contract cache root for a project. */
export function defaultCacheDir(projectRoot: string): string {
    return path.join(projectRoot, '.cache', 'dxo', 'downloads');
}

class IdleCutError extends Error {
    constructor() {
        super('stream idle timeout');
    }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
        if (signal?.aborted) {
            reject(signal.reason ?? new Error('aborted'));
            return;
        }
        const t = setTimeout(resolve, ms);
        signal?.addEventListener(
            'abort',
            () => {
                clearTimeout(t);
                reject(signal.reason ?? new Error('aborted'));
            },
            { once: true },
        );
    });
}

async function sizeOf(p: string): Promise<number | null> {
    try {
        return (await stat(p)).size;
    } catch {
        return null;
    }
}

async function sha256File(p: string): Promise<string> {
    const h = createHash('sha256');
    for await (const chunk of createReadStream(p)) h.update(chunk as Uint8Array);
    return h.digest('hex');
}

/** Parse `Retry-After` (delta-seconds or HTTP date) into ms, capped. */
export function parseRetryAfterMs(value: string | null, capMs: number): number | null {
    if (!value) return null;
    const secs = Number(value.trim());
    if (Number.isFinite(secs) && secs >= 0) return Math.min(secs * 1000, capMs);
    const date = Date.parse(value);
    if (!Number.isNaN(date)) return Math.min(Math.max(0, date - Date.now()), capMs);
    return null;
}

/** Combine an optional caller signal with a timeout controller. */
function linkSignal(user: AbortSignal | undefined, ctrl: AbortController): () => void {
    if (!user) return () => {};
    if (user.aborted) {
        ctrl.abort(user.reason);
        return () => {};
    }
    const onAbort = () => ctrl.abort(user.reason);
    user.addEventListener('abort', onAbort, { once: true });
    return () => user.removeEventListener('abort', onAbort);
}

export class Downloader {
    private readonly cfg: { [K in keyof typeof DEFAULTS]: number } & { cacheDir: string; maxRedirects: number };

    constructor(config: DownloadConfig) {
        this.cfg = {
            cacheDir: config.cacheDir,
            maxAttempts: config.maxAttempts ?? DEFAULTS.maxAttempts,
            maxAttemptsPerMirror: config.maxAttemptsPerMirror ?? DEFAULTS.maxAttemptsPerMirror,
            backoffInitialMs: config.backoffInitialMs ?? DEFAULTS.backoffInitialMs,
            backoffMaxMs: config.backoffMaxMs ?? DEFAULTS.backoffMaxMs,
            retryAfterCapMs: config.retryAfterCapMs ?? DEFAULTS.retryAfterCapMs,
            requestTimeoutMs: config.requestTimeoutMs ?? DEFAULTS.requestTimeoutMs,
            streamIdleTimeoutMs: config.streamIdleTimeoutMs ?? DEFAULTS.streamIdleTimeoutMs,
            maxRedirects: config.maxRedirects ?? 8,
            staleLockAfterMs: config.staleLockAfterMs ?? DEFAULTS.staleLockAfterMs,
        };
        void this.cfg.maxRedirects; // fetch follows redirects; the cap is enforced by the Rust layer / proxies.
    }

    async fetch(spec: AssetSpec, options: DownloadOptions = {}): Promise<DownloadReport> {
        const signal = options.signal;
        if (spec.sources.length === 0) throw new Error(`dxo-download: asset ${spec.id} has no sources`);
        if (!/^[0-9a-f]{64}$/.test(spec.sha256)) throw new Error(`dxo-download: asset ${spec.id} sha256 must be 64 lowercase hex`);
        if (spec.expectedBytes <= 0) throw new Error(`dxo-download: asset ${spec.id} expectedBytes must be positive`);

        const dir = path.join(this.cfg.cacheDir, spec.id);
        await mkdir(dir, { recursive: true });
        const finalPath = path.join(dir, spec.filename);
        const partPath = `${finalPath}.part`;
        const statePath = `${finalPath}.state.json`;
        const lockPath = path.join(dir, '.write.lock');

        // Published copy: verify or discard.
        const published = await sizeOf(finalPath);
        if (published !== null) {
            if (published === spec.expectedBytes) {
                const digest = await sha256File(finalPath);
                if (digest === spec.sha256) {
                    return { path: finalPath, fromCache: true, resumedFrom: spec.expectedBytes, bytesTransferred: 0, attempts: 0, sha256: digest };
                }
            }
            await rm(finalPath, { force: true });
        }

        await this.acquireLock(lockPath);
        try {
            return await this.fetchLocked(spec, finalPath, partPath, statePath, signal, options.onProgress);
        } finally {
            await rm(lockPath, { force: true }).catch(() => {});
        }
    }

    private async acquireLock(lockPath: string): Promise<void> {
        for (let attempt = 0; attempt < 2; attempt++) {
            try {
                const fh = await open(lockPath, 'wx');
                await fh.writeFile(`${process.pid} ${Date.now()}`);
                await fh.close();
                return;
            } catch (e) {
                if ((e as NodeJS.ErrnoException).code !== 'EEXIST') throw e;
                const info = await stat(lockPath).catch(() => null);
                if (info && Date.now() - info.mtimeMs > this.cfg.staleLockAfterMs) {
                    await rm(lockPath, { force: true });
                    continue;
                }
                throw new Error(`dxo-download: another downloader holds ${lockPath}`);
            }
        }
    }

    private async fetchLocked(
        spec: AssetSpec,
        finalPath: string,
        partPath: string,
        statePath: string,
        signal: AbortSignal | undefined,
        onProgress: ((downloaded: number, total: number) => void) | undefined,
    ): Promise<DownloadReport> {
        // Load and validate resume state.
        let state = await this.loadState(statePath, spec);
        let committed = 0;
        if (state) {
            const partSize = await sizeOf(partPath);
            if (partSize === null || partSize < state.committed_bytes) {
                state = null; // part lost or shrank below trusted progress → restart
            } else if (partSize > state.committed_bytes) {
                await this.truncateTo(partPath, state.committed_bytes); // ahead-of-state bytes are untrusted
                committed = state.committed_bytes;
            } else {
                committed = state.committed_bytes;
            }
        }
        if (!state) {
            await this.truncateTo(partPath, 0);
        }
        const resumedFrom = committed;

        let sourceIndex = state?.source_index ?? 0;
        if (sourceIndex < 0 || sourceIndex >= spec.sources.length) sourceIndex = 0;
        let mirrorFailures = 0;
        let attempts = 0;

        while (attempts < this.cfg.maxAttempts) {
            if (signal?.aborted) throw signal.reason ?? new Error('aborted');
            attempts++;
            const source = spec.sources[sourceIndex]!;
            try {
                const outcome = await this.attempt(spec, source, state, committed, partPath, signal, onProgress);
                committed = outcome.committed;
                if (outcome.complete) {
                    state = null;
                    await rm(statePath, { force: true }).catch(() => {});
                    break;
                }
                if (outcome.restart) {
                    committed = 0;
                    state = null;
                    await this.truncateTo(partPath, 0);
                    await rm(statePath, { force: true }).catch(() => {});
                }
                if (outcome.progress) {
                    // Stream cut or early EOF: commit what landed, keep the same mirror.
                    state = {
                        version: 1,
                        id: spec.id,
                        sha256: spec.sha256,
                        expected_bytes: spec.expectedBytes,
                        source_url: source.url,
                        source_index: sourceIndex,
                        committed_bytes: committed,
                        etag: outcome.etag,
                        last_modified: outcome.lastModified,
                    };
                    await this.saveState(statePath, state);
                    mirrorFailures++;
                } else if (outcome.transient) {
                    mirrorFailures++;
                }
                if (outcome.retryAfterMs !== undefined && outcome.retryAfterMs > 0) {
                    await sleep(Math.min(outcome.retryAfterMs, this.cfg.retryAfterCapMs), signal);
                } else if (outcome.progress || outcome.transient) {
                    await sleep(this.backoff(mirrorFailures), signal);
                }
            } catch (e) {
                if (signal?.aborted) throw e;
                if (isPermanent(e)) {
                    if (spec.sources.length <= 1) throw e;
                    sourceIndex = (sourceIndex + 1) % spec.sources.length;
                    mirrorFailures = 0;
                    continue;
                }
                mirrorFailures++;
                await sleep(this.backoff(mirrorFailures), signal);
            }

            if (mirrorFailures >= this.cfg.maxAttemptsPerMirror && spec.sources.length > 1) {
                // Rotate mirror; a different origin may not honour the old
                // validator, so trusted progress is only discarded when
                // another mirror actually exists.
                sourceIndex = (sourceIndex + 1) % spec.sources.length;
                mirrorFailures = 0;
                committed = 0;
                state = null;
                await this.truncateTo(partPath, 0);
                await rm(statePath, { force: true }).catch(() => {});
            }
        }

        if (committed !== spec.expectedBytes) {
            throw new Error(`dxo-download: ${spec.id} exhausted ${this.cfg.maxAttempts} attempts at ${committed}/${spec.expectedBytes} bytes`);
        }

        // Verify before publishing.
        const digest = await sha256File(partPath);
        if (digest !== spec.sha256) {
            await rm(partPath, { force: true });
            await rm(statePath, { force: true }).catch(() => {});
            throw new Error(`dxo-download: ${spec.id} digest mismatch (got ${digest}); asset not published`);
        }
        await rename(partPath, finalPath);
        return { path: finalPath, fromCache: false, resumedFrom, bytesTransferred: committed - resumedFrom, attempts, sha256: digest };
    }

    private async loadState(statePath: string, spec: AssetSpec): Promise<ResumeState | null> {
        try {
            const raw = JSON.parse(await readFile(statePath, 'utf8')) as ResumeState;
            if (raw.version !== 1 || raw.id !== spec.id || raw.sha256 !== spec.sha256 || raw.expected_bytes !== spec.expectedBytes) return null;
            if (typeof raw.committed_bytes !== 'number' || raw.committed_bytes < 0 || raw.committed_bytes > spec.expectedBytes) return null;
            return raw;
        } catch {
            return null;
        }
    }

    private async saveState(statePath: string, state: ResumeState): Promise<void> {
        const tmp = `${statePath}.tmp`;
        await writeFile(tmp, JSON.stringify(state));
        await rename(tmp, statePath);
    }

    private async truncateTo(p: string, size: number): Promise<void> {
        const fh = await open(p, size === 0 ? 'w' : 'r+');
        try {
            await fh.truncate(size);
        } finally {
            await fh.close();
        }
    }

    private backoff(failures: number): number {
        const ms = this.cfg.backoffInitialMs * 2 ** Math.max(0, failures - 1);
        return Math.min(ms, this.cfg.backoffMaxMs);
    }

    /** One HTTP attempt. Returns where the local part stands afterwards. */
    private async attempt(
        spec: AssetSpec,
        source: DownloadSource,
        state: ResumeState | null,
        committed: number,
        partPath: string,
        signal: AbortSignal | undefined,
        onProgress: ((downloaded: number, total: number) => void) | undefined,
    ): Promise<{
        committed: number;
        complete: boolean;
        restart?: boolean;
        progress?: boolean;
        transient?: boolean;
        retryAfterMs?: number;
        etag?: string;
        lastModified?: string;
    }> {
        const headers: Record<string, string> = { ...source.headers };
        if (committed > 0) {
            headers.Range = `bytes=${committed}-`;
            if (state?.etag) headers['If-Range'] = state.etag;
            else if (state?.last_modified) headers['If-Range'] = state.last_modified;
        }

        const ctrl = new AbortController();
        const unlinkSignal = linkSignal(signal, ctrl);
        const requestTimer = setTimeout(() => ctrl.abort(new Error('request timeout')), this.cfg.requestTimeoutMs);
        let res: Response;
        try {
            res = await fetch(source.url, { headers, signal: ctrl.signal, redirect: 'follow' });
        } finally {
            clearTimeout(requestTimer);
            unlinkSignal();
        }

        if (res.status === 416) {
            if (committed >= spec.expectedBytes) return { committed, complete: true };
            return { committed: 0, complete: false, restart: true };
        }
        if (res.status === 429) {
            return { committed, complete: false, transient: true, retryAfterMs: parseRetryAfterMs(res.headers.get('Retry-After'), this.cfg.retryAfterCapMs) ?? this.backoff(1) };
        }
        if (res.status >= 500) {
            return { committed, complete: false, transient: true };
        }
        if (!res.ok) {
            throw new PermanentError(`dxo-download: ${source.url} -> HTTP ${res.status}`);
        }

        const etag = res.headers.get('ETag') ?? undefined;
        const lastModified = res.headers.get('Last-Modified') ?? undefined;

        let start = committed;
        if (res.status === 200 && committed > 0) {
            // Server ignored the Range: the response is the whole asset.
            start = 0;
            await this.truncateTo(partPath, 0);
        }
        if (res.status === 206) {
            const cr = res.headers.get('Content-Range');
            const m = cr ? /^bytes (\d+)-(\d+)\/(\d+|\*)$/.exec(cr) : null;
            if (!m || Number(m[1]) !== committed) {
                throw new PermanentError(`dxo-download: bad Content-Range ${JSON.stringify(cr)} for offset ${committed}`);
            }
            if (m[3] !== '*' && Number(m[3]) !== spec.expectedBytes) {
                throw new PermanentError(`dxo-download: Content-Range total ${m[3]} != expected ${spec.expectedBytes}`);
            }
        }

        if (!res.body) throw new Error('dxo-download: empty response body');
        const reader = res.body.getReader();
        const fh = await open(partPath, 'a');
        let received = 0;
        let cut = false;
        try {
            for (;;) {
                const chunk = await readWithIdle(reader, this.cfg.streamIdleTimeoutMs, signal);
                if (chunk === null) break;
                await fh.write(chunk);
                received += chunk.byteLength;
                onProgress?.(start + received, spec.expectedBytes);
            }
        } catch (e) {
            if (signal?.aborted) throw e;
            if (e instanceof IdleCutError || isTransientNet(e)) cut = true;
            else throw e;
        } finally {
            await fh.sync().catch(() => {});
            await fh.close();
            if (cut || received === 0) {
                await reader.cancel().catch(() => {});
            }
        }

        const next = start + received;
        if (next >= spec.expectedBytes) return { committed: next, complete: next === spec.expectedBytes, etag, lastModified };
        return { committed: next, complete: false, progress: true, etag, lastModified };
    }
}

class PermanentError extends Error {}

function isPermanent(e: unknown): boolean {
    return e instanceof PermanentError;
}

function isTransientNet(e: unknown): boolean {
    const msg = String((e as Error)?.message ?? e);
    return /timeout|network|ECONNRESET|EPIPE|socket|terminated|aborted \(cause: request/i.test(msg);
}

async function readWithIdle(
    reader: ReadableStreamDefaultReader<Uint8Array>,
    idleMs: number,
    signal: AbortSignal | undefined,
): Promise<Uint8Array | null> {
    return new Promise<Uint8Array | null>((resolve, reject) => {
        const timer = setTimeout(() => {
            cleanup();
            reject(new IdleCutError());
        }, idleMs);
        const onAbort = () => {
            cleanup();
            reject(signal?.reason ?? new Error('aborted'));
        };
        function cleanup() {
            clearTimeout(timer);
            signal?.removeEventListener('abort', onAbort);
        }
        if (signal?.aborted) {
            clearTimeout(timer);
            reject(signal.reason ?? new Error('aborted'));
            return;
        }
        signal?.addEventListener('abort', onAbort, { once: true });
        reader.read().then(
            ({ done, value }) => {
                cleanup();
                if (done) resolve(null);
                else resolve(value as Uint8Array);
            },
            (e) => {
                cleanup();
                reject(e);
            },
        );
    });
}

/** Convenience wrapper matching the Rust `Downloader::new(..).fetch(..)` flow. */
export async function downloadAsset(spec: AssetSpec, config: DownloadConfig, options?: DownloadOptions): Promise<DownloadReport> {
    return new Downloader(config).fetch(spec, options);
}
