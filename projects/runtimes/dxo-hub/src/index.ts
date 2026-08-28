/**
 * Artifact hub: local: and hf: providers with digest + offline cache.
 */

import { createHash } from 'node:crypto';
import { createWriteStream, existsSync } from 'node:fs';
import { mkdir, readFile, readdir, rename } from 'node:fs/promises';
import { homedir } from 'node:os';
import path from 'node:path';
import { pipeline } from 'node:stream/promises';

export type HubCacheMode = 'reuse' | 'offline';

export type HubProvider = 'local' | 'hf' | 'modelscope' | 's3' | 'r2';

export type HubModelOptions = {
    revision?: string;
    files?: string[];
    cache?: HubCacheMode;
    signal?: AbortSignal;
    /** Collection / dataset manifest id when resolving a model set (Wave 4). */
    collection?: string;
};

export type HubResolvedArtifact = {
    provider: HubProvider;
    uri: string;
    revision: string;
    digest: string;
    localPath: string;
    files: Record<string, string>;
};

export type Hub = {
    model(ref: string, options?: HubModelOptions): Promise<HubResolvedArtifact>;
};

function defaultCacheRoot(): string {
    return process.env.DXO_HUB_CACHE?.trim() || path.join(homedir(), '.cache', 'dxo', 'hub');
}

function parseRef(ref: string): { provider: HubProvider; rest: string } {
    const schemes: HubProvider[] = ['local', 'hf', 'modelscope', 's3', 'r2'];
    for (const scheme of schemes) {
        const prefix = `${scheme}:`;
        if (ref.startsWith(prefix)) {
            return { provider: scheme, rest: ref.slice(prefix.length) };
        }
    }
    throw new Error(
        `@dxo/hub model() requires explicit scheme local:|hf:|modelscope:|s3:|r2:, got ${JSON.stringify(ref)}`,
    );
}

function unsupportedRemoteProvider(provider: HubProvider, ref: string): never {
    const err = new Error(
        `@dxo/hub provider ${provider} is reserved for hub-remote-cache (Wave 4); ref=${JSON.stringify(ref)}`,
    ) as Error & { code: string; phase: string; recoverable: boolean };
    err.code = 'HUB_PROVIDER_UNAVAILABLE';
    err.phase = 'resolve';
    err.recoverable = false;
    throw err;
}

async function listFilesRecursive(root: string): Promise<string[]> {
    const out: string[] = [];
    async function walk(dir: string, rel = ''): Promise<void> {
        const entries = await readdir(dir, { withFileTypes: true });
        for (const e of entries) {
            const childRel = rel ? `${rel}/${e.name}` : e.name;
            const full = path.join(dir, e.name);
            if (e.isDirectory()) await walk(full, childRel);
            else if (e.isFile()) out.push(childRel.replace(/\\/g, '/'));
        }
    }
    await walk(root);
    return out.sort();
}

async function digestFiles(root: string, files: string[]): Promise<string> {
    const hash = createHash('sha256');
    for (const rel of files) {
        hash.update(rel);
        hash.update('\0');
        hash.update(await readFile(path.join(root, rel)));
        hash.update('\0');
    }
    return hash.digest('hex');
}

function normalizeLocalPath(dir: string): string {
    // local:/C:/foo or local:C:/foo or local:./rel
    let p = dir;
    if (p.startsWith('/') && /^\/[A-Za-z]:/.test(p)) {
        p = p.slice(1);
    }
    return path.resolve(p);
}

async function resolveLocal(dir: string, options: HubModelOptions, uri: string): Promise<HubResolvedArtifact> {
    const root = normalizeLocalPath(dir);
    if (!existsSync(root)) {
        throw new Error(`@dxo/hub local path not found: ${root}`);
    }
    const all = await listFilesRecursive(root);
    const filesList = options.files?.length ? options.files.map((f) => f.replace(/\\/g, '/')) : all;
    for (const f of filesList) {
        if (!existsSync(path.join(root, f))) {
            throw new Error(`@dxo/hub missing file ${f} under ${root}`);
        }
    }
    const sorted = [...filesList].sort();
    const digest = await digestFiles(root, sorted);
    const files: Record<string, string> = {};
    for (const f of sorted) files[f] = path.join(root, f);
    return {
        provider: 'local',
        uri,
        revision: options.revision ?? 'local',
        digest,
        localPath: root,
        files,
    };
}

function hfToken(): string | undefined {
    return process.env.DXO_HF_TOKEN?.trim() || process.env.HF_TOKEN?.trim() || undefined;
}

async function downloadTo(url: string, dest: string, signal?: AbortSignal, token?: string): Promise<void> {
    const headers: Record<string, string> = {};
    if (token) headers.Authorization = `Bearer ${token}`;
    const res = await fetch(url, { headers, signal });
    if (!res.ok || !res.body) {
        throw new Error(`@dxo/hub HF fetch failed ${res.status} ${url}`);
    }
    await mkdir(path.dirname(dest), { recursive: true });
    const tmp = `${dest}.partial`;
    await pipeline(res.body as unknown as NodeJS.ReadableStream, createWriteStream(tmp));
    await rename(tmp, dest);
}

async function resolveHf(repo: string, options: HubModelOptions, uri: string): Promise<HubResolvedArtifact> {
    const revision = options.revision ?? 'main';
    const cacheRoot = defaultCacheRoot();
    const safeRepo = repo.replace(/[\\/]+/g, '--');
    const destRoot = path.join(cacheRoot, 'hf', safeRepo, revision);
    const cacheMode = options.cache ?? 'reuse';
    const wanted = options.files?.length ? options.files : ['config.json'];

    if (cacheMode === 'offline') {
        if (!existsSync(destRoot)) {
            throw new Error(`@dxo/hub offline cache miss: ${destRoot}`);
        }
    } else {
        await mkdir(destRoot, { recursive: true });
        const token = hfToken();
        for (const file of wanted) {
            const dest = path.join(destRoot, file);
            if (existsSync(dest)) continue;
            const url = `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${file
                .split('/')
                .map(encodeURIComponent)
                .join('/')}`;
            await downloadTo(url, dest, options.signal, token);
        }
    }

    const sorted = [...wanted].map((f) => f.replace(/\\/g, '/')).sort();
    for (const f of sorted) {
        if (!existsSync(path.join(destRoot, f))) {
            throw new Error(`@dxo/hub HF file missing after resolve: ${f}`);
        }
    }
    const digest = await digestFiles(destRoot, sorted);
    const files: Record<string, string> = {};
    for (const f of sorted) files[f] = path.join(destRoot, f);
    return {
        provider: 'hf',
        uri,
        revision,
        digest,
        localPath: destRoot,
        files,
    };
}

export function createHub(): Hub {
    return {
        async model(ref, options = {}) {
            if (options.signal?.aborted) {
                throw new Error('@dxo/hub model() aborted');
            }
            const { provider, rest } = parseRef(ref);
            if (provider === 'local') {
                return resolveLocal(rest, options, ref);
            }
            if (provider === 'hf') {
                return resolveHf(rest, options, ref);
            }
            unsupportedRemoteProvider(provider, ref);
        },
    };
}

export function hubVersion(): string {
    return 'dxo-hub@0';
}
