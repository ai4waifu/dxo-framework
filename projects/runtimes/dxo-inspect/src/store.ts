import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readdir, readFile, writeFile, appendFile } from 'node:fs/promises';
import path from 'node:path';
import {
    INSPECT_PROTOCOL,
    INSPECT_PROTOCOL_VERSION,
    type ArtifactKindV0,
    type InspectEventV0,
    type RunMetaV0,
    type RunStatus,
} from './schemas.js';

const EVENTS_FILE = 'events.jsonl';
const META_FILE = 'meta.json';
const ARTIFACTS_DIR = 'artifacts';

export type RunRecorderOptions = {
    /** Root directory; default `{cwd}/.dxo/runs` or `DXO_RUNS_DIR`. */
    root?: string;
    runId?: string;
    label?: string;
    hyperparams?: Record<string, unknown>;
};

export type RunSummary = {
    runId: string;
    meta: RunMetaV0;
};

function nowMs(): number {
    return Date.now();
}

export function defaultRunsRoot(): string {
    const fromEnv = process.env.DXO_RUNS_DIR?.trim();
    if (fromEnv) return path.resolve(fromEnv);
    return path.resolve(process.cwd(), '.dxo', 'runs');
}

function runDir(root: string, runId: string): string {
    return path.join(root, runId);
}

export class RunRecorder {
    readonly runId: string;
    readonly root: string;
    readonly runPath: string;
    #closed = false;
    #meta: RunMetaV0;

    private constructor(root: string, runId: string, meta: RunMetaV0) {
        this.root = root;
        this.runId = runId;
        this.runPath = runDir(root, runId);
        this.#meta = meta;
    }

    static async open(options: RunRecorderOptions = {}): Promise<RunRecorder> {
        const root = options.root ?? defaultRunsRoot();
        const runId = options.runId ?? randomUUID();
        const startedAtMs = nowMs();
        const meta: RunMetaV0 = {
            format: INSPECT_PROTOCOL,
            version: INSPECT_PROTOCOL_VERSION,
            runId,
            startedAtMs,
            label: options.label,
            status: 'running',
            hyperparams: options.hyperparams,
        };
        const dir = runDir(root, runId);
        await mkdir(path.join(dir, ARTIFACTS_DIR), { recursive: true });
        await writeFile(path.join(dir, META_FILE), `${JSON.stringify(meta, null, 2)}\n`, 'utf8');
        const recorder = new RunRecorder(root, runId, meta);
        await recorder.append({
            type: 'run/start',
            runId,
            wallTimeMs: startedAtMs,
            meta: options.hyperparams,
        });
        return recorder;
    }

    async append(event: InspectEventV0): Promise<void> {
        if (this.#closed) throw new Error(`run ${this.runId} is closed`);
        const line = `${JSON.stringify(event)}\n`;
        await appendFile(path.join(this.runPath, EVENTS_FILE), line, 'utf8');
    }

    /** Write artifact bytes and emit artifact/ref. Returns relative URI under run dir. */
    async writeArtifact(name: string, kind: ArtifactKindV0, body: string): Promise<string> {
        const rel = path.join(ARTIFACTS_DIR, name);
        const abs = path.join(this.runPath, rel);
        await writeFile(abs, body, 'utf8');
        const digest = createHash('sha256').update(body).digest('hex');
        await this.append({
            type: 'artifact/ref',
            runId: this.runId,
            wallTimeMs: nowMs(),
            artifact: {
                name,
                kind,
                uri: rel.replace(/\\/g, '/'),
                digest,
                wallTimeMs: nowMs(),
            },
        });
        return rel;
    }

    async close(status: Exclude<RunStatus, 'running'>, error?: string): Promise<void> {
        if (this.#closed) return;
        const endedAtMs = nowMs();
        this.#meta = { ...this.#meta, status, endedAtMs };
        await writeFile(path.join(this.runPath, META_FILE), `${JSON.stringify(this.#meta, null, 2)}\n`, 'utf8');
        await this.append({
            type: 'run/end',
            runId: this.runId,
            wallTimeMs: endedAtMs,
            status,
            error,
        });
        this.#closed = true;
    }
}

export async function listRuns(root: string = defaultRunsRoot()): Promise<RunSummary[]> {
    let entries: string[];
    try {
        entries = await readdir(root);
    } catch {
        return [];
    }
    const out: RunSummary[] = [];
    for (const runId of entries) {
        const meta = await readRunMeta(root, runId);
        if (meta) out.push({ runId, meta });
    }
    out.sort((a, b) => b.meta.startedAtMs - a.meta.startedAtMs);
    return out;
}

export async function readRunMeta(root: string, runId: string): Promise<RunMetaV0 | null> {
    try {
        const text = await readFile(path.join(runDir(root, runId), META_FILE), 'utf8');
        return JSON.parse(text) as RunMetaV0;
    } catch {
        return null;
    }
}

export async function readEvents(root: string, runId: string): Promise<InspectEventV0[]> {
    try {
        const text = await readFile(path.join(runDir(root, runId), EVENTS_FILE), 'utf8');
        return text
            .split('\n')
            .filter((line: string) => line.trim().length > 0)
            .map((line: string) => JSON.parse(line) as InspectEventV0);
    } catch {
        return [];
    }
}
