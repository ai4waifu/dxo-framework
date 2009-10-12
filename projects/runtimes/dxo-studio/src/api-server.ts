import { createServer, type Server } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import {
    defaultRunsRoot,
    listRuns,
    readEvents,
    readRunMeta,
    type RunSummary,
} from '@dxo/inspect';
import type { ArtifactV0, InspectEventV0, MetricV0 } from '@dxo/inspect';

export type InspectApiServerOptions = {
    host?: string;
    port?: number;
    runsRoot?: string;
};

export type InspectApiServer = {
    host: string;
    port: number;
    url: string;
    runsRoot: string;
    server: Server;
    close: () => Promise<void>;
};

function json(res: import('node:http').ServerResponse, status: number, body: unknown): void {
    const text = JSON.stringify(body);
    res.writeHead(status, {
        'content-type': 'application/json; charset=utf-8',
        'content-length': Buffer.byteLength(text),
        'access-control-allow-origin': '*',
    });
    res.end(text);
}

function notFound(res: import('node:http').ServerResponse): void {
    json(res, 404, { error: 'not_found' });
}

function parsePath(url: string): { pathname: string; searchParams: URLSearchParams } {
    const u = new URL(url, 'http://127.0.0.1');
    return { pathname: u.pathname, searchParams: u.searchParams };
}

function metricsFromEvents(events: InspectEventV0[]): MetricV0[] {
    const out: MetricV0[] = [];
    for (const event of events) {
        if (event.type === 'metric/scalar') out.push(event.metric);
    }
    return out;
}

function artifactsFromEvents(events: InspectEventV0[]): ArtifactV0[] {
    const out: ArtifactV0[] = [];
    for (const event of events) {
        if (event.type === 'artifact/ref') out.push(event.artifact);
    }
    return out;
}

async function handleRequest(
    req: import('node:http').IncomingMessage,
    res: import('node:http').ServerResponse,
    runsRoot: string,
): Promise<void> {
    if (req.method !== 'GET') {
        json(res, 405, { error: 'method_not_allowed' });
        return;
    }

    const { pathname } = parsePath(req.url ?? '/');

    if (pathname === '/api/health') {
        json(res, 200, { ok: true, service: 'dxo-studio-api' });
        return;
    }

    if (pathname === '/api/runs') {
        const runs: RunSummary[] = await listRuns(runsRoot);
        json(res, 200, { runs });
        return;
    }

    const runMatch = /^\/api\/runs\/([^/]+)(\/.*)?$/.exec(pathname);
    if (!runMatch) {
        notFound(res);
        return;
    }

    const runId = decodeURIComponent(runMatch[1]!);
    const sub = runMatch[2] ?? '';

    if (sub === '' || sub === '/meta') {
        const meta = await readRunMeta(runsRoot, runId);
        if (!meta) {
            notFound(res);
            return;
        }
        json(res, 200, { meta });
        return;
    }

    if (sub === '/events') {
        const events = await readEvents(runsRoot, runId);
        json(res, 200, { runId, events });
        return;
    }

    if (sub === '/metrics') {
        const events = await readEvents(runsRoot, runId);
        json(res, 200, { runId, metrics: metricsFromEvents(events) });
        return;
    }

    if (sub === '/artifacts') {
        const events = await readEvents(runsRoot, runId);
        json(res, 200, { runId, artifacts: artifactsFromEvents(events) });
        return;
    }

    if (sub === '/checkpoints') {
        const events = await readEvents(runsRoot, runId);
        const checkpoints = artifactsFromEvents(events).filter((a) => a.kind === 'checkpoint');
        json(res, 200, { runId, checkpoints });
        return;
    }

    if (sub === '/model-graph') {
        const abs = path.join(runsRoot, runId, 'artifacts', 'model-graph.json');
        try {
            const body = await readFile(abs, 'utf8');
            json(res, 200, { runId, graph: JSON.parse(body) });
        } catch {
            notFound(res);
        }
        return;
    }

    const artifactMatch = /^\/artifacts\/(.+)$/.exec(sub);
    if (artifactMatch) {
        const rel = decodeURIComponent(artifactMatch[1]!);
        const abs = path.join(runsRoot, runId, rel);
        try {
            const body = await readFile(abs, 'utf8');
            json(res, 200, { runId, path: rel, body });
        } catch {
            notFound(res);
        }
        return;
    }

    notFound(res);
}

/** Loopback HTTP API over the append-only inspect run store. */
export async function createInspectApiServer(
    options: InspectApiServerOptions = {},
): Promise<InspectApiServer> {
    const host = options.host ?? '127.0.0.1';
    const port = options.port ?? 0;
    const runsRoot = options.runsRoot ?? defaultRunsRoot();

    const server = createServer((req, res) => {
        void handleRequest(req, res, runsRoot).catch((err) => {
            const message = err instanceof Error ? err.message : String(err);
            json(res, 500, { error: 'internal_error', message });
        });
    });

    await new Promise<void>((resolve, reject) => {
        server.once('error', reject);
        server.listen(port, host, () => resolve());
    });

    const addr = server.address();
    if (!addr || typeof addr === 'string') {
        throw new Error('failed to bind inspect API server');
    }

    const boundPort = addr.port;
    const url = `http://${host}:${boundPort}`;

    return {
        host,
        port: boundPort,
        url,
        runsRoot,
        server,
        close: () =>
            new Promise((resolve, reject) => {
                server.close((err) => (err ? reject(err) : resolve()));
            }),
    };
}
