import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { batch, dataset } from '@dxo/data';
import { moduleGraphFromModule, serializeModelGraph } from '@dxo/graph';
import { readEvents, readRunMeta, recordTrainIter } from '@dxo/inspect';
import { Linear } from '@dxo/nn';
import { SGD } from '@dxo/optimizer';
import { createInspectApiServer } from '@dxo/studio';
import { Trainer } from '@dxo/train';

/**
 * studio-run-smoke: real Trainer → inspect store → loopback API → refresh-stable reads.
 * Does not scrape stdout. UI is optional; this gate is API-level.
 */

function makeSamples(n: number) {
    const samples = [];
    for (let i = 0; i < n; i++) {
        const a = (i % 10) / 10;
        const b = ((i * 3) % 10) / 10;
        samples.push({
            x: [a, b],
            xShape: [2] as number[],
            y: [2 * a - b],
            yShape: [1] as number[],
        });
    }
    return samples;
}

const runsRoot = await mkdtemp(path.join(tmpdir(), 'dxo-studio-runs-'));

try {
    const model = new Linear(2, 1);
    model.loadState({
        weight: { shape: [2, 1], data: [0.1, -0.1] },
        bias: { shape: [1], data: [0] },
    });

    const trainer = new Trainer({
        model,
        optimizer: new SGD(0.1),
        epochs: 3,
        checkpointEvery: 1,
        batches: () => batch(dataset(makeSamples(16)), { batchSize: 8 }),
    });

    const { runId, status } = await recordTrainIter(trainer.fitIter(), {
        root: runsRoot,
        label: 'studio-smoke',
        hyperparams: { lr: 0.1, epochs: 3 },
    });
    assert.equal(status, 'ok');

    // Attach module graph artifact for Models page / API.
    const { writeFile, mkdir } = await import('node:fs/promises');
    const graph = moduleGraphFromModule(model);
    const artDir = path.join(runsRoot, runId, 'artifacts');
    await mkdir(artDir, { recursive: true });
    await writeFile(path.join(artDir, 'model-graph.json'), serializeModelGraph(graph), 'utf8');

    const meta = await readRunMeta(runsRoot, runId);
    assert.ok(meta);
    assert.equal(meta.status, 'ok');
    assert.equal(meta.label, 'studio-smoke');

    const events = await readEvents(runsRoot, runId);
    assert.ok(events.some((e) => e.type === 'metric/scalar'));
    assert.ok(events.some((e) => e.type === 'artifact/ref'));
    assert.ok(events.some((e) => e.type === 'run/end'));

    const api = await createInspectApiServer({ host: '127.0.0.1', port: 0, runsRoot });
    try {
        const runs1 = await (await fetch(`${api.url}/api/runs`)).json();
        assert.equal(runs1.runs.length, 1);
        assert.equal(runs1.runs[0].runId, runId);

        const metrics = await (await fetch(`${api.url}/api/runs/${runId}/metrics`)).json();
        assert.ok(metrics.metrics.some((m: { name: string }) => m.name === 'loss'));

        const eventsRes = await (await fetch(`${api.url}/api/runs/${runId}/events`)).json();
        assert.ok(eventsRes.events.length >= events.length - 1);

        const ck = await (await fetch(`${api.url}/api/runs/${runId}/checkpoints`)).json();
        assert.ok(ck.checkpoints.length >= 1);

        const graphRes = await (await fetch(`${api.url}/api/runs/${runId}/model-graph`)).json();
        assert.equal(graphRes.graph.view, 'module');
        assert.ok(graphRes.graph.nodes.some((n: { kind: string }) => n.kind === 'Linear'));

        // Refresh / second read — same store, same content.
        const runs2 = await (await fetch(`${api.url}/api/runs`)).json();
        assert.deepEqual(runs2.runs[0].meta.runId, runs1.runs[0].meta.runId);
        assert.equal(runs2.runs[0].meta.status, 'ok');
    } finally {
        await api.close();
    }

    console.log(`studio-run-smoke ok: run=${runId} events=${events.length}`);
} finally {
    await rm(runsRoot, { recursive: true, force: true });
}
