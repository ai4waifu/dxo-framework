import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { RunRecorder } from '@dxo/inspect';
import { createInspectApiServer } from '@dxo/studio';

/**
 * studio-ui-wave2: confusion-matrix, image-samples, compare, binary file API,
 * and process-restart refresh recovery against the same runsRoot.
 */

async function waitForHealth(baseUrl: string, attempts = 80): Promise<void> {
    let last: unknown;
    for (let i = 0; i < attempts; i++) {
        try {
            const res = await fetch(`${baseUrl}/api/health`);
            if (res.ok) return;
            last = new Error(`health status ${res.status}`);
        } catch (err) {
            last = err;
        }
        await new Promise((r) => setTimeout(r, 25));
    }
    throw last instanceof Error ? last : new Error(String(last));
}

async function fetchOk(url: string, attempts = 40): Promise<Response> {
    let last: unknown;
    for (let i = 0; i < attempts; i++) {
        try {
            const res = await fetch(url);
            if (res.ok || res.status < 500) return res;
            last = new Error(`status ${res.status}`);
        } catch (err) {
            last = err;
        }
        await new Promise((r) => setTimeout(r, 25));
    }
    throw last instanceof Error ? last : new Error(String(last));
}

async function readUiSnapshot(baseUrl: string, runId: string) {
    const cm = await (await fetchOk(`${baseUrl}/api/runs/${runId}/confusion-matrix`)).json();
    const imgs = await (await fetchOk(`${baseUrl}/api/runs/${runId}/image-samples`)).json();
    const file = await fetchOk(`${baseUrl}/api/runs/${runId}/files/artifacts/samples/demo.png`);
    assert.equal(file.status, 200);
    const cmp = await (await fetchOk(`${baseUrl}/api/compare?runs=${runId},${runId}`)).json();
    return {
        labels: cm.confusionMatrix.labels,
        sampleCount: imgs.imageSamples.samples.length,
        contentType: file.headers.get('content-type') ?? '',
        seriesCount: cmp.series.length,
    };
}

const runsRoot = await mkdtemp(path.join(tmpdir(), 'dxo-studio-ui-'));

try {
    const recorder = await RunRecorder.open({ root: runsRoot, label: 'ui-wave2', runId: 'wave2-run' });
    await recorder.writeConfusionMatrix({
        labels: ['cat', 'dog'],
        matrix: [
            [4, 1],
            [0, 3],
        ],
    });
    const png = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52]);
    await recorder.writeArtifactBytes('samples/demo.png', 'other', png);
    await recorder.writeImageSamples({
        samples: [{ uri: 'artifacts/samples/demo.png', label: 'cat', pred: 'cat' }],
    });
    await recorder.close('ok');

    const runId = 'wave2-run';
    const api1 = await createInspectApiServer({ host: '127.0.0.1', port: 0, runsRoot });
    let before: Awaited<ReturnType<typeof readUiSnapshot>>;
    try {
        await waitForHealth(api1.url);
        before = await readUiSnapshot(api1.url, runId);
        assert.equal(before.labels.length, 2);
        assert.equal(before.sampleCount, 1);
        assert.ok(before.contentType.includes('image/png'));
        assert.equal(before.seriesCount, 2);
    } finally {
        await api1.close();
    }

    const api2 = await createInspectApiServer({ host: '127.0.0.1', port: 0, runsRoot });
    try {
        await waitForHealth(api2.url);
        const after = await readUiSnapshot(api2.url, runId);
        assert.deepEqual(after, before!);
    } finally {
        await api2.close();
    }

    console.log('studio-ui-wave2 ok: restart-refresh=ok');
} finally {
    await rm(runsRoot, { recursive: true, force: true });
}
