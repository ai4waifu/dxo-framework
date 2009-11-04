import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { RunRecorder } from '@dxo/inspect';
import { createInspectApiServer } from '@dxo/studio';

/**
 * studio-ui-wave2: confusion-matrix, image-samples, compare, and binary file API.
 */

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

    const api = await createInspectApiServer({ host: '127.0.0.1', port: 0, runsRoot });
    try {
        const runId = 'wave2-run';
        const cm = await (await fetch(`${api.url}/api/runs/${runId}/confusion-matrix`)).json();
        assert.equal(cm.confusionMatrix.labels.length, 2);

        const imgs = await (await fetch(`${api.url}/api/runs/${runId}/image-samples`)).json();
        assert.equal(imgs.imageSamples.samples.length, 1);

        const file = await fetch(`${api.url}/api/runs/${runId}/files/artifacts/samples/demo.png`);
        assert.equal(file.status, 200);
        assert.ok((file.headers.get('content-type') ?? '').includes('image/png'));

        const cmp = await (await fetch(`${api.url}/api/compare?runs=${runId},${runId}`)).json();
        assert.equal(cmp.series.length, 2);
    } finally {
        await api.close();
    }

    console.log('studio-ui-wave2 ok');
} finally {
    await rm(runsRoot, { recursive: true, force: true });
}
