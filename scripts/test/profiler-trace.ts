/**
 * profiler-trace: real spans map to kernel graph / Studio timeline;
 * missing Titan data → structured unavailable (never invent start/end).
 */
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { bundleModelGraphs, executionGraphFromModule, kernelGraphFromProfile, moduleGraphFromModule } from '@dxo/graph';
import { profileTraceUnavailable, RunRecorder, type ProfileTraceV0 } from '@dxo/inspect';
import { Linear } from '@dxo/nn';
import { createInspectApiServer } from '@dxo/studio';

const linear = new Linear(2, 1, { requiresGrad: false });
const moduleG = moduleGraphFromModule(linear);
const execG = executionGraphFromModule(linear, [1, 2]);

const emptyKernel = kernelGraphFromProfile(null);
assert.equal(emptyKernel.availability, 'unavailable');
assert.equal(emptyKernel.nodes.length, 0);

const unavailable = profileTraceUnavailable('Titan profile events not exposed on this build');
assert.equal(unavailable.availability, 'unavailable');
assert.equal(unavailable.spans.length, 0);
assert.equal(kernelGraphFromProfile(unavailable).availability, 'unavailable');

const real: ProfileTraceV0 = {
    format: 'dxo-profile',
    version: 0,
    availability: 'ready',
    spans: [
        { name: 'upload', category: 'transfer', startMs: 10, endMs: 12 },
        { name: 'sgemm', category: 'kernel', startMs: 12, endMs: 40 },
        { name: 'readback', category: 'readback', startMs: 40, endMs: 41 },
    ],
};
const kernelG = kernelGraphFromProfile(real);
assert.equal(kernelG.availability, 'ready');
assert.equal(kernelG.nodes.length, 3);
for (const n of kernelG.nodes) {
    assert.equal(typeof n.attrs?.startMs, 'number');
    assert.equal(typeof n.attrs?.endMs, 'number');
}

const bundle = bundleModelGraphs({ module: moduleG, execution: execG, kernel: kernelG });
assert.notEqual(bundle.module.view, bundle.execution.view);
assert.notEqual(bundle.execution.view, bundle.kernel.view);

const runsRoot = await mkdtemp(path.join(tmpdir(), 'dxo-profiler-'));
try {
    const recorder = await RunRecorder.open({ root: runsRoot, label: 'profiler', runId: 'prof-run' });
    await recorder.writeModelGraph(bundle);
    await recorder.writeProfileTrace(unavailable);
    await recorder.close('ok');

    const api = await createInspectApiServer({ host: '127.0.0.1', port: 0, runsRoot });
    try {
        const g = await (await fetch(`${api.url}/api/runs/prof-run/model-graph`)).json();
        assert.equal(g.graph.format, 'dxo-model-graph-bundle');
        assert.equal(g.graph.kernel.view, 'kernel');

        const pr = await (await fetch(`${api.url}/api/runs/prof-run/profile-trace`)).json();
        assert.equal(pr.profileTrace.availability, 'unavailable');
        assert.equal(pr.profileTrace.spans.length, 0);
    } finally {
        await api.close();
    }

    console.log('profiler-trace ok: unavailable without Titan; real spans only when present');
} finally {
    await rm(runsRoot, { recursive: true, force: true });
}
