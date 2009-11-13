import assert from 'node:assert/strict';
import {
    bundleModelGraphs,
    executionGraphFromModule,
    kernelGraphFromProfile,
    kernelGraphUnavailable,
    moduleGraphFromModule,
    moduleGraphFromSequential,
    parseModelGraphArtifact,
} from '@dxo/graph';
import { Linear, Relu, Sequential } from '@dxo/nn';

const linear = new Linear(4, 2, { requiresGrad: false });
const lg = moduleGraphFromModule(linear);
assert.equal(lg.view, 'module');
assert.equal(lg.nodes.length, 3);
assert.equal(lg.edges.length, 2);
assert.ok(lg.nodes.some((n) => n.kind === 'Linear'));

const seq = new Sequential([new Linear(2, 3, { requiresGrad: false }), new Relu(), new Linear(3, 1, { requiresGrad: false })]);
const sg = moduleGraphFromSequential(seq);
assert.equal(sg.nodes.length, 5); // input + 3 layers + output
assert.equal(sg.edges.length, 4);
assert.ok(sg.nodes.some((n) => n.kind === 'Relu'));

const eg = executionGraphFromModule(seq, [1, 2]);
assert.equal(eg.view, 'execution');
assert.equal(eg.availability, 'ready');
assert.ok(eg.nodes.some((n) => n.kind === 'op' && n.label === 'linear'));
assert.ok(eg.nodes.some((n) => n.kind === 'op' && n.label === 'relu'));
// Module graph uses Linear/Relu kinds; execution uses op — must not be identical
assert.notDeepEqual(
    sg.nodes.map((n) => n.kind),
    eg.nodes.map((n) => n.kind),
);

const kgEmpty = kernelGraphFromProfile(null);
assert.equal(kgEmpty.view, 'kernel');
assert.equal(kgEmpty.availability, 'unavailable');
assert.equal(kgEmpty.nodes.length, 0);

const kg = kernelGraphFromProfile({
    format: 'dxo-profile',
    version: 0,
    spans: [
        { name: 'gemm', category: 'kernel', startMs: 1, endMs: 3 },
        { name: 'h2d', category: 'transfer', startMs: 0, endMs: 1 },
    ],
});
assert.equal(kg.availability, 'ready');
assert.equal(kg.nodes.length, 2);
assert.ok(kg.nodes.every((n) => n.attrs && typeof n.attrs.startMs === 'number'));

const bundle = bundleModelGraphs({ module: sg, execution: eg, kernel: kernelGraphUnavailable('no titan') });
assert.equal(bundle.format, 'dxo-model-graph-bundle');
assert.equal(bundle.module.view, 'module');
assert.equal(bundle.execution.view, 'execution');
assert.equal(bundle.kernel.view, 'kernel');

const parsed = parseModelGraphArtifact(sg);
assert.equal(parsed.module.view, 'module');
assert.equal(parsed.kernel.availability, 'unavailable');

console.log('model-graph ok: module / execution / kernel views separated');
