import assert from 'node:assert/strict';
import { moduleGraphFromModule, moduleGraphFromSequential } from '@dxo/graph';
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

console.log('model-graph ok: Linear and Sequential module graphs');
