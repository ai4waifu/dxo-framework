import assert from 'node:assert/strict';
import { tensor } from '@dxo/core';
import { FullyConnected, Sequential } from '@dxo/nn';

const fc = new FullyConnected(2, 3);
fc.weight = tensor(
    [
        1,
        0,
        0, //
        0,
        1,
        0,
    ],
    [2, 3],
);
fc.bias = tensor([0.1, 0.2, 0.3], [3]);

const sample = tensor([1, 2], [1, 2]);
const out = fc.forward(sample);
assert.deepEqual([...out.shape], [1, 3]);
assert.deepEqual(
    (await out.toArray()).map((n) => Math.round(n * 1000) / 1000),
    [1.1, 2.2, 0.3],
);

const mlp = new Sequential([new FullyConnected(2, 4), new FullyConnected(4, 1)]);

const l0 = mlp.layers[0] as FullyConnected;
l0.weight = tensor(
    [
        1,
        0,
        0,
        0, //
        0,
        1,
        0,
        0,
    ],
    [2, 4],
);
l0.bias = tensor([0, 0, 0, 0], [4]);

const l1 = mlp.layers[1] as FullyConnected;
l1.weight = tensor([1, 1, 1, 1], [4, 1]);
l1.bias = tensor([0], [1]);

const batch = tensor(
    [
        1,
        0, //
        0,
        1,
    ],
    [2, 2],
);
const logits = mlp.forward(batch);
assert.deepEqual([...logits.shape], [2, 1]);
assert.deepEqual(
    (await logits.toArray()).map((n) => Math.round(n * 100) / 100),
    [1, 1],
);

const saved = await fc.state();
fc.loadState({
    weight: { shape: [2, 3], data: [0, 0, 0, 0, 0, 0] },
    bias: { shape: [3], data: [0, 0, 0] },
});
const zeroed = fc.forward(sample);
assert.ok((await zeroed.toArray()).every((n) => n === 0));
fc.loadState(saved);
const restored = fc.forward(sample);
assert.deepEqual(await restored.toArray(), await out.toArray());

const outArr = await out.toArray();
const logitsArr = await logits.toArray();
console.log(`nn-forward ok: linear=${outArr}, mlp=${logitsArr}`);
