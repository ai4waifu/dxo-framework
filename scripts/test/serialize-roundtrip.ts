import assert from 'node:assert/strict';
import { tensor } from '@dxo/core';
import { Linear } from '@dxo/nn';
import {
    decodeJson,
    decodeLinearState,
    encodeJson,
    encodeLinearState,
    packTensors,
    STATE_FORMAT,
    STATE_VERSION,
    unpackTensors,
} from '@dxo/serialize';

const model = new Linear(2, 1, { requiresGrad: false });
model.loadState({
    weight: { shape: [2, 1], data: [0.25, -0.5] },
    bias: { shape: [1], data: [0.1] },
});

const doc = encodeLinearState(model.state());
assert.equal(doc.format, STATE_FORMAT);
assert.equal(doc.version, STATE_VERSION);
assert.ok(doc.tensors.weight);
assert.ok(doc.tensors.bias);

const text = encodeJson(doc);
const round = decodeLinearState(decodeJson(text));
assert.deepEqual(round.weight.data, [0.25, -0.5]);
assert.ok(Math.abs(round.bias.data[0]! - 0.1) < 1e-5);

const clone = new Linear(2, 1, { requiresGrad: false });
clone.loadState(round);
const x = tensor([1, 2], [1, 2]);
assert.deepEqual(clone.forward(x).toArray(), model.forward(x).toArray());
assert.deepEqual(clone.weight.toArray(), model.weight.toArray());

const packed = packTensors({ a: { shape: [2], data: [3, 4] } });
assert.deepEqual(unpackTensors(packed).a!.data, [3, 4]);

assert.throws(() => decodeJson(JSON.stringify({ format: 'nope', version: 1, tensors: {} })));
assert.throws(() => decodeJson(JSON.stringify({ format: STATE_FORMAT, version: 99, tensors: {} })));

console.log('serialize-roundtrip ok: Linear state JSON v1');
