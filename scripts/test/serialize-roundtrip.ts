import assert from 'node:assert/strict';
import { tensor } from '@dxo/core';
import { FullyConnected } from '@dxo/nn';
import {
    decodeState,
    encodeState,
} from '@dxo/serialize';

const model = new FullyConnected(2, 1, { requiresGrad: false });
model.loadState({
    weight: { shape: [2, 1], data: [0.25, -0.5] },
    bias: { shape: [1], data: [0.1] },
});

const state = await model.state();
const round = decodeState(encodeState(state, { format: 'safetensors' }), { format: 'safetensors' });
assert.deepEqual(round.weight.data, [0.25, -0.5]);
assert.ok(Math.abs(round.bias.data[0]! - 0.1) < 1e-5);

const clone = new FullyConnected(2, 1, { requiresGrad: false });
clone.loadState(round);
const x = tensor([1, 2], [1, 2]);
assert.deepEqual(await clone.forward(x).toArray(), await model.forward(x).toArray());
assert.deepEqual(await clone.weight.toArray(), await model.weight.toArray());

const stBytes = encodeState({
    weight: round.weight,
    bias: round.bias,
}, { format: 'safetensors' });
const stRound = decodeState(stBytes, { format: 'safetensors' });
assert.deepEqual(stRound.weight!.data, round.weight.data);
assert.deepEqual(stRound.bias!.data, round.bias.data);
assert.deepEqual(stRound.weight!.shape, [2, 1]);

const fromUnaligned = decodeState(Uint8Array.from(stBytes), { format: 'safetensors' });
assert.deepEqual(fromUnaligned.weight!.data, round.weight.data);

console.log('serialize-roundtrip ok (State + safetensors)');
