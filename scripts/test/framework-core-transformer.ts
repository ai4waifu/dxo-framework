/**
 * framework-core-transformer: Embedding / LayerNorm / causal attention + tiny decoder
 * train step, save/load via packTensors, and autograd smoke.
 */
import assert from 'node:assert/strict';
import { embedding, tensor, withoutGrad } from '@dxo/core';
import { Embedding, LayerNormalization, TinyTransformer } from '@dxo/nn';
import { decodeState, encodeState } from '@dxo/serialize';

// --- Embedding gather ---
{
    const emb = new Embedding(4, 2, { requiresGrad: false });
    emb.weight = tensor([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
    const out = emb.forward(tensor([2, 0], [2]));
    assert.deepEqual([...out.shape], [2, 2]);
    assert.deepEqual(await out.toArray(), [5, 6, 1, 2]);
    assert.deepEqual(await embedding(emb.weight, tensor([1], [1])).toArray(), [3, 4]);
}

// --- LayerNorm ---
{
    const ln = new LayerNormalization(2, { requiresGrad: false });
    ln.weight = tensor([1, 1], [2]);
    ln.bias = tensor([0, 0], [2]);
    const y = ln.forward(tensor([1, 2, 3, 4], [2, 2]));
    const v = await y.toArray();
    assert.ok(Math.abs(v[0]! + v[1]!) < 1e-4);
    assert.ok(Math.abs(v[2]! + v[3]!) < 1e-4);
}

// --- Tiny decoder forward + one train step ---
const model = new TinyTransformer(8, 4, 8, 2, 1);
const tokens = tensor([0, 1, 2, 3, 1, 2, 3, 0], [2, 4]);
const logits = model.forward(tokens);
assert.deepEqual([...logits.shape], [2, 4, 8]);

const loss = logits.mean();
model.zeroGrad();
loss.backward();
const grads = model.parameters().map((p) => p.grad);
assert.ok(grads.some((g) => g?.some((x) => Math.abs(x) > 0)));

// --- Save / load roundtrip ---
const named: Record<string, { shape: number[]; data: number[] }> = {};
const params = model.parameters();
for (let i = 0; i < params.length; i++) {
    const p = params[i]!;
    named[`p${i}`] = { shape: [...p.shape], data: await p.toArray() };
}
const restored = decodeState(encodeState(named, { format: 'safetensors' }), { format: 'safetensors' });
assert.deepEqual(Object.keys(restored).sort(), Object.keys(named).sort());
for (const key of Object.keys(named)) {
    assert.deepEqual(restored[key]!.data, named[key]!.data);
    assert.deepEqual(restored[key]!.shape, named[key]!.shape);
}

const logits2 = withoutGrad(() => model.forward(tokens));
assert.deepEqual([...logits2.shape], [2, 4, 8]);

// --- Named state ↔ safetensors reload ---
const namedState = await model.state();
const st = encodeState(namedState, { format: 'safetensors' });
const reloaded = new TinyTransformer(8, 4, 8, 2, 1, { requiresGrad: false });
reloaded.loadState(decodeState(st, { format: 'safetensors' }), { requiresGrad: false });
const logits3 = withoutGrad(() => reloaded.forward(tokens));
const a = await logits2.toArray();
const b = await logits3.toArray();
assert.equal(a.length, b.length);
for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i]! - b[i]!) < 1e-5);
}

console.log(`framework-core-transformer ok: params=${params.length} logits=${(await logits.toArray()).slice(0, 4).map((n) => +n.toFixed(3))}`);
