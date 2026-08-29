/**
 * framework-core-cnn: Conv2d / MaxPool2d / BatchNorm2d + TinyCnn train step and serialize.
 */
import assert from 'node:assert/strict';
import { tensor, withoutGrad } from '@dxo/core';
import { BatchNormalization2d, Convolution2d, MaxPooling2d, TinyCnn } from '@dxo/nn';
import { packTensors, unpackTensors } from '@dxo/serialize';

// --- Conv2d forward shape ---
{
    const conv = new Convolution2d(1, 2, 3, { padding: 1, requiresGrad: false });
    conv.weight = tensor(new Array(2 * 1 * 3 * 3).fill(0.1), [2, 1, 3, 3]);
    conv.bias = tensor([0, 0], [2]);
    const y = conv.forward(tensor(new Array(1 * 1 * 4 * 4).fill(1), [1, 1, 4, 4]));
    assert.deepEqual([...y.shape], [1, 2, 4, 4]);
}

// --- MaxPool2d ---
{
    const pool = new MaxPooling2d(2);
    const y = pool.forward(tensor([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 4]));
    assert.deepEqual([...y.shape], [1, 1, 1, 2]);
    assert.deepEqual(await y.toArray(), [6, 8]);
}

// --- BatchNorm2d + backward ---
{
    const bn = new BatchNormalization2d(1);
    const x = tensor([1, 2, 3, 4], [1, 1, 2, 2], { requiresGrad: true });
    const y = bn.forward(x);
    // E[xhat]=0 ⇒ mean(y) ignores gamma; mean(y^2) ignores beta when mean(y)=0.
    // Combine both so weight and bias grads are observable.
    y.mul(y).mean().add(y.mean()).backward();
    assert.ok(bn.weight.grad?.some((g) => Math.abs(g) > 0));
    assert.ok(bn.bias.grad?.some((g) => Math.abs(g) > 0));
    assert.ok(x.grad?.some((g) => Math.abs(g) > 1e-8));
}

// --- TinyCnn train step ---
const model = new TinyCnn(1, 3, { channels: 4, spatial: 8 });
const images = tensor(new Array(2 * 1 * 8 * 8).fill(0.25), [2, 1, 8, 8], { requiresGrad: false });
const logits = model.forward(images);
assert.deepEqual([...logits.shape], [2, 3]);
const loss = logits.mean();
model.zeroGrad();
loss.backward();
const grads = model.parameters().map((p) => p.grad);
assert.ok(grads.some((g) => g && g.some((x) => Math.abs(x) > 0)));
assert.ok(model.conv.weight.grad?.some((g) => Math.abs(g) > 0));
assert.ok(model.conv.bias.grad?.some((g) => Math.abs(g) > 0));

// --- Named serialize roundtrip ---
const named = await model.state();
const packed = packTensors(named);
const restored = unpackTensors(packed);
const clone = new TinyCnn(1, 3, { channels: 4, spatial: 8, requiresGrad: false });
clone.loadState(restored, { requiresGrad: false });
const a = await withoutGrad(() => model.forward(images)).toArray();
const b = await withoutGrad(() => clone.forward(images)).toArray();
assert.equal(a.length, b.length);
for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i]! - b[i]!) < 1e-5);
}

console.log(`framework-core-cnn ok: params=${model.parameters().length} logits=${a.slice(0, 3).map((n) => +n.toFixed(3))}`);
