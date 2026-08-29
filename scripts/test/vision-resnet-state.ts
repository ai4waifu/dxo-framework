/**
 * vision-resnet-state: DXO-native ResNet-18 state schema + 32x32 feature forward.
 */
import assert from 'node:assert/strict';
import { tensor } from '@dxo/core';
import { ResNet, VisionError } from '@dxo/vision';

const backbone = new ResNet({ depth: 18, trainable: false });
const names = backbone.parameterNames();
assert.ok(names.includes('stem.conv.weight'));
assert.ok(names.includes('stage1.block0.conv1.weight'));
assert.ok(names.includes('stage4.block1.conv2.weight'));
assert.ok(
    names.some((k) => k.includes('down.conv.weight')),
    'expected downsample keys',
);
assert.ok(!names.some((k) => k.includes('fc') || k.includes('classifier')));

const saved = await backbone.state();
assert.deepEqual(Object.keys(saved).sort(), [...names].sort());

const clone = new ResNet({ depth: 18, trainable: false });
clone.loadState(saved, { requiresGrad: false });
const saved2 = await clone.state();
for (const key of names) {
    assert.deepEqual(saved2[key]!.shape, saved[key]!.shape, key);
    assert.deepEqual(saved2[key]!.data, saved[key]!.data, key);
}

// 32x32 → final 1x1 → features [N,512]
const n = 1;
const c = 3;
const hw = 32;
const x = tensor(new Array(n * c * hw * hw).fill(0.01), [n, c, hw, hw]);
const features = clone.forward(x);
assert.deepEqual([...features.shape], [1, 512]);

// depth ≠ 18
{
    const deep = new ResNet({ depth: 34 });
    assert.throws(
        () => deep.forward(x),
        (err: unknown) => err instanceof VisionError && err.code === 'UNSUPPORTED',
    );
    await assert.rejects(
        () => deep.state(),
        (err: unknown) => err instanceof VisionError && err.code === 'UNSUPPORTED',
    );
}

console.log(`vision-resnet-state ok: keys=${names.length} features=${features.shape.join('x')}`);
