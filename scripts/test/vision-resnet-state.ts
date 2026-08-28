/**
 * vision-resnet-state: DXO-native ResNet-18 state schema + minimal forward (32×32 → [N,512]).
 */
import assert from 'node:assert/strict';
import { tensor } from '@dxo/core';
import { ResNet, VisionError } from '@dxo/vision';

const backbone = new ResNet({ depth: 18 });
const names = backbone.parameterNames();
assert.ok(names.includes('stem.conv.weight'));
assert.ok(names.includes('stage1.block0.conv1.weight'));
assert.ok(names.includes('stage4.block1.conv2.weight'));
assert.ok(names.includes('stage2.block0.downsample.conv.weight'));
assert.ok(!names.some((k) => k.includes('fc') || k.includes('numClasses')));
assert.ok(!names.some((k) => k.startsWith('layer') || k === 'conv1.weight'));

const st = await backbone.state();
assert.equal(Object.keys(st).sort().join('\n'), [...names].sort().join('\n'));

const clone = new ResNet({ depth: 18 });
clone.loadState(st, { requiresGrad: false });

const n = 1;
const x = tensor(new Array(n * 3 * 32 * 32).fill(0.01), [n, 3, 32, 32]);
const features = clone.forward(x);
assert.deepEqual([...features.shape], [1, 512]);

assert.throws(
    () => clone.forward(tensor(new Array(1 * 3 * 64 * 64).fill(0.01), [1, 3, 64, 64])),
    (err: unknown) => err instanceof VisionError && err.code === 'UNSUPPORTED',
);

assert.throws(
    () => new ResNet({ depth: 34 }).forward(x),
    (err: unknown) => err instanceof VisionError && err.code === 'UNSUPPORTED',
);

{
    const deep = new ResNet({ depth: 34 });
    assert.equal(deep.parameterNames().length, 0);
    await assert.rejects(
        () => deep.state(),
        (err: unknown) => err instanceof VisionError && err.code === 'UNSUPPORTED',
    );
}

console.log(`vision-resnet-state ok: keys=${names.length} features=${features.shape.join('x')}`);
