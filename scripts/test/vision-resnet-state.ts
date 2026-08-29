/**
 * vision-resnet-state: Living `02` canonical keys + 32x32 feature forward (GPU lane).
 */
import assert from 'node:assert/strict';
import { tensor } from '@dxo/core';
import { ResNet } from '@dxo/vision';

const backbone = new ResNet({ depth: 18, trainable: false });
const names = backbone.parameterNames();
assert.ok(names.includes('stem.convolution.weight'));
assert.ok(names.includes('stem.batch_normalization.weight'));
assert.ok(names.includes('stage_1.block_1.convolution_1.weight'));
assert.ok(names.includes('stage_4.block_2.convolution_2.weight'));
assert.ok(names.some((k) => k.includes('downsample.convolution.weight')));
assert.ok(!names.some((k) => /(^|[.])(fc|bn|conv)([.]|$)/.test(k)));
assert.ok(!names.some((k) => /\.block_0\.|stage[0-9]|conv[0-9]|bn[0-9]/.test(k)));

const saved = await backbone.state();
assert.deepEqual(Object.keys(saved).sort(), [...names].sort());

const clone = new ResNet({ depth: 18, trainable: false });
clone.loadState(saved, { requiresGrad: false });

const x = tensor(new Array(1 * 3 * 32 * 32).fill(0.01), [1, 3, 32, 32]);
const features = clone.forward(x);
assert.deepEqual([...features.shape], [1, 512]);

console.log(`vision-resnet-state ok: keys=${names.length} features=${features.shape.join('x')}`);
