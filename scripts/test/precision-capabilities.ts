/**
 * precision-capabilities: honest f32-only probe (Living `12`).
 */
import assert from 'node:assert/strict';
import { DxoError, precisionCapabilities, resolveWeightDtype } from '@dxo/core';

{
    const caps = precisionCapabilities();
    assert.ok(caps.backend === 'cpu' || caps.backend === 'cuda');
    assert.equal(typeof caps.cudaAvailable, 'boolean');
    assert.deepEqual(caps.weights, ['f32']);
    assert.deepEqual(caps.activations, ['f32']);
    assert.deepEqual(caps.accumulation, ['f32']);
}

{
    assert.equal(resolveWeightDtype('f32'), 'f32');
    assert.equal(resolveWeightDtype('float32'), 'f32');
}

{
    assert.throws(() => resolveWeightDtype('f16'), (err: unknown) => err instanceof DxoError && (err as DxoError).code === 'DXO_PRECISION_DTYPE_UNAVAILABLE');
    assert.throws(() => resolveWeightDtype('int8'), (err: unknown) => err instanceof DxoError && (err as DxoError).code === 'DXO_PRECISION_DTYPE_UNAVAILABLE');
}

console.log('precision-capabilities: ok');
