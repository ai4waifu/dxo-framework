import assert from 'node:assert/strict';
import { createRuntime, version } from '@dxo/lite';

/**
 * M6 / 0.0.8 thin gate: async createRuntime, capabilities, CPU fallback,
 * Promise tensor ops, and hard ban on WebGL as a tensor backend.
 * Node CI has no WebGPU — smoke the CPU path + diagnostic error path.
 */

assert.match(version(), /lite/);

const rt = await createRuntime({ fallback: 'cpu' });
assert.equal(rt.capabilities.backend, 'cpu');
assert.equal(rt.capabilities.webgpu, false);
assert.equal(rt.capabilities.webglTensorBackend, false);
assert.equal(rt.capabilities.dtype.f32, true);

const a = await rt.tensor([1, 2, 3, 4], [2, 2]);
const b = await rt.tensor([5, 6, 7, 8], [2, 2]);
const c = await a.matmul(b);
assert.deepEqual(await c.toArray(), [19, 22, 43, 50]);

const sum = await (await rt.ones([2])).add(await rt.ones([2]));
assert.deepEqual(await sum.toArray(), [2, 2]);

rt.destroy();
await assert.rejects(() => rt.tensor([1], [1]), /destroyed/);

await assert.rejects(
    () => createRuntime({ fallback: 'error' }),
    (err: unknown) => {
        assert.ok(err instanceof Error);
        assert.match(err.message, /WebGPU/i);
        assert.match(err.message, /WebGL is not a DXO tensor backend/);
        return true;
    },
);

console.log('lite-webgpu-smoke ok: cpu fallback + Promise matmul + no-WebGL contract');
