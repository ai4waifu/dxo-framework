import assert from 'node:assert/strict';
import { createRuntime, version } from '@dxo/lite';

/**
 * M6 / 0.0.8 thin gate: async createRuntime, capabilities, CPU fallback,
 * synchronous Tensor composition, async observation barriers, and hard ban on WebGL.
 * Node CI has no WebGPU — smoke the CPU path + diagnostic error path.
 * Also loads interim `@dxo/lite-unknown-wasm32` when the artifact is built.
 */

assert.match(version(), /lite/);

const rt = await createRuntime({ fallback: 'cpu' });
assert.equal(rt.capabilities.backend, 'cpu');
assert.equal(rt.capabilities.webgpu, false);
assert.equal(rt.capabilities.titanWgpuReady, false);
assert.equal(rt.capabilities.webglTensorBackend, false);
assert.equal(rt.capabilities.dtype.f32, true);
assert.ok(rt.capabilities.wasm, 'expected interim WASM kernels — run pnpm build:lite-wasm');
assert.match(rt.capabilities.wasm.version, /dxo-lite-wasm/);
assert.equal(rt.capabilities.wasm.interimHostF32, true);

const a = rt.tensor([1, 2, 3, 4], [2, 2]);
const b = rt.tensor([5, 6, 7, 8], [2, 2]);
const c = a.matmul(b);
await c.ready();
assert.deepEqual(await c.toArray(), [19, 22, 43, 50]);

const sum = rt.ones([2]).add(rt.ones([2]));
assert.deepEqual(await sum.toArray(), [2, 2]);

rt.destroy();
assert.throws(() => rt.tensor([1], [1]), /destroyed/);

await assert.rejects(
    () => createRuntime({ fallback: 'error' }),
    (err: unknown) => {
        assert.ok(err instanceof Error);
        assert.match(err.message, /WebGPU|Titan/i);
        assert.match(err.message, /WebGL is not a DXO tensor backend/);
        return true;
    },
);

console.log('lite-webgpu-smoke ok: WASM interim kernels + sync Tensor chain + no-WebGL contract');
