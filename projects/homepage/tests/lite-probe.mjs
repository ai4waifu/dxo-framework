import assert from 'node:assert/strict';
import { createRuntime, version } from '@dxo/lite';

assert.match(version(), /lite/);

const rt = await createRuntime({ fallback: 'cpu' });
assert.equal(rt.capabilities.backend, 'cpu');
assert.ok(rt.capabilities.wasm, 'build lite-unknown-wasm32 first (pnpm build:lite-wasm)');
const wasmVersion = rt.capabilities.wasm.version;

const a = rt.tensor([1, 2, 3, 4], [2, 2]);
const id = rt.tensor([1, 0, 0, 1], [2, 2]);
const out = await a.matmul(id).toArray();
assert.deepEqual(out, [1, 2, 3, 4]);

rt.destroy();
console.log(`lite-probe ok: version=${version()}, wasm=${wasmVersion}`);
