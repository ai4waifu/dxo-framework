import assert from 'node:assert/strict';
import { createRuntime, version } from '@dxo/lite';

const v = version();
assert.equal(typeof v, 'string');
assert.match(v, /lite/);

const rt = await createRuntime({ fallback: 'cpu' });
assert.equal(rt.capabilities.webglTensorBackend, false);
const backend = rt.capabilities.backend;
const t = await rt.zeros([2, 2]);
assert.equal(t.constructor.name, 'Tensor');
assert.deepEqual(await t.toArray(), [0, 0, 0, 0]);
rt.destroy();

console.log(`lite-probe ok: version=${v}, backend=${backend}`);
