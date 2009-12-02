import assert from 'node:assert/strict';
import { backend, ones, randn, tensor, version, zeros } from '@dxo/core';

const v = version();
assert.equal(typeof v, 'string');
assert.match(v, /^\d+\.\d+\.\d+/);
assert.equal(backend(), 'cpu');

const a = tensor([1, 2, 3, 4], [2, 2]);
const b = tensor([5, 6, 7, 8], [2, 2]);
const c = a.matmul(b);
assert.deepEqual([...c.shape], [2, 2]);
assert.ok(Math.abs((await c.toArray())[0]! - 19) < 1e-5);

const x = tensor([1, 2, 3, 4], [2, 2]);
const bias = tensor([10, 20], [2]);
const y = x.add(bias);
assert.deepEqual(await y.toArray(), [11, 22, 13, 24]);

const z = zeros([3, 2]);
assert.deepEqual([...z.shape], [3, 2]);
assert.ok((await z.toArray()).every((n) => n === 0));

const o = ones([2, 2]);
assert.deepEqual(await o.toArray(), [1, 1, 1, 1]);

const r = randn([4]);
assert.equal((await r.toArray()).length, 4);

const m = tensor([2, 3, 4, 5], [2, 2]);
const scaled = m.mul(bias);
assert.deepEqual(await scaled.toArray(), [20, 60, 40, 100]);

const view = x.reshape([4]);
assert.deepEqual([...view.shape], [4]);
assert.deepEqual(await view.toArray(), [1, 2, 3, 4]);

const t = x.transpose();
assert.deepEqual([...t.shape], [2, 2]);
assert.deepEqual(await t.toArray(), [1, 3, 2, 4]);

const f32 = new Float32Array([1, 0, 0, 1]);
const fromF32 = tensor(f32, [2, 2]);
assert.deepEqual(await fromF32.toArray(), [1, 0, 0, 1]);

const cArr = await c.toArray();
console.log(`tensor-cpu ok: version=${v}, backend=${backend()}, matmul=${cArr[0]}`);
