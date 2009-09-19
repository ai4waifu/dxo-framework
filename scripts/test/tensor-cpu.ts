import assert from 'node:assert/strict';
import { tensor, version, zeros } from '@dxo/core';

const v = version();
assert.equal(typeof v, 'string');
assert.match(v, /^\d+\.\d+\.\d+/);

const a = tensor([1, 2, 3, 4], [2, 2]);
const b = tensor([5, 6, 7, 8], [2, 2]);
const c = a.matmul(b);
assert.deepEqual([...c.shape], [2, 2]);
assert.ok(Math.abs(c.toArray()[0]! - 19) < 1e-5);

const x = tensor([1, 2, 3, 4], [2, 2]);
const bias = tensor([10, 20], [2]);
const y = x.add(bias);
assert.deepEqual(y.toArray(), [11, 22, 13, 24]);

const z = zeros([3, 2]);
assert.deepEqual([...z.shape], [3, 2]);
assert.ok(z.toArray().every((n) => n === 0));

console.log(`tensor-cpu ok: version=${v}, matmul=${c.toArray()[0]}`);
