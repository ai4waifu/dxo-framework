import assert from 'node:assert/strict';
import { Tensor, version, zeros } from '@dxo/core';

const v = version();
assert.equal(typeof v, 'string');
assert.match(v, /^\d+\.\d+\.\d+/);

const t = zeros([1]);
assert.ok(t instanceof Tensor);

console.log(`smoke ok: version=${v}, Tensor=${t.constructor.name}`);
