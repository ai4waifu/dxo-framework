import assert from 'node:assert/strict';
import { Tensor, version } from '@dxo/core';

const v = version();
assert.equal(typeof v, 'string');
assert.match(v, /^\d+\.\d+\.\d+/);

const t = new Tensor();
assert.ok(t instanceof Tensor);

console.log(`smoke ok: version=${v}, Tensor=${t.constructor.name}`);
