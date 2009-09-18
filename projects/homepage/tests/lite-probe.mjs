import assert from 'node:assert/strict';
import { Tensor, version } from '@dxo/lite';

const v = version();
assert.equal(typeof v, 'string');
assert.match(v, /lite/);

const t = new Tensor();
assert.ok(t instanceof Tensor);

console.log(`lite-probe ok: version=${v}, Tensor=${t.constructor.name}`);
