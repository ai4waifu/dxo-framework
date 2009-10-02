import assert from 'node:assert/strict';
import { cudaAvailable, tensor, version } from '@dxo/core';

const v = version();
if (!cudaAvailable()) {
    console.log(`gpu-matmul SKIP: CUDA unavailable (version=${v})`);
    process.exit(0);
}

const lhs = tensor([1, 2, 3, 4], [2, 2]);
const rhs = tensor([5, 6, 7, 8], [2, 2]);
const cpuRef = lhs.matmul(rhs);

const a = lhs.detach().to('cuda');
const b = rhs.detach().to('cuda');
assert.equal(a.device, 'cuda');
assert.equal(b.device, 'cuda');

const c = a.matmul(b);
assert.equal(c.device, 'cuda');
assert.deepEqual([...c.shape], [2, 2]);

for (let i = 0; i < cpuRef.toArray().length; i += 1) {
    assert.ok(Math.abs(c.toArray()[i]! - cpuRef.toArray()[i]!) < 1e-4, `index ${i}`);
}

const back = c.to('cpu');
assert.equal(back.device, 'cpu');
assert.ok(Math.abs(back.toArray()[0]! - 19) < 1e-4);

console.log(`gpu-matmul ok: version=${v}, cuda matmul matches cpu reference`);
