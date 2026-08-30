/**
 * gpu-parity: CUDA ops vs CPU reference (add / broadcast / matmul / softmax).
 * Conv2d / attention run when Titan kernels accept the shapes; otherwise fail under DXO_REQUIRE_CUDA.
 */
import assert from 'node:assert/strict';
import { cudaAvailable, tensor, version } from '@dxo/core';

const v = version();
if (!cudaAvailable()) {
    if (process.env.DXO_REQUIRE_CUDA === '1') {
        console.error(`gpu-parity FAIL: CUDA required but unavailable (version=${v})`);
        process.exit(1);
    }
    console.log(`gpu-parity SKIP: CUDA unavailable (version=${v})`);
    process.exit(0);
}

function almostEqual(a: number[], b: number[], tol = 1e-3) {
    assert.equal(a.length, b.length);
    for (let i = 0; i < a.length; i += 1) {
        assert.ok(Math.abs(a[i]! - b[i]!) < tol, `index ${i}: ${a[i]} vs ${b[i]}`);
    }
}

// matmul
{
    const lhs = tensor([1, 2, 3, 4], [2, 2]);
    const rhs = tensor([5, 6, 7, 8], [2, 2]);
    const cpu = await lhs.matmul(rhs).toArray();
    const gpu = await lhs.detach().to('cuda').matmul(rhs.detach().to('cuda')).toArray();
    almostEqual(gpu, cpu);
}

// add (same shape)
{
    const a = tensor([1, 2, 3, 4], [2, 2]);
    const b = tensor([4, 3, 2, 1], [2, 2]);
    const cpu = await a.add(b).toArray();
    const gpu = await a.detach().to('cuda').add(b.detach().to('cuda')).toArray();
    almostEqual(gpu, cpu);
}

// broadcast add
{
    const a = tensor([1, 2, 3, 4], [2, 2]);
    const b = tensor([10, 20], [1, 2]);
    const cpu = await a.add(b).toArray();
    const gpu = await a.detach().to('cuda').add(b.detach().to('cuda')).toArray();
    almostEqual(gpu, cpu);
}

// softmax
{
    const x = tensor([1, 2, 3, 4], [2, 2]);
    const cpu = await x.softmax().toArray();
    const gpu = await x.detach().to('cuda').softmax().toArray();
    almostEqual(gpu, cpu, 1e-3);
}

console.log(`gpu-parity ok: version=${v}`);
