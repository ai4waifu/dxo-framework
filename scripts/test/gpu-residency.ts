/**
 * gpu-residency: continuous CUDA ops must not host-roundtrip between kernels.
 * Counts explicit uploads (to cuda) + one final readback only.
 */
import assert from 'node:assert/strict';
import {
    cudaAvailable,
    hostTransferCount,
    resetHostTransferCount,
    tensor,
    version,
} from '@dxo/core';

const v = version();
if (!cudaAvailable()) {
    if (process.env.DXO_REQUIRE_CUDA === '1') {
        console.error(`gpu-residency FAIL: CUDA required but unavailable (version=${v})`);
        process.exit(1);
    }
    console.log(`gpu-residency SKIP: CUDA unavailable (version=${v})`);
    process.exit(0);
}

resetHostTransferCount();
const before = hostTransferCount();

const a = tensor([1, 2, 3, 4], [2, 2]).detach().to('cuda');
const b = tensor([5, 6, 7, 8], [2, 2]).detach().to('cuda');
const afterUpload = hostTransferCount();
assert.ok(afterUpload - before >= 2, `expected ≥2 uploads, got ${afterUpload - before}`);

const mid = hostTransferCount();
const c = a.matmul(b).matmul(b).add(a);
const afterOps = hostTransferCount();
assert.equal(
    afterOps,
    mid,
    `device-resident chain must not transfer (mid=${mid} afterOps=${afterOps})`,
);

const out = await c.toArray();
const afterRead = hostTransferCount();
assert.ok(afterRead > afterOps, 'explicit readback must increment transfer counter');
assert.deepEqual([...c.shape], [2, 2]);
assert.ok(out.length === 4);

// device mismatch
{
    const cpu = tensor([1, 2, 3, 4], [2, 2]);
    const gpu = cpu.detach().to('cuda');
    let threw = false;
    try {
        cpu.matmul(gpu);
    } catch {
        threw = true;
    }
    assert.ok(threw, 'device mismatch must throw');
}

console.log(`gpu-residency ok: version=${v} transfers=${hostTransferCount()}`);
