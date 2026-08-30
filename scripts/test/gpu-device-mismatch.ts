/**
 * gpu-device-mismatch: structured errors for cross-device ops.
 */
import assert from 'node:assert/strict';
import { cudaAvailable, tensor, version } from '@dxo/core';

const v = version();
if (!cudaAvailable()) {
    if (process.env.DXO_REQUIRE_CUDA === '1') {
        console.error(`gpu-device-mismatch FAIL: CUDA required but unavailable (version=${v})`);
        process.exit(1);
    }
    console.log(`gpu-device-mismatch SKIP: CUDA unavailable (version=${v})`);
    process.exit(0);
}

const cpu = tensor([1, 2, 3, 4], [2, 2]);
const gpu = cpu.detach().to('cuda');

await assert.rejects(async () => cpu.add(gpu), /device mismatch/i);
await assert.rejects(async () => cpu.matmul(gpu), /device mismatch/i);

console.log(`gpu-device-mismatch ok: version=${v}`);
