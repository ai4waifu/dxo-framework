/**
 * gpu-readback: explicit .to('cpu') / toArray is the only host materialization contract.
 */
import assert from 'node:assert/strict';
import { cudaAvailable, hostTransferCount, resetHostTransferCount, tensor, version } from '@dxo/core';

const v = version();
if (!cudaAvailable()) {
    if (process.env.DXO_REQUIRE_CUDA === '1') {
        console.error(`gpu-readback FAIL: CUDA required but unavailable (version=${v})`);
        process.exit(1);
    }
    console.log(`gpu-readback SKIP: CUDA unavailable (version=${v})`);
    process.exit(0);
}

resetHostTransferCount();
const g = tensor([1, 2, 3, 4], [2, 2]).detach().to('cuda');
const afterUp = hostTransferCount();
const back = g.to('cpu');
assert.equal(back.device, 'cpu');
const afterDown = hostTransferCount();
assert.ok(afterDown > afterUp, 'to(cpu) must readback');
assert.deepEqual(await back.toArray(), [1, 2, 3, 4]);

console.log(`gpu-readback ok: version=${v}`);
