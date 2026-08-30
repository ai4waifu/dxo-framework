/**
 * titan-event-dep: HAL wait_event encodes upload→compute dependency.
 * CPU session always exercises the primitive.
 * With DXO_REQUIRE_CUDA=1, also requires the CUDA session probe (no skip).
 */
import { backend, cudaAvailable, probeEventDep, probeEventDepCuda, version } from '@dxo/core';

probeEventDep();
console.log(`titan-event-dep ok: CPU wait_event via ${backend()} (version=${version()})`);

if (process.env.DXO_REQUIRE_CUDA === '1') {
    if (!cudaAvailable()) {
        console.error('titan-event-dep FAIL: DXO_REQUIRE_CUDA=1 but CUDA unavailable');
        process.exit(1);
    }
    probeEventDepCuda();
    console.log('titan-event-dep ok: CUDA wait_event upload→compute');
} else if (cudaAvailable()) {
    probeEventDepCuda();
    console.log('titan-event-dep ok: CUDA wait_event (optional)');
}
