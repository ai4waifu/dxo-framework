/**
 * rust-error-code: Titan kind → DXO_TITAN_* via HAL/event probe paths (CPU schema).
 */
import assert from 'node:assert/strict';
import { DxoError, probeEventDep, tensor } from '@dxo/core';

// CPU HAL probe should succeed (no throw).
probeEventDep();

// Shape mismatch through native wire carries a stable code.
{
    let caught: DxoError | undefined;
    try {
        // numel mismatch → DXO_TENSOR_INVALID_SHAPE
        tensor([1, 2, 3], [2, 2]);
    } catch (e) {
        caught = DxoError.fromUnknown(e);
    }
    assert.ok(caught);
    assert.equal(caught!.code, 'DXO_TENSOR_INVALID_SHAPE');
}

// Device parse / to('cuda') when CUDA missing → titan/backend code (or skip if CUDA present).
{
    const { cudaAvailable } = await import('@dxo/core');
    if (!cudaAvailable()) {
        const t = tensor([1], [1]).detach();
        let caught: DxoError | undefined;
        try {
            t.to('cuda');
        } catch (e) {
            caught = DxoError.fromUnknown(e);
        }
        assert.ok(caught);
        assert.equal(caught!.code, 'DXO_TITAN_BACKEND_UNAVAILABLE');
        assert.equal(caught!.backend, 'cuda');
        assert.equal(caught!.args?.requested, 'cuda');
    } else {
        // With CUDA, to('cuda') on detached leaf should succeed — still a valid path.
        const t = tensor([1, 2], [2]).detach().to('cuda');
        assert.equal(t.device, 'cuda');
    }
}

console.log('rust-error-code ok: structured Titan/DXO codes on CPU paths');
