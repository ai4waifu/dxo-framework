/**
 * diagnostic-wire: structured DxoError codes + locale formatting (Living 15).
 * code/args stable across locales; message text may differ.
 */
import assert from 'node:assert/strict';
import { DxoError, formatDiagnostic, tensor } from '@dxo/core';

{
    let caught: DxoError | undefined;
    try {
        tensor([1], [1], { device: 'cuda' });
    } catch (e) {
        caught = DxoError.fromUnknown(e);
    }
    assert.ok(caught);
    assert.equal(caught!.code, 'DXO_DEVICE_UNAVAILABLE');
    assert.equal(caught!.args?.device, 'cuda');
    const en = formatDiagnostic(caught!.toDiagnostic(), 'en-US');
    const zh = formatDiagnostic(caught!.toDiagnostic(), 'zh-CN');
    assert.match(en, /cuda/i);
    assert.match(zh, /cuda/i);
    assert.notEqual(en, zh);
    assert.equal(caught!.toDiagnostic().code, 'DXO_DEVICE_UNAVAILABLE');
}

{
    const t = tensor([1, 2], [2], { requiresGrad: true });
    let caught: DxoError | undefined;
    try {
        t.backward();
    } catch (e) {
        caught = DxoError.fromUnknown(e);
    }
    assert.ok(caught);
    assert.equal(caught!.code, 'DXO_TENSOR_NON_SCALAR');
    const diag = caught!.toDiagnostic();
    assert.equal(diag.code, 'DXO_TENSOR_NON_SCALAR');
    assert.ok(diag.args?.shape);
}

{
    const t = tensor([1, 2], [2]);
    let caught: DxoError | undefined;
    try {
        await t.item();
    } catch (e) {
        caught = DxoError.fromUnknown(e);
    }
    assert.ok(caught);
    assert.equal(caught!.code, 'DXO_TENSOR_NON_SCALAR');
}

{
    // Same diagnostic payload across locales (machine fields).
    const err = new DxoError({
        code: 'DXO_TITAN_BACKEND_UNAVAILABLE',
        args: { requested: 'cuda', available: 'cpu' },
        backend: 'cuda',
        operation: 'to',
        message: 'CUDA unavailable (no driver/device)',
    });
    const a = err.toDiagnostic();
    const b = err.toDiagnostic();
    assert.equal(a.code, b.code);
    assert.deepEqual(a.args, b.args);
    assert.equal(a.backend, 'cuda');
    const en = formatDiagnostic(a, 'en-US');
    const zh = formatDiagnostic(a, 'zh-CN');
    assert.notEqual(en, zh);
}

console.log('diagnostic-wire ok: DxoError codes + en/zh catalog');
