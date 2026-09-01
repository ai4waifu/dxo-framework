/**
 * precision-fallback-diagnostics: structured errors for unavailable precision / decode (Living `12`).
 * v0: error path only — silent f16/f32 fallback not wired until mixed-precision parity.
 */
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import {
    DxoError,
    decodeImageBuffer,
    formatDiagnostic,
    resolvePrecisionPolicy,
    resolveWeightDtype,
} from '@dxo/core';
import {
    precisionPolicyFromMetadata,
    stampResolvedPrecision,
    validateManifestPrecision,
} from '@dxo/serialize';

function catchDxo(fn: () => void): DxoError {
    try {
        fn();
    } catch (e) {
        return DxoError.fromUnknown(e);
    }
    throw new Error('expected throw');
}

{
    const err = catchDxo(() => resolveWeightDtype('f16'));
    assert.equal(err.code, 'DXO_PRECISION_DTYPE_UNAVAILABLE');
    assert.ok(err.args?.requested === 'f16' || err.args?.canonical === 'f16');
    const en = formatDiagnostic(err.toDiagnostic(), 'en-US');
    const zh = formatDiagnostic(err.toDiagnostic(), 'zh-CN');
    assert.match(en, /f16/i);
    assert.notEqual(en, zh);
}

{
    const err = catchDxo(() => decodeImageBuffer(Buffer.from([0xff, 0xd8, 0xff]), { format: 'jpeg' }));
    assert.equal(err.code, 'DXO_IMAGE_DECODE_UNSUPPORTED');
    const en = formatDiagnostic(err.toDiagnostic(), 'en-US');
    assert.match(en, /jpeg|format/i);
}

{
    assert.throws(
        () => resolvePrecisionPolicy({ weights: 'f32', fallback: 'f32' }),
        (e: unknown) => e instanceof Error && !(e instanceof DxoError) && /fallback/i.test((e as Error).message),
    );
}

{
    const resolved = resolvePrecisionPolicy({ weights: 'f32', activations: 'f32' });
    assert.deepEqual(resolved, { weights: 'f32', activations: 'f32' });
}

{
    const policy = precisionPolicyFromMetadata({
        'dxo.precision.weights': 'f32',
        'dxo.precision.activations': 'f32',
    });
    assert.deepEqual(policy, { weights: 'f32', activations: 'f32' });
    assert.deepEqual(validateManifestPrecision({
        'dxo.precision.weights': 'f32',
    }), { weights: 'f32', activations: 'f32' });
}

{
    const err = catchDxo(() =>
        validateManifestPrecision({
            'dxo.precision.weights': 'int8',
        }),
    );
    assert.equal(err.code, 'DXO_PRECISION_DTYPE_UNAVAILABLE');
}

{
    const stamped = stampResolvedPrecision({}, { weights: 'f32', activations: 'f32' });
    assert.equal(stamped['dxo.precision.resolved'], 'f32/f32');
}

console.log('precision-fallback-diagnostics: ok');
