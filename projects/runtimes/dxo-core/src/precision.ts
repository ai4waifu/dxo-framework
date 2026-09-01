/**
 * Precision / quantization capability reporting (Living `12`).
 * Honest probe — no silent int8/f16 kernel fallback.
 */

import { loadNative } from './native.js';
import { wrapNative } from './errors.js';
import type { NativePrecisionCapsReport } from './native-types.js';

export type PrecisionCapabilities = NativePrecisionCapsReport;

/** Policy object for precision selection (Living `12` — fields may expand with kernels). */
export type PrecisionPolicy = {
    weights?: 'f32' | 'f16' | 'bf16' | 'int8' | 'int4';
    activations?: 'f32' | 'f16' | 'bf16' | 'int8';
    accumulation?: 'f32' | 'f16';
    fallback?: 'error' | 'f32' | 'f16';
};

/** Probe runtime precision capabilities. */
export function precisionCapabilities(): PrecisionCapabilities {
    return loadNative().precisionCapabilities();
}

/** Resolve a requested weight dtype; throws `DxoError` when unavailable on this build. */
export function resolveWeightDtype(requested: string): string {
    return wrapNative(() => loadNative().resolveWeightDtype(requested));
}

/** Resolve policy weight dtype or throw; no silent fallback unless `fallback` is set (v0: only `error`). */
export function resolvePrecisionPolicy(policy: PrecisionPolicy): { weights: string; activations: string } {
    const weights = resolveWeightDtype(policy.weights ?? 'f32');
    const activations = resolveWeightDtype(policy.activations ?? weights);
    if (policy.fallback && policy.fallback !== 'error') {
        throw new Error('PrecisionPolicy.fallback !== "error" is not wired until mixed-precision parity lands');
    }
    void policy.accumulation;
    return { weights, activations };
}
