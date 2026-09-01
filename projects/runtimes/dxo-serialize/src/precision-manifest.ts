/**
 * Manifest precision fields (Living `12`) — validate against runtime capabilities; no silent fallback.
 */

import { resolvePrecisionPolicy, type PrecisionPolicy } from '@dxo/core';

const KEYS = {
    weights: 'dxo.precision.weights',
    activations: 'dxo.precision.activations',
    accumulation: 'dxo.precision.accumulation',
    fallback: 'dxo.precision.fallback',
} as const;

type ManifestWeightDtype = NonNullable<PrecisionPolicy['weights']>;
type ManifestActivationDtype = NonNullable<PrecisionPolicy['activations']>;
type ManifestAccumulationDtype = NonNullable<PrecisionPolicy['accumulation']>;
type ManifestFallback = NonNullable<PrecisionPolicy['fallback']>;

function pick<K extends string>(metadata: Record<string, string>, key: K): string | undefined {
    const v = metadata[key];
    return typeof v === 'string' && v.trim() ? v.trim() : undefined;
}

/** Parse manifest metadata into a `PrecisionPolicy` (empty fields omitted). */
export function precisionPolicyFromMetadata(metadata: Record<string, string>): PrecisionPolicy {
    const policy: PrecisionPolicy = {};
    const weights = pick(metadata, KEYS.weights);
    const activations = pick(metadata, KEYS.activations);
    const accumulation = pick(metadata, KEYS.accumulation);
    const fallback = pick(metadata, KEYS.fallback);
    if (weights) policy.weights = weights as ManifestWeightDtype;
    if (activations) policy.activations = activations as ManifestActivationDtype;
    if (accumulation) policy.accumulation = accumulation as ManifestAccumulationDtype;
    if (fallback) policy.fallback = fallback as ManifestFallback;
    return policy;
}

/** Resolve manifest precision against runtime capabilities; throws `DxoError` when unavailable. */
export function validateManifestPrecision(metadata: Record<string, string>): { weights: string; activations: string } {
    return resolvePrecisionPolicy(precisionPolicyFromMetadata(metadata));
}

/** Attach resolved precision fields to safetensors metadata (audit trail). */
export function stampResolvedPrecision(
    metadata: Record<string, string>,
    resolved: { weights: string; activations: string },
): Record<string, string> {
    return {
        ...metadata,
        [KEYS.weights]: resolved.weights,
        [KEYS.activations]: resolved.activations,
        'dxo.precision.resolved': `${resolved.weights}/${resolved.activations}`,
    };
}
