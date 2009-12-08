import type { Device } from '@dxo/core';

/** Soft graph port — connectivity only; no labels / language. */
export type TensorPort = {
    name: string;
    dtype: 'f32';
    /** Symbolic dims; numbers are fixed, strings are dynamic axes. */
    shape: ReadonlyArray<number | string>;
};

/** External weight locator (resolved by hub / external `@dxo/resnet` later). */
export type WeightSource =
    | {
          provider: string;
          repo: string;
          revision?: string;
          path: string;
      }
    | { path: string };

export type { Device };

export type ResNetDepth = 18 | 34 | 50 | 101 | 152;

/**
 * ResNet constructor options.
 * Intentionally has **no** `classes` / `numClasses` / label fields.
 */
export type ResNetOptions = {
    depth?: ResNetDepth;
    inChannels?: number;
    zeroInitResidual?: boolean;
    norm?: 'batchnorm';
    weights?: WeightSource | 'none';
    device?: Device;
    trainable?: boolean;
};

export type ResNetSignature = {
    input: { image: TensorPort };
    output: { features: TensorPort };
};

/** Feature channel count after GAP for classic ResNet depths. */
export function resnetFeatureChannels(depth: ResNetDepth): number {
    switch (depth) {
        case 18:
        case 34:
            return 512;
        case 50:
        case 101:
        case 152:
            return 2048;
        default: {
            const _exhaustive: never = depth;
            return _exhaustive;
        }
    }
}
