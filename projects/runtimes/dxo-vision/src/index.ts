/**
 * @dxo/vision — developer preview (API unstable).
 * Image types + ResNet backbone / heads / label decode.
 * Not a closed vision-classify gate. Weights live in external `@dxo/resnet`.
 */

import type { ImageBuffer, Tensor } from '@dxo/core';

export type { TensorStateSlice } from '@dxo/nn';
export { VisionError } from './errors.js';
export {
    Classifier,
    type ClassifierOptions,
    compose,
    LinearHead,
    type LinearHeadOptions,
} from './heads.js';
export {
    type ClassificationDecode,
    type ClassificationTopK,
    type DecodeClassificationOptions,
    decodeClassification,
    defineLabelSpace,
    type LabelSpace,
    type LabelSpaceOptions,
} from './labels.js';
export { defineResNet, RESNET18_STAGE_BLOCKS, RESNET18_STAGE_CHANNELS, ResNet } from './resnet.js';
export type {
    Device,
    ResNetDepth,
    ResNetOptions,
    ResNetSignature,
    TensorPort,
    WeightSource,
} from './types.js';
export { resnetFeatureChannels } from './types.js';

export type ColorSpace = 'rgb' | 'rgba' | 'gray' | 'bgr';
export type ImageLayout = 'HWC' | 'CHW' | 'nchw' | 'nhwc';
export type ImageDtype = 'u8' | 'u16' | 'f32';

export type ImageTensorSpec = {
    layout: 'nchw' | 'nhwc';
    dtype: 'f32';
    channels: 1 | 3 | 4;
};

export type ImageToTensorOptions = {
    device?: 'cpu' | 'cuda' | 'metal';
    copy?: boolean;
    /** Target model layout; vision may transpose/cast before buffer view. */
    layout?: 'nchw' | 'nhwc';
};

/**
 * User-facing image with vision metadata. Does not own a parallel native engine.
 * Pixel carrier is `@dxo/core` ImageBuffer (colorSpace/alphaMode on the buffer).
 */
export interface Image {
    readonly width: number;
    readonly height: number;
    readonly channels: number;
    readonly colorSpace: ColorSpace;
    readonly layout: ImageLayout;
    readonly dtype: ImageDtype;
    readonly alpha: boolean;
    readonly source?: { kind: string; uri?: string };

    /** Underlying typed host buffer when available (no domain semantics). */
    buffer(): ImageBuffer | undefined;
    toTensor(options?: ImageToTensorOptions): Tensor;
    dispose(): void;
    ready(): Promise<void>;
}

export interface ImageBatch {
    readonly size: number;
    readonly images: readonly Image[];
    toTensor(options?: ImageToTensorOptions): Tensor;
    dispose(): void;
}

/** @deprecated Prefer {@link ImageTensorSpec} + {@link Image}. */
export type LegacyImageTensorSpec = ImageTensorSpec;

export function visionVersion(): string {
    return 'dxo-vision@developer-preview';
}

export function unsupportedVisionApi(name: string): never {
    throw new Error(`@dxo/vision ${name} is not wired yet on this developer preview surface`);
}
