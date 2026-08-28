/**
 * Vision product surface (Living `11`).
 * Workspace-only; excluded from placeholder / publish-npm lists.
 *
 * `Image` / `ImageBatch` live here. Pixel storage / zero-copy views are
 * `@dxo/core` `ImageBuffer` — not a second tensor engine.
 */

import type { ImageBuffer, Tensor } from '@dxo/core';

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
    return 'dxo-vision@placeholder';
}

export function unsupportedVisionApi(name: string): never {
    throw new Error(`@dxo/vision ${name} is a workspace stub; real transforms / zoo are not wired yet`);
}
