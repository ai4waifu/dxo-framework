/**
 * ImageBuffer — typed host pixel carrier (Living `11`).
 * Built on TypedBuffer primitives; not an `@dxo/vision` Image.
 */

import type { Tensor } from './index.js';
import type { BufferToTensorOptions, HostDtype, TypedBuffer } from './typed-buffer.js';

export type ImageBufferDtype = Extract<HostDtype, 'u8' | 'u16' | 'f32'>;
export type ImageBufferLayout = 'HWC' | 'CHW';
export type ImageBufferColorSpace = 'rgb' | 'rgba' | 'gray' | 'bgr' | 'unknown';
export type ImageBufferAlphaMode = 'opaque' | 'straight' | 'premultiplied' | 'none';

export type ImageBufferToTensorOptions = BufferToTensorOptions;

/**
 * Opaque typed host buffer for image pixels — not a vision domain object.
 */
export interface ImageBuffer extends TypedBuffer {
    readonly width: number;
    readonly height: number;
    readonly channels: number;
    readonly dtype: ImageBufferDtype;
    readonly layout: ImageBufferLayout;
    readonly stride: readonly number[];
    readonly colorSpace: ImageBufferColorSpace;
    readonly alphaMode: ImageBufferAlphaMode;

    toTensor(options?: ImageBufferToTensorOptions): Tensor;
}

export type DecodeImageBridgeOptions = {
    /** Hint only; codec provider may ignore unsupported formats. */
    format?: 'png' | 'jpeg' | 'webp' | 'raw';
};

/**
 * Native decode bridge entry (stub). Codec providers plug in later;
 * full JPEG/PNG product APIs stay out of `@dxo/core`.
 */
export function decodeImageBuffer(_bytes: Uint8Array, _options?: DecodeImageBridgeOptions): Promise<ImageBuffer> {
    return Promise.reject(new Error('@dxo/core decodeImageBuffer is not wired yet'));
}

export function unsupportedImageBufferApi(name: string): never {
    throw new Error(`@dxo/core ${name} is reserved for the ImageBuffer napi surface; not an @dxo/vision Image API`);
}
