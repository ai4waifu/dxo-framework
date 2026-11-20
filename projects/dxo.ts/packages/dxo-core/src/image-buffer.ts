/**
 * ImageBuffer — typed host pixel carrier (Living `11`).
 * Built on TypedBuffer primitives; not an `@dxo/vision` Image.
 */

import { Buffer } from 'node:buffer';
import { Tensor } from './index.js';
import { wrapNative } from './errors.js';
import { loadNative } from './native.js';
import type { NativeImageBuffer } from './native-types.js';
import type { BufferToTensorOptions, HostDtype, TypedBuffer } from './typed-buffer.js';

export type ImageBufferDtype = Extract<HostDtype, 'u8' | 'u16' | 'f32'>;
export type ImageBufferLayout = 'HWC' | 'CHW';
export type ImageBufferColorSpace = 'rgb' | 'rgba' | 'gray' | 'bgr' | 'unknown';
export type ImageBufferAlphaMode = 'opaque' | 'straight' | 'premultiplied' | 'none';

export type ImageBufferToTensorOptions = BufferToTensorOptions & {
    /** Scale u8 pixels to `[0,1]` when converting to f32 NCHW (default `true`). */
    normalize?: boolean;
};

export type CreateImageBufferFromPixelsOptions = {
    width: number;
    height: number;
    channels: number;
    dtype: ImageBufferDtype;
    layout: ImageBufferLayout;
    colorSpace: ImageBufferColorSpace;
    alphaMode: ImageBufferAlphaMode;
    data: Uint8Array | Buffer;
};

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

    pixelBytes(): Buffer;
    toTensor(options?: ImageBufferToTensorOptions): Tensor;
}

export type DecodeImageBridgeOptions = {
    /** Hint only; codec provider may reject unsupported formats. */
    format?: 'png' | 'jpeg' | 'webp' | 'raw';
};

class ImageBufferImpl implements ImageBuffer {
    readonly #handle: NativeImageBuffer;

    constructor(handle: NativeImageBuffer) {
        this.#handle = handle;
    }

    get width(): number {
        return this.#handle.width;
    }

    get height(): number {
        return this.#handle.height;
    }

    get channels(): number {
        return this.#handle.channels;
    }

    get dtype(): ImageBufferDtype {
        return this.#handle.dtype as ImageBufferDtype;
    }

    get layout(): ImageBufferLayout {
        return this.#handle.layout as ImageBufferLayout;
    }

    get colorSpace(): ImageBufferColorSpace {
        return this.#handle.colorSpace as ImageBufferColorSpace;
    }

    get alphaMode(): ImageBufferAlphaMode {
        return this.#handle.alphaMode as ImageBufferAlphaMode;
    }

    get shape(): readonly number[] {
        return [this.height, this.width, this.channels];
    }

    get stride(): readonly number[] {
        if (this.layout === 'CHW') {
            const plane = this.height * this.width;
            return [plane, this.width, 1];
        }
        return [this.width * this.channels, this.channels, 1];
    }

    pixelBytes(): Buffer {
        return this.#handle.pixelBytes();
    }

    toTensor(options?: ImageBufferToTensorOptions): Tensor {
        const normalize = options?.normalize ?? true;
        let t = new Tensor(wrapNative(() => this.#handle.toTensor(normalize)));
        if (options?.device === 'cuda') {
            t = t.to('cuda');
        }
        return t;
    }

    dispose(): void {
        // Host bytes owned by native handle; no separate teardown in v0.
    }

    ready(): Promise<void> {
        return Promise.resolve();
    }
}

/** Construct ImageBuffer from explicit pixel bytes + layout metadata. */
export function createImageBufferFromPixels(options: CreateImageBufferFromPixelsOptions): ImageBuffer {
    const native = loadNative();
    const data = Buffer.from(options.data);
    const handle = wrapNative(() =>
        native.createImageBufferFromPixels(
            options.width,
            options.height,
            options.channels,
            options.dtype,
            options.layout,
            options.colorSpace,
            options.alphaMode,
            data,
        ),
    );
    return new ImageBufferImpl(handle);
}

/** Native decode bridge (v0: PNG; JPEG/WebP return structured errors). */
export function decodeImageBuffer(bytes: Uint8Array, options?: DecodeImageBridgeOptions): Promise<ImageBuffer> {
    const native = loadNative();
    const handle = wrapNative(() => native.decodeImageBuffer(Buffer.from(bytes), options?.format ?? null));
    return Promise.resolve(new ImageBufferImpl(handle));
}

export function unsupportedImageBufferApi(name: string): never {
    throw new Error(`@dxo/core ${name} is reserved for the ImageBuffer napi surface; not an @dxo/vision Image API`);
}
