/**
 * Vision-facing Image wrapper over `@dxo/core` ImageBuffer.
 */

import {
    createImageBufferFromPixels,
    decodeImageBuffer,
    type ImageBuffer,
    type ImageBufferAlphaMode,
    type ImageBufferColorSpace,
    type ImageBufferDtype,
    type ImageBufferLayout,
    type Tensor,
} from '@dxo/core';
import type {
    ColorSpace,
    Image,
    ImageDtype,
    ImageLayout,
    ImageToTensorOptions,
} from './index.js';

export type FromPixelsOptions = {
    width: number;
    height: number;
    channels: number;
    colorSpace: ColorSpace;
    layout?: ImageLayout;
    dtype?: ImageDtype;
    alpha?: boolean;
    data: Uint8Array;
    source?: { kind: string; uri?: string };
};

export type FromDecodeOptions = {
    format?: 'png' | 'jpeg' | 'webp';
    source?: { kind: string; uri?: string };
};

function toBufferColorSpace(cs: ColorSpace): ImageBufferColorSpace {
    return cs;
}

function toBufferLayout(layout: ImageLayout): ImageBufferLayout {
    if (layout === 'CHW' || layout === 'nchw') return 'CHW';
    return 'HWC';
}

function toBufferAlphaMode(alpha: boolean, colorSpace: ColorSpace): ImageBufferAlphaMode {
    if (colorSpace === 'rgba') return 'straight';
    return alpha ? 'straight' : 'opaque';
}

export class VisionImage implements Image {
    readonly #buffer: ImageBuffer;
    readonly #meta: {
        layout: ImageLayout;
        dtype: ImageDtype;
        alpha: boolean;
        source?: { kind: string; uri?: string };
    };

    constructor(
        buffer: ImageBuffer,
        meta: {
            layout: ImageLayout;
            dtype: ImageDtype;
            alpha: boolean;
            source?: { kind: string; uri?: string };
        },
    ) {
        this.#buffer = buffer;
        this.#meta = meta;
    }

    get width(): number {
        return this.#buffer.width;
    }

    get height(): number {
        return this.#buffer.height;
    }

    get channels(): number {
        return this.#buffer.channels;
    }

    get colorSpace(): ColorSpace {
        return this.#buffer.colorSpace === 'unknown' ? 'rgb' : (this.#buffer.colorSpace as ColorSpace);
    }

    get layout(): ImageLayout {
        return this.#meta.layout;
    }

    get dtype(): ImageDtype {
        return this.#meta.dtype;
    }

    get alpha(): boolean {
        return this.#meta.alpha;
    }

    get source(): { kind: string; uri?: string } | undefined {
        return this.#meta.source;
    }

    buffer(): ImageBuffer | undefined {
        return this.#buffer;
    }

    toTensor(options?: ImageToTensorOptions): Tensor {
        void options?.layout;
        return this.#buffer.toTensor({
            device: options?.device,
            copy: options?.copy,
            normalize: true,
        });
    }

    dispose(): void {
        this.#buffer.dispose();
    }

    ready(): Promise<void> {
        return this.#buffer.ready();
    }

    static fromPixels(options: FromPixelsOptions): VisionImage {
        const layout = options.layout ?? 'HWC';
        const dtype = options.dtype ?? 'u8';
        const buffer = createImageBufferFromPixels({
            width: options.width,
            height: options.height,
            channels: options.channels,
            dtype: dtype as ImageBufferDtype,
            layout: toBufferLayout(layout),
            colorSpace: toBufferColorSpace(options.colorSpace),
            alphaMode: toBufferAlphaMode(options.alpha ?? false, options.colorSpace),
            data: options.data,
        });
        return new VisionImage(buffer, {
            layout,
            dtype,
            alpha: options.alpha ?? options.colorSpace === 'rgba',
            source: options.source,
        });
    }

    static async fromDecode(bytes: Uint8Array, options?: FromDecodeOptions): Promise<VisionImage> {
        const buffer = await decodeImageBuffer(bytes, { format: options?.format });
        const alpha = buffer.alphaMode !== 'none' && buffer.alphaMode !== 'opaque';
        return new VisionImage(buffer, {
            layout: 'HWC',
            dtype: buffer.dtype as ImageDtype,
            alpha,
            source: options?.source ?? { kind: 'decode' },
        });
    }
}

export function fromPixels(options: FromPixelsOptions): VisionImage {
    return VisionImage.fromPixels(options);
}

export async function fromDecode(bytes: Uint8Array, options?: FromDecodeOptions): Promise<VisionImage> {
    return VisionImage.fromDecode(bytes, options);
}
