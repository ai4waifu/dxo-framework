/**
 * Shared typed-buffer primitives (Living `00` / `11`).
 *
 * napi owns storage · view · copy · upload · readback · codec bridge · lifetime.
 * Domain packages own meaning · preprocessing · batching · model/product workflow.
 * Prefer these few primitives over one lifecycle model per domain.
 */

import type { Tensor } from './index.js';

export type HostDtype = 'u8' | 'u16' | 'i32' | 'i64' | 'f16' | 'f32' | 'bf16';

export type BufferToTensorOptions = {
    device?: 'cpu' | 'cuda' | 'metal';
    /** `false` when allowed: zero-copy view; `true`: explicit copy. */
    copy?: boolean;
};

/** Host typed storage with shape / stride — domain-agnostic. */
export interface TypedBuffer {
    readonly dtype: HostDtype;
    readonly shape: readonly number[];
    readonly stride: readonly number[];
    toTensor(options?: BufferToTensorOptions): Tensor;
    dispose(): void;
    ready(): Promise<void>;
}

/** View over TypedBuffer or Tensor storage without owning a second copy policy. */
export interface TensorView {
    readonly buffer: TypedBuffer;
    readonly offset: number;
    readonly shape: readonly number[];
    readonly stride: readonly number[];
    toTensor(options?: BufferToTensorOptions): Tensor;
}

/** Device-resident opaque storage (upload / readback via barriers). */
export interface DeviceBuffer {
    readonly device: 'cpu' | 'cuda' | 'metal';
    readonly nbytes: number;
    readback(options?: { copy?: boolean }): Promise<TypedBuffer>;
    dispose(): void;
    ready(): Promise<void>;
}

/** Memory-mapped or streamed weight / artifact handle. */
export interface MappedFile {
    readonly path?: string;
    readonly nbytes: number;
    slice(offset: number, length: number): TypedBuffer;
    dispose(): void;
}

export interface StreamHandle {
    read(nbytes: number): Promise<TypedBuffer>;
    dispose(): void;
}

/** Opaque codec / decoder bridge (JPEG, audio, video, …) — not product APIs. */
export interface CodecHandle {
    readonly kind: string;
    close(): void;
    ready(): Promise<void>;
}

export function unsupportedTypedBufferApi(name: string): never {
    throw new Error(`@dxo/core ${name} is reserved for the typed-buffer napi surface; not a domain product API`);
}
