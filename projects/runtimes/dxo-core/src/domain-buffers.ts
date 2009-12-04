/**
 * Domain-shaped typed buffers on shared primitives (Living `11`).
 * Contracts only — native bindings land per gate; no product workflows here.
 */

import type { Tensor } from './index.js';
import type {
    BufferToTensorOptions,
    DeviceBuffer,
    HostDtype,
    MappedFile,
    TypedBuffer,
} from './typed-buffer.js';

export type TokenBufferDtype = Extract<HostDtype, 'i32' | 'i64'>;

/** Token id / mask / position carrier — Tokenizer rules stay in `@dxo/llm`. */
export interface TokenBuffer extends TypedBuffer {
    readonly dtype: TokenBufferDtype;
    readonly length: number;
}

/** Sample-rate aware audio host buffer — Audio/Spectrogram semantics stay in TS. */
export interface AudioBuffer extends TypedBuffer {
    readonly sampleRate: number;
    readonly channels: number;
    readonly interleaved: boolean;
    readonly frameCount: number;
}

/** Opaque video frame handle — VideoClip / sampling stay in vision or a future video package. */
export interface VideoFrameBuffer {
    readonly timestampUs: number;
    readonly durationUs?: number;
    readonly width: number;
    readonly height: number;
    readonly pixelFormat: string;
    readonly hardwareDecoderOwned: boolean;
    pixels(): TypedBuffer | undefined;
    dispose(): void;
    ready(): Promise<void>;
}

/** Ragged / packed batch: values + offsets + lengths (+ optional mask). */
export interface BatchBuffer {
    readonly values: TypedBuffer;
    readonly offsets: TypedBuffer;
    readonly lengths: TypedBuffer;
    readonly mask?: TypedBuffer;
    toTensor(options?: BufferToTensorOptions): Tensor;
    dispose(): void;
}

export interface IndexBuffer extends TypedBuffer {
    readonly dtype: TokenBufferDtype;
}

export interface SparseBuffer {
    readonly format: 'coo' | 'csr';
    readonly indices: TypedBuffer;
    readonly values: TypedBuffer;
    readonly shape: readonly number[];
    dispose(): void;
}

/** Device-resident or host generation output — explicit readback only. */
export interface OutputBuffer {
    readonly device: 'cpu' | 'cuda' | 'metal';
    readonly handle: DeviceBuffer;
    readback(options?: { copy?: boolean; maxSide?: number }): Promise<TypedBuffer>;
    dispose(): void;
    ready(): Promise<void>;
}

/** Checkpoint / safetensors weight block — state/manifest stay in `@dxo/serialize`. */
export interface StateBuffer {
    readonly name?: string;
    readonly mapped?: MappedFile;
    readonly host?: TypedBuffer;
    readonly device?: DeviceBuffer;
    dispose(): void;
    ready(): Promise<void>;
}
