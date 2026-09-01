import { decodeSafetensors as nativeDecode, encodeSafetensors as nativeEncode, type SafetensorBufferEntry } from '@dxo/core';

export const STATE_FORMAT = 'safetensors' as const;
export type DxoStateFormat = typeof STATE_FORMAT;
export type Dtype = 'f32';

export interface TensorStateSlice {
    shape: number[];
    data: number[];
}

/** Buffer-backed slice for Rust safetensors encode (checkpoint hot path). */
export interface StateBufferSlice {
    shape: number[];
    data: Buffer;
}

export type State = Record<string, TensorStateSlice>;
export type StateBuffers = Record<string, StateBufferSlice>;

function numel(shape: number[]): number {
    return shape.reduce((n, d) => n * d, 1);
}

function validateState(state: State): void {
    if (!state || typeof state !== 'object' || Array.isArray(state)) throw new Error('state must be a named tensor map');
    if (Object.keys(state).length === 0) throw new Error('state must contain at least one tensor');
    for (const [name, slice] of Object.entries(state)) {
        if (!slice || !Array.isArray(slice.shape) || !Array.isArray(slice.data)) throw new Error(`invalid tensor state '${name}'`);
        const expected = numel(slice.shape);
        if (slice.data.length !== expected) throw new Error(`tensor '${name}' length ${slice.data.length} != product(shape)=${expected}`);
    }
}

function validateStateBuffers(buffers: StateBuffers): void {
    if (!buffers || typeof buffers !== 'object' || Array.isArray(buffers)) throw new Error('state buffers must be a named tensor map');
    if (Object.keys(buffers).length === 0) throw new Error('state buffers must contain at least one tensor');
    for (const [name, slice] of Object.entries(buffers)) {
        if (!slice || !Array.isArray(slice.shape) || !Buffer.isBuffer(slice.data)) {
            throw new Error(`invalid buffer state '${name}'`);
        }
        const expected = numel(slice.shape) * 4;
        if (slice.data.length !== expected) {
            throw new Error(`tensor '${name}' byte length ${slice.data.length} != product(shape)*4=${expected}`);
        }
    }
}

function floatDataToBuffer(data: number[]): Buffer {
    const buf = Buffer.allocUnsafe(data.length * 4);
    for (let i = 0; i < data.length; i++) {
        buf.writeFloatLE(data[i]!, i * 4);
    }
    return buf;
}

function bufferToFloatArray(data: Buffer): number[] {
    const out: number[] = [];
    for (let i = 0; i < data.length; i += 4) {
        out.push(data.readFloatLE(i));
    }
    return out;
}

function entriesFromState(state: State): SafetensorBufferEntry[] {
    return Object.keys(state)
        .sort()
        .map((name) => {
            const slice = state[name]!;
            return { name, shape: slice.shape, data: floatDataToBuffer(slice.data) };
        });
}

function entriesFromBuffers(buffers: StateBuffers): SafetensorBufferEntry[] {
    return Object.keys(buffers)
        .sort()
        .map((name) => {
            const slice = buffers[name]!;
            return { name, shape: slice.shape, data: slice.data };
        });
}

/** Encode named f32 tensor buffers + optional metadata (Rust codec). */
export function encodeStateBuffers(buffers: StateBuffers, metadata?: Record<string, string>): Uint8Array {
    validateStateBuffers(buffers);
    const metadataJson = metadata && Object.keys(metadata).length ? JSON.stringify(metadata) : undefined;
    return nativeEncode(entriesFromBuffers(buffers), metadataJson);
}

/** Encode a full checkpoint document (model buffers + optional metadata). */
export function encodeCheckpoint(options: {
    model: StateBuffers;
    optimizer?: SafetensorBufferEntry[];
    metadata?: Record<string, string>;
}): Uint8Array {
    validateStateBuffers(options.model);
    const entries = [...entriesFromBuffers(options.model)];
    if (options.optimizer?.length) {
        entries.push(...options.optimizer);
    }
    const metadataJson =
        options.metadata && Object.keys(options.metadata).length ? JSON.stringify(options.metadata) : undefined;
    return nativeEncode(entries, metadataJson);
}

export function encodeState(state: State, options: { format?: DxoStateFormat; metadata?: Record<string, string> } = {}): Uint8Array {
    if (options.format !== undefined && options.format !== STATE_FORMAT) throw new Error(`unsupported DXO state format '${options.format}'`);
    validateState(state);
    const metadataJson = options.metadata && Object.keys(options.metadata).length ? JSON.stringify(options.metadata) : undefined;
    return nativeEncode(entriesFromState(state), metadataJson);
}

export function decodeState(bytes: Uint8Array, options: { format?: DxoStateFormat } = {}): State {
    if (options.format !== undefined && options.format !== STATE_FORMAT) throw new Error(`unsupported DXO state format '${options.format}'`);
    const decoded = nativeDecode(bytes);
    const out: State = {};
    for (const entry of decoded.tensors) {
        if (entry.name.startsWith('optimizer.')) continue;
        out[entry.name] = { shape: entry.shape, data: bufferToFloatArray(entry.data) };
    }
    return out;
}

/** Decode to buffer slices (restore hot path — use with `tensorFromF32Buffer`). */
export function decodeStateBuffers(bytes: Uint8Array): { tensors: StateBuffers; metadata: Record<string, string> } {
    const decoded = nativeDecode(bytes);
    const tensors: StateBuffers = {};
    for (const entry of decoded.tensors) {
        if (entry.name.startsWith('optimizer.')) continue;
        tensors[entry.name] = { shape: entry.shape, data: entry.data };
    }
    let metadata: Record<string, string> = {};
    try {
        metadata = JSON.parse(decoded.metadataJson) as Record<string, string>;
    } catch {
        metadata = {};
    }
    return { tensors, metadata };
}

/** Decode including optimizer.* tensor keys as raw buffer entries. */
export function decodeCheckpoint(bytes: Uint8Array): {
    model: StateBuffers;
    optimizer: SafetensorBufferEntry[];
    metadata: Record<string, string>;
} {
    const decoded = nativeDecode(bytes);
    const model: StateBuffers = {};
    const optimizer: SafetensorBufferEntry[] = [];
    for (const entry of decoded.tensors) {
        if (entry.name.startsWith('optimizer.')) {
            optimizer.push(entry);
        } else {
            model[entry.name] = { shape: entry.shape, data: entry.data };
        }
    }
    let metadata: Record<string, string> = {};
    try {
        metadata = JSON.parse(decoded.metadataJson) as Record<string, string>;
    } catch {
        metadata = {};
    }
    return { model, optimizer, metadata };
}

export { decodeSafetensors, encodeSafetensors, type SafetensorBufferEntry } from '@dxo/core';
export type SafetensorSlice = TensorStateSlice;
