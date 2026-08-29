import { decodeSafetensors, encodeSafetensors } from './safetensors.js';
export const STATE_FORMAT = 'safetensors' as const;
export type DxoStateFormat = typeof STATE_FORMAT;
export type Dtype = 'f32';
export interface TensorStateSlice { shape: number[]; data: number[]; }
export type State = Record<string, TensorStateSlice>;
function numel(shape: number[]): number { return shape.reduce((n, d) => n * d, 1); }
function validateState(state: State): void {
    if (!state || typeof state !== 'object' || Array.isArray(state)) throw new Error('state must be a named tensor map');
    if (Object.keys(state).length === 0) throw new Error('state must contain at least one tensor');
    for (const [name, slice] of Object.entries(state)) {
        if (!slice || !Array.isArray(slice.shape) || !Array.isArray(slice.data)) throw new Error(`invalid tensor state '${name}'`);
        const expected = numel(slice.shape);
        if (slice.data.length !== expected) throw new Error(`tensor '${name}' length ${slice.data.length} != product(shape)=${expected}`);
    }
}
export function encodeState(state: State, options: { format?: DxoStateFormat } = {}): Uint8Array {
    if (options.format !== undefined && options.format !== STATE_FORMAT) throw new Error(`unsupported DXO state format '${options.format}'`);
    validateState(state); return encodeSafetensors(state);
}
export function decodeState(bytes: Uint8Array, options: { format?: DxoStateFormat } = {}): State {
    if (options.format !== undefined && options.format !== STATE_FORMAT) throw new Error(`unsupported DXO state format '${options.format}'`);
    return decodeSafetensors(bytes) as State;
}
export { decodeSafetensors, encodeSafetensors, type SafetensorSlice } from './safetensors.js';
