/* tslint:disable */
/* eslint-disable */

/**
 * Element-wise f32 add; lengths must match.
 */
export function addF32(a: Float32Array, b: Float32Array): Float32Array;

/**
 * True when this module is the interim host-f32 path (not Titan wgpu yet).
 */
export function isInterimHostF32(): boolean;

/**
 * Row-major f32 matmul for rank-2 tensors: `(m,k) × (k,n) → (m,n)`.
 *
 * `a` length must be `m * k`; `b` length must be `k * n`.
 */
export function matmulF32(a: Float32Array, a_rows: number, a_cols: number, b: Float32Array, b_cols: number): Float32Array;

/**
 * Package / ABI identity (not npm semver).
 */
export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly addF32: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly isInterimHostF32: () => number;
    readonly matmulF32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number, number];
    readonly version: () => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
