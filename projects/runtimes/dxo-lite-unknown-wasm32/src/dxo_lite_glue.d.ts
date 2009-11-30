/**
 * Ambient types for wasm-bindgen glue under `lib/` (generated; not edited by hand).
 * Real `.d.ts` is emitted next to `lib/dxo_lite.js` by wasm-pack.
 */
declare module '../lib/dxo_lite.js' {
    export default function init(module_or_path?: { module_or_path?: BufferSource | string | URL } | BufferSource | string | URL): Promise<unknown>;
    export function addF32(a: Float32Array, b: Float32Array): Float32Array;
    export function matmulF32(
        a: Float32Array,
        a_rows: number,
        a_cols: number,
        b: Float32Array,
        b_cols: number,
    ): Float32Array;
    export function version(): string;
    export function isInterimHostF32(): boolean;
}
