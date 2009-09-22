/**
 * Browser / Worker WebGPU runtime facade (M0 stub).
 * Target: TS facade → wasm-bindgen → lite-engine (Rust + wgpu) → WebGPU.
 * Explicit CPU/WASM fallback only; WebGL is not a tensor backend.
 */
export function version(): string {
    return '0.1.0-dxo-lite';
}

/** Placeholder tensor view (M0; async runtime API lands in M6). */
export class Tensor {}
