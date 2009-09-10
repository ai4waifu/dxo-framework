import { loadNative } from './native.js';

/** @returns {string} */
export function version() {
    const native = loadNative();
    return native.version();
}

/** Empty tensor shell (M0); backed by Rust `dxo_napi::Tensor`. */
export class Tensor {
    constructor() {
        const native = loadNative();
        new native.Tensor();
    }
}
