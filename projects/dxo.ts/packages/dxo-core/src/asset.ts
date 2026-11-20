import { wrapNative } from './errors.js';
import { loadNative } from './native.js';
import type { NativeDxoAsset } from './native-types.js';

/**
 * Opened `.dxo` asset container (main file + optional numbered volumes).
 *
 * Parsing, digest verification and chunk reads all happen in Rust (`data-binary-dxo`);
 * this class only moves buffers across the napi boundary. No `dxo-build` CLI is spawned.
 */
export class DxoAsset {
    readonly #handle: NativeDxoAsset;

    constructor(path: string) {
        this.#handle = wrapNative(() => new (loadNative().DxoAsset)(path));
    }

    /** Index metadata as a JSON string (kind, semantic digest, format, tensors, splits, weights, provenance). */
    metadataJson(): string {
        return wrapNative(() => this.#handle.metadataJson());
    }

    /** Parsed index metadata document. */
    metadata(): Record<string, unknown> {
        return JSON.parse(this.metadataJson()) as Record<string, unknown>;
    }

    /** Full acceptance: every volume shell, every chunk digest, every tensor logical digest. */
    verifyAll(): void {
        wrapNative(() => this.#handle.verifyAll());
    }

    /** Complete logical bytes of one tensor. */
    readTensor(key: string): Buffer {
        return wrapNative(() => this.#handle.readTensor(key));
    }

    /** Contiguous sample range of one dataset field (row-major packed bytes). */
    readSamples(split: string, field: string, startSample: number, count: number): Buffer {
        return wrapNative(() => this.#handle.readSamples(split, field, startSample, count));
    }
}
