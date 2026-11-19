import { Buffer } from 'node:buffer';
import { wrapNative } from './errors.js';
import { loadNative } from './native.js';
import type { NativeGraphMlp, NativeParameterInfo, SafetensorBufferEntry } from './native-types.js';

/** Options for {@link GraphMlp}. */
export interface GraphMlpOptions {
    /** Layer widths; the last value is the class count. */
    sizes: readonly number[];
    /** Fixed rows per batch. Default: 64. */
    batchSize?: number;
    /** Deterministic init seed. Default: 0. */
    seed?: number;
    /** Adam learning rate. Default: 0.01. */
    learningRate?: number;
}

/** Classification metrics from {@link GraphMlp.evaluate}. */
export interface ClassificationReport {
    /** Mean softmax cross entropy. */
    meanLoss: number;
    /** Row-wise argmax hit rate over full batches. */
    accuracy: number;
    /** Number of full batches evaluated. */
    batches: number;
}

/** Parameter summary. */
export interface ParameterInfo {
    name: string;
    shape: readonly number[];
}

function asBuffer(values: Float32Array): Buffer {
    return Buffer.from(values.buffer, values.byteOffset, values.byteLength);
}

function asF32(buffer: Buffer): Float32Array {
    const copy = new Uint8Array(buffer.byteLength);
    copy.set(buffer);
    return new Float32Array(copy.buffer);
}

/**
 * Fixed-batch MLP trainer backed by the Boost training graph and Adam.
 *
 * The graph, reverse pass and optimizer live in Rust (`dl-graph` / `dl-train`); this class only
 * moves f32 buffers across the napi boundary. Every call consumes whole batches — trailing rows
 * that cannot fill a batch are ignored, so callers slicing datasets should drop them explicitly.
 */
export class GraphMlp {
    readonly #handle: NativeGraphMlp;

    private constructor(handle: NativeGraphMlp) {
        this.#handle = handle;
    }

    /** Build the training graph and optimizer. */
    static create(options: GraphMlpOptions): GraphMlp {
        return wrapNative(
            () =>
                new GraphMlp(
                    new (loadNative().GraphMlp)(
                        [...options.sizes],
                        options.batchSize ?? null,
                        options.seed ?? null,
                        options.learningRate ?? null,
                    ),
                ),
        );
    }

    get sizes(): readonly number[] {
        return this.#handle.sizes;
    }

    get batchSize(): number {
        return this.#handle.batchSize;
    }

    /** Run one epoch over whole batches; returns the mean cross entropy. */
    fitEpoch(inputs: Float32Array, targets: Float32Array): number {
        return wrapNative(() => this.#handle.fitEpoch(asBuffer(inputs), asBuffer(targets)));
    }

    /** Evaluate without updating parameters. */
    evaluate(inputs: Float32Array, targets: Float32Array): ClassificationReport {
        const report = wrapNative(() => this.#handle.evaluate(asBuffer(inputs), asBuffer(targets)));
        return { meanLoss: report.meanLoss, accuracy: report.accuracy, batches: report.batches };
    }

    /** Logits for whole batches, row-major `[rows, classes]`. */
    predict(inputs: Float32Array): Float32Array {
        return asF32(wrapNative(() => this.#handle.predict(asBuffer(inputs))));
    }

    /** Parameter names and shapes. */
    parameters(): ParameterInfo[] {
        return wrapNative(() => this.#handle.parameters()).map((entry: NativeParameterInfo) => ({
            name: entry.name,
            shape: entry.shape,
        }));
    }

    /** Read one parameter as f32 values. */
    parameter(name: string): Float32Array {
        return asF32(wrapNative(() => this.#handle.parameter(name)));
    }

    /** Overwrite one parameter; the shape must match the graph declaration. */
    setParameter(name: string, values: Float32Array): void {
        wrapNative(() => this.#handle.setParameter(name, asBuffer(values)));
    }

    /** Snapshot every parameter as safetensors-ready entries (encode with `encodeSafetensors`). */
    stateEntries(): SafetensorBufferEntry[] {
        return this.parameters().map((entry) => ({
            name: entry.name,
            shape: [...entry.shape],
            data: Buffer.from(this.parameter(entry.name).buffer.slice(0)),
        }));
    }

    /** Restore from decoded safetensors entries (unknown names are rejected by the graph). */
    loadEntries(entries: readonly SafetensorBufferEntry[]): void {
        for (const entry of entries) {
            wrapNative(() => this.#handle.setParameter(entry.name, Buffer.from(entry.data)));
        }
    }
}
