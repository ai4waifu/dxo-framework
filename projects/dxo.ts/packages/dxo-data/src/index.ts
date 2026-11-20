import { type Tensor, tensor } from '@dxo/core';

/** One training/eval example with explicit shapes (flat row-major payloads). */
export interface Sample {
    x: number[];
    xShape: number[];
    y?: number[];
    yShape?: number[];
}

/** Mini-batch of tensors; `x` always present, `y` if every sample had a target. */
export interface Batch {
    x: Tensor;
    y?: Tensor;
}

export interface BatchOptions {
    batchSize: number;
    /** Drop a trailing incomplete batch (default false). */
    dropLast?: boolean;
}

/** Wrap an array as a sync iterable dataset. */
export function dataset(samples: readonly Sample[]): Iterable<Sample> {
    return {
        *[Symbol.iterator]() {
            for (const s of samples) yield s;
        },
    };
}

function numel(shape: number[]): number {
    return shape.reduce((n, d) => n * d, 1);
}

function assertSample(s: Sample, index: number): void {
    if (s.x.length !== numel(s.xShape)) {
        throw new Error(`sample[${index}].x length ${s.x.length} != product(xShape)=${numel(s.xShape)}`);
    }
    if (s.y !== undefined) {
        if (!s.yShape) throw new Error(`sample[${index}].yShape required when y is set`);
        if (s.y.length !== numel(s.yShape)) {
            throw new Error(`sample[${index}].y length ${s.y.length} != product(yShape)=${numel(s.yShape)}`);
        }
    }
}

function stackAxis0(flats: number[][], itemShape: number[]): Tensor {
    const n = flats.length;
    const data = flats.flat();
    return tensor(data, [n, ...itemShape]);
}

function takeBatch(buf: Sample[]): Batch {
    const first = buf[0]!;
    const xShape = first.xShape;
    for (let i = 1; i < buf.length; i++) {
        const s = buf[i]!;
        if (s.xShape.length !== xShape.length || s.xShape.some((d, j) => d !== xShape[j])) {
            throw new Error('batch samples must share identical xShape');
        }
    }
    const x = stackAxis0(
        buf.map((s) => s.x),
        xShape,
    );
    const allY = buf.every((s) => s.y !== undefined && s.yShape);
    if (!allY) return { x };
    const yShape = buf[0]!.yShape!;
    for (let i = 1; i < buf.length; i++) {
        const s = buf[i]!;
        if (s.yShape!.length !== yShape.length || s.yShape!.some((d, j) => d !== yShape[j])) {
            throw new Error('batch samples must share identical yShape');
        }
    }
    const y = stackAxis0(
        buf.map((s) => s.y!),
        yShape,
    );
    return { x, y };
}

/** Sync batching over an iterable of samples. */
export function* batch(samples: Iterable<Sample>, options: BatchOptions): Generator<Batch, void, undefined> {
    const { batchSize, dropLast = false } = options;
    if (!(batchSize > 0)) throw new Error('batchSize must be positive');
    const buf: Sample[] = [];
    let index = 0;
    for (const s of samples) {
        assertSample(s, index++);
        buf.push(s);
        if (buf.length === batchSize) {
            yield takeBatch(buf);
            buf.length = 0;
        }
    }
    if (buf.length > 0 && !dropLast) yield takeBatch(buf);
}

/** Async batching over an async iterable of samples. */
export async function* batchAsync(samples: AsyncIterable<Sample>, options: BatchOptions): AsyncGenerator<Batch, void, undefined> {
    const { batchSize, dropLast = false } = options;
    if (!(batchSize > 0)) throw new Error('batchSize must be positive');
    const buf: Sample[] = [];
    let index = 0;
    for await (const s of samples) {
        assertSample(s, index++);
        buf.push(s);
        if (buf.length === batchSize) {
            yield takeBatch(buf);
            buf.length = 0;
        }
    }
    if (buf.length > 0 && !dropLast) yield takeBatch(buf);
}

export type DataLoaderSource = Iterable<Sample> | AsyncIterable<Sample>;

function isAsyncIterable(source: DataLoaderSource): source is AsyncIterable<Sample> {
    return typeof (source as AsyncIterable<Sample>)[Symbol.asyncIterator] === 'function';
}

/**
 * Thin loader: sync sources → sync batches; async sources → async batches.
 * Prefer `batch` / `batchAsync` when you want an explicit return type.
 */
export function dataLoader(source: Iterable<Sample>, options: BatchOptions): Iterable<Batch>;
export function dataLoader(source: AsyncIterable<Sample>, options: BatchOptions): AsyncIterable<Batch>;
export function dataLoader(source: DataLoaderSource, options: BatchOptions): Iterable<Batch> | AsyncIterable<Batch> {
    if (isAsyncIterable(source)) return batchAsync(source, options);
    return batch(source, options);
}

/** @deprecated Prefer `dataLoader` / `batch` — kept as a recognizable name. */
export const DataLoader = dataLoader;
