import { Buffer } from 'node:buffer';
import { loadNative } from './native.js';
import type { NativeTensor } from './native-types.js';

export type Device = 'cpu' | 'cuda' | 'metal';

export type DType = 'f32' | 'f16' | 'bf16' | 'i64' | 'bool';

export type TensorData = number[] | Float32Array;

export interface TensorOptions {
    device?: Device;
    requiresGrad?: boolean;
}

export type { NativeAddon, NativeTensor } from './native-types.js';

/**
 * Dense float32 tensor (CPU preview + optional CUDA matmul spike).
 *
 * Contract (G3 / 0.0.4+, G4 / 0.0.6 CUDA):
 * - Ops are eager; Tape records when `requiresGrad` and grad mode is enabled.
 * - `backward()` requires a scalar (`numel === 1`, usually shape `[1]`).
 * - `grad` is a row-major copy or `undefined`.
 * - Factory helpers create CPU tensors; use `.to('cuda')` on detached tensors when CUDA is available.
 */
export class Tensor {
    readonly #handle: NativeTensor;

    constructor(handle: NativeTensor) {
        this.#handle = handle;
    }

    get shape(): readonly number[] {
        return this.#handle.shape;
    }

    get device(): Device {
        const d = this.#handle.device;
        if (d === 'cpu' || d === 'cuda') return d;
        throw new Error(`unexpected native device tag: ${d}`);
    }

    get requiresGrad(): boolean {
        return this.#handle.requiresGrad;
    }

    get dtype(): DType {
        const d = this.#handle.dtype;
        if (d === 'f32' || d === 'f16' || d === 'bf16' || d === 'i64' || d === 'bool') return d;
        return 'f32';
    }

    /** Native handle for inter-op calls within `@dxo/core`. */
    get nativeHandle(): NativeTensor {
        return this.#handle;
    }

    /** Accumulated gradient (row-major), or `undefined` if absent. */
    get grad(): number[] | undefined {
        return this.#handle.grad ?? undefined;
    }

    /** Number of elements (shape product). */
    numel(): number {
        return this.shape.reduce((n, d) => n * d, 1);
    }

    add(other: Tensor): Tensor {
        return new Tensor(this.#handle.add(other.#handle));
    }

    /** `this - other` (broadcast-aware). */
    sub(other: Tensor): Tensor {
        return new Tensor(this.#handle.sub(other.#handle));
    }

    neg(): Tensor {
        return new Tensor(this.#handle.neg());
    }

    div(other: Tensor): Tensor {
        return new Tensor(this.#handle.div(other.#handle));
    }

    mul(other: Tensor): Tensor {
        return new Tensor(this.#handle.mul(other.#handle));
    }

    matmul(other: Tensor): Tensor {
        return new Tensor(this.#handle.matmul(other.#handle));
    }

    reshape(shape: readonly number[]): Tensor {
        return new Tensor(this.#handle.reshape([...shape]));
    }

    transpose(): Tensor {
        return new Tensor(this.#handle.transpose());
    }

    relu(): Tensor {
        return new Tensor(this.#handle.relu());
    }

    /** Sum all elements → scalar tensor of shape `[1]`. */
    sum(): Tensor {
        return new Tensor(this.#handle.sum());
    }

    /** Mean of all elements → scalar tensor of shape `[1]`. */
    mean(): Tensor {
        return new Tensor(this.#handle.mean());
    }

    max(): Tensor {
        return new Tensor(this.#handle.maxAll());
    }

    softmax(): Tensor {
        return new Tensor(this.#handle.softmax());
    }

    logSoftmax(): Tensor {
        return new Tensor(this.#handle.logSoftmax());
    }

    narrow(dim: number, start: number, len: number): Tensor {
        return new Tensor(this.#handle.narrow(dim, start, len));
    }

    conv2d(weight: Tensor, bias: Tensor | null, stride: number, padding: number): Tensor {
        return new Tensor(this.#handle.conv2d(weight.nativeHandle, bias ? bias.nativeHandle : null, stride, padding));
    }

    maxPool2d(kernel: number, stride: number, padding: number): Tensor {
        return new Tensor(this.#handle.maxPool2d(kernel, stride, padding));
    }

    batchNorm2d(gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor {
        return new Tensor(this.#handle.batchNorm2d(gamma.nativeHandle, beta.nativeHandle, eps));
    }

    /** LayerNorm over the last dimension. */
    layerNorm(weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
        return new Tensor(this.#handle.layerNorm(weight.nativeHandle, bias.nativeHandle, eps));
    }

    /** Batch matmul `[B,M,K] @ [B,K,N]`. */
    bmm(other: Tensor): Tensor {
        return new Tensor(this.#handle.bmm(other.nativeHandle));
    }

    /** Swap the last two axes. */
    transposeLast(): Tensor {
        return new Tensor(this.#handle.transposeLast());
    }

    /** Swap two axes. */
    transposeDims(dim0: number, dim1: number): Tensor {
        return new Tensor(this.#handle.transposeDims(dim0, dim1));
    }

    /** Scaled dot-product attention; `this`/`k`/`v` are `[B,H,T,D]`. */
    scaledDotProductAttention(k: Tensor, v: Tensor, causal = false): Tensor {
        return new Tensor(this.#handle.scaledDotProductAttention(k.nativeHandle, v.nativeHandle, causal));
    }

    /** Retag logical dtype without changing host f32 payload. */
    castDtype(dtype: DType): Tensor {
        return new Tensor(this.#handle.castDtype(dtype));
    }

    /** Values only — drops requiresGrad and tape edges. */
    detach(): Tensor {
        return new Tensor(this.#handle.detach());
    }

    /** Move to `cpu` or `cuda` (CUDA requires detached tensor in this preview). */
    to(device: 'cpu' | 'cuda'): Tensor {
        return new Tensor(this.#handle.to(device));
    }

    /** Clear accumulated gradient on this leaf/intermediate slot. */
    zeroGrad(): void {
        this.#handle.zeroGrad();
    }

    /** Reverse-mode from a scalar output. */
    backward(): void {
        this.#handle.backward();
    }

    /** CPU preview: resolves immediately once host values are available. */
    async ready(): Promise<void> {}

    toFloat32Array(): Promise<Float32Array> {
        return Promise.resolve(Float32Array.from(this.#handle.toArray()));
    }

    async toArray(): Promise<number[]> {
        return this.#handle.toArray();
    }

    async item(): Promise<number> {
        if (this.numel() !== 1) {
            throw new Error(`item() requires a scalar tensor, got shape [${this.shape.join(',')}]`);
        }
        const values = await this.toArray();
        return values[0]!;
    }
}

function assertCpu(options: Pick<TensorOptions, 'device'>): void {
    if (options.device && options.device !== 'cpu') {
        throw new Error(`device '${options.device}' is not available in this preview (cpu only)`);
    }
}

/** Create a CPU tensor from flat data + shape. */
export function tensor(data: TensorData, shape: readonly number[], options: TensorOptions = {}): Tensor {
    assertCpu(options);
    const native = loadNative();
    const rg = options.requiresGrad ?? false;
    if (data instanceof Float32Array) {
        const buf = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
        return new Tensor(native.tensorF32(buf, [...shape], rg));
    }
    const flat = flattenData(data);
    return new Tensor(native.tensor(flat, [...shape], rg));
}

export function zeros(shape: readonly number[], options: TensorOptions = {}): Tensor {
    assertCpu(options);
    const native = loadNative();
    return new Tensor(native.zeros([...shape], options.requiresGrad ?? false));
}

export function ones(shape: readonly number[], options: TensorOptions = {}): Tensor {
    assertCpu(options);
    const native = loadNative();
    return new Tensor(native.ones([...shape], options.requiresGrad ?? false));
}

export function randn(shape: readonly number[], options: TensorOptions = {}): Tensor {
    assertCpu(options);
    const native = loadNative();
    return new Tensor(native.randn([...shape], options.requiresGrad ?? false));
}

/** Gather embedding rows: `weight` `[vocab, dim]`, `indices` integer ids. */
export function embedding(weight: Tensor, indices: Tensor): Tensor {
    return new Tensor(loadNative().embedding(weight.nativeHandle, indices.nativeHandle));
}

/** Host draw for sync init paths (e.g. Linear weight init). */
export function randnValues(shape: readonly number[]): number[] {
    return loadNative()
        .randn([...shape], false)
        .toArray();
}

export function version(): string {
    return loadNative().version();
}

/** Engine backend label (e.g. `titan-cpu`). */
export function backend(): string {
    return loadNative().backend();
}

/** Whether Titan CUDA Driver API is available on this machine. */
export function cudaAvailable(): boolean {
    return loadNative().cudaAvailable();
}

/** Probe Titan HAL `wait_event` via the CPU session (does not use JS await as a substitute). */
export function probeTitanEventDep(): void {
    loadNative().probeTitanEventDep();
}

/** Whether the current thread records autograd ops. */
export function isGradEnabled(): boolean {
    return loadNative().isGradEnabled();
}

/**
 * Run `fn` with tape recording disabled, then restore the previous flag.
 * Uses a thread-local scope stack (not a permanent process global).
 */
export function withoutGrad<T>(run: () => T): T {
    const native = loadNative();
    const prev = native.setGradEnabled(false);
    try {
        return run();
    } finally {
        native.setGradEnabled(prev);
    }
}

function flattenData(data: number[]): number[] {
    return data.flat(Infinity) as number[];
}
