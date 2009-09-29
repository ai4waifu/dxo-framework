import { Buffer } from 'node:buffer';
import { loadNative } from './native.js';
import type { NativeTensor } from './native-types.js';

export type Device = 'cpu' | 'cuda' | 'metal';

export type TensorData = number[] | Float32Array;

export interface TensorOptions {
    device?: Device;
    requiresGrad?: boolean;
}

export type { NativeAddon, NativeTensor } from './native-types.js';

/**
 * Dense float32 tensor (CPU preview).
 *
 * Contract (G3 / 0.0.4):
 * - Ops are eager; Tape records when `requiresGrad` and grad mode is enabled.
 * - `backward()` requires a scalar (`numel === 1`, usually shape `[1]`).
 * - `grad` is a row-major copy or `undefined`.
 * - Only `device: 'cpu'` is available in this slice.
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
        return 'cpu';
    }

    get requiresGrad(): boolean {
        return this.#handle.requiresGrad;
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

    /** `this - other` via add/mul (broadcast-aware). */
    sub(other: Tensor): Tensor {
        return this.add(other.mul(tensor([-1], [1])));
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
        const n = this.numel();
        if (n === 0) throw new Error('mean of empty tensor');
        return this.sum().mul(tensor([1 / n], [1]));
    }

    /** Values only — drops requiresGrad and tape edges. */
    detach(): Tensor {
        return new Tensor(this.#handle.detach());
    }

    /** Clear accumulated gradient on this leaf/intermediate slot. */
    zeroGrad(): void {
        this.#handle.zeroGrad();
    }

    /** Reverse-mode from a scalar output. */
    backward(): void {
        this.#handle.backward();
    }

    toFloat32Array(): Float32Array {
        return Float32Array.from(this.#handle.toArray());
    }

    toArray(): number[] {
        return this.#handle.toArray();
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

export function version(): string {
    return loadNative().version();
}

/** Engine backend label (e.g. `titan-cpu`). */
export function backend(): string {
    return loadNative().backend();
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
