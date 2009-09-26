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

    add(other: Tensor): Tensor {
        return new Tensor(this.#handle.add(other.#handle));
    }

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

    sum(): Tensor {
        return new Tensor(this.#handle.sum());
    }

    detach(): Tensor {
        return new Tensor(this.#handle.detach());
    }

    zeroGrad(): void {
        this.#handle.zeroGrad();
    }

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
        throw new Error('only cpu device is available in this slice');
    }
}

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

export function backend(): string {
    return loadNative().backend();
}

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
