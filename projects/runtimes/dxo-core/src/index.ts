import { loadNative } from './native.js';
import type { NativeTensor } from './native-types.js';

export type Device = 'cpu' | 'cuda' | 'metal';

export type TensorData = number[] | Float32Array;

export interface TensorOptions {
    device?: Device;
    requiresGrad?: boolean;
}

export type { NativeTensor, NativeAddon } from './native-types.js';

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
        return false;
    }

    add(other: Tensor): Tensor {
        return new Tensor(this.#handle.add(other.#handle));
    }

    matmul(other: Tensor): Tensor {
        return new Tensor(this.#handle.matmul(other.#handle));
    }

    reshape(shape: readonly number[]): Tensor {
        return new Tensor(this.#handle.reshape([...shape]));
    }

    relu(): Tensor {
        return new Tensor(this.#handle.relu());
    }

    toFloat32Array(): Float32Array {
        return Float32Array.from(this.#handle.toArray());
    }

    toArray(): number[] {
        return this.#handle.toArray();
    }
}

export function tensor(data: TensorData, shape: readonly number[], options: TensorOptions = {}): Tensor {
    if (options.device && options.device !== 'cpu') {
        throw new Error('only cpu device is available in this slice');
    }
    if (options.requiresGrad) {
        throw new Error('requiresGrad is not available before autograd (M2)');
    }
    const flat = flattenData(data);
    const native = loadNative();
    return new Tensor(native.tensor(flat, [...shape]));
}

export function zeros(shape: readonly number[], options: Pick<TensorOptions, 'device'> = {}): Tensor {
    if (options.device && options.device !== 'cpu') {
        throw new Error('only cpu device is available in this slice');
    }
    const native = loadNative();
    return new Tensor(native.zeros([...shape]));
}

export function version(): string {
    return loadNative().version();
}

export function withoutGrad<T>(run: () => T): T {
    return run();
}

function flattenData(data: TensorData): number[] {
    if (data instanceof Float32Array) {
        return [...data];
    }
    return data.flat(Infinity) as number[];
}
