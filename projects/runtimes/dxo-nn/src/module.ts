import { Tensor, tensor, zeros } from '@dxo/core';

export interface TensorStateSlice {
    shape: number[];
    data: number[];
}

export interface LinearState {
    weight: TensorStateSlice;
    bias: TensorStateSlice;
}

export abstract class Module {
    abstract forward(x: Tensor): Tensor;

    parameters(): Tensor[] {
        const out: Tensor[] = [];
        for (const value of Object.values(this) as unknown[]) {
            if (value instanceof Tensor) out.push(value);
            if (value instanceof Module) out.push(...value.parameters());
        }
        return out;
    }
}

export function relu(x: Tensor): Tensor {
    return x.relu();
}

/** Fully-connected: `y = relu(x @ weight + bias)`; weight `[inFeatures, outFeatures]`. */
export class Linear extends Module {
    weight: Tensor;
    bias: Tensor;

    constructor(
        readonly inFeatures: number,
        readonly outFeatures: number,
    ) {
        super();
        this.weight = zeros([inFeatures, outFeatures]);
        this.bias = zeros([outFeatures]);
    }

    forward(x: Tensor): Tensor {
        return relu(x.matmul(this.weight).add(this.bias));
    }

    state(): LinearState {
        return {
            weight: { shape: [...this.weight.shape], data: this.weight.toArray() },
            bias: { shape: [...this.bias.shape], data: this.bias.toArray() },
        };
    }

    loadState(saved: LinearState): void {
        this.weight = tensor(saved.weight.data, saved.weight.shape);
        this.bias = tensor(saved.bias.data, saved.bias.shape);
    }
}

export class Sequential extends Module {
    constructor(readonly layers: Module[]) {
        super();
    }

    forward(x: Tensor): Tensor {
        let out = x;
        for (const layer of this.layers) {
            out = layer.forward(out);
        }
        return out;
    }
}
