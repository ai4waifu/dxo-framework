import { randnValues, Tensor, tensor, zeros } from '@dxo/core';

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

    zeroGrad(): void {
        for (const p of this.parameters()) p.zeroGrad();
    }
}

export function relu(x: Tensor): Tensor {
    return x.relu();
}

/** Element-wise ReLU module. */
export class Relu extends Module {
    forward(x: Tensor): Tensor {
        return x.relu();
    }
}

/** Fully-connected affine map: `y = x @ weight + bias` (no activation).
 *
 * Weight layout: `[inFeatures, outFeatures]`. Default leaves use `requiresGrad: true`.
 * After `optimizer.step(parameters())`, call `loadParameters` to install new leaves.
 */
export class Linear extends Module {
    weight: Tensor;
    bias: Tensor;

    constructor(
        readonly inFeatures: number,
        readonly outFeatures: number,
        opts: { requiresGrad?: boolean } = {},
    ) {
        super();
        const rg = opts.requiresGrad ?? true;
        const scale = Math.sqrt(2 / (inFeatures + outFeatures));
        const raw = randnValues([inFeatures, outFeatures]).map((v) => v * scale);
        this.weight = tensor(raw, [inFeatures, outFeatures], { requiresGrad: rg });
        this.bias = zeros([outFeatures], { requiresGrad: rg });
    }

    forward(x: Tensor): Tensor {
        return x.matmul(this.weight).add(this.bias);
    }

    /** Replace parameter leaves after an optimizer step. */
    loadParameters(params: Tensor[]): void {
        if (params.length < 2) throw new Error('Linear expects [weight, bias]');
        this.weight = params[0]!;
        this.bias = params[1]!;
    }

    async state(): Promise<LinearState> {
        return {
            weight: { shape: [...this.weight.shape], data: await this.weight.toArray() },
            bias: { shape: [...this.bias.shape], data: await this.bias.toArray() },
        };
    }

    loadState(saved: LinearState): void {
        this.weight = tensor(saved.weight.data, saved.weight.shape, { requiresGrad: true });
        this.bias = tensor(saved.bias.data, saved.bias.shape, { requiresGrad: true });
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
