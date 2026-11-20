import type { Tensor } from '@dxo/core';
import type { TensorStateSlice } from '@dxo/nn';
import { BatchNormalization2d, Convolution2d, NeuralNetwork, ReLU } from '@dxo/nn';

/**
 * DXO-native BasicBlock.
 * Canonical keys (Living `02`): `{prefix}.convolution_{1|2}` / `batch_normalization_{1|2}`
 * and optional `{prefix}.downsample.convolution|batch_normalization` (singleton downsample).
 */
export class BasicBlock extends NeuralNetwork {
    protected semanticName(): string {
        return 'basic_block';
    }
    readonly convolution_1: Convolution2d;
    readonly batch_normalization_1: BatchNormalization2d;
    readonly convolution_2: Convolution2d;
    readonly batch_normalization_2: BatchNormalization2d;
    readonly relu: ReLU;
    readonly downConvolution: Convolution2d | null;
    readonly downBatchNormalization: BatchNormalization2d | null;
    constructor(inChannels: number, outChannels: number, opts: { stride?: number; requiresGrad?: boolean } = {}) {
        super();
        const stride = opts.stride ?? 1;
        const rg = opts.requiresGrad ?? true;
        this.convolution_1 = new Convolution2d(inChannels, outChannels, 3, { stride, padding: 1, bias: false, requiresGrad: rg });
        this.batch_normalization_1 = new BatchNormalization2d(outChannels, { requiresGrad: rg });
        this.convolution_2 = new Convolution2d(outChannels, outChannels, 3, { padding: 1, bias: false, requiresGrad: rg });
        this.batch_normalization_2 = new BatchNormalization2d(outChannels, { requiresGrad: rg });
        this.relu = new ReLU();
        this.registerChild(this.convolution_1, { name: 'convolution_1', mode: 'repeatable' });
        this.registerChild(this.batch_normalization_1, { name: 'batch_normalization_1', mode: 'repeatable' });
        this.registerChild(this.convolution_2, { name: 'convolution_2', mode: 'repeatable' });
        this.registerChild(this.batch_normalization_2, { name: 'batch_normalization_2', mode: 'repeatable' });
        this.registerChild(this.relu, { name: 'relu' });
        if (stride !== 1 || inChannels !== outChannels) {
            this.downConvolution = new Convolution2d(inChannels, outChannels, 1, { stride, bias: false, requiresGrad: rg });
            this.downBatchNormalization = new BatchNormalization2d(outChannels, { requiresGrad: rg });
            const downsample = new Downsample(this.downConvolution, this.downBatchNormalization);
            this.registerChild(downsample, { name: 'downsample' });
        } else {
            this.downConvolution = null;
            this.downBatchNormalization = null;
        }
    }

    /** @deprecated Use downConvolution; kept for callers checking downsample presence. */
    get downConvolutionLayer(): Convolution2d | null {
        return this.downConvolution;
    }

    forward(x: Tensor): Tensor {
        let h = this.relu.forward(this.batch_normalization_1.forward(this.convolution_1.forward(x)));
        h = this.batch_normalization_2.forward(this.convolution_2.forward(h));
        let skip = x;
        if (this.downConvolution && this.downBatchNormalization) {
            skip = this.downBatchNormalization.forward(this.downConvolution.forward(x));
        }
        return this.relu.forward(h.add(skip));
    }

    async state(): Promise<Record<string, TensorStateSlice>> {
        const p = this.canonicalName();
        const c1 = await this.convolution_1.state();
        const b1 = await this.batch_normalization_1.state();
        const c2 = await this.convolution_2.state();
        const b2 = await this.batch_normalization_2.state();
        const out: Record<string, TensorStateSlice> = {
            [`${p}.convolution_1.weight`]: c1.weight,
            [`${p}.batch_normalization_1.weight`]: b1.weight!,
            [`${p}.batch_normalization_1.bias`]: b1.bias!,
            [`${p}.convolution_2.weight`]: c2.weight,
            [`${p}.batch_normalization_2.weight`]: b2.weight!,
            [`${p}.batch_normalization_2.bias`]: b2.bias!,
        };
        if (this.downConvolution && this.downBatchNormalization) {
            const dc = await this.downConvolution.state();
            const db = await this.downBatchNormalization.state();
            out[`${p}.downsample.convolution.weight`] = dc.weight;
            out[`${p}.downsample.batch_normalization.weight`] = db.weight!;
            out[`${p}.downsample.batch_normalization.bias`] = db.bias!;
        }
        return out;
    }

    loadState(saved: Record<string, TensorStateSlice>, opts: { requiresGrad?: boolean } = {}): void {
        const p = this.canonicalName();
        const rg = opts.requiresGrad ?? true;
        const need = (k: string) => {
            const s = saved[k];
            if (!s) throw new Error(`BasicBlock.loadState: missing '${k}'`);
            return s;
        };
        this.convolution_1.loadState({ weight: need(`${p}.convolution_1.weight`) }, { requiresGrad: rg });
        this.batch_normalization_1.loadState(
            { weight: need(`${p}.batch_normalization_1.weight`), bias: need(`${p}.batch_normalization_1.bias`) },
            { requiresGrad: rg },
        );
        this.convolution_2.loadState({ weight: need(`${p}.convolution_2.weight`) }, { requiresGrad: rg });
        this.batch_normalization_2.loadState(
            { weight: need(`${p}.batch_normalization_2.weight`), bias: need(`${p}.batch_normalization_2.bias`) },
            { requiresGrad: rg },
        );
        if (this.downConvolution && this.downBatchNormalization) {
            this.downConvolution.loadState(
                {
                    weight: need(`${p}.downsample.convolution.weight`),
                },
                { requiresGrad: rg },
            );
            this.downBatchNormalization.loadState(
                {
                    weight: need(`${p}.downsample.batch_normalization.weight`),
                    bias: need(`${p}.downsample.batch_normalization.bias`),
                },
                { requiresGrad: rg },
            );
        }
    }
}

class Downsample extends NeuralNetwork {
    constructor(
        readonly convolution: Convolution2d,
        readonly batchNormalization: BatchNormalization2d,
    ) {
        super();
        this.registerChild(convolution, { name: 'convolution' });
        this.registerChild(batchNormalization, { name: 'batch_normalization' });
    }

    protected semanticName(): string {
        return 'downsample';
    }
    forward(x: Tensor): Tensor {
        return this.batchNormalization.forward(this.convolution.forward(x));
    }
}
