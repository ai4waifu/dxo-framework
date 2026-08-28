import type { Tensor } from '@dxo/core';
import { BatchNorm2d, Conv2d, Module, Relu, type TensorStateSlice } from '@dxo/nn';

/** DXO-native residual block (not a torchvision key mirror). */
export class BasicBlock extends Module {
    readonly conv1: Conv2d;
    readonly bn1: BatchNorm2d;
    readonly conv2: Conv2d;
    readonly bn2: BatchNorm2d;
    readonly relu: Relu;
    readonly downConv: Conv2d | null;
    readonly downBn: BatchNorm2d | null;

    constructor(inChannels: number, outChannels: number, opts: { stride?: number; requiresGrad?: boolean } = {}) {
        super();
        const stride = opts.stride ?? 1;
        const rg = opts.requiresGrad ?? true;
        this.conv1 = new Conv2d(inChannels, outChannels, 3, { stride, padding: 1, requiresGrad: rg });
        this.bn1 = new BatchNorm2d(outChannels, { requiresGrad: rg });
        this.conv2 = new Conv2d(outChannels, outChannels, 3, { stride: 1, padding: 1, requiresGrad: rg });
        this.bn2 = new BatchNorm2d(outChannels, { requiresGrad: rg });
        this.relu = new Relu();
        if (stride !== 1 || inChannels !== outChannels) {
            this.downConv = new Conv2d(inChannels, outChannels, 1, { stride, requiresGrad: rg });
            this.downBn = new BatchNorm2d(outChannels, { requiresGrad: rg });
        } else {
            this.downConv = null;
            this.downBn = null;
        }
    }

    forward(x: Tensor): Tensor {
        let h = this.relu.forward(this.bn1.forward(this.conv1.forward(x)));
        h = this.bn2.forward(this.conv2.forward(h));
        let skip = x;
        if (this.downConv && this.downBn) {
            skip = this.downBn.forward(this.downConv.forward(x));
        }
        return this.relu.forward(h.add(skip));
    }

    async state(prefix: string): Promise<Record<string, TensorStateSlice>> {
        const out: Record<string, TensorStateSlice> = {};
        const c1 = await this.conv1.state();
        const b1 = await this.bn1.state();
        const c2 = await this.conv2.state();
        const b2 = await this.bn2.state();
        out[`${prefix}.conv1.weight`] = c1.weight;
        out[`${prefix}.conv1.bias`] = c1.bias;
        out[`${prefix}.bn1.weight`] = b1.weight;
        out[`${prefix}.bn1.bias`] = b1.bias;
        out[`${prefix}.conv2.weight`] = c2.weight;
        out[`${prefix}.conv2.bias`] = c2.bias;
        out[`${prefix}.bn2.weight`] = b2.weight;
        out[`${prefix}.bn2.bias`] = b2.bias;
        if (this.downConv && this.downBn) {
            const dc = await this.downConv.state();
            const db = await this.downBn.state();
            out[`${prefix}.downsample.conv.weight`] = dc.weight;
            out[`${prefix}.downsample.conv.bias`] = dc.bias;
            out[`${prefix}.downsample.bn.weight`] = db.weight;
            out[`${prefix}.downsample.bn.bias`] = db.bias;
        }
        return out;
    }

    loadState(prefix: string, saved: Record<string, TensorStateSlice>, opts: { requiresGrad?: boolean } = {}): void {
        const rg = opts.requiresGrad ?? true;
        const take = (k: string) => {
            const t = saved[k];
            if (!t) throw new Error(`BasicBlock.loadState: missing '${k}'`);
            return t;
        };
        this.conv1.loadState({ weight: take(`${prefix}.conv1.weight`), bias: take(`${prefix}.conv1.bias`) }, { requiresGrad: rg });
        this.bn1.loadState({ weight: take(`${prefix}.bn1.weight`), bias: take(`${prefix}.bn1.bias`) }, { requiresGrad: rg });
        this.conv2.loadState({ weight: take(`${prefix}.conv2.weight`), bias: take(`${prefix}.conv2.bias`) }, { requiresGrad: rg });
        this.bn2.loadState({ weight: take(`${prefix}.bn2.weight`), bias: take(`${prefix}.bn2.bias`) }, { requiresGrad: rg });
        if (this.downConv && this.downBn) {
            this.downConv.loadState(
                {
                    weight: take(`${prefix}.downsample.conv.weight`),
                    bias: take(`${prefix}.downsample.conv.bias`),
                },
                { requiresGrad: rg },
            );
            this.downBn.loadState(
                {
                    weight: take(`${prefix}.downsample.bn.weight`),
                    bias: take(`${prefix}.downsample.bn.bias`),
                },
                { requiresGrad: rg },
            );
        }
    }
}
