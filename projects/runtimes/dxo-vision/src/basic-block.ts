import type { Tensor } from '@dxo/core';
import type { TensorStateSlice } from '@dxo/nn';
import { BatchNorm2d, Conv2d, Module, Relu } from '@dxo/nn';

/**
 * DXO-native BasicBlock (not a torchvision key mirror).
 * Keys: `{prefix}.conv1|bn1|conv2|bn2[.weight|.bias]` and optional `{prefix}.down.conv|bn`.
 */
export class BasicBlock extends Module {
    readonly conv1: Conv2d;
    readonly bn1: BatchNorm2d;
    readonly conv2: Conv2d;
    readonly bn2: BatchNorm2d;
    readonly relu: Relu;
    readonly downConv: Conv2d | null;
    readonly downBn: BatchNorm2d | null;
    readonly prefix: string;

    constructor(prefix: string, inChannels: number, outChannels: number, opts: { stride?: number; requiresGrad?: boolean } = {}) {
        super();
        this.prefix = prefix;
        const stride = opts.stride ?? 1;
        const rg = opts.requiresGrad ?? true;
        this.conv1 = new Conv2d(inChannels, outChannels, 3, { stride, padding: 1, requiresGrad: rg });
        this.bn1 = new BatchNorm2d(outChannels, { requiresGrad: rg });
        this.conv2 = new Conv2d(outChannels, outChannels, 3, { padding: 1, requiresGrad: rg });
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

    async state(): Promise<Record<string, TensorStateSlice>> {
        const p = this.prefix;
        const c1 = await this.conv1.state();
        const b1 = await this.bn1.state();
        const c2 = await this.conv2.state();
        const b2 = await this.bn2.state();
        const out: Record<string, TensorStateSlice> = {
            [`${p}.conv1.weight`]: c1.weight,
            [`${p}.conv1.bias`]: c1.bias,
            [`${p}.bn1.weight`]: b1.weight,
            [`${p}.bn1.bias`]: b1.bias,
            [`${p}.conv2.weight`]: c2.weight,
            [`${p}.conv2.bias`]: c2.bias,
            [`${p}.bn2.weight`]: b2.weight,
            [`${p}.bn2.bias`]: b2.bias,
        };
        if (this.downConv && this.downBn) {
            const dc = await this.downConv.state();
            const db = await this.downBn.state();
            out[`${p}.down.conv.weight`] = dc.weight;
            out[`${p}.down.conv.bias`] = dc.bias;
            out[`${p}.down.bn.weight`] = db.weight;
            out[`${p}.down.bn.bias`] = db.bias;
        }
        return out;
    }

    loadState(saved: Record<string, TensorStateSlice>, opts: { requiresGrad?: boolean } = {}): void {
        const p = this.prefix;
        const rg = opts.requiresGrad ?? true;
        const need = (k: string) => {
            const s = saved[k];
            if (!s) throw new Error(`BasicBlock.loadState: missing '${k}'`);
            return s;
        };
        this.conv1.loadState({ weight: need(`${p}.conv1.weight`), bias: need(`${p}.conv1.bias`) }, { requiresGrad: rg });
        this.bn1.loadState({ weight: need(`${p}.bn1.weight`), bias: need(`${p}.bn1.bias`) }, { requiresGrad: rg });
        this.conv2.loadState({ weight: need(`${p}.conv2.weight`), bias: need(`${p}.conv2.bias`) }, { requiresGrad: rg });
        this.bn2.loadState({ weight: need(`${p}.bn2.weight`), bias: need(`${p}.bn2.bias`) }, { requiresGrad: rg });
        if (this.downConv && this.downBn) {
            this.downConv.loadState({ weight: need(`${p}.down.conv.weight`), bias: need(`${p}.down.conv.bias`) }, { requiresGrad: rg });
            this.downBn.loadState({ weight: need(`${p}.down.bn.weight`), bias: need(`${p}.down.bn.bias`) }, { requiresGrad: rg });
        }
    }
}
