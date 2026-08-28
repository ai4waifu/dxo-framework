import type { Tensor } from '@dxo/core';
import { BatchNorm2d, Conv2d, MaxPool2d, Module, Relu, type TensorStateSlice } from '@dxo/nn';
import { BasicBlock } from './basic-block.js';
import { unsupported, VisionError } from './errors.js';
import {
    type ResNetDepth,
    type ResNetOptions,
    type ResNetSignature,
    resnetFeatureChannels,
    type TensorPort,
    type WeightSource,
} from './types.js';

/** DXO-native ResNet-18 layout: stem + stage{1..4}.block{i}.* (no torchvision key mirror, no fc). */
export const RESNET18_STAGE_BLOCKS = [2, 2, 2, 2] as const;
export const RESNET18_STAGE_CHANNELS = [64, 128, 256, 512] as const;

function buildSignature(depth: ResNetDepth, inChannels: number): ResNetSignature {
    const features = resnetFeatureChannels(depth);
    return {
        input: {
            image: {
                name: 'image',
                dtype: 'f32',
                shape: ['batch', inChannels, 'height', 'width'],
            },
        },
        output: {
            features: {
                name: 'features',
                dtype: 'f32',
                shape: ['batch', features],
            },
        },
    };
}

/**
 * ResNet backbone only — `forward` yields feature Tensor, never labels.
 * depth=18 is wired (DXO keys). Prefer input spatial 32×32 so final map is 1×1 (no avgPool yet).
 */
export class ResNet extends Module {
    readonly depth: ResNetDepth;
    readonly inChannels: number;
    readonly zeroInitResidual: boolean;
    readonly norm: 'batchnorm';
    readonly device: ResNetOptions['device'];
    readonly signature: ResNetSignature;
    #trainable: boolean;
    #ready = true;

    readonly stemConv: Conv2d | null = null;
    readonly stemBn: BatchNorm2d | null = null;
    readonly stemRelu: Relu | null = null;
    readonly stemPool: MaxPool2d | null = null;
    readonly stages: BasicBlock[][] | null = null;

    constructor(options: ResNetOptions = {}) {
        super();
        this.depth = options.depth ?? 18;
        this.inChannels = options.inChannels ?? 3;
        this.zeroInitResidual = options.zeroInitResidual ?? false;
        this.norm = options.norm ?? 'batchnorm';
        this.device = options.device ?? 'cpu';
        this.#trainable = options.trainable ?? true;
        this.signature = buildSignature(this.depth, this.inChannels);
        if (options.weights && options.weights !== 'none') {
            this.#ready = false;
        }

        if (this.depth === 18) {
            const rg = this.#trainable;
            this.stemConv = new Conv2d(this.inChannels, 64, 7, { stride: 2, padding: 3, requiresGrad: rg });
            this.stemBn = new BatchNorm2d(64, { requiresGrad: rg });
            this.stemRelu = new Relu();
            this.stemPool = new MaxPool2d(3, { stride: 2, padding: 1 });
            const stages: BasicBlock[][] = [];
            let inCh = 64;
            for (let s = 0; s < RESNET18_STAGE_BLOCKS.length; s++) {
                const outCh = RESNET18_STAGE_CHANNELS[s]!;
                const nBlocks = RESNET18_STAGE_BLOCKS[s]!;
                const blocks: BasicBlock[] = [];
                for (let b = 0; b < nBlocks; b++) {
                    const stride = b === 0 && s > 0 ? 2 : 1;
                    blocks.push(new BasicBlock(inCh, outCh, { stride, requiresGrad: rg }));
                    inCh = outCh;
                }
                stages.push(blocks);
            }
            this.stages = stages;
        }
    }

    features(): TensorPort {
        return this.signature.output.features;
    }

    freeze(): void {
        this.#trainable = false;
    }

    unfreeze(): void {
        this.#trainable = true;
    }

    get trainable(): boolean {
        return this.#trainable;
    }

    /** Flat DXO state keys for the wired graph (empty when depth is not built). */
    parameterNames(): string[] {
        if (!this.stages || !this.stemConv) return [];
        const names: string[] = ['stem.conv.weight', 'stem.conv.bias', 'stem.bn.weight', 'stem.bn.bias'];
        for (let s = 0; s < this.stages.length; s++) {
            const blocks = this.stages[s]!;
            for (let b = 0; b < blocks.length; b++) {
                const p = `stage${s + 1}.block${b}`;
                names.push(
                    `${p}.conv1.weight`,
                    `${p}.conv1.bias`,
                    `${p}.bn1.weight`,
                    `${p}.bn1.bias`,
                    `${p}.conv2.weight`,
                    `${p}.conv2.bias`,
                    `${p}.bn2.weight`,
                    `${p}.bn2.bias`,
                );
                if (blocks[b]!.downConv) {
                    names.push(
                        `${p}.downsample.conv.weight`,
                        `${p}.downsample.conv.bias`,
                        `${p}.downsample.bn.weight`,
                        `${p}.downsample.bn.bias`,
                    );
                }
            }
        }
        return names;
    }

    async state(): Promise<Record<string, TensorStateSlice>> {
        if (!this.stemConv || !this.stemBn || !this.stages) {
            unsupported('ResNet.state', `depth=${this.depth}; only depth=18 exposes DXO state schema`);
        }
        const out: Record<string, TensorStateSlice> = {};
        const sc = await this.stemConv.state();
        const sb = await this.stemBn.state();
        out['stem.conv.weight'] = sc.weight;
        out['stem.conv.bias'] = sc.bias;
        out['stem.bn.weight'] = sb.weight;
        out['stem.bn.bias'] = sb.bias;
        for (let s = 0; s < this.stages.length; s++) {
            const blocks = this.stages[s]!;
            for (let b = 0; b < blocks.length; b++) {
                Object.assign(out, await blocks[b]!.state(`stage${s + 1}.block${b}`));
            }
        }
        return out;
    }

    loadState(saved: Record<string, TensorStateSlice>, opts: { requiresGrad?: boolean } = {}): void {
        if (!this.stemConv || !this.stemBn || !this.stages) {
            unsupported('ResNet.loadState', `depth=${this.depth}; only depth=18 accepts DXO state`);
        }
        const rg = opts.requiresGrad ?? true;
        const take = (k: string) => {
            const t = saved[k];
            if (!t) throw new VisionError('MISSING_STATE_KEY', `ResNet.loadState: missing '${k}'`);
            return t;
        };
        this.stemConv.loadState({ weight: take('stem.conv.weight'), bias: take('stem.conv.bias') }, { requiresGrad: rg });
        this.stemBn.loadState({ weight: take('stem.bn.weight'), bias: take('stem.bn.bias') }, { requiresGrad: rg });
        for (let s = 0; s < this.stages.length; s++) {
            const blocks = this.stages[s]!;
            for (let b = 0; b < blocks.length; b++) {
                blocks[b]!.loadState(`stage${s + 1}.block${b}`, saved, { requiresGrad: rg });
            }
        }
    }

    forward(image: Tensor): Tensor {
        if (!this.stemConv || !this.stemBn || !this.stemRelu || !this.stemPool || !this.stages) {
            unsupported('ResNet.forward', `depth=${this.depth}; only depth=18 is wired`);
        }
        if (image.shape.length !== 4) {
            throw new VisionError('MODEL_INPUT_SHAPE_MISMATCH', `ResNet.forward expects NCHW rank 4, got [${image.shape.join(',')}]`);
        }
        if (image.shape[1] !== this.inChannels) {
            throw new VisionError('MODEL_INPUT_SHAPE_MISMATCH', `ResNet.forward expected ${this.inChannels} channels, got ${image.shape[1]}`);
        }

        let h = this.stemRelu.forward(this.stemBn.forward(this.stemConv.forward(image)));
        h = this.stemPool.forward(h);
        for (const blocks of this.stages) {
            for (const block of blocks) {
                h = block.forward(h);
            }
        }

        const n = h.shape[0]!;
        const c = h.shape[1]!;
        const hh = h.shape[2]!;
        const ww = h.shape[3]!;
        if (hh !== 1 || ww !== 1) {
            unsupported('ResNet.forward', `spatial ${hh}x${ww} needs GAP (use 32x32 input for depth=18 preview, or wait for avgPool)`);
        }
        return h.reshape([n, c]);
    }

    async load(_weights: WeightSource, _options?: { scope?: 'all' | 'backbone' }): Promise<void> {
        unsupported('ResNet.load', 'use loadState / loadWeights (0.0.13+) with DXO-key safetensors');
    }

    async ready(): Promise<void> {
        if (!this.#ready) {
            unsupported('ResNet.ready', 'deferred weight load not wired');
        }
    }
}

export function defineResNet(options?: ResNetOptions): ResNet {
    return new ResNet(options);
}
