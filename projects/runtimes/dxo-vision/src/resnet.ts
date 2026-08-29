import type { Tensor } from '@dxo/core';
import { BatchNorm2d, Conv2d, MaxPool2d, Module, Relu, type TensorStateSlice } from '@dxo/nn';
import { BasicBlock } from './basic-block.js';
import { unsupported, VisionError } from './errors.js';
import type { Neural } from './neural.js';
import {
    type ResNetDepth,
    type ResNetOptions,
    type ResNetSignature,
    resnetFeatureChannels,
    type TensorPort,
    type WeightSource,
} from './types.js';

/** DXO-native ResNet-18 layout: stages [2,2,2,2], channels 64→512. No classification `fc`. */
const RESNET18_BLOCKS = [2, 2, 2, 2] as const;
const RESNET18_CHANNELS = [64, 128, 256, 512] as const;

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
 * ResNet backbone Neural — `forward` yields feature Tensor, never labels.
 * DXO-native parameter names (`stem.*` / `stageN.blockM.*`); not a torchvision key mirror.
 * depth=18 is wired; other depths throw `UNSUPPORTED` on forward/state.
 * Extends `@dxo/nn` Module for parameter walk; public contract is Neural.
 */
export class ResNet extends Module implements Neural<Tensor, Tensor> {
    readonly depth: ResNetDepth;
    readonly inChannels: number;
    readonly zeroInitResidual: boolean;
    readonly norm: 'batchnorm';
    readonly device: ResNetOptions['device'];
    readonly signature: ResNetSignature;
    #trainable: boolean;
    #ready = true;

    /** Present only when depth === 18. */
    stemConv: Conv2d | null = null;
    stemBn: BatchNorm2d | null = null;
    stemRelu: Relu | null = null;
    stemPool: MaxPool2d | null = null;
    stages: BasicBlock[][] | null = null;

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
            this.#buildResNet18(options.trainable ?? true);
        }
    }

    #buildResNet18(requiresGrad: boolean): void {
        const rg = requiresGrad;
        this.stemConv = new Conv2d(this.inChannels, 64, 7, { stride: 2, padding: 3, requiresGrad: rg });
        this.stemBn = new BatchNorm2d(64, { requiresGrad: rg });
        this.stemRelu = new Relu();
        this.stemPool = new MaxPool2d(3, { stride: 2, padding: 1 });
        const stages: BasicBlock[][] = [];
        let inCh = 64;
        for (let s = 0; s < RESNET18_BLOCKS.length; s++) {
            const outCh = RESNET18_CHANNELS[s]!;
            const nBlocks = RESNET18_BLOCKS[s]!;
            const stage: BasicBlock[] = [];
            for (let b = 0; b < nBlocks; b++) {
                const stride = s > 0 && b === 0 ? 2 : 1;
                stage.push(new BasicBlock(`stage${s + 1}.block${b}`, inCh, outCh, { stride, requiresGrad: rg }));
                inCh = outCh;
            }
            stages.push(stage);
        }
        this.stages = stages;
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

    /** Flat DXO state keys (depth=18 only). */
    parameterNames(): string[] {
        if (this.depth !== 18 || !this.stages) {
            return [];
        }
        // Synchronous name list without reading tensor data.
        const names: string[] = ['stem.conv.weight', 'stem.conv.bias', 'stem.bn.weight', 'stem.bn.bias'];
        for (const stage of this.stages) {
            for (const block of stage) {
                const p = block.prefix;
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
                if (block.downConv) {
                    names.push(`${p}.down.conv.weight`, `${p}.down.conv.bias`, `${p}.down.bn.weight`, `${p}.down.bn.bias`);
                }
            }
        }
        return names;
    }

    async state(): Promise<Record<string, TensorStateSlice>> {
        if (this.depth !== 18 || !this.stemConv || !this.stemBn || !this.stages) {
            unsupported('ResNet.state', `depth=${this.depth}; only depth=18 state schema is wired`);
        }
        const c = await this.stemConv.state();
        const b = await this.stemBn.state();
        const out: Record<string, TensorStateSlice> = {
            'stem.conv.weight': c.weight,
            'stem.conv.bias': c.bias,
            'stem.bn.weight': b.weight,
            'stem.bn.bias': b.bias,
        };
        for (const stage of this.stages) {
            for (const block of stage) {
                Object.assign(out, await block.state());
            }
        }
        return out;
    }

    loadState(saved: Record<string, TensorStateSlice>, opts: { requiresGrad?: boolean } = {}): void {
        if (this.depth !== 18 || !this.stemConv || !this.stemBn || !this.stages) {
            unsupported('ResNet.loadState', `depth=${this.depth}; only depth=18 state schema is wired`);
        }
        const rg = opts.requiresGrad ?? true;
        const need = (k: string) => {
            const s = saved[k];
            if (!s) throw new VisionError('MISSING_STATE_KEY', `ResNet.loadState: missing '${k}'`);
            return s;
        };
        this.stemConv.loadState({ weight: need('stem.conv.weight'), bias: need('stem.conv.bias') }, { requiresGrad: rg });
        this.stemBn.loadState({ weight: need('stem.bn.weight'), bias: need('stem.bn.bias') }, { requiresGrad: rg });
        for (const stage of this.stages) {
            for (const block of stage) {
                block.loadState(saved, { requiresGrad: rg });
            }
        }
    }

    forward(image: Tensor): Tensor {
        if (this.depth !== 18 || !this.stemConv || !this.stemBn || !this.stemRelu || !this.stemPool || !this.stages) {
            unsupported('ResNet.forward', `depth=${this.depth}; only depth=18 forward is wired`);
        }
        if (image.shape.length !== 4) {
            throw new VisionError('MODEL_INPUT_SHAPE_MISMATCH', `ResNet.forward expects NCHW rank 4, got [${image.shape.join(',')}]`);
        }
        if (image.shape[1] !== this.inChannels) {
            throw new VisionError('MODEL_INPUT_SHAPE_MISMATCH', `ResNet.forward expected ${this.inChannels} channels, got ${image.shape[1]}`);
        }

        let h = this.stemRelu.forward(this.stemBn.forward(this.stemConv.forward(image)));
        h = this.stemPool.forward(h);
        for (const stage of this.stages) {
            for (const block of stage) {
                h = block.forward(h);
            }
        }
        const n = h.shape[0]!;
        const c = h.shape[1]!;
        const spatial = (h.shape[2] ?? 1) * (h.shape[3] ?? 1);
        if (spatial !== 1) {
            unsupported('ResNet.forward', `final spatial ${h.shape[2]}x${h.shape[3]} != 1x1 (use 32x32 input for depth=18 without avgPool)`);
        }
        return h.reshape([n, c]);
    }

    async load(_weights: WeightSource, _options?: { scope?: 'all' | 'backbone' }): Promise<void> {
        unsupported('ResNet.load', 'use loadWeights in a later preview; remote sources not wired');
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
