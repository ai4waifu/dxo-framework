import type { Tensor } from '@dxo/core';
import { Module } from '@dxo/nn';
import { unsupported } from './errors.js';
import {
    type ResNetDepth,
    type ResNetOptions,
    type ResNetSignature,
    type TensorPort,
    type WeightSource,
    resnetFeatureChannels,
} from './types.js';

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
 * Numerical full forward waits on spatial pool / stage wiring; preview throws `UNSUPPORTED`.
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
            // Weight materialization is hub/serialize work; keep ctor sync and constructible.
            this.#ready = false;
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

    forward(_image: Tensor): Tensor {
        unsupported(
            'ResNet.forward',
            `depth=${this.depth}; full stem/stages/GAP not wired on this preview surface`,
        );
    }

    async load(_weights: WeightSource, _options?: { scope?: 'all' | 'backbone' }): Promise<void> {
        unsupported('ResNet.load', 'weight materialization lands with hub / serialize');
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
