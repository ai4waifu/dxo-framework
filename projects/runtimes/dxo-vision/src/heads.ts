import type { Tensor } from '@dxo/core';
import { Linear, NeuralNetwork } from '@dxo/nn';
import { VisionError } from './errors.js';
import type { ResNet } from './resnet.js';
import type { TensorPort } from './types.js';

export type LinearHeadOptions = {
    /** Feature dim; inferred from `input` port or by `compose` when omitted. */
    input?: number | TensorPort;
    /** Logits dim — class count for the head only, not ResNet. */
    output: number;
    bias?: boolean;
};

function resolveInFeatures(input: number | TensorPort): number {
    if (typeof input === 'number') return input;
    const last = input.shape[input.shape.length - 1];
    if (typeof last === 'number') return last;
    throw new VisionError('INVALID_HEAD_INPUT', 'LinearHead input port must end with a fixed feature dimension');
}

/** Feature → logits only; no label names / language. */
export class LinearHead extends NeuralNetwork {
    readonly outFeatures: number;
    #inFeatures: number | undefined;
    #linear: Linear | undefined;

    constructor(options: LinearHeadOptions) {
        super();
        this.outFeatures = options.output;
        if (options.input !== undefined) {
            this.bindInput(resolveInFeatures(options.input));
        }
    }

    get inFeatures(): number | undefined {
        return this.#inFeatures;
    }

    /** Wire feature size (used by `compose` when ctor omitted `input`). */
    bindInput(inFeatures: number): void {
        if (this.#inFeatures !== undefined && this.#inFeatures !== inFeatures) {
            throw new VisionError('HEAD_FEATURE_MISMATCH', `LinearHead already bound to inFeatures=${this.#inFeatures}, got ${inFeatures}`);
        }
        if (this.#linear && this.#inFeatures === inFeatures) return;
        this.#inFeatures = inFeatures;
        this.#linear = new Linear(inFeatures, this.outFeatures);
    }

    forward(features: Tensor): Tensor {
        if (!this.#linear) {
            const last = features.shape[features.shape.length - 1];
            if (typeof last !== 'number') {
                throw new VisionError('INVALID_HEAD_INPUT', 'cannot infer LinearHead input from features');
            }
            this.bindInput(last);
        }
        return this.#linear!.forward(features);
    }

    override parameters(): Tensor[] {
        return this.#linear ? this.#linear.parameters() : [];
    }
}

export type ClassifierOptions = {
    backbone: ResNet;
    head: LinearHead;
};

/** Backbone + head; `forward` returns logits Tensor. */
export class Classifier extends NeuralNetwork {
    readonly backbone: ResNet;
    readonly head: LinearHead;

    constructor(options: ClassifierOptions) {
        super();
        this.backbone = options.backbone;
        this.head = options.head;
    }

    forward(image: Tensor): Tensor {
        const features = this.backbone.forward(image);
        return this.head.forward(features);
    }
}

/** Compose ResNet + LinearHead into a Classifier. */
export function compose(backbone: ResNet, head: LinearHead): Classifier {
    const expected = backbone.features();
    const last = expected.shape[expected.shape.length - 1];
    if (typeof last !== 'number') {
        throw new VisionError('INVALID_BACKBONE', 'backbone features port must end with a fixed dim');
    }
    if (head.inFeatures !== undefined && head.inFeatures !== last) {
        throw new VisionError('HEAD_FEATURE_MISMATCH', `LinearHead inFeatures=${head.inFeatures} does not match backbone features=${last}`);
    }
    head.bindInput(last);
    return new Classifier({ backbone, head });
}
