import type { Tensor } from '@dxo/core';
import type { TensorStateSlice } from '@dxo/nn';
import type { TensorPort } from './types.js';

/** Soft ports for graph connectivity — no labels / language. */
export type NeuralSignature = {
    input: Record<string, TensorPort>;
    output: Record<string, TensorPort>;
};

/**
 * Public DXO compute contract (Living `14`).
 * `@dxo/nn` `Module` may remain as an implementation base; cross-package APIs use Neural.
 */
export interface Neural<I = Tensor, O = Tensor> {
    forward(input: I): O;
    readonly signature: NeuralSignature;
    parameters(): Iterable<Tensor>;
    state(): Promise<Record<string, TensorStateSlice>>;
}
