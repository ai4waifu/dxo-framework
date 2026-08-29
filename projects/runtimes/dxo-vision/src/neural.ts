import type { TensorPort } from './types.js';

/** Soft ports for graph connectivity — no labels / language. */
export type NetworkPorts = {
    input: Record<string, TensorPort>;
    output: Record<string, TensorPort>;
};

/**
 * Public DXO compute contract (Living `14`).
 * `@dxo/nn` `Neural` is the implementation base; this contract adds ports.
 */
