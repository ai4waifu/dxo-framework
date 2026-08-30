import {
    adamStep,
    backwardSgdStep,
    createAdamState,
    sgdStep,
    type AdamStateHandle,
    type Tensor,
} from '@dxo/core';

/**
 * Optimizer contract (G3 / train-batch):
 * `step` returns new `requiresGrad` leaves; callers must reassign onto the module
 * (e.g. `linear.loadParameters(opt.step(linear.parameters()))`).
 * Updates run in Rust (`sgdStep` / `adamStep`) — no per-parameter `toArray` roundtrips.
 */
export interface Optimizer {
    step(params: Tensor[]): Promise<Tensor[]>;
    /**
     * When set, Trainer may fuse `loss.backward()` + SGD in one native call using this lr.
     * Only SGD exposes this; Adam keeps separate backward + `step`.
     */
    readonly fusedSgdLr?: number;
}

/**
 * Vanilla SGD: `p ← p - lr * grad`.
 * Implemented as one native batch call; does not mutate input tensors.
 */
export class SGD implements Optimizer {
    readonly fusedSgdLr: number;

    constructor(readonly lr: number) {
        if (!(lr > 0)) throw new Error('SGD lr must be positive');
        this.fusedSgdLr = lr;
    }

    async step(params: Tensor[]): Promise<Tensor[]> {
        return sgdStep(params, this.lr);
    }
}

/** Adam (β1=0.9, β2=0.999, ε=1e-8); moment state held in native `AdamState`. */
export class Adam implements Optimizer {
    readonly #state: AdamStateHandle;

    constructor(
        readonly lr: number,
        readonly beta1 = 0.9,
        readonly beta2 = 0.999,
        readonly eps = 1e-8,
    ) {
        if (!(lr > 0)) throw new Error('Adam lr must be positive');
        this.#state = createAdamState();
    }

    async step(params: Tensor[]): Promise<Tensor[]> {
        return adamStep(params, this.#state, this.lr, this.beta1, this.beta2, this.eps);
    }
}

/**
 * Fused train step helper: `loss.backward()` + SGD in one engine call.
 * Trainer may use this instead of separate backward + `optimizer.step`.
 */
export async function sgdTrainStep(loss: Tensor, params: Tensor[], lr: number): Promise<Tensor[]> {
    return backwardSgdStep(loss, params, lr);
}
