import { type Tensor, tensor } from '@dxo/core';

/**
 * Optimizer contract (G3 / 0.0.4):
 * `step` returns new `requiresGrad` leaves; callers must reassign onto the module
 * (e.g. `linear.loadParameters(opt.step(linear.parameters()))`).
 */
export interface Optimizer {
    /** Apply one update; returns new leaf tensors (caller reassigns onto the module). */
    step(params: Tensor[]): Tensor[];
}

/**
 * Vanilla SGD: `p ← p - lr * grad`.
 * Does not mutate input tensors in place.
 */
export class SGD implements Optimizer {
    constructor(readonly lr: number) {
        if (!(lr > 0)) throw new Error('SGD lr must be positive');
    }

    step(params: Tensor[]): Tensor[] {
        return params.map((p) => {
            const g = p.grad;
            if (!g) return p;
            const data = p.toArray();
            if (data.length !== g.length) {
                throw new Error(`SGD grad length mismatch: param=${data.length} grad=${g.length}`);
            }
            const next = data.map((v, i) => v - this.lr * g[i]!);
            return tensor(next, [...p.shape], { requiresGrad: true });
        });
    }
}

/** Adam (β1=0.9, β2=0.999, ε=1e-8); moment state keyed by parameter index. */
export class Adam implements Optimizer {
    private m: number[][] = [];
    private v: number[][] = [];
    private t = 0;

    constructor(
        readonly lr: number,
        readonly beta1 = 0.9,
        readonly beta2 = 0.999,
        readonly eps = 1e-8,
    ) {
        if (!(lr > 0)) throw new Error('Adam lr must be positive');
    }

    step(params: Tensor[]): Tensor[] {
        this.t += 1;
        return params.map((p, idx) => {
            const g = p.grad;
            if (!g) return p;
            const data = p.toArray();
            if (!this.m[idx] || this.m[idx]!.length !== data.length) {
                this.m[idx] = new Array(data.length).fill(0);
                this.v[idx] = new Array(data.length).fill(0);
            }
            const m = this.m[idx]!;
            const v = this.v[idx]!;
            const next = new Array<number>(data.length);
            for (let i = 0; i < data.length; i++) {
                m[i] = this.beta1 * m[i]! + (1 - this.beta1) * g[i]!;
                v[i] = this.beta2 * v[i]! + (1 - this.beta2) * g[i]! * g[i]!;
                const mHat = m[i]! / (1 - this.beta1 ** this.t);
                const vHat = v[i]! / (1 - this.beta2 ** this.t);
                next[i] = data[i]! - (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
            }
            return tensor(next, [...p.shape], { requiresGrad: true });
        });
    }
}
