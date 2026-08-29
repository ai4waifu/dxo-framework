import assert from 'node:assert/strict';
import { tensor } from '@dxo/core';
import { FullyConnected } from '@dxo/nn';
import { SGD } from '@dxo/optimizer';

/**
 * Tiny linearly-separable stand-in for an MNIST linear probe:
 * features = [x0, x1], target = 2*x0 - x1 (MSE regression).
 * Gate: mean train loss must drop by >50% over SGD steps.
 */
function makeBatch(n: number): { x: ReturnType<typeof tensor>; y: ReturnType<typeof tensor> } {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < n; i++) {
        const a = (i % 10) / 10;
        const b = ((i * 3) % 10) / 10;
        xs.push(a, b);
        ys.push(2 * a - b);
    }
    return {
        x: tensor(xs, [n, 2]),
        y: tensor(ys, [n, 1]),
    };
}

const n = 32;
const model = new FullyConnected(2, 1);
// Deterministic small init so the gate is reproducible.
model.loadState({
    weight: { shape: [2, 1], data: [0.1, -0.1] },
    bias: { shape: [1], data: [0] },
});
const opt = new SGD(0.1);
const { x, y } = makeBatch(n);

async function forwardLoss(): Promise<{ value: number; backward: () => void }> {
    model.zeroGrad();
    const pred = model.forward(x);
    const diff = pred.sub(y);
    const loss = diff.mul(diff).mean();
    const value = await loss.item();
    return {
        value,
        backward: () => {
            loss.backward();
        },
    };
}

const first = await forwardLoss();
const loss0 = first.value;
first.backward();
model.loadParameters(await opt.step(model.parameters()));

for (let step = 1; step < 80; step++) {
    const turn = await forwardLoss();
    assert.ok(Number.isFinite(turn.value), `non-finite loss at step ${step}: ${turn.value}`);
    turn.backward();
    model.loadParameters(await opt.step(model.parameters()));
}

const loss1 = (await forwardLoss()).value;
assert.ok(loss0 > 1e-4, `initial loss too small: ${loss0}`);
assert.ok(Number.isFinite(loss1), `final loss non-finite: ${loss1}`);
assert.ok(loss1 < loss0 * 0.5, `loss did not drop enough: ${loss0} -> ${loss1}`);

console.log(`mnist-linear ok: loss ${loss0.toFixed(4)} -> ${loss1.toFixed(4)}`);
