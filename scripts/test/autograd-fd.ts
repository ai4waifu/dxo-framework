import assert from 'node:assert/strict';
import { backend, tensor, withoutGrad } from '@dxo/core';

assert.equal(backend(), 'titan-cpu');

function almostEqual(a: number[], b: number[], tol = 5e-2): void {
    assert.equal(a.length, b.length);
    for (let i = 0; i < a.length; i++) {
        assert.ok(Math.abs(a[i]! - b[i]!) < tol, `i=${i} got=${a[i]} want=${b[i]}`);
    }
}

function finiteDiff(data: number[], lossAt: (x: number[]) => number, eps = 1e-3): number[] {
    const out = new Array<number>(data.length);
    for (let i = 0; i < data.length; i++) {
        const xp = data.slice();
        const xm = data.slice();
        xp[i]! += eps;
        xm[i]! -= eps;
        out[i] = (lossAt(xp) - lossAt(xm)) / (2 * eps);
    }
    return out;
}

// matmul + sum
{
    const xData = [1, 2, 3, 4];
    const wData = [0.5, 0, 0, 0.5];
    const x = tensor(xData, [2, 2], { requiresGrad: true });
    const w = tensor(wData, [2, 2], { requiresGrad: true });
    const y = x.matmul(w).sum();
    y.backward();
    const fdX = finiteDiff(
        xData,
        (d) =>
            tensor(d, [2, 2])
                .matmul(tensor(wData, [2, 2]))
                .sum()
                .toArray()[0]!,
    );
    const fdW = finiteDiff(
        wData,
        (d) =>
            tensor(xData, [2, 2])
                .matmul(tensor(d, [2, 2]))
                .sum()
                .toArray()[0]!,
    );
    almostEqual(x.grad!, fdX);
    almostEqual(w.grad!, fdW);
}

// add + mul + relu + sum (broadcast bias)
{
    const xData = [1, -2, 3, -4];
    const bData = [0.5, -0.25];
    const x = tensor(xData, [2, 2], { requiresGrad: true });
    const b = tensor(bData, [2], { requiresGrad: true });
    const scale = tensor([2, 2, 2, 2], [2, 2]);
    const y = x.add(b).mul(scale).relu().sum();
    y.backward();
    const fdX = finiteDiff(
        xData,
        (d) =>
            tensor(d, [2, 2])
                .add(tensor(bData, [2]))
                .mul(scale)
                .relu()
                .sum()
                .toArray()[0]!,
    );
    almostEqual(x.grad!, fdX, 8e-2);
}

// withoutGrad disables tape
{
    const x = tensor([1, -1], [2], { requiresGrad: true });
    const y = withoutGrad(() => x.relu());
    assert.equal(y.requiresGrad, false);
}

console.log('autograd-fd ok: matmul/add/mul/relu finite-diff aligned');
