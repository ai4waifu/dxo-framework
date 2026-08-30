/**
 * train-batch-rust: Rust batch zeroGrad / SGD / Adam / fused backward+SGD.
 * CPU-only slice; Trainer orchestration stays in TypeScript.
 */
import assert from 'node:assert/strict';
import { tensor, zeroGrads } from '@dxo/core';
import { FullyConnected } from '@dxo/nn';
import { Adam, SGD, sgdTrainStep } from '@dxo/optimizer';
import { Trainer } from '@dxo/train';

{
    const model = new FullyConnected(2, 1);
    model.loadState({
        weight: { shape: [2, 1], data: [0.5, -0.5] },
        bias: { shape: [1], data: [0] },
    });
    const x = tensor([1, 2], [1, 2]);
    const y = tensor([0], [1, 1]);
    model.zeroGrad();
    const pred = model.forward(x);
    const diff = pred.sub(y);
    const loss = diff.mul(diff).mean();
    loss.backward();
    assert.ok(model.weight.grad);
    const before = (await model.weight.toArray()).slice();
    const updated = await new SGD(0.1).step(model.parameters());
    assert.equal(updated.length, 2);
    assert.equal(updated[0]!.requiresGrad, true);
    assert.notEqual(updated[0], model.weight);
    model.loadParameters(updated);
    assert.notDeepEqual(await model.weight.toArray(), before);
}

{
    const w = tensor([1, 2], [2], { requiresGrad: true });
    const loss = w.sum();
    loss.backward();
    assert.ok(w.grad);
    zeroGrads([w]);
    assert.equal(w.grad, undefined);
}

{
    const model = new FullyConnected(2, 1);
    model.loadState({
        weight: { shape: [2, 1], data: [1, -1] },
        bias: { shape: [1], data: [0] },
    });
    const x = tensor([1, 1], [1, 2]);
    const y = tensor([1], [1, 1]);
    model.zeroGrad();
    const pred = model.forward(x);
    const err = pred.sub(y);
    const loss = err.mul(err).mean();
    const before = (await model.weight.toArray()).slice();
    const next = await sgdTrainStep(loss, model.parameters(), 0.05);
    model.loadParameters(next);
    assert.notDeepEqual(await model.weight.toArray(), before);
}

{
    const model = new FullyConnected(2, 1);
    model.loadState({
        weight: { shape: [2, 1], data: [0.4, -0.2] },
        bias: { shape: [1], data: [0.1] },
    });
    const opt = new Adam(0.05);
    model.zeroGrad();
    const pred = model.forward(tensor([1, 0], [1, 2]));
    const loss = pred.sub(tensor([0], [1, 1])).mul(pred.sub(tensor([0], [1, 1]))).mean();
    loss.backward();
    const before = (await model.weight.toArray()).slice();
    model.loadParameters(await opt.step(model.parameters()));
    assert.notDeepEqual(await model.weight.toArray(), before);
}

{
    const model = new FullyConnected(4, 1);
    const batches = () => [
        {
            x: tensor(
                [
                    [0, 0, 1, 1],
                    [1, 1, 0, 0],
                ].flat(),
                [2, 4],
            ),
            y: tensor([[0], [1]].flat(), [2, 1]),
        },
    ];
    const trainer = new Trainer({
        model,
        optimizer: new SGD(0.2),
        batches,
        epochs: 3,
    });
    const summary = await trainer.fit();
    assert.equal(summary.aborted, false);
    assert.equal(summary.epochs, 3);
    assert.ok(summary.finalMeanLoss !== undefined);
}

console.log('train-batch-rust ok: zeroGrads / sgdStep / adamStep / fused / Trainer');
