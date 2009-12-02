import assert from 'node:assert/strict';
import { backend, isGradEnabled, tensor, withoutGrad } from '@dxo/core';
import { Linear, Relu, Sequential } from '@dxo/nn';
import { SGD } from '@dxo/optimizer';

assert.equal(backend(), 'cpu');
assert.equal(isGradEnabled(), true);

// Linear is affine-only (negative pre-activation must survive).
{
    const fc = new Linear(1, 1, { requiresGrad: false });
    fc.weight = tensor([-2], [1, 1]);
    fc.bias = tensor([0], [1]);
    const out = fc.forward(tensor([1], [1, 1]));
    assert.equal((await out.toArray())[0], -2);
}

// Relu is explicit.
{
    const net = new Sequential([new Linear(1, 1, { requiresGrad: false }), new Relu()]);
    (net.layers[0] as Linear).weight = tensor([-2], [1, 1]);
    (net.layers[0] as Linear).bias = tensor([0], [1]);
    const out = net.forward(tensor([1], [1, 1]));
    assert.equal((await out.toArray())[0], 0);
}

// withoutGrad restores flag.
{
    assert.equal(isGradEnabled(), true);
    withoutGrad(() => {
        assert.equal(isGradEnabled(), false);
        const t = tensor([1, -1], [2], { requiresGrad: true }).relu();
        assert.equal(t.requiresGrad, false);
    });
    assert.equal(isGradEnabled(), true);
}

// SGD returns new leaves; loadParameters installs them.
{
    const model = new Linear(2, 1);
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
    assert.deepEqual([...loss.shape], [1]);
    loss.backward();
    assert.ok(model.weight.grad);
    const before = (await model.weight.toArray()).slice();
    const opt = new SGD(0.1);
    const updated = await opt.step(model.parameters());
    assert.equal(updated.length, 2);
    assert.equal(updated[0]!.requiresGrad, true);
    assert.notEqual(updated[0], model.weight);
    model.loadParameters(updated);
    assert.notDeepEqual(await model.weight.toArray(), before);
}

// backward rejects non-scalars (contract).
{
    const t = tensor([1, 2], [2], { requiresGrad: true });
    assert.throws(() => t.backward());
}

console.log('g3-contract ok: Linear/Relu/SGD/withoutGrad/mean/scalar-backward');
