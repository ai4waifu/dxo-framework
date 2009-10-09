import assert from 'node:assert/strict';
import { batch, dataset } from '@dxo/data';
import { Linear } from '@dxo/nn';
import { SGD } from '@dxo/optimizer';
import { decodeLinearState, STATE_FORMAT, STATE_VERSION } from '@dxo/serialize';
import { type TrainEvent, Trainer } from '@dxo/train';

/**
 * G5 / trainer-loop: fitIter + AbortSignal + checkpoint round-trip on a
 * linearly separable MSE probe (same target family as mnist-linear).
 */
function makeSamples(n: number) {
    const samples = [];
    for (let i = 0; i < n; i++) {
        const a = (i % 10) / 10;
        const b = ((i * 3) % 10) / 10;
        samples.push({
            x: [a, b],
            xShape: [2] as number[],
            y: [2 * a - b],
            yShape: [1] as number[],
        });
    }
    return samples;
}

const samples = makeSamples(32);
const model = new Linear(2, 1);
model.loadState({
    weight: { shape: [2, 1], data: [0.1, -0.1] },
    bias: { shape: [1], data: [0] },
});

const epochs = 20;
const trainer = new Trainer({
    model,
    optimizer: new SGD(0.1),
    epochs,
    checkpointEvery: 10,
    batches: () => batch(dataset(samples), { batchSize: 8 }),
});

const events: TrainEvent[] = [];
const epochMeans: number[] = [];

for await (const event of trainer.fitIter()) {
    events.push(event);
    if (event.type === 'epoch_end') epochMeans.push(event.meanLoss);
}

const loss0 = epochMeans[0];
const loss1 = epochMeans[epochMeans.length - 1];
assert.ok(loss0 !== undefined && loss0 > 1e-4, `initial mean loss too small: ${loss0}`);
assert.ok(loss1 !== undefined && Number.isFinite(loss1), `final mean loss bad: ${loss1}`);

const done = events.find((e) => e.type === 'done');
assert.ok(done && done.type === 'done');
assert.equal(done.epochs, epochs);
assert.ok(done.steps > 0);
assert.ok(loss1! < loss0! * 0.5, `mean loss did not drop: ${loss0} -> ${loss1}`);

const checkpoints = events.filter((e) => e.type === 'checkpoint');
assert.equal(checkpoints.length, 2); // epochs 10 and 20
const lastCk = checkpoints[checkpoints.length - 1]!;
assert.ok(lastCk.type === 'checkpoint');
assert.equal(lastCk.document.format, STATE_FORMAT);
assert.equal(lastCk.document.version, STATE_VERSION);

const restored = new Linear(2, 1, { requiresGrad: false });
restored.loadState(decodeLinearState(lastCk.document));
assert.deepEqual(await restored.weight.toArray(), await model.weight.toArray());
assert.deepEqual(await restored.bias.toArray(), await model.bias.toArray());

// Abort mid-run
const model2 = new Linear(2, 1);
model2.loadState({
    weight: { shape: [2, 1], data: [0.1, -0.1] },
    bias: { shape: [1], data: [0] },
});
const ac = new AbortController();
const trainer2 = new Trainer({
    model: model2,
    optimizer: new SGD(0.1),
    epochs: 10,
    batches: () => batch(dataset(samples), { batchSize: 8 }),
});
let sawAbort = false;
let batchCount = 0;
for await (const event of trainer2.fitIter({ signal: ac.signal })) {
    if (event.type === 'batch') {
        batchCount += 1;
        if (batchCount === 2) ac.abort();
    }
    if (event.type === 'aborted') {
        sawAbort = true;
        assert.equal(event.reason, 'signal');
    }
}
assert.ok(sawAbort, 'expected aborted event');

const summary = await new Trainer({
    model: new Linear(2, 1),
    optimizer: new SGD(0.05),
    epochs: 1,
    batches: () => batch(dataset(samples), { batchSize: 16 }),
}).fit();
assert.equal(summary.aborted, false);
assert.equal(summary.epochs, 1);
assert.ok(summary.lastCheckpoint);

console.log(`trainer-loop ok: mean ${loss0!.toFixed(4)} -> ${loss1!.toFixed(4)}; abort@${batchCount}; ckpts=${checkpoints.length}`);
