import assert from 'node:assert/strict';
import { type Batch, batch, batchAsync, dataLoader, dataset } from '@dxo/data';

const samples = [
    { x: [1, 0], xShape: [2] as number[], y: [1], yShape: [1] as number[] },
    { x: [0, 1], xShape: [2], y: [0], yShape: [1] },
    { x: [1, 1], xShape: [2], y: [1], yShape: [1] },
];

const ds = dataset(samples);
const batches = [...batch(ds, { batchSize: 2 })];
assert.equal(batches.length, 2);
assert.deepEqual([...batches[0]!.x.shape], [2, 2]);
assert.deepEqual(await batches[0]!.x.toArray(), [1, 0, 0, 1]);
assert.deepEqual([...batches[0]!.y!.shape], [2, 1]);
assert.deepEqual(await batches[0]!.y!.toArray(), [1, 0]);
assert.deepEqual([...batches[1]!.x.shape], [1, 2]);

const dropped = [...batch(ds, { batchSize: 2, dropLast: true })];
assert.equal(dropped.length, 1);

async function* asyncSamples() {
    for (const s of samples) yield s;
}

const asyncBatches: Batch[] = [];
for await (const b of batchAsync(asyncSamples(), { batchSize: 3 })) {
    asyncBatches.push(b);
}
assert.equal(asyncBatches.length, 1);
assert.deepEqual([...asyncBatches[0]!.x.shape], [3, 2]);

const viaLoader = [...dataLoader(ds, { batchSize: 2, dropLast: true })];
assert.equal(viaLoader.length, 1);

assert.throws(() => [...batch(dataset([{ x: [1], xShape: [2] }]), { batchSize: 1 })]);

console.log('data-iter ok: sync/async batch + DataLoader alias');
