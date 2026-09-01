/**
 * checkpoint-stream: Rust safetensors codec + buffer checkpoint hot path.
 */
import assert from 'node:assert/strict';
import { decodeSafetensors, encodeSafetensors, tensor } from '@dxo/core';
import { batch, dataset } from '@dxo/data';
import { FullyConnected } from '@dxo/nn';
import { Adam } from '@dxo/optimizer';
import {
    decodeCheckpoint,
    decodeState,
    decodeStateBuffers,
    encodeCheckpoint,
    encodeStateBuffers,
} from '@dxo/serialize';
import { Trainer } from '@dxo/train';

// Rust encode/decode roundtrip with metadata
{
    const entries = [
        {
            name: 'weight',
            shape: [2, 1],
            data: Buffer.from(Float32Array.from([0.1, -0.2]).buffer),
        },
    ];
    const bytes = encodeSafetensors(entries, JSON.stringify({ 'dxo.checkpoint.version': '1', epoch: '1' }));
    const decoded = decodeSafetensors(bytes);
    assert.equal(decoded.tensors.length, 1);
    assert.equal(decoded.tensors[0]!.name, 'weight');
    const meta = JSON.parse(decoded.metadataJson) as Record<string, string>;
    assert.equal(meta.epoch, '1');
}

// FullyConnected buffer path (no number[] on encode)
{
    const model = new FullyConnected(2, 1, { requiresGrad: false });
    model.loadState({
        weight: { shape: [2, 1], data: [0.25, -0.5] },
        bias: { shape: [1], data: [0.1] },
    });
    const buffers = model.stateBuffers();
    const bytes = encodeStateBuffers(buffers, { 'dxo.checkpoint.version': '1' });
    const { tensors } = decodeStateBuffers(bytes);
    const restored = new FullyConnected(2, 1, { requiresGrad: false });
    restored.loadStateFromBuffers(tensors);
    assert.deepEqual(await restored.weight.toArray(), await model.weight.toArray());
}

// Adam optimizer tensors in checkpoint
{
    const model = new FullyConnected(2, 1);
    const opt = new Adam(0.05);
    const params = model.parameters();
    const loss = model.forward(tensor([1, 2], [1, 2])).sum();
    loss.backward();
    model.loadParameters(await opt.step(params));
    const shapes = params.map((p) => [...p.shape]);
    const bytes = encodeCheckpoint({
        model: model.stateBuffers(),
        optimizer: opt.checkpointEntries(shapes),
        metadata: { 'optimizer.adam.step': String(opt.checkpointStepCount) },
    });
    const ck = decodeCheckpoint(bytes);
    assert.ok(ck.optimizer.some((e) => e.name === 'optimizer.adam.0.m'));
    const opt2 = new Adam(0.05);
    opt2.restoreFromCheckpoint(params.length, ck.optimizer);
    opt2.checkpointStepCount = Number(ck.metadata['optimizer.adam.step'] ?? 0);
    assert.equal(opt2.checkpointStepCount, opt.checkpointStepCount);
}

// Trainer emits Rust-encoded checkpoints
{
    const samples = Array.from({ length: 16 }, (_, i) => ({
        x: [(i % 5) / 5, ((i * 2) % 5) / 5],
        xShape: [2] as number[],
        y: [1],
        yShape: [1] as number[],
    }));
    const model = new FullyConnected(2, 1);
    const trainer = new Trainer({
        model,
        optimizer: new Adam(0.1),
        epochs: 2,
        checkpointEvery: 2,
        batches: () => batch(dataset(samples), { batchSize: 4 }),
    });
    let ckBytes: Uint8Array | undefined;
    for await (const event of trainer.fitIter()) {
        if (event.type === 'checkpoint') ckBytes = event.bytes;
    }
    assert.ok(ckBytes);
    const state = decodeState(ckBytes!);
    const restored = new FullyConnected(2, 1, { requiresGrad: false });
    restored.loadState(state);
    assert.ok((await restored.weight.toArray()).length === 2);
}

console.log('checkpoint-stream ok: Rust safetensors + buffer checkpoint path');
