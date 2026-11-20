/**
 * dxo-asset: `.dxo` 容器读回验收。
 *
 * 对磁盘上存在的数据集资产逐个：napi `DxoAsset` 打开 → `verifyAll`（全分卷 / 全 chunk /
 * 全张量逻辑摘要）→ 逐张量 `readTensor` 对照索引里的 `logical_bytes` / `logical_sha256` →
 * 每个 split field 抽 3 个样本走 `readSamples`，校验定长 field 的字节数。
 *
 * 资产位置：`DXO_ASSET_PATH` 优先；否则扫描同级 `dxo-*` 数据集仓。一个都找不到时跳过（CI 无数据）。
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DxoAsset } from '@dxo/core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

const CANDIDATES = [
    '../dxo-mnist/data/mnist.dxo',
    '../dxo-cifar/data/cifar10.dxo',
    '../dxo-coco/data/coco-sample.dxo',
    '../dxo-ffhq/data/ffhq-sample.dxo',
    '../dxo-imagenet/data/imagenet-sample.dxo',
];

const ELEM_BYTES: Record<string, number> = { u8: 1, f32: 4 };

type TensorMeta = { key: string; dtype: string; shape: number[]; logical_bytes: number; logical_sha256: string };
type FieldMeta = { name: string; role: string; tensor_key: string };
type SplitMeta = { name: string; sample_count: number; fields: FieldMeta[] };
type AssetMeta = {
    kind: string;
    semantic_sha256: string;
    tensors: TensorMeta[];
    dataset?: { splits: SplitMeta[] };
};

const explicit = process.env.DXO_ASSET_PATH?.trim();
const files = explicit
    ? [path.resolve(root, explicit)]
    : CANDIDATES.map((p) => path.resolve(root, p)).filter((p) => existsSync(p));

if (files.length === 0) {
    console.log('dxo-asset: skipped（磁盘上没有 .dxo 资产；可用 DXO_ASSET_PATH 指定）');
    process.exit(0);
}

for (const file of files) {
    assert.ok(existsSync(file), `资产不存在: ${file}`);
    const started = Date.now();
    const asset = new DxoAsset(file);
    const meta = JSON.parse(asset.metadataJson()) as AssetMeta;

    assert.equal(meta.kind, 'dataset', `${file}: kind 必须为 dataset`);
    assert.match(meta.semantic_sha256, /^[0-9a-f]{64}$/, `${file}: semantic_sha256 非法`);
    assert.ok(meta.tensors.length > 0, `${file}: 张量表为空`);

    // 全量验收：分卷壳、chunk 摘要、张量逻辑摘要
    asset.verifyAll();

    let sampled = 0;
    for (const tensor of meta.tensors) {
        const bytes = asset.readTensor(tensor.key);
        assert.equal(bytes.length, tensor.logical_bytes, `${file}: ${tensor.key} 字节数与索引不符`);
        const elems = tensor.shape.reduce((acc, n) => acc * n, 1);
        const width = ELEM_BYTES[tensor.dtype];
        assert.ok(width, `${file}: ${tensor.key} 未知 dtype ${tensor.dtype}`);
        assert.equal(tensor.logical_bytes, elems * width, `${file}: ${tensor.key} shape 与字节数不自洽`);
        const digest = createHash('sha256').update(bytes).digest('hex');
        assert.equal(digest, tensor.logical_sha256, `${file}: ${tensor.key} 逻辑摘要不符`);

        const split = meta.dataset?.splits.find((s) => s.fields.some((f) => f.tensor_key === tensor.key));
        const field = split?.fields.find((f) => f.tensor_key === tensor.key);
        if (!split || !field || split.sample_count === 0) continue;
        const perSample = tensor.logical_bytes / split.sample_count;
        if (!Number.isInteger(perSample) || perSample === 0) continue;
        const count = Math.min(3, split.sample_count);
        const slice = asset.readSamples(split.name, field.name, 0, count);
        assert.equal(slice.length, count * perSample, `${file}: ${split.name}.${field.name} 抽样字节数不符`);
        sampled += count;
    }

    console.log(
        `dxo-asset ok: ${path.basename(file)} tensors=${meta.tensors.length} ` +
            `samples=${sampled} semantic=${meta.semantic_sha256.slice(0, 16)}… ${Date.now() - started}ms`,
    );
}

console.log(`dxo-asset: ${files.length} 个资产通过`);
