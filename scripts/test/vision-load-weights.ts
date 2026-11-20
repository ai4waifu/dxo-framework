/**
 * vision-load-weights: DXO-key safetensors roundtrip via loadWeights (GPU verify lane).
 * Does not translate torchvision names — convert offline in external scripts.
 */
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { encodeState } from '@dxo/serialize';
import { loadWeights, ResNet, VisionError } from '@dxo/vision';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

const src = new ResNet({ depth: 18, trainable: false });
const names = src.parameterNames();
assert.ok(names.length > 0);
assert.ok(!names.some((k) => /(^|[.])fc([.]|$)/i.test(k)));

const st = await src.state();
const bytes = encodeState(st);

// bytes path
{
    const dst = new ResNet({ depth: 18, trainable: false });
    await loadWeights(dst, { bytes, scope: 'backbone' });
    const loaded = await dst.state();
    assert.deepEqual(Object.keys(loaded).sort(), [...names].sort());
    for (const key of names) {
        assert.deepEqual(loaded[key]!.shape, st[key]!.shape, key);
        assert.deepEqual(loaded[key]!.data, st[key]!.data, key);
    }
}

// file path + ignore foreign fc.* under backbone scope
{
    const dir = await mkdtemp(path.join(tmpdir(), 'dxo-vision-lw-'));
    const file = path.join(dir, 'resnet18.safetensors');
    const withFc = {
        ...st,
        'fc.weight': { shape: [1000, 512], data: new Array(1000 * 512).fill(0) },
        'fc.bias': { shape: [1000], data: new Array(1000).fill(0) },
    };
    await writeFile(file, encodeState(withFc));
    const dst = new ResNet({ depth: 18, trainable: false });
    await loadWeights(dst, { path: file, scope: 'backbone' });
    assert.deepEqual((await dst.state())['stem.convolution.weight']!.shape, st['stem.convolution.weight']!.shape);

    await assert.rejects(
        () => loadWeights(new ResNet({ depth: 18, trainable: false }), { path: file, scope: 'all' }),
        (err: unknown) => err instanceof VisionError && err.code === 'UNKNOWN_WEIGHT_KEY',
    );
}

// `.dxo` weights 容器：与 safetensors 路径逐参数位级一致（资产不在时跳过）
{
    const dir = path.resolve(root, '../dxo-imagenet/weights/resnet18');
    const safetensors = path.join(dir, 'model.safetensors');
    const asset = path.join(dir, 'resnet18.dxo');
    if (existsSync(safetensors) && existsSync(asset)) {
        const fromSt = new ResNet({ depth: 18, trainable: false });
        await loadWeights(fromSt, { path: safetensors, scope: 'backbone' });
        const fromDxo = new ResNet({ depth: 18, trainable: false });
        await loadWeights(fromDxo, { assetPath: asset, scope: 'backbone' });
        const a = await fromSt.state();
        const b = await fromDxo.state();
        for (const key of names) {
            assert.deepEqual(b[key]!.shape, a[key]!.shape, key);
            assert.deepEqual(b[key]!.data, a[key]!.data, key);
        }
        await assert.rejects(
            () => loadWeights(new ResNet({ depth: 18, trainable: false }), { path: safetensors, assetPath: asset }),
            (err: unknown) => err instanceof VisionError && err.code === 'INVALID_LOAD_SOURCE',
        );
        console.log(`vision-load-weights dxo asset ok: ${path.basename(asset)} (${names.length} params)`);
    }
}

console.log(`vision-load-weights ok: keys=${names.length}`);
