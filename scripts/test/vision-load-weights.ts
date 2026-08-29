/**
 * vision-load-weights: DXO-key safetensors roundtrip via loadWeights (GPU verify lane).
 * Does not translate torchvision names — convert offline in external scripts.
 */
import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { encodeSafetensors } from '@dxo/serialize';
import { loadWeights, ResNet, VisionError } from '@dxo/vision';

const src = new ResNet({ depth: 18, trainable: false });
const names = src.parameterNames();
assert.ok(names.length > 0);
assert.ok(!names.some((k) => /(^|[.])fc([.]|$)/i.test(k)));

const st = await src.state();
const bytes = encodeSafetensors(st);

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
    const dir = await mkdtemp(join(tmpdir(), 'dxo-vision-lw-'));
    const file = join(dir, 'resnet18.safetensors');
    const withFc = {
        ...st,
        'fc.weight': { shape: [1000, 512], data: new Array(1000 * 512).fill(0) },
        'fc.bias': { shape: [1000], data: new Array(1000).fill(0) },
    };
    await writeFile(file, encodeSafetensors(withFc));
    const dst = new ResNet({ depth: 18, trainable: false });
    await loadWeights(dst, { path: file, scope: 'backbone' });
    assert.deepEqual((await dst.state())['stem.conv.weight']!.shape, st['stem.conv.weight']!.shape);

    await assert.rejects(
        () => loadWeights(new ResNet({ depth: 18, trainable: false }), { path: file, scope: 'all' }),
        (err: unknown) => err instanceof VisionError && err.code === 'UNKNOWN_WEIGHT_KEY',
    );
}

console.log(`vision-load-weights ok: keys=${names.length}`);
