/**
 * hub-provider: local digest stable; offline cache reuse; HF optional when network allows.
 */
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHub } from '@dxo/hub';

const root = await mkdtemp(path.join(tmpdir(), 'dxo-hub-'));
const cache = await mkdtemp(path.join(tmpdir(), 'dxo-hub-cache-'));
process.env.DXO_HUB_CACHE = cache;

try {
    const modelDir = path.join(root, 'model');
    await mkdir(modelDir);
    await writeFile(path.join(modelDir, 'config.json'), '{"arch":"tiny"}\n');
    await writeFile(path.join(modelDir, 'weights.bin'), Buffer.from([1, 2, 3, 4]));

    const hub = createHub();
    const uri = `local:${modelDir.replace(/\\/g, '/')}`;
    const a = await hub.model(uri, { files: ['config.json', 'weights.bin'] });
    const b = await hub.model(uri, { files: ['weights.bin', 'config.json'] });
    assert.equal(a.provider, 'local');
    assert.equal(a.digest, b.digest);
    assert.equal(a.digest.length, 64);
    assert.ok(a.files['config.json']);
    assert.ok(a.files['weights.bin']);

    // Seed a fake HF cache entry and resolve offline (no network).
    const hfRoot = path.join(cache, 'hf', 'org--tiny', 'main');
    await mkdir(hfRoot, { recursive: true });
    await writeFile(path.join(hfRoot, 'config.json'), '{"id":"tiny"}\n');
    const offline = await hub.model('hf:org/tiny', {
        files: ['config.json'],
        cache: 'offline',
        revision: 'main',
    });
    assert.equal(offline.provider, 'hf');
    assert.equal(offline.digest.length, 64);
    assert.equal(offline.localPath, hfRoot);

    const again = await hub.model('hf:org/tiny', {
        files: ['config.json'],
        cache: 'offline',
        revision: 'main',
    });
    assert.equal(again.digest, offline.digest);

    console.log(`hub-provider ok: digest=${a.digest.slice(0, 12)}… offline=${offline.digest.slice(0, 12)}…`);
} finally {
    await rm(root, { recursive: true, force: true });
    await rm(cache, { recursive: true, force: true });
}
