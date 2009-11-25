/**
 * Thin entry: pack Modal runner to artifacts/ then exec it.
 * Prefer CI calling `artifacts/dxo-modal-runner/run.mjs` after `pack-modal-runner.mjs`.
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const pack = path.join(root, 'scripts/ci/pack-modal-runner.mjs');
const run = path.join(root, 'artifacts/dxo-modal-runner/run.mjs');

const packed = spawnSync(process.execPath, [pack], { cwd: root, stdio: 'inherit' });
if ((packed.status ?? 1) !== 0) process.exit(packed.status ?? 1);

const r = spawnSync(process.execPath, [run, ...process.argv.slice(2)], {
    cwd: root,
    stdio: 'inherit',
    env: process.env,
});
process.exit(r.status ?? 1);
