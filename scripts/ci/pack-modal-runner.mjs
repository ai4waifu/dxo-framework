/**
 * Pack an installable Modal GPU runner into artifacts/dxo-modal-runner/
 * (real npm deps — no pnpm workspace: protocol).
 *
 * Output: artifacts/dxo-modal-runner/{package.json,run.mjs,node_modules/...}
 */
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const src = path.join(root, 'scripts/ci/modal-runner');
const out = path.join(root, 'artifacts/dxo-modal-runner');

if (!existsSync(path.join(src, 'package.json')) || !existsSync(path.join(src, 'run.mjs'))) {
    console.error('pack-modal-runner: missing scripts/ci/modal-runner/{package.json,run.mjs}');
    process.exit(1);
}

mkdirSync(path.join(root, 'artifacts'), { recursive: true });
if (existsSync(out)) rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

cpSync(path.join(src, 'package.json'), path.join(out, 'package.json'));
cpSync(path.join(src, 'run.mjs'), path.join(out, 'run.mjs'));

console.log('pack-modal-runner: npm install (isolated package)…');
const npm = spawnSync('npm', ['install', '--omit=dev', '--no-fund', '--no-audit'], {
    cwd: out,
    stdio: 'inherit',
    shell: process.platform === 'win32',
});
if (npm.status !== 0) process.exit(npm.status ?? 1);

if (!existsSync(path.join(out, 'node_modules/modal'))) {
    console.error('pack-modal-runner: modal not installed under artifacts/dxo-modal-runner');
    process.exit(1);
}

const pkg = JSON.parse(readFileSync(path.join(out, 'package.json'), 'utf8'));
writeFileSync(
    path.join(out, '.packed.json'),
    JSON.stringify({ name: pkg.name, packedAt: new Date().toISOString(), modal: '0.10.0' }, null, 2),
);

console.log(`pack-modal-runner: ready ${out}`);
