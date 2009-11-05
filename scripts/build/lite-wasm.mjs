import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Build dxo-lite-wasm (wasm32-unknown-unknown) into @dxo/lite-unknown-wasm32/dist.
 *
 * wasm-pack `--out-dir` is resolved relative to the **crate** directory, not cwd.
 */

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const crate = path.join(root, 'projects/compilers/dxo-lite-wasm');
const outPkg = path.join(root, 'projects/runtimes/dxo-lite-unknown-wasm32');
const dist = path.join(outPkg, 'dist');
/** Staging dir under the crate so wasm-pack out-dir stays simple. */
const staging = path.join(crate, '.wasm-pack-out');

function run(cmd, args, cwd) {
    // No shell: paths with spaces (Windows) must not be re-split by cmd.exe.
    const r = spawnSync(cmd, args, { cwd, stdio: 'inherit', shell: false });
    if (r.error) {
        console.error(r.error);
        process.exit(1);
    }
    if (r.status !== 0) {
        process.exit(r.status ?? 1);
    }
}

if (existsSync(staging)) {
    rmSync(staging, { recursive: true, force: true });
}
mkdirSync(staging, { recursive: true });
mkdirSync(dist, { recursive: true });

run(
    'wasm-pack',
    ['build', '--target', 'web', '--release', '--out-dir', '.wasm-pack-out', '--out-name', 'dxo_lite'],
    crate,
);

for (const name of readdirSync(staging)) {
    // Drop wasm-pack's generated package.json / README — we own those at package root.
    if (name === 'package.json' || name === 'README.md' || name === '.gitignore') continue;
    copyFileSync(path.join(staging, name), path.join(dist, name));
}

rmSync(staging, { recursive: true, force: true });

// Remove stale nested output from older broken out-dir paths (crate-relative mis-resolve).
const staleNested = path.join(crate, 'projects');
if (existsSync(staleNested)) {
    rmSync(staleNested, { recursive: true, force: true });
}

const stalePkgJson = path.join(dist, 'package.json');
if (existsSync(stalePkgJson)) unlinkSync(stalePkgJson);

console.log(`lite-wasm → ${dist}`);
for (const name of readdirSync(dist)) {
    console.log(`  ${name}`);
}
