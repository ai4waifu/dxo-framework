/**
 * Pack a minimal Linux GPU-verify bundle (no Rust toolchain, no monorepo root).
 *
 * Expects: already ran `pnpm build:ts` + `pnpm build:native` on linux-x64.
 * Output: `artifacts/dxo-gpu-verify.tgz`
 */
import { spawnSync } from 'node:child_process';
import {
    cpSync,
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const outDir = path.join(root, 'artifacts');
const outTar = path.join(outDir, 'dxo-gpu-verify.tgz');

if (process.platform !== 'linux' || process.arch !== 'x64') {
    console.error(`pack-gpu-artifact: must run on linux-x64 (got ${process.platform}/${process.arch})`);
    process.exit(1);
}

const corePkg = path.join(root, 'projects/runtimes/dxo-core');
const linuxPkg = path.join(root, 'projects/runtimes/dxo-linux-x64');
const nodeFile = path.join(linuxPkg, 'dxo.linux-x64-gnu.node');

if (!existsSync(path.join(corePkg, 'dist/index.js'))) {
    console.error('pack-gpu-artifact: missing @dxo/core dist — run pnpm build:ts');
    process.exit(1);
}
if (!existsSync(nodeFile)) {
    console.error('pack-gpu-artifact: missing linux native addon — run pnpm build:native');
    process.exit(1);
}

const stage = mkdtempSync(path.join(tmpdir(), 'dxo-gpu-pack-'));
try {
    const vendorCore = path.join(stage, 'vendor/dxo-core');
    const vendorLinux = path.join(stage, 'vendor/dxo-linux-x64');
    mkdirSync(vendorCore, { recursive: true });
    mkdirSync(vendorLinux, { recursive: true });
    mkdirSync(path.join(stage, 'scripts'), { recursive: true });

    cpSync(path.join(corePkg, 'dist'), path.join(vendorCore, 'dist'), { recursive: true });
    writeFileSync(
        path.join(vendorCore, 'package.json'),
        JSON.stringify(
            {
                name: '@dxo/core',
                version: '0.1.0',
                type: 'module',
                main: './dist/index.js',
                exports: { '.': { default: './dist/index.js' } },
                optionalDependencies: { '@dxo/dxo-linux-x64': '*' },
            },
            null,
            2,
        ),
    );

    cpSync(nodeFile, path.join(vendorLinux, 'dxo.linux-x64-gnu.node'));
    writeFileSync(
        path.join(vendorLinux, 'package.json'),
        JSON.stringify(
            {
                name: '@dxo/dxo-linux-x64',
                version: '0.1.0',
                os: ['linux'],
                cpu: ['x64'],
                main: 'dxo.linux-x64-gnu.node',
            },
            null,
            2,
        ),
    );

    const verifyScripts = ['gpu-matmul.ts', 'titan-event-dep.ts'];
    for (const name of verifyScripts) {
        const src = path.join(root, 'scripts/test', name);
        if (existsSync(src)) cpSync(src, path.join(stage, 'scripts', name));
    }

    writeFileSync(
        path.join(stage, 'package.json'),
        JSON.stringify(
            {
                name: 'dxo-gpu-verify-bundle',
                private: true,
                type: 'module',
                dependencies: {
                    '@dxo/core': 'file:./vendor/dxo-core',
                    '@dxo/dxo-linux-x64': 'file:./vendor/dxo-linux-x64',
                    tsx: '4.23.12',
                },
            },
            null,
            2,
        ),
    );

    writeFileSync(
        path.join(stage, 'run-verifies.mjs'),
        `#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const tests = process.argv.slice(2);
const defaults = ['gpu-matmul.ts', 'titan-event-dep.ts'];
const list = (tests.length ? tests : defaults).filter((n) => existsSync(path.join(root, 'scripts', n)));
if (!list.length) {
  console.error('no verify scripts found');
  process.exit(1);
}
for (const name of list) {
  console.log('---', name);
  const r = spawnSync(process.execPath, ['--import', 'tsx', path.join(root, 'scripts', name)], {
    cwd: root,
    stdio: 'inherit',
    env: { ...process.env, DXO_REQUIRE_CUDA: '1' },
  });
  if ((r.status ?? 1) !== 0) process.exit(r.status ?? 1);
}
console.log('gpu-verify bundle ok');
`,
    );

    console.log('pack-gpu-artifact: npm install (CPU)…');
    const npm = spawnSync('npm', ['install', '--omit=dev', '--no-fund', '--no-audit'], {
        cwd: stage,
        stdio: 'inherit',
        shell: process.platform === 'win32',
    });
    if (npm.status !== 0) process.exit(npm.status ?? 1);

    mkdirSync(outDir, { recursive: true });
    if (existsSync(outTar)) rmSync(outTar);

    const tar = spawnSync('tar', ['-czf', outTar, '-C', stage, '.'], { stdio: 'inherit' });
    if (tar.status !== 0) process.exit(tar.status ?? 1);

    const bytes = readFileSync(outTar).byteLength;
    console.log(`pack-gpu-artifact: wrote ${outTar} (${(bytes / (1024 * 1024)).toFixed(2)} MiB)`);
} finally {
    rmSync(stage, { recursive: true, force: true });
}
