/**
 * Pack a minimal Linux GPU-verify bundle (no Rust toolchain, no monorepo root).
 *
 * Expects: already built `@dxo/core` dist + linux native on linux-x64.
 * Emits a **standalone** npm tree under the tarball (file: deps only — no workspace:).
 * Output: `artifacts/dxo-gpu-verify.tgz`
 */
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(import.meta.url);
const tsxCli = require.resolve('tsx/cli');
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

    // Suite list from scripts/verify/suites.ts (requiresGpu || includeInGpuArtifact).
    const listR = spawnSync(process.execPath, [tsxCli, path.join(root, 'scripts/verify/run.ts'), '--list-gpu-artifact'], {
        cwd: root,
        encoding: 'utf8',
    });
    if (listR.status !== 0) {
        console.error(listR.stderr || listR.stdout || 'failed to read gpu artifact suites from registry');
        process.exit(listR.status ?? 1);
    }
    const verifyScripts = listR.stdout
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean);
    if (!verifyScripts.length) {
        console.error('pack-gpu-artifact: registry returned no GPU suites');
        process.exit(1);
    }
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

    const defaultsJson = JSON.stringify(verifyScripts);
    const dxoCommit =
        process.env.GITHUB_SHA || spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).stdout?.trim() || 'unknown';
    writeFileSync(
        path.join(stage, 'evidence-manifest.json'),
        JSON.stringify(
            {
                schema: 'dxo-gpu-evidence/v0',
                packedAt: new Date().toISOString(),
                dxoCommit,
                githubRef: process.env.GITHUB_REF ?? null,
                githubRunId: process.env.GITHUB_RUN_ID ?? null,
                suites: verifyScripts,
                requireCuda: true,
                notes: 'Fill driver/capability/realDevice on Modal runner after nvidia-smi + probe.',
            },
            null,
            2,
        ),
    );
    writeFileSync(
        path.join(stage, 'run-verifies.mjs'),
        `#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const tests = process.argv.slice(2);
const defaults = ${defaultsJson};
const list = (tests.length ? tests : defaults).filter((n) => existsSync(path.join(root, 'scripts', n)));
if (!list.length) {
  console.error('no verify scripts found');
  process.exit(1);
}

const results = [];
for (const name of list) {
  console.log('---', name);
  const started = Date.now();
  const r = spawnSync(process.execPath, ['--import', 'tsx', path.join(root, 'scripts', name)], {
    cwd: root,
    stdio: 'inherit',
    env: { ...process.env, DXO_REQUIRE_CUDA: '1' },
  });
  const status = r.status ?? 1;
  results.push({ suite: name, status, ms: Date.now() - started, skipped: false });
  if (status !== 0) {
    writeEvidence(results, false);
    process.exit(status);
  }
}
writeEvidence(results, true);
console.log('gpu-verify bundle ok');

function writeEvidence(suiteResults, ok) {
  const manifestPath = path.join(root, 'evidence-manifest.json');
  let base = {};
  try { base = JSON.parse(readFileSync(manifestPath, 'utf8')); } catch {}
  let fingerprint = null;
  try {
    const probe = spawnSync(process.execPath, ['--import', 'tsx', '-e',
      "import { cudaAvailable, cudaCapabilityFingerprint } from '@dxo/core'; console.log(cudaAvailable() ? cudaCapabilityFingerprint() : '')"],
      { cwd: root, encoding: 'utf8', env: { ...process.env, DXO_REQUIRE_CUDA: '1' } });
    fingerprint = (probe.stdout || '').trim() || null;
  } catch {}
  const out = {
    ...base,
    completedAt: new Date().toISOString(),
    ok,
    realDevice: true,
    capabilityFingerprint: fingerprint,
    suiteResults,
  };
  writeFileSync(path.join(root, 'evidence-result.json'), JSON.stringify(out, null, 2));
  console.log('wrote evidence-result.json');
}
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
