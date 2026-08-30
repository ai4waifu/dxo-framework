#!/usr/bin/env node
/**
 * Orchestrate Modal T4 Sandbox from a **packed** npm install (no pnpm workspace:).
 *
 * Env: MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, DXO_MODAL_GPU, DXO_MODAL_APP
 * Usage: node run.mjs --artifact /path/to/dxo-gpu-verify.tgz
 */
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ModalClient } from 'modal';

function argValue(flag, fallback) {
    const i = process.argv.indexOf(flag);
    if (i >= 0 && process.argv[i + 1]) return process.argv[i + 1];
    return fallback;
}

const here = path.dirname(fileURLToPath(import.meta.url));
const artifact = path.resolve(argValue('--artifact', path.join(process.cwd(), 'artifacts/dxo-gpu-verify.tgz')));
const gpu = process.env.DXO_MODAL_GPU?.trim() || 'T4';
const appName = process.env.DXO_MODAL_APP?.trim() || 'dxo-gpu-ci';
const remoteDir = '/tmp/dxo-gpu-verify';
const remoteTar = '/tmp/dxo-gpu-verify.tgz';

if (!process.env.MODAL_TOKEN_ID || !process.env.MODAL_TOKEN_SECRET) {
    console.error('modal-gpu-verify: set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET');
    process.exit(1);
}
if (!existsSync(artifact)) {
    console.error(`modal-gpu-verify: artifact not found: ${artifact}`);
    process.exit(1);
}

async function execChecked(sb, command, params = {}) {
    console.log('$', command.join(' '));
    const proc = await sb.exec(command, params);
    const stdout = await proc.stdout.readText();
    const stderr = await proc.stderr.readText();
    if (stdout) process.stdout.write(stdout);
    if (stderr) process.stderr.write(stderr);
    const code = await proc.wait();
    if (code !== 0) {
        throw new Error(`command failed (${code}): ${command.join(' ')}`);
    }
    return { stdout, stderr, code };
}

const modal = new ModalClient();
const app = await modal.apps.fromName(appName, { createIfMissing: true });
const image = modal.images.fromRegistry('node:22-bookworm');

console.log(`modal-gpu-verify: runner=${here} gpu=${gpu} app=${appName}`);
const sb = await modal.sandboxes.create(app, image, {
    gpu,
    timeoutMs: 20 * 60 * 1000,
    memoryMiB: 4096,
    cpu: 2,
});

try {
    console.log(`modal-gpu-verify: sandbox=${sb.sandboxId}`);
    const smi = await execChecked(sb, ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']);

    console.log(`modal-gpu-verify: upload ${artifact}`);
    await sb.filesystem.copyFromLocal(artifact, remoteTar);
    await execChecked(sb, ['bash', '-lc', `rm -rf ${remoteDir} && mkdir -p ${remoteDir} && tar -xzf ${remoteTar} -C ${remoteDir}`]);

    // Stamp driver evidence into the packed manifest before verifies run.
    await execChecked(sb, [
        'bash',
        '-lc',
        `cd ${remoteDir} && node -e ${JSON.stringify(
            `const fs=require('fs');const p='evidence-manifest.json';const m=JSON.parse(fs.readFileSync(p,'utf8'));m.gpu=${JSON.stringify(gpu)};m.nvidiaSmi=${JSON.stringify(smi.stdout.trim())};m.modalSandboxId=${JSON.stringify(sb.sandboxId)};fs.writeFileSync(p,JSON.stringify(m,null,2));`,
        )}`,
    ]);

    await execChecked(sb, ['bash', '-lc', `cd ${remoteDir} && DXO_REQUIRE_CUDA=1 node run-verifies.mjs`]);

    // Best-effort: print evidence result for GHA logs.
    await execChecked(sb, ['bash', '-lc', `cd ${remoteDir} && test -f evidence-result.json && cat evidence-result.json || true`]);

    console.log('modal-gpu-verify: ok');
} finally {
    await sb.terminate();
    console.log('modal-gpu-verify: sandbox terminated');
}
