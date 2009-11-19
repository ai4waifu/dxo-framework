/**
 * Orchestrate Modal T4 Sandbox: upload prebuilt GPU-verify bundle, run verifies, terminate.
 *
 * Runs on a cheap CPU host (GitHub Actions). GPU wall-clock is only Sandbox lifetime.
 *
 * Env:
 *   MODAL_TOKEN_ID / MODAL_TOKEN_SECRET — required
 *   DXO_MODAL_GPU — default "T4"
 *   DXO_MODAL_APP — default "dxo-gpu-ci"
 *
 * Usage:
 *   node scripts/ci/modal-gpu-verify.mjs [--artifact artifacts/dxo-gpu-verify.tgz]
 */
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ModalClient } from 'modal';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

function argValue(flag, fallback) {
    const i = process.argv.indexOf(flag);
    if (i >= 0 && process.argv[i + 1]) return process.argv[i + 1];
    return fallback;
}

const artifact = path.resolve(
    argValue('--artifact', path.join(root, 'artifacts/dxo-gpu-verify.tgz')),
);
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
// Node 22 + Modal host NVIDIA driver (libcuda). No Rust compile on GPU.
const image = modal.images.fromRegistry('node:22-bookworm');

console.log(`modal-gpu-verify: creating Sandbox gpu=${gpu} app=${appName}`);
const sb = await modal.sandboxes.create(app, image, {
    gpu,
    timeoutMs: 20 * 60 * 1000,
    memoryMiB: 4096,
    cpu: 2,
});

try {
    console.log(`modal-gpu-verify: sandbox=${sb.sandboxId}`);
    await execChecked(sb, ['nvidia-smi']);

    console.log(`modal-gpu-verify: upload ${artifact}`);
    await sb.filesystem.copyFromLocal(artifact, remoteTar);
    await execChecked(sb, [
        'bash',
        '-lc',
        `rm -rf ${remoteDir} && mkdir -p ${remoteDir} && tar -xzf ${remoteTar} -C ${remoteDir}`,
    ]);

    await execChecked(sb, ['bash', '-lc', `cd ${remoteDir} && node run-verifies.mjs`]);

    console.log('modal-gpu-verify: ok');
} finally {
    await sb.terminate();
    console.log('modal-gpu-verify: sandbox terminated');
}
