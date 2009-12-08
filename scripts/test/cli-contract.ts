/**
 * cli-contract: help/version, exit codes, doctor --json (napi), inspect list/show (napi),
 * unsupported commands → 2, studio api-only start/stop.
 */
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { RunRecorder } from '@dxo/inspect';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const bin = path.join(root, 'projects/runtimes/dxo/bin/dxo.mjs');
const require = createRequire(import.meta.url);
// Ensure dist exists (CI builds first).
require.resolve('@dxo/dxo');

function runDxo(
    args: string[],
    opts: { env?: NodeJS.ProcessEnv; cwd?: string } = {},
): { status: number | null; stdout: string; stderr: string } {
    const r = spawnSync(process.execPath, [bin, ...args], {
        cwd: opts.cwd ?? root,
        encoding: 'utf8',
        env: { ...process.env, ...opts.env },
    });
    return {
        status: r.status,
        stdout: r.stdout ?? '',
        stderr: r.stderr ?? '',
    };
}

// --- help / version ---
{
    const help = runDxo(['--help']);
    assert.equal(help.status, 0);
    assert.match(help.stdout, /doctor|studio|inspect/i);

    const ver = runDxo(['version', '--json']);
    assert.equal(ver.status, 0);
    const body = JSON.parse(ver.stdout);
    assert.equal(body.cli.name, '@dxo/dxo');
    assert.ok(typeof body.core.version === 'string' && body.core.version.length > 0);
}

// --- unknown / unsupported → 2 ---
{
    const unknown = runDxo(['not-a-command']);
    assert.equal(unknown.status, 2);

    const runCmd = runDxo(['run', 'entry.ts']);
    assert.equal(runCmd.status, 2);
    assert.match(runCmd.stderr, /not supported/i);

    const serve = runDxo(['serve']);
    assert.equal(serve.status, 2);
}

// --- doctor --json (napi doctorReport) ---
{
    const doc = runDxo(['doctor', '--json']);
    assert.equal(doc.status, 0, doc.stderr);
    const report = JSON.parse(doc.stdout);
    assert.equal(report.ok, true);
    assert.ok(report.native.loaded);
    assert.ok(report.engine);
    assert.equal(report.engine.ok, true);
    assert.ok(typeof report.engine.version === 'string');
    assert.ok(typeof report.engine.backend === 'string');
    assert.ok(typeof report.engine.abi === 'string');
    assert.equal(typeof report.engine.cudaAvailable, 'boolean');
}

// --- inspect list/show via napi Rust store ---
const runsRoot = await mkdtemp(path.join(tmpdir(), 'dxo-cli-inspect-'));
try {
    const recorder = await RunRecorder.open({
        root: runsRoot,
        label: 'cli-contract',
        runId: 'cli-run-1',
    });
    await recorder.append({
        type: 'metric/scalar',
        runId: 'cli-run-1',
        wallTimeMs: Date.now(),
        metric: { name: 'loss', value: 0.5, step: 1, wallTimeMs: Date.now() },
    });
    await recorder.writeArtifact('ckpt.json', 'checkpoint', '{"ok":true}');
    await recorder.close('ok');

    const list = runDxo(['inspect', 'list', '--runs-dir', runsRoot, '--json']);
    assert.equal(list.status, 0, list.stderr);
    const listed = JSON.parse(list.stdout);
    assert.equal(listed.runsRoot, runsRoot);
    assert.ok(listed.runs.some((r: { runId: string }) => r.runId === 'cli-run-1'));

    const show = runDxo(['inspect', 'show', 'cli-run-1', '--runs-dir', runsRoot, '--json']);
    assert.equal(show.status, 0, show.stderr);
    const shown = JSON.parse(show.stdout);
    assert.equal(shown.meta.runId, 'cli-run-1');
    assert.equal(shown.meta.status, 'ok');
    assert.ok(shown.summary.eventCount >= 1);
    assert.ok(shown.summary.metricScalarCount >= 1);
} finally {
    await rm(runsRoot, { recursive: true, force: true });
}

// --- studio api-only smoke (napi createInspectApiServer) ---
{
    const runs = await mkdtemp(path.join(tmpdir(), 'dxo-cli-studio-'));
    await mkdir(runs, { recursive: true });
    try {
        const child = spawn(process.execPath, [bin, 'studio', '--api-only', '--port', '0', '--runs-dir', runs, '--json'], {
            cwd: root,
            env: process.env,
            stdio: ['ignore', 'pipe', 'pipe'],
        });

        let stdout = '';
        let stderr = '';
        child.stdout?.on('data', (buf: Buffer) => {
            stdout += buf.toString('utf8');
        });
        child.stderr?.on('data', (buf: Buffer) => {
            stderr += buf.toString('utf8');
        });

        const apiUrl = await new Promise<string>((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error(`studio start timeout\nstdout=${stdout}\nstderr=${stderr}`)), 20_000);
            const tryParse = () => {
                const line = stdout
                    .trim()
                    .split(/\r?\n/)
                    .find((l) => l.startsWith('{'));
                if (!line) return;
                try {
                    const body = JSON.parse(line) as { apiUrl?: string };
                    if (body.apiUrl) {
                        clearTimeout(timer);
                        resolve(body.apiUrl);
                    }
                } catch {
                    /* partial JSON */
                }
            };
            child.stdout?.on('data', tryParse);
            child.on('error', (err) => {
                clearTimeout(timer);
                reject(err);
            });
            child.on('exit', (code) => {
                clearTimeout(timer);
                reject(new Error(`studio exited early: ${code}\nstdout=${stdout}\nstderr=${stderr}`));
            });
            tryParse();
        });

        let healthOk = false;
        let lastErr: unknown;
        for (let i = 0; i < 80; i++) {
            try {
                const health = await fetch(`${apiUrl}/api/health`);
                if (health.ok) {
                    healthOk = true;
                    break;
                }
                lastErr = new Error(`health status ${health.status}`);
            } catch (err) {
                lastErr = err;
            }
            await new Promise((r) => setTimeout(r, 25));
        }
        assert.ok(healthOk, `studio health failed: ${String(lastErr)}`);

        const exited = new Promise<number | null>((resolve) => {
            child.once('exit', (code) => resolve(code));
        });
        child.kill('SIGINT');
        const code = await exited;
        // Unix: 130; some Windows hosts may surface null/1 — accept CANCEL contract when available.
        if (process.platform === 'win32') {
            assert.ok(code === 130 || code === 1 || code === 0 || code === null, `exit=${code}`);
        } else {
            assert.equal(code, 130);
        }
    } finally {
        await rm(runs, { recursive: true, force: true });
    }
}

// invalid cwd → 2
{
    const bad = runDxo(['doctor', '--cwd', path.join(tmpdir(), 'dxo-missing-cwd-xyz')]);
    assert.equal(bad.status, 2);
}

console.log('cli-contract ok');
