/**
 * Unified verify runner.
 *
 * Usage:
 *   pnpm verify -- <id>
 *   pnpm verify --group <group>
 *   pnpm verify --ci          # ordinary host CI groups (cpu + contract + product)
 *   pnpm verify               # same as --ci
 *   pnpm verify --list-gpu-artifact  # basenames for Modal packer
 */
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
    CI_GROUPS,
    SUITE_BY_ID,
    SUITES,
    type SuiteDef,
    type SuiteGroup,
    selectCiSuites,
    selectGpuArtifactSuites,
    selectGroupSuites,
} from './suites.ts';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(import.meta.url);
const tsxCli = require.resolve('tsx/cli');

function usage(exitCode: number): never {
    const ids = SUITES.map((s) => s.id).join('|');
    const groups = [...new Set(SUITES.map((s) => s.group))].join('|');
    console.error(`usage:
  pnpm verify -- <${ids}>
  pnpm verify --group <${groups}>
  pnpm verify --ci
  pnpm verify --list-gpu-artifact`);
    process.exit(exitCode);
}

function parseArgs(argv: string[]): { mode: 'id' | 'group' | 'ci' | 'list-gpu-artifact'; value?: string } {
    const args = argv.filter((a) => a !== '--');
    if (args.length === 0 || args[0] === '--ci' || args[0] === 'ci') {
        return { mode: 'ci' };
    }
    if (args[0] === '--list-gpu-artifact' || args[0] === 'list-gpu-artifact') {
        return { mode: 'list-gpu-artifact' };
    }
    if (args[0] === '--group' || args[0] === 'group') {
        const group = args[1];
        if (!group) usage(1);
        return { mode: 'group', value: group };
    }
    if (args[0]?.startsWith('-')) {
        usage(1);
    }
    return { mode: 'id', value: args[0] };
}

function runSuite(suite: SuiteDef): void {
    const script = path.join(root, suite.script);
    console.log(`--- verify ${suite.id} (${suite.group})`);
    const r = spawnSync(process.execPath, [tsxCli, script], {
        cwd: root,
        stdio: 'inherit',
        env: process.env,
        timeout: suite.timeoutMs,
    });
    if (r.error) {
        console.error(`verify ${suite.id} failed: ${r.error.message}`);
        process.exit(1);
    }
    if (r.signal) {
        console.error(`verify ${suite.id} killed by signal ${r.signal}`);
        process.exit(1);
    }
    if ((r.status ?? 1) !== 0) {
        process.exit(r.status ?? 1);
    }
}

function runAll(suites: SuiteDef[]): void {
    if (!suites.length) {
        console.error('no suites selected');
        process.exit(1);
    }
    for (const suite of suites) {
        runSuite(suite);
    }
    console.log(`verify ok: ${suites.length} suite(s)`);
}

const parsed = parseArgs(process.argv.slice(2));

if (parsed.mode === 'list-gpu-artifact') {
    for (const s of selectGpuArtifactSuites()) {
        const base = s.script.split(/[\\/]/).pop();
        if (base) console.log(base);
    }
    process.exit(0);
}

if (parsed.mode === 'ci') {
    const suites = selectCiSuites();
    console.log(`verify --ci: groups=${CI_GROUPS.join(',')} count=${suites.length}`);
    runAll(suites);
} else if (parsed.mode === 'group') {
    const group = parsed.value as SuiteGroup;
    const known = new Set(SUITES.map((s) => s.group));
    if (!known.has(group)) {
        console.error(`unknown group: ${group}`);
        usage(1);
    }
    runAll(selectGroupSuites(group));
} else {
    const id = parsed.value!;
    const suite = SUITE_BY_ID.get(id);
    if (!suite) {
        console.error(`unknown suite: ${id}`);
        usage(1);
    }
    runSuite(suite);
    console.log(`verify ok: ${suite.id}`);
}
