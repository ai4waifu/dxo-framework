import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const id = process.argv[2];
const require = createRequire(import.meta.url);

const suites = {
    smoke: 'scripts/test/smoke.ts',
    'tensor-cpu': 'scripts/test/tensor-cpu.ts',
    'autograd-fd': 'scripts/test/autograd-fd.ts',
    'nn-forward': 'scripts/test/nn-forward.ts',
    'mnist-linear': 'scripts/test/mnist-linear.ts',
};

if (!id || !(id in suites)) {
    console.error(`usage: pnpm verify -- <${Object.keys(suites).join('|')}>`);
    process.exit(1);
}

const script = path.join(root, suites[id]);
const tsxCli = require.resolve('tsx/cli');
const r = spawnSync(process.execPath, [tsxCli, script], {
    cwd: root,
    stdio: 'inherit',
    env: process.env,
});
process.exit(r.status ?? 1);
