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
    'g3-contract': 'scripts/test/g3-contract.ts',
    'data-iter': 'scripts/test/data-iter.ts',
    'serialize-roundtrip': 'scripts/test/serialize-roundtrip.ts',
    'trainer-loop': 'scripts/test/trainer-loop.ts',
    'gpu-matmul': 'scripts/test/gpu-matmul.ts',
    'lite-webgpu-smoke': 'scripts/test/lite-webgpu-smoke.ts',
    'runtime-contract-lite': 'scripts/test/runtime-contract-lite.ts',
    'runtime-contract-core': 'scripts/test/runtime-contract-core.ts',
    'model-graph': 'scripts/test/model-graph.ts',
    'studio-run-smoke': 'scripts/test/studio-run-smoke.ts',
    'studio-ui-wave2': 'scripts/test/studio-ui-wave2.ts',
    'model-app-image': 'scripts/test/model-app-image.ts',
    'framework-core-transformer': 'scripts/test/framework-core-transformer.ts',
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
