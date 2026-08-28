/**
 * Backward-compatible entry: `node scripts/verify.mjs …` → tsx scripts/verify/run.ts
 */
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);
const tsxCli = require.resolve('tsx/cli');
const runTs = path.join(root, 'scripts/verify/run.ts');

const r = spawnSync(process.execPath, [tsxCli, runTs, ...process.argv.slice(2)], {
    cwd: root,
    stdio: 'inherit',
    env: process.env,
});
process.exit(r.status ?? 1);
