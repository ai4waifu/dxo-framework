import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const pkgPath = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../package.json');

type Pkg = { name: string; version: string };

export function cliPackage(): Pkg {
    return require(pkgPath) as Pkg;
}
