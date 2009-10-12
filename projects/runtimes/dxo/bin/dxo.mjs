#!/usr/bin/env node
import { version } from '@dxo/core';

const args = process.argv.slice(2);
const cmd = args[0] ?? 'help';

function printHelp() {
    console.log(`dxo ${version()}

Usage:
  dxo version
  dxo studio [--port <apiPort>] [--ui-port <uiPort>] [--runs-dir <path>] [--api-only]

  studio  Start local Studio workbench (loopback).
          API default: http://127.0.0.1:4310
`);
}

function flagValue(name) {
    const i = args.indexOf(name);
    if (i < 0 || i + 1 >= args.length) return undefined;
    return args[i + 1];
}

if (cmd === 'version' || cmd === '-v' || cmd === '--version') {
    console.log(version());
    process.exit(0);
}

if (cmd === 'help' || cmd === '-h' || cmd === '--help') {
    printHelp();
    process.exit(0);
}

if (cmd === 'studio') {
    const { startStudio } = await import('@dxo/studio');
    const port = Number(flagValue('--port') ?? 4310);
    const uiPort = flagValue('--ui-port') ? Number(flagValue('--ui-port')) : undefined;
    const runsDir = flagValue('--runs-dir');
    const apiOnly = args.includes('--api-only');

    const studio = await startStudio({
        host: '127.0.0.1',
        port,
        runsRoot: runsDir,
        ui: !apiOnly,
        uiPort,
    });

    console.log(`dxo studio API ${studio.url}`);
    console.log(`runs root: ${studio.runsRoot}`);
    if (!apiOnly) {
        console.log('UI: vmz dev (projects/studio) — set window.__DXO_STUDIO_API__ if API port differs');
    }

    const shutdown = async () => {
        await studio.closeAll();
        process.exit(0);
    };
    process.on('SIGINT', () => void shutdown());
    process.on('SIGTERM', () => void shutdown());
} else {
    console.error(`unknown command: ${cmd}`);
    printHelp();
    process.exit(1);
}
