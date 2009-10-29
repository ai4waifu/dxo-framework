#!/usr/bin/env node
import { version } from '@dxo/core';

const args = process.argv.slice(2);
const cmd = args[0] ?? 'help';

function printHelp() {
    console.log(`dxo ${version()}

Usage:
  dxo version
  dxo studio [--port <apiPort>] [--webui-port <port>] [--runs-dir <path>] [--api-only]

  studio  VMZ watch serve + loopback inspect API (browser).
          Desktop GUI: double-click dxo-studio.exe (Tauri).
          API default: http://127.0.0.1:4310
          WebUI default: http://127.0.0.1:5173 (VMZ dev)
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
    const webuiPort = flagValue('--webui-port') ? Number(flagValue('--webui-port')) : undefined;
    const legacyUiPort = flagValue('--ui-port') ? Number(flagValue('--ui-port')) : undefined;
    const runsDir = flagValue('--runs-dir');
    const apiOnly = args.includes('--api-only');

    const studio = await startStudio({
        host: '127.0.0.1',
        port,
        runsRoot: runsDir,
        webui: !apiOnly,
        webuiPort: webuiPort ?? legacyUiPort,
    });

    console.log(`dxo studio API ${studio.url}`);
    console.log(`runs root: ${studio.runsRoot}`);
    if (studio.webuiUrl) {
        console.log(`WebUI (VMZ watch): ${studio.webuiUrl}`);
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
