import { type ChildProcess, spawn } from 'node:child_process';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInspectApiServer, type InspectApiServer, type InspectApiServerOptions } from './api-server.js';

export type StartStudioOptions = InspectApiServerOptions & {
    /** Start VMZ watch serve for browser WebUI. Default true. */
    webui?: boolean;
    /** VMZ dev port; omit to auto-pick from 5173. */
    webuiPort?: number;
};

export type StudioProcess = InspectApiServer & {
    webui?: ChildProcess;
    webuiUrl?: string;
    closeAll: () => Promise<void>;
};

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function spawnVmzDev(apiUrl: string, port?: number): ChildProcess {
    const env = {
        ...process.env,
        DXO_STUDIO_API: apiUrl,
        ...(port ? { PORT: String(port), VMZ_PORT: String(port) } : {}),
    };
    const isWin = process.platform === 'win32';
    const cmd = isWin ? 'pnpm.cmd' : 'pnpm';
    const args = ['exec', 'vmz', 'dev', '.'];
    if (port) {
        args.push('--port', String(port));
    }
    return spawn(cmd, args, {
        cwd: packageRoot,
        env,
        stdio: 'inherit',
        shell: isWin,
    });
}

async function writeStudioApiBootstrap(apiUrl: string): Promise<void> {
    const out = path.join(packageRoot, 'public', 'dxo-studio-api.js');
    await writeFile(out, `window.__DXO_STUDIO_API__=${JSON.stringify(apiUrl)};\n`, 'utf8');
}

/**
 * `dxo studio`: Rust inspect API + VMZ watch serve → open in browser.
 * Desktop GUI is `dxo-studio.exe` (Tauri), not this path.
 */
export async function startStudio(options: StartStudioOptions = {}): Promise<StudioProcess> {
    const api = await createInspectApiServer({
        host: options.host ?? '127.0.0.1',
        port: options.port ?? 4310,
        runsRoot: options.runsRoot,
    });

    const webuiEnabled = options.webui !== false;
    let webui: ChildProcess | undefined;
    let webuiUrl: string | undefined;

    if (webuiEnabled) {
        await writeStudioApiBootstrap(api.url);
        webui = spawnVmzDev(api.url, options.webuiPort);
        webuiUrl = options.webuiPort
            ? `http://127.0.0.1:${options.webuiPort}`
            : 'http://127.0.0.1:5173 (VMZ default; check terminal if port differs)';
    }

    const closeAll = async (): Promise<void> => {
        if (webui && !webui.killed) {
            webui.kill();
        }
        await api.close();
    };

    return { ...api, webui, webuiUrl, closeAll };
}

/** @deprecated Use {@link startStudio}. */
export function createStudio(options: StartStudioOptions = {}): { kind: 'launcher'; options: StartStudioOptions } {
    return { kind: 'launcher', options };
}

export function studioVersion(): string {
    return 'dxo-studio@0.1.0-preview';
}
