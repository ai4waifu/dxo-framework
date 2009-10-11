import { spawn, type ChildProcess } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInspectApiServer, type InspectApiServer, type InspectApiServerOptions } from './api-server.js';

export type StartStudioOptions = InspectApiServerOptions & {
    /** Start the VMZ dev server for `projects/studio`. Default true. */
    ui?: boolean;
    /** VMZ dev port; 0 picks a free port when supported. */
    uiPort?: number;
};

export type StudioProcess = InspectApiServer & {
    ui?: ChildProcess;
    closeAll: () => Promise<void>;
};

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const studioAppRoot = path.resolve(packageRoot, '../../studio');

function spawnVmzDev(apiUrl: string, port?: number): ChildProcess {
    const env = {
        ...process.env,
        DXO_STUDIO_API: apiUrl,
        ...(port ? { PORT: String(port), VMZ_PORT: String(port) } : {}),
    };
    const isWin = process.platform === 'win32';
    const cmd = isWin ? 'pnpm.cmd' : 'pnpm';
    return spawn(cmd, ['exec', 'vmz', 'dev', '.'], {
        cwd: studioAppRoot,
        env,
        stdio: 'inherit',
        shell: isWin,
    });
}

/**
 * Start loopback inspect API and optionally the VMZ Studio dev app.
 * Binds API to 127.0.0.1 by default.
 */
export async function startStudio(options: StartStudioOptions = {}): Promise<StudioProcess> {
    const api = await createInspectApiServer({
        host: options.host ?? '127.0.0.1',
        port: options.port ?? 4310,
        runsRoot: options.runsRoot,
    });

    const uiEnabled = options.ui !== false;
    let ui: ChildProcess | undefined;
    if (uiEnabled) {
        ui = spawnVmzDev(api.url, options.uiPort);
    }

    const closeAll = async (): Promise<void> => {
        if (ui && !ui.killed) {
            ui.kill();
        }
        await api.close();
    };

    return { ...api, ui, closeAll };
}

/** @deprecated Use {@link startStudio}. */
export function createStudio(options: StartStudioOptions = {}): { kind: 'launcher'; options: StartStudioOptions } {
    return { kind: 'launcher', options };
}

export function studioVersion(): string {
    return 'dxo-studio@0.1.0-preview';
}
