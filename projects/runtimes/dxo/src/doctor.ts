import { type NativeDoctorReport, doctorReport as nativeDoctorReport } from '@dxo/core';
import { resolveNativePath } from '@dxo/core/native';
import { CliError, EXIT } from './errors.js';
import { cliPackage } from './pkg.js';

export type DoctorReport = {
    ok: boolean;
    cli: { name: string; version: string };
    runtime: { node: string; platform: string; arch: string };
    native: { path: string | null; loaded: boolean; error?: string };
    /** napi `doctorReport` payload — sole engine/backend truth. */
    engine: NativeDoctorReport | null;
};

export function runDoctor(): DoctorReport {
    const pkg = cliPackage();
    const report: DoctorReport = {
        ok: false,
        cli: { name: pkg.name, version: pkg.version },
        runtime: {
            node: process.version,
            platform: process.platform,
            arch: process.arch,
        },
        native: { path: null, loaded: false },
        engine: null,
    };

    try {
        const nativePath = resolveNativePath();
        const engine = nativeDoctorReport();
        report.native = { path: nativePath, loaded: true };
        report.engine = engine;
        report.ok = engine.ok;
    } catch (err) {
        report.ok = false;
        report.native = {
            path: null,
            loaded: false,
            error: err instanceof Error ? err.message : String(err),
        };
    }

    return report;
}

export function doctorOrThrow(): DoctorReport {
    const report = runDoctor();
    if (!report.ok || !report.engine) {
        throw new CliError('NATIVE_UNAVAILABLE', report.native.error ?? 'native addon unavailable', EXIT.ENV, report);
    }
    return report;
}

export function formatDoctorText(report: DoctorReport): string {
    const engine = report.engine;
    const lines = [
        `CLI    ${report.cli.name}@${report.cli.version}`,
        `node   ${report.runtime.node} (${report.runtime.platform}/${report.runtime.arch})`,
        `native ${report.native.loaded ? report.native.path : `unavailable: ${report.native.error ?? 'unknown'}`}`,
    ];
    if (engine) {
        lines.push(`core   ${engine.version}`);
        lines.push(`abi    ${engine.abi}`);
        lines.push(`backend ${engine.backend}  cuda=${engine.cudaAvailable}`);
    }
    lines.push(report.ok ? 'ok' : 'env capability insufficient');
    return lines.join('\n');
}
