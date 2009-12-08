import { existsSync } from 'node:fs';
import path from 'node:path';
import { version as coreVersion, defaultInspectRunsRoot, listInspectRuns, readInspectEvents, readInspectRunMeta } from '@dxo/core';
import { startStudio } from '@dxo/studio';
import { cac } from 'cac';
import { formatDoctorText, runDoctor } from './doctor.js';
import { CliError, EXIT, unsupportedCommand } from './errors.js';
import type { GlobalFlags } from './output.js';
import { writeError, writeResult } from './output.js';
import { cliPackage } from './pkg.js';

type SharedOptions = GlobalFlags & {
    cwd?: string;
    config?: string;
    runsDir?: string;
    host?: string;
    port?: string | number;
    webuiPort?: string | number;
    apiOnly?: boolean;
};

function applyCwd(cwd: string | undefined): void {
    if (!cwd) return;
    const abs = path.resolve(cwd);
    if (!existsSync(abs)) {
        throw new CliError('INVALID_CWD', `cwd does not exist: ${abs}`, EXIT.USAGE, { cwd: abs });
    }
    process.chdir(abs);
}

function resolveConfigPath(config: string | undefined): string | undefined {
    if (!config) return undefined;
    const abs = path.resolve(config);
    if (!existsSync(abs)) {
        throw new CliError('INVALID_CONFIG', `config file not found: ${abs}`, EXIT.USAGE, {
            config: abs,
        });
    }
    return abs;
}

function runsRootOrDefault(runsDir: string | undefined): string {
    if (runsDir) return path.resolve(runsDir);
    return defaultInspectRunsRoot();
}

function waitForSignal(): Promise<'SIGINT' | 'SIGTERM'> {
    return new Promise((resolve) => {
        // Keep the event loop alive when stdin is ignored (CI / spawn).
        const keepAlive = setInterval(() => {}, 60_000);
        const finish = (signal: 'SIGINT' | 'SIGTERM') => {
            clearInterval(keepAlive);
            process.off('SIGINT', onSigInt);
            process.off('SIGTERM', onSigTerm);
            resolve(signal);
        };
        const onSigInt = () => finish('SIGINT');
        const onSigTerm = () => finish('SIGTERM');
        process.on('SIGINT', onSigInt);
        process.on('SIGTERM', onSigTerm);
    });
}

function createCli() {
    const pkg = cliPackage();
    const cli = cac('dxo');

    cli.option('--cwd <path>', 'Project directory (no parent walk)');
    cli.option('--config <path>', 'Explicit dxo.config path');
    cli.option('--json', 'Emit one JSON document on stdout');
    cli.option('--quiet', 'Suppress informational stdout');
    cli.option('--verbose', 'Include diagnostic details / stacks on stderr');
    cli.option('--no-color', 'Disable ANSI color');

    cli.help();
    cli.version(pkg.version);

    cli.command('version', 'Print CLI and napi core version').action((opts: SharedOptions) => {
        applyCwd(opts.cwd);
        resolveConfigPath(opts.config);
        // Force napi load — version() is native.
        const body = {
            cli: { name: pkg.name, version: pkg.version },
            core: { version: coreVersion() },
        };
        writeResult(opts, body, () => `${pkg.name}@${pkg.version}  core=${body.core.version}`);
    });

    cli.command('doctor', 'Diagnose Node + napi engine/backend (doctorReport)').action((opts: SharedOptions) => {
        applyCwd(opts.cwd);
        resolveConfigPath(opts.config);
        const report = runDoctor();
        writeResult(opts, report, () => formatDoctorText(report));
        if (!report.ok) {
            throw new CliError('NATIVE_UNAVAILABLE', report.native.error ?? 'native addon unavailable', EXIT.ENV, report);
        }
    });

    cli.command('inspect <action> [runId]', 'Inspect local runs via napi Rust store')
        .option('--runs-dir <path>', 'Run store root (default: napi defaultInspectRunsRoot)')
        .action((action: string, runId: string | undefined, opts: SharedOptions) => {
            applyCwd(opts.cwd);
            resolveConfigPath(opts.config);
            const root = runsRootOrDefault(opts.runsDir);

            if (action === 'list') {
                const runs = listInspectRuns(root);
                const body = {
                    runsRoot: root,
                    runs: runs.map((r) => ({
                        runId: r.runId,
                        status: r.meta.status,
                        label: r.meta.label ?? null,
                        startedAtMs: r.meta.startedAtMs,
                        endedAtMs: r.meta.endedAtMs ?? null,
                    })),
                };
                writeResult(opts, body, () => {
                    if (!body.runs.length) return `no runs under ${root}`;
                    return body.runs.map((r) => `${r.runId}  ${r.status}  ${r.label ?? '-'}  ${r.startedAtMs}`).join('\n');
                });
                return;
            }

            if (action === 'show') {
                if (!runId) {
                    throw new CliError('USAGE', 'inspect show requires <run-id>', EXIT.USAGE);
                }
                const meta = readInspectRunMeta(runId, root);
                if (!meta) {
                    throw new CliError('RUN_NOT_FOUND', `run not found: ${runId}`, EXIT.FAIL, {
                        runId,
                        runsRoot: root,
                    });
                }
                const events = readInspectEvents(runId, root);
                const artifacts = events.filter(
                    (e) => e && typeof e === 'object' && 'type' in e && (e as { type: string }).type === 'artifact/ref',
                );
                const metrics = events.filter(
                    (e) => e && typeof e === 'object' && 'type' in e && (e as { type: string }).type === 'metric/scalar',
                );
                const body = {
                    runsRoot: root,
                    meta: {
                        format: meta.format,
                        version: meta.version,
                        runId: meta.runId,
                        startedAtMs: meta.startedAtMs,
                        endedAtMs: meta.endedAtMs ?? null,
                        label: meta.label ?? null,
                        status: meta.status,
                        hyperparams: meta.hyperparamsJson ? JSON.parse(meta.hyperparamsJson) : null,
                    },
                    summary: {
                        eventCount: events.length,
                        metricScalarCount: metrics.length,
                        artifactCount: artifacts.length,
                    },
                };
                writeResult(opts, body, () =>
                    [
                        `run     ${body.meta.runId}`,
                        `status  ${body.meta.status}`,
                        `label   ${body.meta.label ?? '-'}`,
                        `events  ${body.summary.eventCount}`,
                        `metrics ${body.summary.metricScalarCount}`,
                        `artifacts ${body.summary.artifactCount}`,
                    ].join('\n'),
                );
                return;
            }

            throw new CliError('UNKNOWN_COMMAND', `unknown inspect action: ${action}`, EXIT.USAGE, { action });
        });

    cli.command('studio', 'Start loopback Studio (napi inspect HTTP serve + optional WebUI)')
        .option('--host <host>', 'Bind host', { default: '127.0.0.1' })
        .option('--port <port>', 'API port (0 = ephemeral)')
        .option('--webui-port <port>', 'VMZ WebUI port')
        .option('--runs-dir <path>', 'Run store root')
        .option('--api-only', 'Do not start WebUI')
        .action(async (opts: SharedOptions) => {
            applyCwd(opts.cwd);
            resolveConfigPath(opts.config);
            const port = opts.port === undefined || opts.port === '' ? 4310 : Number(opts.port);
            if (!Number.isFinite(port) || port < 0) {
                throw new CliError('INVALID_PORT', `invalid --port: ${opts.port}`, EXIT.USAGE);
            }
            const webuiPort = opts.webuiPort === undefined || opts.webuiPort === '' ? undefined : Number(opts.webuiPort);
            if (webuiPort !== undefined && (!Number.isFinite(webuiPort) || webuiPort <= 0)) {
                throw new CliError('INVALID_PORT', `invalid --webui-port: ${opts.webuiPort}`, EXIT.USAGE);
            }

            // startStudio → createInspectApiServer → napi create_inspect_api_server
            const studio = await startStudio({
                host: opts.host ?? '127.0.0.1',
                port,
                runsRoot: opts.runsDir ? path.resolve(opts.runsDir) : undefined,
                webui: !opts.apiOnly,
                webuiPort,
            });

            const body = {
                apiUrl: studio.url,
                host: studio.host,
                port: studio.port,
                runsRoot: studio.runsRoot,
                webuiUrl: studio.webuiUrl ?? null,
            };
            writeResult(opts, body, () => {
                const lines = [`dxo studio API ${studio.url}`, `runs ${studio.runsRoot}`];
                if (studio.webuiUrl) lines.push(`webui ${studio.webuiUrl}`);
                lines.push('waiting for SIGINT/SIGTERM…');
                return lines.join('\n');
            });

            const signal = await waitForSignal();
            await studio.closeAll();
            throw new CliError('CANCELLED', signal === 'SIGINT' ? 'cancelled by SIGINT' : 'cancelled by SIGTERM', EXIT.CANCEL);
        });

    cli.command('run <entry>', 'Run an explicit TS/JS entry (not yet supported)').action(() => {
        throw unsupportedCommand('run');
    });

    cli.command('graph <entry>', 'Export model graph (not yet supported)').action(() => {
        throw unsupportedCommand('graph');
    });

    cli.command('model <action> [source]', 'Model commands (verify not yet supported)').action((action: string) => {
        if (action === 'verify') throw unsupportedCommand('model verify');
        throw new CliError('UNKNOWN_COMMAND', `unknown model action: ${action}`, EXIT.USAGE);
    });

    // Future `dxo serve` must call a napi-backed runtime (Living 13 / @dxo/serve).
    // Do not invent a parallel TS HTTP server here.
    cli.command('serve', 'Inference serve (unsupported until napi serve surface exists)').action(() => {
        throw unsupportedCommand('serve');
    });

    cli.on('command:*', () => {
        const name = cli.args.join(' ') || '(empty)';
        throw new CliError('UNKNOWN_COMMAND', `unknown command: ${name}`, EXIT.USAGE, {
            args: cli.args,
        });
    });

    return cli;
}

function exitCodeFromUnknown(err: unknown): number {
    if (err && typeof err === 'object' && 'exitCode' in err) {
        return Number((err as CliError).exitCode);
    }
    return EXIT.FAIL;
}

/**
 * CLI entry used by `bin/dxo.mjs` and verify harnesses.
 * Always pass args without the node/script prefix; this rebuilds argv for cac.
 */
export async function main(argv: string[] = process.argv.slice(2)): Promise<number> {
    const flags: GlobalFlags = {
        json: argv.includes('--json'),
        quiet: argv.includes('--quiet'),
        verbose: argv.includes('--verbose'),
        color: !argv.includes('--no-color'),
    };

    const cli = createCli();
    const cacArgv = ['node', 'dxo', ...argv];

    try {
        cli.parse(cacArgv, { run: false });
        if (cli.options.help) {
            process.exitCode = EXIT.OK;
            return EXIT.OK;
        }
        if (cli.options.version && !cli.matchedCommandName) {
            process.exitCode = EXIT.OK;
            return EXIT.OK;
        }

        const result = cli.runMatchedCommand();
        if (result && typeof (result as Promise<unknown>).then === 'function') {
            await result;
        }

        if (!cli.matchedCommand && cli.args[0]) {
            throw new CliError('UNKNOWN_COMMAND', `unknown command: ${cli.args.join(' ')}`, EXIT.USAGE, { args: cli.args });
        }

        if (!cli.matchedCommand && !cli.options.version) {
            cli.outputHelp();
        }

        process.exitCode = EXIT.OK;
        return EXIT.OK;
    } catch (err) {
        if (typeof err === 'string') {
            writeError(flags, new CliError('USAGE', err, EXIT.USAGE));
            process.exitCode = EXIT.USAGE;
            return EXIT.USAGE;
        }
        if (err instanceof Error && /unknown option|missing required/i.test(err.message)) {
            writeError(flags, new CliError('USAGE', err.message, EXIT.USAGE));
            process.exitCode = EXIT.USAGE;
            return EXIT.USAGE;
        }
        writeError(flags, err);
        const code = exitCodeFromUnknown(err);
        process.exitCode = code;
        return code;
    }
}
