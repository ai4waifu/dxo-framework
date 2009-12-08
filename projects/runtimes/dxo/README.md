# `@dxo/dxo`

The DXO command-line entry point for Node.js projects.

> Developer preview. The CLI is private and its API is unstable. Only commands implemented in the installed build are
> executable.

## Install and invoke

```bash
pnpm exec dxo --help
pnpm exec dxo version
```

For a published release:

```bash
pnpm add -D @dxo/dxo
pnpm exec dxo doctor
```

The CLI does not install a runtime, download model code, or select a remote provider implicitly.

## Commands

### `version`

Prints CLI, `@dxo/core`, and native runtime version information.

```bash
dxo version
dxo --version
```

### `doctor`

Checks Node, native addon loading, backend capabilities, ABI compatibility, and relevant paths.

```bash
dxo doctor
dxo doctor --json > doctor.json
```

`--json` writes one machine-readable document to stdout; diagnostics go to stderr.

### `studio`

Starts the local Studio inspect API and, unless disabled, the VMZ development UI. It listens on loopback by default.

```bash
dxo studio
dxo studio --port 4310 --webui-port 5173 --runs-dir ./.dxo/runs
dxo studio --api-only
```

`--runs-dir` selects the append-only run store. The process shuts down cleanly on `SIGINT` or `SIGTERM`.

### Planned v0 commands

These commands are part of the CLI contract but may not yet be present in the current preview build:

```text
dxo run <entry>                       Execute an explicit TS/JS entry
dxo inspect list                      List local runs
dxo inspect show <run-id>             Show run metadata and event summary
dxo graph <entry>                     Export a DXO model graph or trace
dxo model verify <source>             Verify manifest, digest, and state compatibility
```

An unavailable command must fail with exit code `2`. `serve`, remote providers, quantization, and multi-GPU commands are
outside the current preview surface. When `serve` lands, it must call a napi-backed runtime — the CLI must not ship a
parallel TypeScript HTTP server.

## Common options

| Option            | Meaning                                           |
|-------------------|---------------------------------------------------|
| `--help`, `-h`    | Show help for the current command                 |
| `--version`, `-v` | Print version information                         |
| `--cwd <path>`    | Use an explicit project directory                 |
| `--config <path>` | Load an explicit config file                      |
| `--json`          | Emit one JSON result on stdout; logs go to stderr |
| `--quiet`         | Suppress informational output                     |
| `--verbose`       | Include diagnostic logging and stack details      |
| `--no-color`      | Disable ANSI color output                         |

Configuration is read from the current directory unless `--cwd` or `--config` is supplied. The CLI does not walk parent
directories.

## Exit codes

| Code  | Meaning                                                           |
|-------|-------------------------------------------------------------------|
| `0`   | Success                                                           |
| `1`   | User entry, runtime, or model verification failure                |
| `2`   | Invalid arguments, configuration, unknown, or unsupported command |
| `78`  | Required environment or capability is unavailable                 |
| `130` | Cancelled by `SIGINT`                                             |

Errors have a stable `code`, human-readable message, and optional structured `details`. Stack traces are never written
to stdout; use `--verbose` when diagnosing locally.

## Scope and security

The CLI is an orchestrator. Tensor execution belongs to `@dxo/core`; run protocols to `@dxo/inspect`; model graphs to
`@dxo/graph`; artifact and state handling to `@dxo/hub` and `@dxo/serialize`; the local UI to `@dxo/studio`.

Network access and remote artifacts are opt-in. Keep tokens and private keys out of config files. The CLI executes Node
code with normal user permissions and is not a sandbox.

## Development status

The current implementation is intentionally small. Conformance work covers help/version, structured `doctor`, argument
errors, signal cancellation, Studio lifecycle, and JSON output for inspect commands.

