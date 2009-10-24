# 🧠 DXO Framework

**TypeScript-first deep learning for Node.js.** DXO combines a Rust/Titan engine with napi-rs bindings and a small,
composable TypeScript surface. This repository is an early developer preview; APIs are unstable while the runtime
contracts settle.

[![CI](https://github.com/ai4waifu/dxo-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/ai4waifu/dxo-framework/actions/workflows/ci.yml)
`0.0.x` · Apache-2.0

## 🤖 Agent-first quick start

DXO is designed to be explored and extended with an AI coding agent. Give the agent the repository context first, then
ask for one verifiable change at a time. The agent should read the workspace `AGENTS.md` and the design source before
editing code; implementation README files are intentionally only short package entry points.

### Install skills

Install the skills your agent needs before starting a DXO task. In Codex, ask the agent to run the built-in skill
installer (or use the helper script directly):

```text
Install the relevant DXO development skills from the curated catalog. At minimum install the skill for
skill installation and the skill that matches my task (for example Cloudflare, Workers, SDK, or browser control).
After installation, tell me which skills were added and use them on the next turn.
```

For a local scripted install, use the installer supplied by your Codex installation:

```bash
python scripts/install-skill-from-github.py --repo openai/skills --path skills/.curated/<skill-name>
```

System skills are preinstalled. Do not copy skill files into this repository or commit them here.

### Prompt template

Copy this prompt into your agent and replace the bracketed fields:

```text
You are working on DXO Framework. First read AGENTS.md, then read the relevant design documents and roadmap entries.
Task: [one concrete outcome]
Scope: [files or package, if known]
Constraints: preserve the Rust/napi boundary, do not invent APIs, and keep preview/placeholder status honest.
Verification: run [specific pnpm verify gate or test].
Delivery: make the smallest reviewable change, show the diff, and report any unrelated pre-existing changes without
including them.
```

Example:

```text
You are working on DXO Framework. Read AGENTS.md and the runtime-contract design first.
Implement the next runtime contract test vector for @dxo/lite only.
Do not change public API names or add WebGL support. Run pnpm verify -- runtime-contract-lite and summarize failures.
```

The agent should stop and ask when the requested behavior is not defined by the design source. Keep commits focused and
use the repository's gitmoji convention.

## ✨ What is included

- Eager CPU tensors and scalar autograd in `@dxo/core`
- TypeScript modules, optimizers, datasets, serialization, and a CPU training loop
- An explicit browser/Worker facade in `@dxo/lite` (WebGPU probing; no WebGL tensor backend)
- Rust native bindings with platform packages selected through npm optional dependencies

GPU, model-zoo, serving, and Studio features are still being built. Check each package README for its exact contract.

## Quick start

```bash
pnpm install
pnpm build
pnpm verify -- smoke
```

```typescript
import {backend, tensor, version} from '@dxo/core';
import {Linear} from '@dxo/nn';
import {SGD} from '@dxo/optimizer';

console.log(version(), backend()); // e.g. 0.1.0 titan-cpu

const model = new Linear(2, 1);
const x = tensor([1, 0, 0, 1], [2, 2]);
const y = model.forward(x);
const loss = y.mean(); // scalar shape [1]
```

## Verify gates

```bash
pnpm verify -- smoke
pnpm verify -- tensor-cpu
pnpm verify -- autograd-fd
pnpm verify -- mnist-linear
pnpm verify -- g3-contract
pnpm verify -- data-iter
pnpm verify -- serialize-roundtrip
pnpm verify -- trainer-loop
pnpm verify -- gpu-matmul
pnpm verify -- lite-webgpu-smoke
```

## Layout

| Path                               | npm package           |
|------------------------------------|-----------------------|
| `projects/compilers/dxo-core`      | *(Rust crate)*        |
| `projects/compilers/dxo-napi`      | *(Rust napi cdylib)*  |
| `projects/runtimes/dxo-core`       | `@dxo/core`           |
| `projects/runtimes/dxo-nn`         | `@dxo/nn`             |
| `projects/runtimes/dxo-optimizer`  | `@dxo/optimizer`      |
| `projects/runtimes/dxo-data`       | `@dxo/data`           |
| `projects/runtimes/dxo-serialize`  | `@dxo/serialize`      |
| `projects/runtimes/dxo-train`      | `@dxo/train`          |
| `projects/runtimes/dxo-lite`       | `@dxo/lite`           |
| `projects/runtimes/dxo`            | `@dxo/dxo` (CLI)      |
| `projects/runtimes/dxo-<platform>` | `@dxo/dxo-<platform>` |

## License

Apache-2.0
