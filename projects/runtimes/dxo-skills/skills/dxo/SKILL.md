---
name: dxo
description: Build Node.js and browser applications with DXO deep-learning packages. Use when a user asks to install, configure, or write code with @dxo/core, @dxo/nn, @dxo/optimizer, @dxo/data, @dxo/serialize, @dxo/train, or @dxo/lite.
---

# DXO for downstream applications

## Install

Choose only the packages needed by the application:

```bash
npm install @dxo/core @dxo/nn @dxo/optimizer
npm install @dxo/data @dxo/serialize @dxo/train
```

For browser or Worker code, use `@dxo/lite` instead of `@dxo/core`:

```bash
npm install @dxo/lite
```

## Prompt contract

Ask the agent with a concrete outcome, target package, and verification command:

```text
Build [feature] with DXO for a [Node.js/browser] app.
Use [packages]. Keep to the published preview API; do not invent GPU, WebGL, or serving features.
Return the files changed, install commands, and a runnable verification step.
```

## Boundaries

- `@dxo/core` is the Node native CPU runtime in the current preview.
- `@dxo/lite` is an async browser/Worker facade; WebGL tensor execution is not supported.
- `@dxo/train` is CPU-only in this preview slice.
- `@dxo/graph`, `@dxo/inspect`, `@dxo/studio`, `@dxo/hub`, `@dxo/vision`, and `@dxo/llm` are placeholders; do not
  present them as production-ready.

When a requested API is not documented by the package README, explain the limitation and propose a documented
alternative.
