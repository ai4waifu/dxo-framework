# 🤖 @dxo/skills

Agent Skills for downstream users who ask an AI coding agent to build with DXO. The bundle teaches an agent how to install
DXO packages, choose the right runtime, and write prompts that stay within the published preview contract.

## 🚀 Try with npx

```bash
npx skills add ai4waifu/dxo-framework --skill dxo
```

If your agent supports npm package skill sources, install this package and copy the bundled `skills/dxo` directory into
your agent's skill directory.

The package is intentionally small and has no runtime dependencies. It is for downstream project work, not DXO
Framework contributor workflows.

## 🧪 Local development

From the monorepo root:

```bash
pnpm --filter @dxo/skills exec node bin/dxo-skills.mjs prompt
```

This package is an early developer-preview utility and is not part of the core tensor runtime.
