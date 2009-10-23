# 🤖 @dxo/skills

Agent-first onboarding helper for DXO Framework. It prints a repository-aware prompt that asks an AI coding agent to
read the workspace rules and design source before changing implementation code.

## 🚀 Try with npx

```bash
npx @dxo/skills prompt
```

The package is intentionally small and has no runtime dependencies. It does not install Codex skills itself; use your
agent's skill installer for that step, then use the generated prompt to start a DXO task.

## 🧪 Local development

From the monorepo root:

```bash
pnpm --filter @dxo/skills exec node bin/dxo-skills.mjs prompt
```

This package is an early developer-preview utility and is not part of the core tensor runtime.
