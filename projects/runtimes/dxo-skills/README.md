# 🤖 @dxo/skills

## Give your AI coding agent a TypeScript-native deep-learning toolkit

`@dxo/skills` is the fastest way to start building with DXO through an AI coding agent. It teaches your agent how to choose DXO packages, compose tensors and model workflows, handle streaming and cancellation, and produce examples that are ready to adapt into a real application.

## Start in seconds

```bash
npx skills add ai4waifu/dxo-framework --skill dxo
```

After installation, open a new request with a concrete outcome. The skill helps your agent turn that request into a small, understandable TypeScript application instead of a generic deep-learning tutorial.

## Copy your first prompt

```text
Build a small image-classification application with DXO and TypeScript.
Use @dxo/vision for image input and preprocessing, load a model from a local artifact,
return the top five predictions as JSON, and include a runnable example.
Prefer the smallest clear public API and explain how I can adapt it to my Node.js service.
```

## Explore more with prompts

```text
Create a streaming text-generation feature with @dxo/llm.
Accept a prompt, emit text chunks as they arrive, support AbortSignal cancellation,
and show the smallest complete TypeScript example.
```

```text
Build a prompt-to-image demo with @dxo/diffuser.
Show progress and preview events, make generation cancellable, and save the final image.
Keep the code easy to move into an existing Node.js application.
```

```text
Create a tiny training experiment with @dxo/data, @dxo/nn, @dxo/optimizer, and @dxo/train.
Use a synthetic dataset, stream useful progress, save the model state, and explain each part
in terms a TypeScript developer can maintain.
```

## What the skill helps your agent do

- Pick the right DXO package for a product feature.
- Prefer TypeScript-native patterns over translated Python APIs.
- Keep tensors, images, tokens, and generated outputs on efficient data paths.
- Design long-running work around `AsyncIterable` and `AbortSignal`.
- Make model loading, preprocessing, and results reproducible.
- Explain trade-offs in plain language and keep examples focused.

## Works with your workflow

Use the skill with Codex, Claude Code, Cursor, or another agent that supports the Skills ecosystem. It complements your agent; your project, source code, credentials, and deployment choices remain yours.

DXO is an early developer preview, so ask your agent to verify the APIs used by an example and to keep unsupported capabilities explicit.
