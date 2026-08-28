# 🧠 DXO

## TypeScript-first deep learning, without the Python translation layer

[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![npm](https://img.shields.io/npm/v/%40dxo%2Fcore)](https://www.npmjs.com/package/@dxo/core)

DXO is a deep-learning framework designed for TypeScript developers.

It does not mechanically copy Python APIs into JavaScript; it uses the conventions that make TypeScript readable,
composable, and pleasant to operate in production.

## Why Pythonic API translations feel wrong in TypeScript

Python-first frameworks are brilliant research tools.

But when a model becomes a product, teams often rebuild the surrounding system in TypeScript: APIs, workers, queues,
dashboards, desktop tools, and edge clients.

DXO keeps the model and the product in one language and one runtime.

| The pain of a direct Python API port                                                                                            | The DXO answer                                                                                   |
|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **“What does this argument mean?”** Long positional calls, overloaded flags, and magic defaults.                                | Explicit option objects with editor-visible types and sensible TypeScript naming.                |
| **“Why did this fail only at runtime?”** Shapes, devices, dtypes, and missing fields discovered after the job starts.           | Contracts visible during authoring, with structured errors when a capability is unavailable.     |
| **“How do I stop this request?”** A blocking training or generation call with no natural cancellation path.                     | `AbortSignal` travels through the operation and releases work predictably.                       |
| **“How do I stream this?”** Callbacks bolted onto an API designed to return one final value.                                    | `AsyncIterable` is the normal result for tokens, previews, progress, and metrics.                |
| **“Why is my device copying everything?”** Python objects, JSON, and process boundaries between model and product.              | Typed buffers and tensor views make ownership, copies, and device transfers explicit.            |
| **“Why does this feel unlike the rest of my codebase?”** A Python object lifecycle translated into classes and mutable globals. | Small functions, immutable configuration, composable handles, and ordinary Node.js control flow. |
| **“How do I turn the demo into a feature?”** A research notebook that needs a second rewrite for the real product.              | The same TypeScript model code can live in an API route, worker, desktop app, or edge function.  |

The result is not “Python, but in JavaScript.” It is deep learning shaped around TypeScript habits: explicit objects
instead of magic positional arguments, ordinary async iteration instead of framework-specific callbacks, cancellation
that travels with long-running work, and types that explain what can happen next.

## 🤖 Start with an AI coding agent

Install the DXO skill first:

```bash
npx skills add ai4waifu/dxo-framework --skill dxo
```

Then give your agent a concrete product request:

```text
Build a TypeScript image-classification endpoint with DXO.
Use @dxo/vision for image input and preprocessing, load a local model,
return the top five labels as JSON, and include a small runnable example.
```

For language applications:

```text
Create a streaming text-generation feature with @dxo/llm.
Accept a prompt, emit text chunks as they arrive, support AbortSignal cancellation,
and show the smallest complete TypeScript example.
```

## The TypeScript-friendly advantage

- **Types that explain the model** — tensor shapes, devices, media buffers, model state, and generated events are
  visible while you code.
- **Native performance where it matters** — a Rust execution engine handles tensor storage and numerical work without
  making your application leave Node.js.
- **One composable workflow** — move from tensors to neural networks, training, model loading, vision, language, and
  generation without changing languages or service boundaries.
- **Efficient data paths** — typed buffers and tensor views reduce unnecessary copies between media, models, and
  devices.
- **Built for real products** — stream progress, cancel long-running work, integrate with existing services, and keep
  model behavior reproducible.

```ts
for await (const event of pipeline.generate({prompt, signal})) {
    if (event.type === 'preview') updatePreview(event.image);
    if (event.type === 'text') response.write(event.text);
}
```

The same control flow works in an HTTP handler, a queue worker, a desktop app, or an edge function. No adapter layer is
needed just to make inference feel natural in your product.

## Start with a tensor, grow into an application

```ts
import {tensor} from '@dxo/core';
import {Linear} from '@dxo/nn';

const model = new Linear(2, 1);
const input = tensor([1, 2], [1, 2]);
const prediction = model.forward(input);

console.log(await prediction.toArray());
```

The same foundation supports training loops, image pipelines, tokenizers, language models, and diffusion workflows.

## A focused ecosystem

| Package          | Use it for                                        |
|------------------|---------------------------------------------------|
| `@dxo/core`      | Tensors, autograd, and runtime execution          |
| `@dxo/nn`        | Reusable neural-network modules                   |
| `@dxo/optimizer` | Parameter updates and optimization                |
| `@dxo/data`      | Datasets, batching, and async data sources        |
| `@dxo/train`     | Cancellable asynchronous training                 |
| `@dxo/serialize` | Portable model state and weights                  |
| `@dxo/vision`    | Image and video inference                         |
| `@dxo/llm`       | Tokenizers and streaming language-model workflows |
| `@dxo/diffuser`  | Diffusion and multimodal generation               |
| `@dxo/lite`      | Browser, Worker, and edge runtimes                |

## From idea to shipped feature

DXO is shaped around the way an intelligent feature actually grows:

1. Start with a typed tensor or a real input such as an image, audio clip, or prompt.
2. Compose a model with ordinary TypeScript functions and objects.
3. Stream predictions, previews, metrics, or tokens as work happens.
4. Cancel expensive work when a request ends or a user changes their mind.
5. Move the same workflow into an API route, queue worker, desktop tool, or edge application.

There is no separate “research API” to abandon when the prototype becomes a product. The model, data path, and control
flow remain understandable to the team that owns the application.

## Built for intelligent products

DXO fits the places where models create user value:

- Personalization and ranking inside a Node.js service.
- Image understanding in a content or commerce workflow.
- Streaming assistants with responsive cancellation.
- Generative media tools with live previews and progress.
- Batch jobs that share code with an interactive product.
- Browser and edge experiences that use the lightweight runtime.

The ecosystem is intentionally composable. Use only `@dxo/core` for tensor work, add `@dxo/train` when you need
optimization, or move directly to `@dxo/vision`, `@dxo/llm`, and `@dxo/diffuser` when the product problem is already
clear.

## A better default for TypeScript teams

Your application already has a language, a package manager, a service framework, an observability stack, and a
deployment story. DXO lets deep learning participate in that system directly. Types describe the boundaries, async
iteration carries live work, and the agent that helps write your application can help shape the model workflow too.

## Keep building

Pick the package that matches your idea, ask your agent for the smallest working feature, and grow from there. DXO is
open source under the Apache-2.0 license.
