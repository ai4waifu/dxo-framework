# 💬 @dxo/llm

**Workspace-only — not on npm publish / placeholder OIDC lists until gates close.**

Tokenizer and language-model product surface (`GenerationConfig`, streaming `generate`, AbortSignal). Inference serving remains `@dxo/serve` (later). Prefer this package name over a vague `@dxo/text` umbrella.

## Preview API

```ts
import { createTokenizer, generate, loadTinyTransformerSafetensors } from '@dxo/llm';
import { TinyTransformer } from '@dxo/nn';

const tok = await createTokenizer('dxo-char-v0');
const model = new TinyTransformer(tok.vocabSize, 64, 32, 4, 2);
// loadTinyTransformerSafetensors(model, bytes);

const ids = tok.encode('hello').inputIds[0]!;
for await (const step of generate(model, /* Tensor [1,T] */, { maxTokens: 8, temperature: 0 })) {
  console.log(step.tokenId);
}
```
