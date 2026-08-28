/**
 * model-runtime-text: char tokenizer + safetensors TinyTransformer load + generate stream/abort.
 */
import assert from 'node:assert/strict';
import { tensor, withoutGrad } from '@dxo/core';
import { createTokenizer, encodeTinyTransformerSafetensors, generate, loadTinyTransformerSafetensors } from '@dxo/llm';
import { TinyTransformer } from '@dxo/nn';

const tok = await createTokenizer('dxo-char-v0');
assert.ok(tok.vocabSize > 4);
assert.equal(tok.decode(tok.encode('Hi', { addSpecialTokens: false }).inputIds[0]!), 'Hi');

const enc = tok.encode('ab');
assert.equal(enc.inputIds.length, 1);
assert.ok(enc.inputIds[0]!.includes(tok.bosTokenId));
assert.ok(enc.inputIds[0]!.includes(tok.eosTokenId));

const vocab = tok.vocabSize;
const model = new TinyTransformer(vocab, 16, 16, 2, 1, { requiresGrad: false });

// Deterministic-ish: one train-free forward then checkpoint
const promptIds = enc.inputIds[0]!;
const prompt = tensor(promptIds, [1, promptIds.length]);
const logitsA = withoutGrad(() => model.forward(prompt));
const ref = await logitsA.toArray();

const bytes = await encodeTinyTransformerSafetensors(model);
const clone = new TinyTransformer(vocab, 16, 16, 2, 1, { requiresGrad: false });
loadTinyTransformerSafetensors(clone, bytes, { requiresGrad: false });
const logitsB = withoutGrad(() => clone.forward(prompt));
const loaded = await logitsB.toArray();
assert.equal(loaded.length, ref.length);
for (let i = 0; i < ref.length; i++) {
    assert.ok(Math.abs(loaded[i]! - ref[i]!) < 1e-5, `logit[${i}] mismatch`);
}

const out: number[] = [];
for await (const step of generate(clone, prompt, { maxTokens: 4, temperature: 0, eosTokenId: tok.eosTokenId })) {
    out.push(step.tokenId);
    assert.ok(step.tokenId >= 0 && step.tokenId < vocab);
}
assert.equal(out.length, 4);

const ac = new AbortController();
ac.abort();
let aborted = false;
try {
    for await (const _ of generate(clone, prompt, { maxTokens: 2 }, { signal: ac.signal })) {
        // unreachable
    }
} catch (err) {
    aborted = err instanceof Error && err.name === 'AbortError';
}
assert.ok(aborted, 'expected AbortError');

console.log(`model-runtime-text ok: vocab=${vocab} gen=${out.join(',')} text="${tok.decode(out)}"`);
