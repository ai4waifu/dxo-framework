/**
 * model-runtime-text: char tokenizer + safetensors TinyTransformer load + generate stream/abort.
 */
import assert from 'node:assert/strict';
import { type Tensor, tensor, withoutGrad } from '@dxo/core';
import { createTokenizer, encodeTinyTransformerSafetensors, generate, loadTinyTransformerSafetensors } from '@dxo/llm';
import { NeuralNetwork, TinyTransformer } from '@dxo/nn';

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

/** Deterministic LM: always argmax to a fixed non-special id so maxTokens is platform-stable. */
class StubLm extends NeuralNetwork {
    private readonly vs: number;
    private readonly ft: number;

    constructor(vocabSize: number, forcedToken: number) {
        super();
        this.vs = vocabSize;
        this.ft = forcedToken;
    }

    forward(tokens: Tensor): Tensor {
        const t = tokens.shape[1]!;
        const v = this.vs;
        const data: number[] = [];
        for (let i = 0; i < t * v; i++) data.push(-1e3);
        for (let i = 0; i < t; i++) data[i * v + this.ft] = 10;
        return tensor(data, [1, t, v]);
    }
}

// Avoid special ids 0..3 (pad/bos/eos/unk); do not pass eosTokenId so decoding cannot stop early.
const forced = Math.min(7, vocab - 1);
assert.ok(forced > tok.eosTokenId);
const stub = new StubLm(vocab, forced);
const stubPrompt = tensor([tok.bosTokenId], [1, 1]);
const out: number[] = [];
for await (const step of generate(stub, stubPrompt, { maxTokens: 4, temperature: 0 })) {
    out.push(step.tokenId);
    assert.ok(step.tokenId >= 0 && step.tokenId < vocab);
}
assert.equal(out.length, 4, `stub generate length: tokens=${JSON.stringify(out)}`);
assert.ok(
    out.every((id) => id === forced),
    `stub generate ids=${JSON.stringify(out)} forced=${forced}`,
);

const ac = new AbortController();
ac.abort();
let aborted = false;
try {
    for await (const _ of generate(stub, stubPrompt, { maxTokens: 2 }, { signal: ac.signal })) {
        // unreachable
    }
} catch (err) {
    aborted = err instanceof Error && err.name === 'AbortError';
}
assert.ok(aborted, 'expected AbortError');

// Keep one real TinyTransformer generate smoke (length may vary by platform float path / early EOS).
const live: number[] = [];
for await (const step of generate(clone, prompt, { maxTokens: 2, temperature: 0 })) {
    live.push(step.tokenId);
}
assert.ok(live.length >= 1 && live.length <= 2, `live generate length=${live.length}`);

console.log(`model-runtime-text ok: vocab=${vocab} stub=${out.join(',')} live=${live.join(',')} text="${tok.decode(live)}"`);
