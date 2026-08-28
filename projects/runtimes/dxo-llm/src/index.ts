/**
 * Language-model product surface: tokenizer + generate.
 * Workspace-only until model-runtime-text / related gates close (not on publish-npm).
 */

import { tensor, withoutGrad, type Tensor, type TokenBuffer } from '@dxo/core';
import type { Module } from '@dxo/nn';
import { TinyTransformer } from '@dxo/nn';
import { decodeSafetensors, encodeSafetensors, type SafetensorSlice } from '@dxo/serialize';

export type TokenizerEncodeOptions = {
    addSpecialTokens?: boolean;
};

export type EncodedBatch = {
    inputIds: number[][];
    attentionMask?: number[][];
};

/**
 * Product-facing token batch. Prefer TokenBuffer for ids/mask when napi lands;
 * nested number[][] is the interim Wave 3 surface only.
 */
export type TokenizedInput = {
    inputIds: number[][] | TokenBuffer;
    attentionMask?: number[][] | TokenBuffer;
    positionIds?: number[][] | TokenBuffer;
    offsets?: TokenBuffer;
};

export type Tokenizer = {
    readonly name: string;
    readonly vocabSize: number;
    readonly bosTokenId: number;
    readonly eosTokenId: number;
    readonly padTokenId: number;
    readonly unkTokenId: number;
    encode(texts: string | string[], options?: TokenizerEncodeOptions): EncodedBatch;
    decode(ids: number[]): string;
};

export type GenerationConfig = {
    maxTokens?: number;
    temperature?: number;
    topP?: number;
    eosTokenId?: number;
};

export type GenerateContext = {
    signal?: AbortSignal;
};

export type GeneratedToken = {
    tokenId: number;
    text?: string;
};

const CHAR_V0 = 'dxo-char-v0';

/** Built-in char vocabulary: specials + printable ASCII 32..126. */
function buildCharVocab(): { idToChar: string[]; charToId: Map<string, number> } {
    const idToChar: string[] = ['<pad>', '<bos>', '<eos>', '<unk>'];
    const charToId = new Map<string, number>();
    for (let i = 0; i < idToChar.length; i++) charToId.set(idToChar[i]!, i);
    for (let c = 32; c <= 126; c++) {
        const ch = String.fromCharCode(c);
        charToId.set(ch, idToChar.length);
        idToChar.push(ch);
    }
    return { idToChar, charToId };
}

function createCharTokenizer(): Tokenizer {
    const { idToChar, charToId } = buildCharVocab();
    const bosTokenId = 1;
    const eosTokenId = 2;
    const padTokenId = 0;
    const unkTokenId = 3;

    return {
        name: CHAR_V0,
        vocabSize: idToChar.length,
        bosTokenId,
        eosTokenId,
        padTokenId,
        unkTokenId,
        encode(texts, options = {}) {
            const list = Array.isArray(texts) ? texts : [texts];
            const addSpecial = options.addSpecialTokens ?? true;
            const inputIds = list.map((text) => {
                const ids: number[] = [];
                if (addSpecial) ids.push(bosTokenId);
                for (const ch of text) {
                    ids.push(charToId.get(ch) ?? unkTokenId);
                }
                if (addSpecial) ids.push(eosTokenId);
                return ids;
            });
            const maxLen = Math.max(0, ...inputIds.map((r) => r.length));
            const attentionMask = inputIds.map((row) => {
                const mask = row.map(() => 1);
                while (mask.length < maxLen) mask.push(0);
                return mask;
            });
            const padded = inputIds.map((row) => {
                const copy = [...row];
                while (copy.length < maxLen) copy.push(padTokenId);
                return copy;
            });
            return { inputIds: padded, attentionMask };
        },
        decode(ids) {
            let out = '';
            for (const id of ids) {
                if (id === bosTokenId || id === eosTokenId || id === padTokenId) continue;
                if (id === unkTokenId) {
                    out += '�';
                    continue;
                }
                const ch = idToChar[id];
                if (ch !== undefined) out += ch;
            }
            return out;
        },
    };
}

/** Resolve a built-in tokenizer by name (path-based HF load is later). */
export async function createTokenizer(nameOrPath: string): Promise<Tokenizer> {
    if (nameOrPath === CHAR_V0 || nameOrPath === 'char' || nameOrPath === 'dxo-char') {
        return createCharTokenizer();
    }
    throw new Error(`@dxo/llm createTokenizer: unknown tokenizer '${nameOrPath}' (v0 supports '${CHAR_V0}')`);
}

function assertNotAborted(signal?: AbortSignal): void {
    if (signal?.aborted) {
        const err = new Error('generate aborted');
        err.name = 'AbortError';
        throw err;
    }
}

function sampleToken(logits: number[], temperature: number, topP: number): number {
    if (!(temperature > 0) || !Number.isFinite(temperature)) {
        let best = 0;
        for (let i = 1; i < logits.length; i++) {
            if (logits[i]! > logits[best]!) best = i;
        }
        return best;
    }

    const scaled = logits.map((v) => v / temperature);
    const max = Math.max(...scaled);
    let weights = scaled.map((v) => Math.exp(v - max));
    const sum = weights.reduce((a, b) => a + b, 0);
    weights = weights.map((w) => w / sum);

    if (topP > 0 && topP < 1) {
        const order = weights.map((w, i) => ({ w, i })).sort((a, b) => b.w - a.w);
        let cum = 0;
        const keep = new Set<number>();
        for (const item of order) {
            keep.add(item.i);
            cum += item.w;
            if (cum >= topP) break;
        }
        weights = weights.map((w, i) => (keep.has(i) ? w : 0));
        const renorm = weights.reduce((a, b) => a + b, 0) || 1;
        weights = weights.map((w) => w / renorm);
    }

    let r = Math.random();
    for (let i = 0; i < weights.length; i++) {
        r -= weights[i]!;
        if (r <= 0) return i;
    }
    return weights.length - 1;
}

/**
 * Autoregressive decode: yields one token at a time.
 * Requires `tokens` shape `[1, T]` (batch 1) for the streaming preview path.
 */
export async function* generate(
    model: Module,
    tokens: Tensor,
    config: GenerationConfig = {},
    ctx: GenerateContext = {},
): AsyncGenerator<GeneratedToken> {
    const maxTokens = config.maxTokens ?? 16;
    const temperature = config.temperature ?? 0;
    const topP = config.topP ?? 1;
    const eosTokenId = config.eosTokenId;

    if (tokens.shape.length !== 2 || tokens.shape[0] !== 1) {
        throw new Error(`generate expects tokens shape [1, T], got [${tokens.shape.join(',')}]`);
    }

    let cur = tokens;

    for (let step = 0; step < maxTokens; step++) {
        assertNotAborted(ctx.signal);
        const logits = withoutGrad(() => model.forward(cur));
        if (logits.shape.length !== 3 || logits.shape[0] !== 1) {
            throw new Error(`generate: model must return [1,T,V], got [${logits.shape.join(',')}]`);
        }
        const t = logits.shape[1]!;
        const v = logits.shape[2]!;
        const flat = await logits.toArray();
        const last = flat.slice((t - 1) * v, t * v);
        const tokenId = sampleToken(last, temperature, topP);

        yield { tokenId };

        if (eosTokenId !== undefined && tokenId === eosTokenId) return;

        const prev = await cur.toArray();
        cur = tensor([...prev, tokenId], [1, t + 1]);
    }
}

/** Encode TinyTransformer state as safetensors bytes. */
export async function encodeTinyTransformerSafetensors(model: TinyTransformer): Promise<Uint8Array> {
    return encodeSafetensors(await model.state());
}

/** Load TinyTransformer weights from safetensors (F32). */
export function loadTinyTransformerSafetensors(model: TinyTransformer, bytes: Uint8Array, opts: { requiresGrad?: boolean } = {}): void {
    const slices = decodeSafetensors(bytes) as Record<string, SafetensorSlice>;
    model.loadState(slices, opts);
}

export function llmVersion(): string {
    return 'dxo-llm@0.1.0-preview';
}
