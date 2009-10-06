/**
 * LLM / tokenizer product stub.
 * Workspace-only; excluded from placeholder / publish-npm lists.
 * Serving / continuous batching stays in `@dxo/serve` (not scaffolded here).
 */

export type TokenizerEncodeOptions = {
    addSpecialTokens?: boolean;
};

export type EncodedBatch = {
    inputIds: number[][];
    attentionMask?: number[][];
};

export type Tokenizer = {
    readonly kind: 'placeholder';
    encode(texts: string | string[], options?: TokenizerEncodeOptions): EncodedBatch;
};

export function createTokenizer(_name?: string): Tokenizer {
    return {
        kind: 'placeholder',
        encode() {
            throw new Error('@dxo/llm createTokenizer().encode() is a workspace stub');
        },
    };
}

export function llmVersion(): string {
    return 'dxo-llm@placeholder';
}
