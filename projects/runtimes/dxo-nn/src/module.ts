import { embedding, randnValues, Tensor, tensor, ones, zeros } from '@dxo/core';

export interface TensorStateSlice {
    shape: number[];
    data: number[];
}

export interface LinearState {
    weight: TensorStateSlice;
    bias: TensorStateSlice;
}

export abstract class Module {
    abstract forward(x: Tensor): Tensor;

    parameters(): Tensor[] {
        const out: Tensor[] = [];
        const visit = (value: unknown): void => {
            if (value instanceof Tensor) {
                out.push(value);
                return;
            }
            if (value instanceof Module) {
                out.push(...value.parameters());
                return;
            }
            if (Array.isArray(value)) {
                for (const item of value) visit(item);
            }
        };
        for (const value of Object.values(this)) visit(value);
        return out;
    }

    zeroGrad(): void {
        for (const p of this.parameters()) p.zeroGrad();
    }
}

export function relu(x: Tensor): Tensor {
    return x.relu();
}

/** Element-wise ReLU module. */
export class Relu extends Module {
    forward(x: Tensor): Tensor {
        return x.relu();
    }
}

/** Fully-connected affine map: `y = x @ weight + bias` (no activation).
 *
 * Weight layout: `[inFeatures, outFeatures]`. Default leaves use `requiresGrad: true`.
 * After `optimizer.step(parameters())`, call `loadParameters` to install new leaves.
 */
export class Linear extends Module {
    weight: Tensor;
    bias: Tensor;

    constructor(
        readonly inFeatures: number,
        readonly outFeatures: number,
        opts: { requiresGrad?: boolean } = {},
    ) {
        super();
        const rg = opts.requiresGrad ?? true;
        const scale = Math.sqrt(2 / (inFeatures + outFeatures));
        const raw = randnValues([inFeatures, outFeatures]).map((v) => v * scale);
        this.weight = tensor(raw, [inFeatures, outFeatures], { requiresGrad: rg });
        this.bias = zeros([outFeatures], { requiresGrad: rg });
    }

    forward(x: Tensor): Tensor {
        if (x.shape.length === 2) {
            return x.matmul(this.weight).add(this.bias);
        }
        if (x.shape.length < 2) {
            throw new Error(`Linear expects rank >= 2, got [${x.shape.join(',')}]`);
        }
        const last = x.shape[x.shape.length - 1]!;
        if (last !== this.inFeatures) {
            throw new Error(`Linear inFeatures ${this.inFeatures} != last dim ${last}`);
        }
        const leading = x.shape.slice(0, -1);
        const flat = leading.reduce((n, d) => n * d, 1);
        const y = x.reshape([flat, last]).matmul(this.weight).add(this.bias);
        return y.reshape([...leading, this.outFeatures]);
    }

    /** Replace parameter leaves after an optimizer step. */
    loadParameters(params: Tensor[]): void {
        if (params.length < 2) throw new Error('Linear expects [weight, bias]');
        this.weight = params[0]!;
        this.bias = params[1]!;
    }

    async state(): Promise<LinearState> {
        return {
            weight: { shape: [...this.weight.shape], data: await this.weight.toArray() },
            bias: { shape: [...this.bias.shape], data: await this.bias.toArray() },
        };
    }

    loadState(saved: LinearState): void {
        this.weight = tensor(saved.weight.data, saved.weight.shape, { requiresGrad: true });
        this.bias = tensor(saved.bias.data, saved.bias.shape, { requiresGrad: true });
    }
}

export class Sequential extends Module {
    constructor(readonly layers: Module[]) {
        super();
    }

    forward(x: Tensor): Tensor {
        let out = x;
        for (const layer of this.layers) {
            out = layer.forward(out);
        }
        return out;
    }
}

export interface EmbeddingState {
    weight: TensorStateSlice;
}

/** Token embedding: `forward(indices)` gathers rows from `[vocab, dim]`. */
export class Embedding extends Module {
    weight: Tensor;

    constructor(
        readonly numEmbeddings: number,
        readonly embeddingDim: number,
        opts: { requiresGrad?: boolean } = {},
    ) {
        super();
        const rg = opts.requiresGrad ?? true;
        const scale = Math.sqrt(1 / embeddingDim);
        const raw = randnValues([numEmbeddings, embeddingDim]).map((v) => v * scale);
        this.weight = tensor(raw, [numEmbeddings, embeddingDim], { requiresGrad: rg });
    }

    forward(indices: Tensor): Tensor {
        return embedding(this.weight, indices);
    }

    loadParameters(params: Tensor[]): void {
        if (params.length < 1) throw new Error('Embedding expects [weight]');
        this.weight = params[0]!;
    }

    async state(): Promise<EmbeddingState> {
        return { weight: { shape: [...this.weight.shape], data: await this.weight.toArray() } };
    }

    loadState(saved: EmbeddingState): void {
        this.weight = tensor(saved.weight.data, saved.weight.shape, { requiresGrad: true });
    }
}

export interface LayerNormState {
    weight: TensorStateSlice;
    bias: TensorStateSlice;
}

/** LayerNorm over the last dimension. */
export class LayerNorm extends Module {
    weight: Tensor;
    bias: Tensor;

    constructor(
        readonly normalizedShape: number,
        opts: { eps?: number; requiresGrad?: boolean } = {},
    ) {
        super();
        const rg = opts.requiresGrad ?? true;
        this.eps = opts.eps ?? 1e-5;
        this.weight = ones([normalizedShape], { requiresGrad: rg });
        this.bias = zeros([normalizedShape], { requiresGrad: rg });
    }

    readonly eps: number;

    forward(x: Tensor): Tensor {
        return x.layerNorm(this.weight, this.bias, this.eps);
    }

    loadParameters(params: Tensor[]): void {
        if (params.length < 2) throw new Error('LayerNorm expects [weight, bias]');
        this.weight = params[0]!;
        this.bias = params[1]!;
    }

    async state(): Promise<LayerNormState> {
        return {
            weight: { shape: [...this.weight.shape], data: await this.weight.toArray() },
            bias: { shape: [...this.bias.shape], data: await this.bias.toArray() },
        };
    }

    loadState(saved: LayerNormState): void {
        this.weight = tensor(saved.weight.data, saved.weight.shape, { requiresGrad: true });
        this.bias = tensor(saved.bias.data, saved.bias.shape, { requiresGrad: true });
    }
}

/**
 * Causal multi-head self-attention over `[B, T, C]`.
 * Head dim must divide `embedDim`.
 */
export class MultiheadAttention extends Module {
    qkv: Linear;
    proj: Linear;

    constructor(
        readonly embedDim: number,
        readonly numHeads: number,
        opts: { requiresGrad?: boolean } = {},
    ) {
        super();
        if (embedDim % numHeads !== 0) {
            throw new Error(`embedDim ${embedDim} must be divisible by numHeads ${numHeads}`);
        }
        this.headDim = embedDim / numHeads;
        this.qkv = new Linear(embedDim, embedDim * 3, opts);
        this.proj = new Linear(embedDim, embedDim, opts);
    }

    readonly headDim: number;

    forward(x: Tensor): Tensor {
        const [b, t, c] = x.shape;
        if (b === undefined || t === undefined || c === undefined || x.shape.length !== 3) {
            throw new Error(`MultiheadAttention expects [B,T,C], got [${x.shape.join(',')}]`);
        }
        if (c !== this.embedDim) {
            throw new Error(`expected embedDim ${this.embedDim}, got ${c}`);
        }
        const qkv = this.qkv.forward(x); // [B,T,3C]
        const packed = qkv.reshape([b, t, 3, this.numHeads, this.headDim]);
        // Split Q/K/V along axis 2 then move heads: [B,H,T,D]
        const q = packed.narrow(2, 0, 1).reshape([b, t, this.numHeads, this.headDim]).transposeDims(1, 2);
        const k = packed.narrow(2, 1, 1).reshape([b, t, this.numHeads, this.headDim]).transposeDims(1, 2);
        const v = packed.narrow(2, 2, 1).reshape([b, t, this.numHeads, this.headDim]).transposeDims(1, 2);
        const attn = q.scaledDotProductAttention(k, v, true); // [B,H,T,D]
        const merged = attn.transposeDims(1, 2).reshape([b, t, this.embedDim]);
        return this.proj.forward(merged);
    }
}

/** One decoder block: LN → causal MHA → residual → LN → MLP → residual. */
export class TransformerBlock extends Module {
    ln1: LayerNorm;
    attn: MultiheadAttention;
    ln2: LayerNorm;
    fc1: Linear;
    fc2: Linear;

    constructor(embedDim: number, numHeads: number, opts: { requiresGrad?: boolean } = {}) {
        super();
        this.ln1 = new LayerNorm(embedDim, opts);
        this.attn = new MultiheadAttention(embedDim, numHeads, opts);
        this.ln2 = new LayerNorm(embedDim, opts);
        this.fc1 = new Linear(embedDim, embedDim * 4, opts);
        this.fc2 = new Linear(embedDim * 4, embedDim, opts);
    }

    forward(x: Tensor): Tensor {
        const h = x.add(this.attn.forward(this.ln1.forward(x)));
        const mlp = this.fc2.forward(this.fc1.forward(this.ln2.forward(h)).relu());
        return h.add(mlp);
    }
}

/** Tiny decoder-only LM: token + position embed → N blocks → LN → tied logits via Linear. */
export class TinyTransformer extends Module {
    tokEmbed: Embedding;
    posEmbed: Embedding;
    blocks: TransformerBlock[];
    lnF: LayerNorm;
    head: Linear;

    constructor(
        readonly vocabSize: number,
        readonly maxSeqLen: number,
        readonly embedDim: number,
        readonly numHeads: number,
        readonly numLayers: number,
        opts: { requiresGrad?: boolean } = {},
    ) {
        super();
        this.tokEmbed = new Embedding(vocabSize, embedDim, opts);
        this.posEmbed = new Embedding(maxSeqLen, embedDim, opts);
        this.blocks = Array.from({ length: numLayers }, () => new TransformerBlock(embedDim, numHeads, opts));
        this.lnF = new LayerNorm(embedDim, opts);
        this.head = new Linear(embedDim, vocabSize, opts);
    }

    /** `tokens` shape `[B, T]` with integer ids stored as f32. */
    forward(tokens: Tensor): Tensor {
        const [b, t] = tokens.shape;
        if (b === undefined || t === undefined || tokens.shape.length !== 2) {
            throw new Error(`TinyTransformer expects [B,T], got [${tokens.shape.join(',')}]`);
        }
        if (t > this.maxSeqLen) {
            throw new Error(`seq len ${t} exceeds maxSeqLen ${this.maxSeqLen}`);
        }
        const posIds = tensor(
            Array.from({ length: b * t }, (_, i) => i % t),
            [b, t],
        );
        let x = this.tokEmbed.forward(tokens).add(this.posEmbed.forward(posIds));
        for (const block of this.blocks) {
            x = block.forward(x);
        }
        return this.head.forward(this.lnF.forward(x));
    }
}
