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

    /** Flat named tensors for safetensors / checkpoint (stable keys). */
    async state(): Promise<Record<string, TensorStateSlice>> {
        const out: Record<string, TensorStateSlice> = {};
        const put = async (key: string, t: Tensor) => {
            out[key] = { shape: [...t.shape], data: await t.toArray() };
        };
        await put('tok_embed.weight', this.tokEmbed.weight);
        await put('pos_embed.weight', this.posEmbed.weight);
        for (let i = 0; i < this.blocks.length; i++) {
            const block = this.blocks[i]!;
            const p = `blocks.${i}`;
            await put(`${p}.ln1.weight`, block.ln1.weight);
            await put(`${p}.ln1.bias`, block.ln1.bias);
            await put(`${p}.attn.qkv.weight`, block.attn.qkv.weight);
            await put(`${p}.attn.qkv.bias`, block.attn.qkv.bias);
            await put(`${p}.attn.proj.weight`, block.attn.proj.weight);
            await put(`${p}.attn.proj.bias`, block.attn.proj.bias);
            await put(`${p}.ln2.weight`, block.ln2.weight);
            await put(`${p}.ln2.bias`, block.ln2.bias);
            await put(`${p}.fc1.weight`, block.fc1.weight);
            await put(`${p}.fc1.bias`, block.fc1.bias);
            await put(`${p}.fc2.weight`, block.fc2.weight);
            await put(`${p}.fc2.bias`, block.fc2.bias);
        }
        await put('ln_f.weight', this.lnF.weight);
        await put('ln_f.bias', this.lnF.bias);
        await put('head.weight', this.head.weight);
        await put('head.bias', this.head.bias);
        return out;
    }

    loadState(saved: Record<string, TensorStateSlice>, opts: { requiresGrad?: boolean } = {}): void {
        const rg = opts.requiresGrad ?? true;
        const take = (key: string): Tensor => {
            const slice = saved[key];
            if (!slice) throw new Error(`TinyTransformer.loadState: missing '${key}'`);
            return tensor(slice.data, slice.shape, { requiresGrad: rg });
        };
        this.tokEmbed.weight = take('tok_embed.weight');
        this.posEmbed.weight = take('pos_embed.weight');
        for (let i = 0; i < this.blocks.length; i++) {
            const block = this.blocks[i]!;
            const p = `blocks.${i}`;
            block.ln1.weight = take(`${p}.ln1.weight`);
            block.ln1.bias = take(`${p}.ln1.bias`);
            block.attn.qkv.weight = take(`${p}.attn.qkv.weight`);
            block.attn.qkv.bias = take(`${p}.attn.qkv.bias`);
            block.attn.proj.weight = take(`${p}.attn.proj.weight`);
            block.attn.proj.bias = take(`${p}.attn.proj.bias`);
            block.ln2.weight = take(`${p}.ln2.weight`);
            block.ln2.bias = take(`${p}.ln2.bias`);
            block.fc1.weight = take(`${p}.fc1.weight`);
            block.fc1.bias = take(`${p}.fc1.bias`);
            block.fc2.weight = take(`${p}.fc2.weight`);
            block.fc2.bias = take(`${p}.fc2.bias`);
        }
        this.lnF.weight = take('ln_f.weight');
        this.lnF.bias = take('ln_f.bias');
        this.head.weight = take('head.weight');
        this.head.bias = take('head.bias');
    }
}

export interface Conv2dState {
    weight: TensorStateSlice;
    bias: TensorStateSlice;
}

/** 2D convolution NCHW / OIHW. */
export class Conv2d extends Module {
    weight: Tensor;
    bias: Tensor;

    constructor(
        readonly inChannels: number,
        readonly outChannels: number,
        readonly kernelSize: number,
        opts: { stride?: number; padding?: number; requiresGrad?: boolean } = {},
    ) {
        super();
        this.stride = opts.stride ?? 1;
        this.padding = opts.padding ?? 0;
        const rg = opts.requiresGrad ?? true;
        const fanIn = inChannels * kernelSize * kernelSize;
        const scale = Math.sqrt(2 / fanIn);
        const n = outChannels * inChannels * kernelSize * kernelSize;
        const raw = randnValues([n]).map((v) => v * scale);
        this.weight = tensor(raw, [outChannels, inChannels, kernelSize, kernelSize], { requiresGrad: rg });
        this.bias = zeros([outChannels], { requiresGrad: rg });
    }

    readonly stride: number;
    readonly padding: number;

    forward(x: Tensor): Tensor {
        return x.conv2d(this.weight, this.bias, this.stride, this.padding);
    }

    async state(): Promise<Conv2dState> {
        return {
            weight: { shape: [...this.weight.shape], data: await this.weight.toArray() },
            bias: { shape: [...this.bias.shape], data: await this.bias.toArray() },
        };
    }

    loadState(saved: Conv2dState, opts: { requiresGrad?: boolean } = {}): void {
        const rg = opts.requiresGrad ?? true;
        this.weight = tensor(saved.weight.data, saved.weight.shape, { requiresGrad: rg });
        this.bias = tensor(saved.bias.data, saved.bias.shape, { requiresGrad: rg });
    }
}

/** Max pooling 2D NCHW. */
export class MaxPool2d extends Module {
    constructor(
        readonly kernelSize: number,
        opts: { stride?: number; padding?: number } = {},
    ) {
        super();
        this.stride = opts.stride ?? kernelSize;
        this.padding = opts.padding ?? 0;
    }

    readonly stride: number;
    readonly padding: number;

    forward(x: Tensor): Tensor {
        return x.maxPool2d(this.kernelSize, this.stride, this.padding);
    }
}

export interface BatchNorm2dState {
    weight: TensorStateSlice;
    bias: TensorStateSlice;
}

/** Batch norm 2D (per-batch stats, training-style). */
export class BatchNorm2d extends Module {
    weight: Tensor;
    bias: Tensor;

    constructor(
        readonly numFeatures: number,
        opts: { eps?: number; requiresGrad?: boolean } = {},
    ) {
        super();
        this.eps = opts.eps ?? 1e-5;
        const rg = opts.requiresGrad ?? true;
        this.weight = ones([numFeatures], { requiresGrad: rg });
        this.bias = zeros([numFeatures], { requiresGrad: rg });
    }

    readonly eps: number;

    forward(x: Tensor): Tensor {
        return x.batchNorm2d(this.weight, this.bias, this.eps);
    }

    async state(): Promise<BatchNorm2dState> {
        return {
            weight: { shape: [...this.weight.shape], data: await this.weight.toArray() },
            bias: { shape: [...this.bias.shape], data: await this.bias.toArray() },
        };
    }

    loadState(saved: BatchNorm2dState, opts: { requiresGrad?: boolean } = {}): void {
        const rg = opts.requiresGrad ?? true;
        this.weight = tensor(saved.weight.data, saved.weight.shape, { requiresGrad: rg });
        this.bias = tensor(saved.bias.data, saved.bias.shape, { requiresGrad: rg });
    }
}

/** Tiny CNN: Conv → BN → ReLU → Pool → Linear. */
export class TinyCnn extends Module {
    conv: Conv2d;
    bn: BatchNorm2d;
    pool: MaxPool2d;
    fc: Linear;

    constructor(
        readonly inChannels: number,
        readonly numClasses: number,
        opts: { channels?: number; spatial?: number; requiresGrad?: boolean } = {},
    ) {
        super();
        const ch = opts.channels ?? 4;
        const spatial = opts.spatial ?? 8;
        const rg = opts.requiresGrad ?? true;
        this.conv = new Conv2d(inChannels, ch, 3, { padding: 1, requiresGrad: rg });
        this.bn = new BatchNorm2d(ch, { requiresGrad: rg });
        this.pool = new MaxPool2d(2);
        const flat = ch * Math.floor(spatial / 2) * Math.floor(spatial / 2);
        this.fc = new Linear(flat, numClasses, { requiresGrad: rg });
    }

    forward(x: Tensor): Tensor {
        let h = this.conv.forward(x);
        h = this.bn.forward(h);
        h = relu(h);
        h = this.pool.forward(h);
        const n = h.shape[0]!;
        return this.fc.forward(h.reshape([n, h.numel() / n]));
    }

    async state(): Promise<Record<string, TensorStateSlice>> {
        const c = await this.conv.state();
        const b = await this.bn.state();
        const f = await this.fc.state();
        return {
            'conv.weight': c.weight,
            'conv.bias': c.bias,
            'bn.weight': b.weight,
            'bn.bias': b.bias,
            'fc.weight': f.weight,
            'fc.bias': f.bias,
        };
    }

    loadState(saved: Record<string, TensorStateSlice>, opts: { requiresGrad?: boolean } = {}): void {
        const rg = opts.requiresGrad ?? true;
        const take = (k: string) => {
            const t = saved[k];
            if (!t) throw new Error(`TinyCnn missing state key ${k}`);
            return tensor(t.data, t.shape, { requiresGrad: rg });
        };
        this.conv.weight = take('conv.weight');
        this.conv.bias = take('conv.bias');
        this.bn.weight = take('bn.weight');
        this.bn.bias = take('bn.bias');
        this.fc.weight = take('fc.weight');
        this.fc.bias = take('fc.bias');
    }
}
