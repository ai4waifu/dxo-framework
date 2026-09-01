import { Buffer } from 'node:buffer';
import { embedding, ones, randnValues, Tensor, tensor, tensorFromF32Buffer, zeros, zeroGrads } from '@dxo/core';

export interface TensorStateSlice {
    shape: number[];
    data: number[];
}

/** Buffer-backed slice for Rust safetensors checkpoint restore. */
export interface StateBufferSlice {
    shape: number[];
    data: Buffer;
}

export type State = Record<string, TensorStateSlice>;

export interface LinearState {
    weight: TensorStateSlice;
    bias: TensorStateSlice;
}

export abstract class NeuralNetwork {
    #parent: NeuralNetwork | undefined;
    #localName: string | undefined;
    #children = new Map<string, NeuralNetwork>();
    #nextOrdinals = new Map<string, number>();
    #reserved = new Set(['weight', 'bias', 'running_mean', 'running_variance', 'num_batches_tracked', 'state', 'grad']);

    abstract forward(x: Tensor): Tensor;

    protected registerChild<T extends NeuralNetwork>(
        child: T,
        options: { name?: string; semanticName?: string; mode?: 'singleton' | 'repeatable' } = {},
    ): T {
        if ((child as NeuralNetwork) === this) throw new Error('DUPLICATE_CHILD_NAME: a NeuralNetwork cannot register itself');
        const semanticName = options.semanticName ?? child.semanticName();
        const mode = options.mode ?? 'singleton';
        validateName(semanticName, false);
        const localName = options.name ?? (mode === 'singleton' ? semanticName : this.allocateName(semanticName));
        validateName(localName, true);
        if (this.#reserved.has(localName)) throw new Error(`RESERVED_NAME: '${localName}'`);
        if (this.#children.has(localName)) throw new Error(`DUPLICATE_CHILD_NAME: '${localName}'`);
        if (
            mode === 'singleton' &&
            options.name === undefined &&
            [...this.#children.values()].some((item) => item.semanticName() === semanticName)
        ) {
            throw new Error(`DUPLICATE_SINGLETON_NODE: '${semanticName}'`);
        }
        if (child.#parent) throw new Error('DUPLICATE_CHILD_NAME: child is already registered');
        child.#parent = this;
        child.#localName = localName;
        this.#children.set(localName, child);
        return child;
    }

    protected semanticName(): string {
        throw new Error('INVALID_CANONICAL_NAME: custom NeuralNetwork must declare semanticName');
    }

    protected canonicalPath(): string {
        const own = this.#localName ?? '';
        if (!this.#parent) return own;
        const parent = this.#parent.canonicalPath();
        return parent ? `${parent}.${own}` : own;
    }

    /** Internal canonical path used by composite Neural implementations. */
    readonly canonicalName = (): string => this.canonicalPath();

    protected registeredChildren(): ReadonlyMap<string, NeuralNetwork> {
        return this.#children;
    }

    private allocateName(semanticName: string): string {
        let ordinal = this.#nextOrdinals.get(semanticName) ?? 1;
        for (;;) {
            const candidate = `${semanticName}_${ordinal}`;
            ordinal += 1;
            if (!this.#children.has(candidate)) {
                this.#nextOrdinals.set(semanticName, ordinal);
                return candidate;
            }
        }
    }

    parameters(): Tensor[] {
        const out: Tensor[] = [];
        const visit = (value: unknown): void => {
            if (value instanceof Tensor) {
                out.push(value);
                return;
            }
            if (value instanceof NeuralNetwork) {
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
        zeroGrads(this.parameters());
    }
}

export abstract class Layer extends NeuralNetwork {}

function validateName(name: string, segment: boolean): void {
    const pattern = segment ? /^[a-z][a-z0-9_]*$/ : /^[a-z][a-z0-9_]*$/;
    if (!pattern.test(name) || (!segment && name.includes('.'))) {
        throw new Error(`INVALID_CANONICAL_NAME: '${name}'`);
    }
}

export function relu(x: Tensor): Tensor {
    return x.relu();
}

/** Element-wise ReLU module. */
export class ReLU extends Layer {
    protected semanticName(): string {
        return 'relu';
    }
    forward(x: Tensor): Tensor {
        return x.relu();
    }
}

/** Fully-connected affine map: `y = x @ weight + bias` (no activation).
 *
 * Weight layout: `[inFeatures, outFeatures]`. Default leaves use `requiresGrad: true`.
 * After `optimizer.step(parameters())`, call `loadParameters` to install new leaves.
 */
export class FullyConnected extends Layer {
    weight: Tensor;
    bias: Tensor;

    protected semanticName(): string {
        return 'fully_connected';
    }

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
            throw new Error(`FullyConnected expects rank >= 2, got [${x.shape.join(',')}]`);
        }
        const last = x.shape[x.shape.length - 1]!;
        if (last !== this.inFeatures) {
            throw new Error(`FullyConnected inFeatures ${this.inFeatures} != last dim ${last}`);
        }
        const leading = x.shape.slice(0, -1);
        const flat = leading.reduce((n, d) => n * d, 1);
        const y = x.reshape([flat, last]).matmul(this.weight).add(this.bias);
        return y.reshape([...leading, this.outFeatures]);
    }

    /** Replace parameter leaves after an optimizer step. */
    loadParameters(params: Tensor[]): void {
        if (params.length < 2) throw new Error('FullyConnected expects [weight, bias]');
        this.weight = params[0]!;
        this.bias = params[1]!;
    }

    async state(): Promise<LinearState> {
        return {
            weight: { shape: [...this.weight.shape], data: await this.weight.toArray() },
            bias: { shape: [...this.bias.shape], data: await this.bias.toArray() },
        };
    }

    /** Buffer-backed state for Rust safetensors encode (checkpoint hot path). */
    stateBuffers(): Record<string, StateBufferSlice> {
        return {
            weight: { shape: [...this.weight.shape], data: this.weight.toF32Buffer() },
            bias: { shape: [...this.bias.shape], data: this.bias.toF32Buffer() },
        };
    }

    loadState(saved: LinearState): void {
        this.weight = tensor(saved.weight.data, saved.weight.shape, { requiresGrad: true });
        this.bias = tensor(saved.bias.data, saved.bias.shape, { requiresGrad: true });
    }

    /** Restore from decoded safetensors buffers (no intermediate `number[]`). */
    loadStateFromBuffers(saved: Record<string, StateBufferSlice>, opts: { requiresGrad?: boolean } = {}): void {
        const rg = opts.requiresGrad ?? true;
        const weight = saved.weight;
        const bias = saved.bias;
        if (!weight || !bias) throw new Error('FullyConnected.loadStateFromBuffers: missing weight/bias');
        this.weight = tensorFromF32Buffer(weight.data, weight.shape, { requiresGrad: rg });
        this.bias = tensorFromF32Buffer(bias.data, bias.shape, { requiresGrad: rg });
    }
}

export class Sequential extends NeuralNetwork {
    constructor(readonly layers: NeuralNetwork[]) {
        super();
        this.layers = layers.map((layer) => this.registerChild(layer, { mode: 'repeatable' }));
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
export class Embedding extends Layer {
    weight: Tensor;

    protected semanticName(): string {
        return 'embedding';
    }

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
export class LayerNormalization extends Layer {
    weight: Tensor;
    bias: Tensor;

    protected semanticName(): string {
        return 'layer_normalization';
    }

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
        if (params.length < 2) throw new Error('LayerNormalization expects [weight, bias]');
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
export class SelfAttention extends Layer {
    qkv: FullyConnected;
    proj: FullyConnected;

    protected semanticName(): string {
        return 'self_attention';
    }

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
        this.qkv = new FullyConnected(embedDim, embedDim * 3, opts);
        this.proj = new FullyConnected(embedDim, embedDim, opts);
    }

    readonly headDim: number;

    forward(x: Tensor): Tensor {
        const [b, t, c] = x.shape;
        if (b === undefined || t === undefined || c === undefined || x.shape.length !== 3) {
            throw new Error(`SelfAttention expects [B,T,C], got [${x.shape.join(',')}]`);
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
export class TransformerBlock extends NeuralNetwork {
    ln1: LayerNormalization;
    attn: SelfAttention;
    ln2: LayerNormalization;
    fc1: FullyConnected;
    fc2: FullyConnected;

    protected semanticName(): string {
        return 'transformer_block';
    }

    constructor(embedDim: number, numHeads: number, opts: { requiresGrad?: boolean } = {}) {
        super();
        this.ln1 = new LayerNormalization(embedDim, opts);
        this.attn = new SelfAttention(embedDim, numHeads, opts);
        this.ln2 = new LayerNormalization(embedDim, opts);
        this.fc1 = new FullyConnected(embedDim, embedDim * 4, opts);
        this.fc2 = new FullyConnected(embedDim * 4, embedDim, opts);
        this.registerChild(this.ln1, { name: 'layer_normalization_1', mode: 'repeatable' });
        this.registerChild(this.attn, { name: 'self_attention', mode: 'singleton' });
        this.registerChild(this.ln2, { name: 'layer_normalization_2', mode: 'repeatable' });
        this.registerChild(this.fc1, { name: 'fully_connected_1', mode: 'repeatable' });
        this.registerChild(this.fc2, { name: 'fully_connected_2', mode: 'repeatable' });
    }

    forward(x: Tensor): Tensor {
        const h = x.add(this.attn.forward(this.ln1.forward(x)));
        const mlp = this.fc2.forward(this.fc1.forward(this.ln2.forward(h)).relu());
        return h.add(mlp);
    }
}

/** Tiny decoder-only LM: token + position embed → N blocks → LN → tied logits via Linear. */
export class TinyTransformer extends NeuralNetwork {
    tokEmbed: Embedding;
    posEmbed: Embedding;
    blocks: TransformerBlock[];
    lnF: LayerNormalization;
    head: FullyConnected;

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
        this.lnF = new LayerNormalization(embedDim, opts);
        this.head = new FullyConnected(embedDim, vocabSize, opts);
        this.registerChild(this.tokEmbed, { name: 'token_embedding' });
        this.registerChild(this.posEmbed, { name: 'position_embedding' });
        for (const block of this.blocks) this.registerChild(block, { mode: 'repeatable', semanticName: 'transformer_block' });
        this.registerChild(this.lnF, { name: 'layer_normalization_final' });
        this.registerChild(this.head, { name: 'fully_connected_head' });
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
    bias?: TensorStateSlice;
}

/** 2D convolution NCHW / OIHW. */
export class Convolution2d extends Layer {
    weight: Tensor;
    bias: Tensor | null;

    protected semanticName(): string {
        return 'convolution';
    }

    constructor(
        readonly inChannels: number,
        readonly outChannels: number,
        readonly kernelSize: number,
        opts: { stride?: number; padding?: number; bias?: boolean; requiresGrad?: boolean } = {},
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
        this.bias = opts.bias === false ? null : zeros([outChannels], { requiresGrad: rg });
    }

    readonly stride: number;
    readonly padding: number;

    forward(x: Tensor): Tensor {
        return x.conv2d(this.weight, this.bias, this.stride, this.padding);
    }

    async state(): Promise<Conv2dState> {
        const state: Conv2dState = {
            weight: { shape: [...this.weight.shape], data: await this.weight.toArray() },
        };
        if (this.bias) state.bias = { shape: [...this.bias.shape], data: await this.bias.toArray() };
        return state;
    }

    loadState(saved: Conv2dState, opts: { requiresGrad?: boolean } = {}): void {
        const rg = opts.requiresGrad ?? true;
        this.weight = tensor(saved.weight.data, saved.weight.shape, { requiresGrad: rg });
        this.bias = saved.bias ? tensor(saved.bias.data, saved.bias.shape, { requiresGrad: rg }) : null;
    }
}

/** Max pooling 2D NCHW. */
export class MaxPooling2d extends Layer {
    protected semanticName(): string {
        return 'max_pooling';
    }
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
    weight?: TensorStateSlice;
    bias?: TensorStateSlice;
    running_mean?: TensorStateSlice;
    running_variance?: TensorStateSlice;
    num_batches_tracked?: TensorStateSlice;
}

/** Batch norm 2D (per-batch stats, training-style). */
export class BatchNormalization2d extends Layer {
    weight: Tensor | null;
    bias: Tensor | null;
    readonly runningMean: Tensor | null;
    readonly runningVariance: Tensor | null;
    readonly numBatchesTracked: Tensor | null;

    protected semanticName(): string {
        return 'batch_normalization';
    }

    constructor(
        readonly numFeatures: number,
        opts: { eps?: number; affine?: boolean; trackRunningStatistics?: boolean; requiresGrad?: boolean } = {},
    ) {
        super();
        this.eps = opts.eps ?? 1e-5;
        const rg = opts.requiresGrad ?? true;
        const affine = opts.affine ?? true;
        const track = opts.trackRunningStatistics ?? false;
        this.weight = affine ? ones([numFeatures], { requiresGrad: rg }) : null;
        this.bias = affine ? zeros([numFeatures], { requiresGrad: rg }) : null;
        this.runningMean = track ? zeros([numFeatures]) : null;
        this.runningVariance = track ? ones([numFeatures]) : null;
        this.numBatchesTracked = track ? zeros([1]) : null;
    }

    readonly eps: number;

    forward(x: Tensor): Tensor {
        return x.batchNorm2d(this.weight ?? ones([this.numFeatures]), this.bias ?? zeros([this.numFeatures]), this.eps);
    }

    async state(): Promise<BatchNorm2dState> {
        const out: BatchNorm2dState = {};
        if (this.weight) out.weight = { shape: [...this.weight.shape], data: await this.weight.toArray() };
        if (this.bias) out.bias = { shape: [...this.bias.shape], data: await this.bias.toArray() };
        if (this.runningMean) out.running_mean = { shape: [...this.runningMean.shape], data: await this.runningMean.toArray() };
        if (this.runningVariance) out.running_variance = { shape: [...this.runningVariance.shape], data: await this.runningVariance.toArray() };
        if (this.numBatchesTracked)
            out.num_batches_tracked = { shape: [...this.numBatchesTracked.shape], data: await this.numBatchesTracked.toArray() };
        return out;
    }

    loadState(saved: BatchNorm2dState, opts: { requiresGrad?: boolean } = {}): void {
        const rg = opts.requiresGrad ?? true;
        this.weight = saved.weight ? tensor(saved.weight.data, saved.weight.shape, { requiresGrad: rg }) : null;
        this.bias = saved.bias ? tensor(saved.bias.data, saved.bias.shape, { requiresGrad: rg }) : null;
    }
}

/** Tiny CNN: Conv → BN → ReLU → Pool → Linear. */
export class TinyCnn extends NeuralNetwork {
    conv: Convolution2d;
    bn: BatchNormalization2d;
    pool: MaxPooling2d;
    fc: FullyConnected;

    constructor(
        readonly inChannels: number,
        readonly numClasses: number,
        opts: { channels?: number; spatial?: number; requiresGrad?: boolean } = {},
    ) {
        super();
        const ch = opts.channels ?? 4;
        const spatial = opts.spatial ?? 8;
        const rg = opts.requiresGrad ?? true;
        this.conv = new Convolution2d(inChannels, ch, 3, { padding: 1, requiresGrad: rg });
        this.bn = new BatchNormalization2d(ch, { requiresGrad: rg });
        this.pool = new MaxPooling2d(2);
        const flat = ch * Math.floor(spatial / 2) * Math.floor(spatial / 2);
        this.fc = new FullyConnected(flat, numClasses, { requiresGrad: rg });
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
            'conv.bias': c.bias!,
            'bn.weight': b.weight!,
            'bn.bias': b.bias!,
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
