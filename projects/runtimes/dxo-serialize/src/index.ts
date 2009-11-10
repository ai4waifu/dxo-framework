/** Wire format name for DXO preview checkpoints. */
export const STATE_FORMAT = 'dxo-state' as const;

/** Current document version (bump only with a migration path). */
export const STATE_VERSION = 1 as const;

export type Dtype = 'f32';

/** One dense tensor payload (row-major f32 as JS numbers). */
export interface TensorBlob {
    shape: number[];
    data: number[];
    dtype: Dtype;
}

/** Versioned named-tensor document. */
export interface StateDocumentV1 {
    format: typeof STATE_FORMAT;
    version: typeof STATE_VERSION;
    tensors: Record<string, TensorBlob>;
}

export type StateDocument = StateDocumentV1;

export interface TensorStateSlice {
    shape: number[];
    data: number[];
}

/** `@dxo/nn` Linear.state() shape (duplicated to avoid a hard nn dependency). */
export interface LinearState {
    weight: TensorStateSlice;
    bias: TensorStateSlice;
}

function numel(shape: number[]): number {
    return shape.reduce((n, d) => n * d, 1);
}

function asBlob(slice: TensorStateSlice): TensorBlob {
    if (slice.data.length !== numel(slice.shape)) {
        throw new Error(`tensor blob length ${slice.data.length} != product(shape)=${numel(slice.shape)}`);
    }
    return {
        shape: [...slice.shape],
        data: [...slice.data],
        dtype: 'f32',
    };
}

function fromBlob(blob: TensorBlob): TensorStateSlice {
    if (blob.dtype !== 'f32') throw new Error(`unsupported dtype ${blob.dtype}`);
    if (blob.data.length !== numel(blob.shape)) {
        throw new Error(`tensor blob length ${blob.data.length} != product(shape)=${numel(blob.shape)}`);
    }
    return { shape: [...blob.shape], data: [...blob.data] };
}

/** Pack a map of tensor slices into a v1 state document. */
export function packTensors(tensors: Record<string, TensorStateSlice>): StateDocumentV1 {
    const out: Record<string, TensorBlob> = {};
    for (const [name, slice] of Object.entries(tensors)) {
        out[name] = asBlob(slice);
    }
    return { format: STATE_FORMAT, version: STATE_VERSION, tensors: out };
}

/** Unpack tensor slices from a v1 state document. */
export function unpackTensors(doc: StateDocument): Record<string, TensorStateSlice> {
    assertDocument(doc);
    const out: Record<string, TensorStateSlice> = {};
    for (const [name, blob] of Object.entries(doc.tensors)) {
        out[name] = fromBlob(blob);
    }
    return out;
}

/** Encode `@dxo/nn` Linear.state() into a versioned document. */
export function encodeLinearState(state: LinearState): StateDocumentV1 {
    return packTensors({
        weight: state.weight,
        bias: state.bias,
    });
}

/** Decode a document produced by `encodeLinearState`. */
export function decodeLinearState(doc: StateDocument): LinearState {
    const tensors = unpackTensors(doc);
    const weight = tensors.weight;
    const bias = tensors.bias;
    if (!weight || !bias) throw new Error('Linear state requires weight and bias tensors');
    return { weight, bias };
}

export function encodeJson(doc: StateDocument): string {
    assertDocument(doc);
    return JSON.stringify(doc);
}

export function decodeJson(text: string): StateDocument {
    const doc = JSON.parse(text) as StateDocument;
    assertDocument(doc);
    return doc;
}

export {
    decodeSafetensors,
    encodeSafetensors,
    type SafetensorSlice,
} from './safetensors.js';

function assertDocument(doc: StateDocument): asserts doc is StateDocumentV1 {
    if (!doc || typeof doc !== 'object') throw new Error('invalid state document');
    if (doc.format !== STATE_FORMAT) throw new Error(`unexpected format '${String((doc as { format?: string }).format)}'`);
    if (doc.version !== STATE_VERSION) {
        throw new Error(`unsupported state version ${String((doc as { version?: number }).version)}`);
    }
    if (!doc.tensors || typeof doc.tensors !== 'object') throw new Error('state document missing tensors');
}
