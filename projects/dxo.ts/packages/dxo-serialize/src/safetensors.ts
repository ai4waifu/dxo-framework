/**
 * Safetensors codec (HF layout) ↔ DXO tensor slice maps.
 * v0 supports **F32** tensors only (matches current training state).
 */

export type SafetensorSlice = {
    shape: number[];
    data: number[];
};

const DTYPE_F32 = 'F32';
const HEADER_ALIGN = 8;

type SafetensorInfo = {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
};

type SafetensorHeader = Record<string, SafetensorInfo | Record<string, string>>;

function numel(shape: number[]): number {
    return shape.reduce((n, d) => n * d, 1);
}

function alignUp(n: number, align: number): number {
    return Math.ceil(n / align) * align;
}

function isTensorInfo(value: unknown): value is SafetensorInfo {
    if (!value || typeof value !== 'object') return false;
    const v = value as SafetensorInfo;
    return typeof v.dtype === 'string' && Array.isArray(v.shape) && Array.isArray(v.data_offsets) && v.data_offsets.length === 2;
}

/** Encode named f32 tensors to a safetensors binary buffer. */
export function encodeSafetensors(tensors: Record<string, SafetensorSlice>): Uint8Array {
    const names = Object.keys(tensors).sort();
    if (names.length === 0) throw new Error('encodeSafetensors: empty tensor map');

    const header: SafetensorHeader = {};
    const chunks: Float32Array[] = [];
    let offset = 0;

    for (const name of names) {
        const slice = tensors[name]!;
        const n = numel(slice.shape);
        if (slice.data.length !== n) {
            throw new Error(`encodeSafetensors: '${name}' length ${slice.data.length} != product(shape)=${n}`);
        }
        const byteLen = n * 4;
        header[name] = {
            dtype: DTYPE_F32,
            shape: [...slice.shape],
            data_offsets: [offset, offset + byteLen],
        };
        chunks.push(Float32Array.from(slice.data));
        offset += byteLen;
    }

    const headerJson = Buffer.from(JSON.stringify(header), 'utf8');
    const headerLen = alignUp(headerJson.length, HEADER_ALIGN);
    const headerPadded = Buffer.alloc(headerLen, 0x20);
    headerJson.copy(headerPadded);

    const out = Buffer.alloc(8 + headerLen + offset);
    out.writeBigUInt64LE(BigInt(headerLen), 0);
    headerPadded.copy(out, 8);

    let dataAt = 8 + headerLen;
    for (const chunk of chunks) {
        Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength).copy(out, dataAt);
        dataAt += chunk.byteLength;
    }

    return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
}

/** Decode a safetensors buffer into named f32 tensor slices. */
export function decodeSafetensors(bytes: Uint8Array): Record<string, SafetensorSlice> {
    if (bytes.byteLength < 8) throw new Error('decodeSafetensors: buffer too short');
    const view = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const headerLen = Number(view.readBigUInt64LE(0));
    if (headerLen <= 0 || 8 + headerLen > view.byteLength) {
        throw new Error(`decodeSafetensors: invalid header length ${headerLen}`);
    }

    const headerText = view
        .subarray(8, 8 + headerLen)
        .toString('utf8')
        .trim();
    let header: SafetensorHeader;
    try {
        header = JSON.parse(headerText) as SafetensorHeader;
    } catch (err) {
        throw new Error(`decodeSafetensors: bad header JSON (${err instanceof Error ? err.message : String(err)})`);
    }

    const dataBase = 8 + headerLen;
    const out: Record<string, SafetensorSlice> = {};

    for (const [name, info] of Object.entries(header)) {
        if (name === '__metadata__') continue;
        if (!isTensorInfo(info)) {
            throw new Error(`decodeSafetensors: invalid tensor entry '${name}'`);
        }
        if (info.dtype !== DTYPE_F32) {
            throw new Error(`decodeSafetensors: unsupported dtype '${info.dtype}' for '${name}' (v0 is F32-only)`);
        }
        const [begin, end] = info.data_offsets;
        if (begin < 0 || end < begin || dataBase + end > view.byteLength) {
            throw new Error(`decodeSafetensors: bad data_offsets for '${name}'`);
        }
        const expected = numel(info.shape) * 4;
        if (end - begin !== expected) {
            throw new Error(`decodeSafetensors: '${name}' byte span ${end - begin} != ${expected}`);
        }
        // Copy into a fresh buffer so Float32Array alignment is always valid
        // even when the source Uint8Array is a non-aligned subarray view.
        const raw = Buffer.allocUnsafe(end - begin);
        view.copy(raw, 0, dataBase + begin, dataBase + end);
        const f32 = new Float32Array(raw.buffer, raw.byteOffset, numel(info.shape));
        out[name] = {
            shape: [...info.shape],
            data: Array.from(f32),
        };
    }

    return out;
}
