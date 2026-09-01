/**
 * image-buffer: typed host pixel carrier + PNG decode bridge.
 */
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { createImageBufferFromPixels, decodeImageBuffer, DxoError } from '@dxo/core';

// HWC u8 → NCHW f32 normalize
{
    const data = Uint8Array.from([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
    const buf = createImageBufferFromPixels({
        width: 2,
        height: 2,
        channels: 3,
        dtype: 'u8',
        layout: 'HWC',
        colorSpace: 'rgb',
        alphaMode: 'opaque',
        data,
    });
    assert.equal(buf.width, 2);
    assert.equal(buf.height, 2);
    assert.equal(buf.channels, 3);
    assert.equal(buf.pixelBytes().length, 12);
    const t = buf.toTensor({ normalize: true });
    assert.deepEqual([...t.shape], [1, 3, 2, 2]);
    const arr = await t.toArray();
    assert.ok(Math.abs(arr[0]! - 1.0) < 1e-6);
    assert.ok(Math.abs(arr[5]! - 1.0) < 1e-6);
}

// Minimal 1×1 red PNG decode
{
    const png = Buffer.from(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
        'base64',
    );
    const buf = await decodeImageBuffer(png, { format: 'png' });
    assert.equal(buf.width, 1);
    assert.equal(buf.height, 1);
    assert.ok(buf.channels === 3 || buf.channels === 4);
    const t = buf.toTensor();
    assert.deepEqual([...t.shape], [1, buf.channels, 1, 1]);
}

// JPEG explicitly rejected (no silent fallback)
{
    assert.throws(
        () => decodeImageBuffer(Buffer.from([0xff, 0xd8, 0xff]), { format: 'jpeg' }),
        (err: unknown) => err instanceof DxoError && (err as DxoError).code === 'DXO_IMAGE_DECODE_UNSUPPORTED',
    );
}

console.log('image-buffer: ok');
