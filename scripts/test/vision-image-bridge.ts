/**
 * vision-image-bridge: @dxo/vision Image over @dxo/core ImageBuffer.
 */
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { fromDecode, fromPixels } from '@dxo/vision';

{
    const data = Uint8Array.from([128, 64, 32]);
    const img = fromPixels({
        width: 1,
        height: 1,
        channels: 3,
        colorSpace: 'rgb',
        data,
    });
    assert.equal(img.width, 1);
    assert.equal(img.channels, 3);
    const t = img.toTensor();
    assert.deepEqual([...t.shape], [1, 3, 1, 1]);
    img.dispose();
}

{
    const png = Buffer.from(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
        'base64',
    );
    const img = await fromDecode(png, { format: 'png', source: { kind: 'test' } });
    assert.equal(img.width, 1);
    assert.equal(img.source?.kind, 'test');
    const t = img.toTensor();
    assert.equal(t.shape[0], 1);
    assert.ok(t.shape[1]! >= 3);
    img.dispose();
}

console.log('vision-image-bridge: ok');
