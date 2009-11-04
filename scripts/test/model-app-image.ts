import assert from 'node:assert/strict';
import { defineModelApp, image, labels } from '@dxo/ui';

/**
 * model-app-image: loopback image upload, streaming-ish run, cancel smoke.
 */

const app = defineModelApp({
    title: 'demo-classifier',
    input: image({ sources: ['upload'] }),
    output: labels({ limit: 3 }),
    async *run(input, ctx) {
        ctx.progress?.({ stage: 'decode' });
        if (ctx.signal.aborted) throw new DOMException('aborted', 'AbortError');
        await new Promise((r) => setTimeout(r, 5));
        ctx.progress?.({ stage: 'infer' });
        yield { stage: 'partial', size: input.bytes.length };
        return {
            label: input.bytes.length > 0 ? 1 : 0,
            scores: [0.1, 0.9],
        };
    },
});

const server = await app.serve({ host: '127.0.0.1', port: 0 });

try {
    const health = await (await fetch(`${server.url}/health`)).json();
    assert.equal(health.ok, true);

    const form = new FormData();
    form.append('image', new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }), 'x.png');
    const res = await fetch(`${server.url}/run`, { method: 'POST', body: form });
    assert.equal(res.status, 200);
    const body = await res.json();
    assert.equal(body.output.label, 1);
    assert.deepEqual(body.output.scores, [0.1, 0.9]);

    const controller = new AbortController();
    const slow = fetch(`${server.url}/run`, {
        method: 'POST',
        body: form,
        signal: controller.signal,
    });
    controller.abort();
    await slow.catch((err) => {
        assert.equal(err.name, 'AbortError');
    });

    console.log(`model-app-image ok: ${server.url}`);
} finally {
    await server.close();
}
