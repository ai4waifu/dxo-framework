/** Default Studio API base (loopback). Override with window.__DXO_STUDIO_API__. */
export function studioApiBase(): string {
    const w = globalThis as typeof globalThis & { __DXO_STUDIO_API__?: string };
    if (typeof w.__DXO_STUDIO_API__ === 'string' && w.__DXO_STUDIO_API__.length > 0) {
        return w.__DXO_STUDIO_API__.replace(/\/$/, '');
    }
    return 'http://127.0.0.1:4310';
}

let shimReady: Promise<void> | null = null;

/**
 * Load `public/dxo-studio-api.js` (the launcher overwrites it with the bound
 * loopback port) before the first API read. Optional — missing shim keeps the
 * compiled-in default.
 */
export function ensureStudioApiShim(): Promise<void> {
    if (shimReady) return shimReady;
    if (typeof document === 'undefined') {
        shimReady = Promise.resolve();
        return shimReady;
    }
    shimReady = new Promise((resolve) => {
        const el = document.createElement('script');
        el.src = '/dxo-studio-api.js';
        el.dataset.dxoStudioApi = '1';
        el.onload = () => resolve();
        el.onerror = () => resolve();
        document.head.appendChild(el);
    });
    return shimReady;
}

export async function apiGet<T>(path: string): Promise<T> {
    await ensureStudioApiShim();
    const url = `${studioApiBase()}${path.startsWith('/') ? path : `/${path}`}`;
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`Studio API ${res.status} for ${path}`);
    }
    return (await res.json()) as T;
}

/** Loopback URL for run artifact bytes (images, etc.). */
export function fileUrl(runId: string, rel: string): string {
    const encoded = rel.split('/').map(encodeURIComponent).join('/');
    return `${studioApiBase()}/api/runs/${encodeURIComponent(runId)}/files/${encoded}`;
}

/** Default `@dxo/ui` model-app serve base for Playground. */
export function modelAppBase(): string {
    const w = globalThis as typeof globalThis & { __DXO_MODEL_APP__?: string };
    if (typeof w.__DXO_MODEL_APP__ === 'string' && w.__DXO_MODEL_APP__.length > 0) {
        return w.__DXO_MODEL_APP__.replace(/\/$/, '');
    }
    return 'http://127.0.0.1:7860';
}
