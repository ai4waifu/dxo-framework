/** Default Studio API base (loopback). Override with window.__DXO_STUDIO_API__. */
export function studioApiBase(): string {
    const w = globalThis as typeof globalThis & { __DXO_STUDIO_API__?: string };
    if (typeof w.__DXO_STUDIO_API__ === 'string' && w.__DXO_STUDIO_API__.length > 0) {
        return w.__DXO_STUDIO_API__.replace(/\/$/, '');
    }
    return 'http://127.0.0.1:4310';
}

export async function apiGet<T>(path: string): Promise<T> {
    const url = `${studioApiBase()}${path.startsWith('/') ? path : `/${path}`}`;
    const res = await fetch(url);
    if (!res.ok) {
        throw new Error(`Studio API ${res.status} for ${path}`);
    }
    return (await res.json()) as T;
}
