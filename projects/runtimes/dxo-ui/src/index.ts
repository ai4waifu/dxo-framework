/**
 * Placeholder for Gradio-like model apps (Living `10` U3).
 * Rendering/serve will use VMZ later; this package stays protocol-facing.
 */

export type ModelAppDefinition = {
    title: string;
    // Full component schema lands in U3; keep opaque for the stub.
    input?: unknown;
    output?: unknown;
    run?: (...args: never[]) => AsyncIterable<unknown> | Promise<unknown>;
};

export type ModelApp = {
    readonly kind: 'placeholder';
    readonly title: string;
    serve(options?: { host?: string; port?: number }): Promise<never>;
};

export function defineModelApp(def: ModelAppDefinition): ModelApp {
    const title = def.title || 'untitled';
    return {
        kind: 'placeholder',
        title,
        async serve() {
            throw new Error(`@dxo/ui defineModelApp('${title}').serve() is a placeholder; VMZ serve lands in a later slice`);
        },
    };
}

export function uiVersion(): string {
    return 'dxo-ui@placeholder';
}
