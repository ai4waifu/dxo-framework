/**
 * Minimal locale catalog for `@dxo/core` diagnostics (Living 15).
 * Not a public `@dxo/i18n` product pack — mechanism + core templates only.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { DxoDiagnostic } from './errors.js';

export type DxoLocale = string;

type Catalog = Record<string, string>;

const cache = new Map<string, Catalog>();

const CODE_TO_KEY: Record<string, string> = {
    DXO_DEVICE_UNAVAILABLE: 'dxo.backend.device_unavailable',
    DXO_BACKEND_UNAVAILABLE: 'dxo.backend.unavailable',
    DXO_TENSOR_NON_SCALAR: 'dxo.tensor.non_scalar',
    DXO_TENSOR_INVALID_SHAPE: 'dxo.tensor.invalid_shape',
    DXO_TENSOR_REQUIRES_DETACH: 'dxo.tensor.requires_detach',
    DXO_TENSOR_BROADCAST_INCOMPATIBLE: 'dxo.tensor.broadcast_incompatible',
    DXO_AUTOGRAD_FAILED: 'dxo.autograd.failed',
    DXO_DEVICE_ERROR: 'dxo.backend.device_error',
    DXO_TITAN_DEVICE_NOT_FOUND: 'dxo.titan.device_not_found',
    DXO_TITAN_DEVICE_LOST: 'dxo.titan.device_lost',
    DXO_TITAN_CROSS_DEVICE: 'dxo.titan.cross_device',
    DXO_TITAN_CROSS_STREAM: 'dxo.titan.cross_stream',
    DXO_TITAN_EVENT_WAIT_FAILED: 'dxo.titan.event_wait_failed',
    DXO_TITAN_ALLOCATION_FAILED: 'dxo.titan.allocation_failed',
    DXO_TITAN_INVALID_ABI: 'dxo.titan.invalid_abi',
    DXO_TITAN_KERNEL_UNSUPPORTED: 'dxo.titan.kernel_unsupported',
    DXO_TITAN_KERNEL_LAUNCH_FAILED: 'dxo.titan.kernel_launch_failed',
    DXO_TITAN_READBACK_FAILED: 'dxo.titan.readback_failed',
    DXO_TITAN_UPLOAD_FAILED: 'dxo.titan.upload_failed',
    DXO_TITAN_BACKEND_UNAVAILABLE: 'dxo.titan.backend_unavailable',
    DXO_TITAN_UNKNOWN: 'dxo.titan.unknown',
    DXO_UNKNOWN: 'dxo.unknown',
};

function packageRoot(): string {
    return join(dirname(fileURLToPath(import.meta.url)), '..');
}

function normalizeLocale(tag: string): string {
    const t = tag.trim().replace(/_/g, '-');
    if (!t) return 'en-US';
    const lower = t.toLowerCase();
    if (lower === 'en' || lower.startsWith('en-')) return 'en-US';
    if (lower === 'zh' || lower.startsWith('zh-')) return 'zh-CN';
    return t;
}

function flatten(obj: unknown, prefix = ''): Catalog {
    const out: Catalog = {};
    if (!obj || typeof obj !== 'object') return out;
    for (const [k, v] of Object.entries(obj as Record<string, unknown>)) {
        const key = prefix ? `${prefix}.${k}` : k;
        if (v && typeof v === 'object' && !Array.isArray(v)) {
            Object.assign(out, flatten(v, key));
        } else if (typeof v === 'string') {
            out[key] = v;
        }
    }
    return out;
}

function loadCatalog(locale: string): Catalog {
    const normalized = normalizeLocale(locale);
    const hit = cache.get(normalized);
    if (hit) return hit;
    const path = join(packageRoot(), 'locales', normalized, 'core.json');
    try {
        const raw = JSON.parse(readFileSync(path, 'utf8')) as unknown;
        const flat = flatten(raw);
        cache.set(normalized, flat);
        return flat;
    } catch {
        if (normalized !== 'en-US') return loadCatalog('en-US');
        const empty: Catalog = {};
        cache.set(normalized, empty);
        return empty;
    }
}

function interpolate(template: string, args?: Record<string, string | number | boolean>): string {
    if (!args) return template;
    return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (_, name: string) => {
        const v = args[name];
        return v === undefined ? `{${name}}` : String(v);
    });
}

/** Resolve locale: explicit → DXO_LOCALE → LANG → en-US. */
export function resolveLocale(explicit?: string | null): string {
    if (explicit && explicit.trim()) return normalizeLocale(explicit);
    const env = process.env.DXO_LOCALE || process.env.LC_ALL || process.env.LANG;
    if (env && env.trim()) return normalizeLocale(env.split('.')[0] ?? env);
    return 'en-US';
}

/** Format a diagnostic with a locale catalog (falls back to `message` / code). */
export function formatDiagnostic(diag: DxoDiagnostic, locale?: string | null): string {
    const loc = resolveLocale(locale);
    const catalog = loadCatalog(loc);
    const key = CODE_TO_KEY[diag.code];
    const template = (key && catalog[key]) || (loc !== 'en-US' ? loadCatalog('en-US')[key ?? ''] : undefined);
    if (template) return interpolate(template, diag.args);
    return diag.message || diag.code;
}
