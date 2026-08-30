/**
 * Single source of truth for verify suites.
 * `pnpm verify`, `pnpm test:verify`, ordinary CI, and GPU artifact packing all read this registry.
 */

export type SuiteGroup = 'cpu' | 'contract' | 'product' | 'gpu-smoke' | 'gpu-parity' | 'gpu-residency' | 'gpu-model' | 'gpu-precision';

export type SuitePlatform = 'linux' | 'darwin' | 'win32' | 'all';

export type SuiteBackend = 'cpu' | 'cuda' | 'metal' | 'webgpu' | 'none';

export type SuiteDef = {
    id: string;
    /** Path relative to repo root. */
    script: string;
    group: SuiteGroup;
    packages: string[];
    platforms: SuitePlatform[];
    backend: SuiteBackend[];
    requiresGpu: boolean;
    requiresNetwork: boolean;
    allowSkip: boolean;
    timeoutMs: number;
    /** Pack into Modal GPU artifact even when not requiresGpu (e.g. CPU probe also run under CUDA). */
    includeInGpuArtifact?: boolean;
};

export const SUITES: SuiteDef[] = [
    {
        id: 'smoke',
        script: 'scripts/test/smoke.ts',
        group: 'cpu',
        packages: ['@dxo/core'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'tensor-cpu',
        script: 'scripts/test/tensor-cpu.ts',
        group: 'cpu',
        packages: ['@dxo/core'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'autograd-fd',
        script: 'scripts/test/autograd-fd.ts',
        group: 'cpu',
        packages: ['@dxo/core'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 120_000,
    },
    {
        id: 'nn-forward',
        script: 'scripts/test/nn-forward.ts',
        group: 'cpu',
        packages: ['@dxo/core', '@dxo/nn'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'mnist-linear',
        script: 'scripts/test/mnist-linear.ts',
        group: 'cpu',
        packages: ['@dxo/core', '@dxo/nn', '@dxo/optimizer'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 120_000,
    },
    {
        id: 'data-iter',
        script: 'scripts/test/data-iter.ts',
        group: 'cpu',
        packages: ['@dxo/data'],
        platforms: ['all'],
        backend: ['none'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'serialize-roundtrip',
        script: 'scripts/test/serialize-roundtrip.ts',
        group: 'cpu',
        packages: ['@dxo/serialize', '@dxo/nn'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'trainer-loop',
        script: 'scripts/test/trainer-loop.ts',
        group: 'cpu',
        packages: ['@dxo/train', '@dxo/nn', '@dxo/optimizer', '@dxo/data', '@dxo/serialize'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 120_000,
    },
    {
        id: 'train-batch-rust',
        script: 'scripts/test/train-batch-rust.ts',
        group: 'cpu',
        packages: ['@dxo/core', '@dxo/nn', '@dxo/optimizer', '@dxo/train'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 120_000,
    },
    {
        id: 'framework-core-cnn',
        script: 'scripts/test/framework-core-cnn.ts',
        group: 'cpu',
        packages: ['@dxo/core', '@dxo/nn', '@dxo/serialize'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 180_000,
    },
    {
        id: 'framework-core-transformer',
        script: 'scripts/test/framework-core-transformer.ts',
        group: 'cpu',
        packages: ['@dxo/core', '@dxo/nn', '@dxo/serialize'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 180_000,
    },
    {
        id: 'g3-contract',
        script: 'scripts/test/g3-contract.ts',
        group: 'contract',
        packages: ['@dxo/core', '@dxo/nn', '@dxo/optimizer'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'diagnostic-wire',
        script: 'scripts/test/diagnostic-wire.ts',
        group: 'contract',
        packages: ['@dxo/core'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'rust-error-code',
        script: 'scripts/test/rust-error-code.ts',
        group: 'contract',
        packages: ['@dxo/core'],
        platforms: ['all'],
        backend: ['cpu', 'cuda'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
        includeInGpuArtifact: true,
    },
    {
        id: 'lite-webgpu-smoke',
        script: 'scripts/test/lite-webgpu-smoke.ts',
        group: 'contract',
        packages: ['@dxo/lite'],
        platforms: ['all'],
        backend: ['cpu', 'webgpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'runtime-contract-lite',
        script: 'scripts/test/runtime-contract-lite.ts',
        group: 'contract',
        packages: ['@dxo/lite'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'runtime-contract-core',
        script: 'scripts/test/runtime-contract-core.ts',
        group: 'contract',
        packages: ['@dxo/core'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'model-graph',
        script: 'scripts/test/model-graph.ts',
        group: 'contract',
        packages: ['@dxo/graph', '@dxo/nn'],
        platforms: ['all'],
        backend: ['none'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'profiler-trace',
        script: 'scripts/test/profiler-trace.ts',
        group: 'contract',
        packages: ['@dxo/graph', '@dxo/inspect'],
        platforms: ['all'],
        backend: ['none'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'cli-contract',
        script: 'scripts/test/cli-contract.ts',
        group: 'contract',
        packages: ['@dxo/dxo', '@dxo/core', '@dxo/studio'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 180_000,
    },
    {
        id: 'vision-compose-contract',
        script: 'scripts/test/vision-compose-contract.ts',
        group: 'contract',
        packages: ['@dxo/vision', '@dxo/core', '@dxo/nn'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    // Vision GPU-model suites need multi-package pack; run under Modal only after pack vendors vision/nn/serialize.
    {
        id: 'vision-resnet-state',
        script: 'scripts/test/vision-resnet-state.ts',
        /** Heavy ResNet forward — GPU verify lane; not in core-only Modal artifact yet. */
        group: 'gpu-model',
        packages: ['@dxo/vision', '@dxo/core', '@dxo/nn'],
        platforms: ['all'],
        backend: ['cuda'],
        requiresGpu: true,
        requiresNetwork: false,
        allowSkip: true,
        timeoutMs: 120_000,
        includeInGpuArtifact: false,
    },
    {
        id: 'vision-load-weights',
        script: 'scripts/test/vision-load-weights.ts',
        /** Safetensors encode/decode + ResNet loadState — GPU lane; not in core-only Modal artifact yet. */
        group: 'gpu-model',
        packages: ['@dxo/vision', '@dxo/core', '@dxo/nn', '@dxo/serialize'],
        platforms: ['all'],
        backend: ['cuda'],
        requiresGpu: true,
        requiresNetwork: false,
        allowSkip: true,
        timeoutMs: 180_000,
        includeInGpuArtifact: false,
    },
    {
        id: 'titan-event-dep',
        script: 'scripts/test/titan-event-dep.ts',
        group: 'contract',
        packages: ['@dxo/core'],
        platforms: ['all'],
        backend: ['cpu', 'cuda'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
        includeInGpuArtifact: true,
    },
    {
        id: 'studio-run-smoke',
        script: 'scripts/test/studio-run-smoke.ts',
        group: 'product',
        packages: ['@dxo/studio', '@dxo/inspect', '@dxo/train', '@dxo/graph', '@dxo/nn', '@dxo/data'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 180_000,
    },
    {
        id: 'studio-ui-wave2',
        script: 'scripts/test/studio-ui-wave2.ts',
        group: 'product',
        packages: ['@dxo/studio', '@dxo/inspect'],
        platforms: ['all'],
        backend: ['none'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'model-app-image',
        script: 'scripts/test/model-app-image.ts',
        group: 'product',
        packages: ['@dxo/ui'],
        platforms: ['all'],
        backend: ['none'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'hub-provider',
        script: 'scripts/test/hub-provider.ts',
        group: 'product',
        packages: ['@dxo/hub'],
        platforms: ['all'],
        backend: ['none'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'model-runtime-text',
        script: 'scripts/test/model-runtime-text.ts',
        group: 'product',
        packages: ['@dxo/llm', '@dxo/nn'],
        platforms: ['all'],
        backend: ['cpu'],
        requiresGpu: false,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 180_000,
    },
    {
        id: 'gpu-matmul',
        script: 'scripts/test/gpu-matmul.ts',
        group: 'gpu-smoke',
        packages: ['@dxo/core'],
        platforms: ['linux'],
        backend: ['cuda'],
        requiresGpu: true,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 120_000,
    },
    {
        id: 'gpu-parity',
        script: 'scripts/test/gpu-parity.ts',
        group: 'gpu-parity',
        packages: ['@dxo/core'],
        platforms: ['linux'],
        backend: ['cuda'],
        requiresGpu: true,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 180_000,
    },
    {
        id: 'gpu-residency',
        script: 'scripts/test/gpu-residency.ts',
        group: 'gpu-residency',
        packages: ['@dxo/core'],
        platforms: ['linux'],
        backend: ['cuda'],
        requiresGpu: true,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 120_000,
    },
    {
        id: 'gpu-readback',
        script: 'scripts/test/gpu-readback.ts',
        group: 'gpu-residency',
        packages: ['@dxo/core'],
        platforms: ['linux'],
        backend: ['cuda'],
        requiresGpu: true,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
    {
        id: 'gpu-device-mismatch',
        script: 'scripts/test/gpu-device-mismatch.ts',
        group: 'gpu-residency',
        packages: ['@dxo/core'],
        platforms: ['linux'],
        backend: ['cuda'],
        requiresGpu: true,
        requiresNetwork: false,
        allowSkip: false,
        timeoutMs: 60_000,
    },
];

export const SUITE_BY_ID: ReadonlyMap<string, SuiteDef> = new Map(SUITES.map((s) => [s.id, s]));

export const CI_GROUPS: SuiteGroup[] = ['cpu', 'contract', 'product'];

export function suiteMatchesPlatform(suite: SuiteDef, platform: NodeJS.Platform = process.platform): boolean {
    if (suite.platforms.includes('all')) return true;
    if (platform === 'linux' || platform === 'darwin' || platform === 'win32') {
        return suite.platforms.includes(platform);
    }
    return false;
}

/** Suites run by ordinary host CI / `pnpm test:verify`. */
export function selectCiSuites(platform: NodeJS.Platform = process.platform): SuiteDef[] {
    return SUITES.filter((s) => CI_GROUPS.includes(s.group) && !s.requiresGpu && !s.requiresNetwork && suiteMatchesPlatform(s, platform));
}

export function selectGroupSuites(group: SuiteGroup, platform: NodeJS.Platform = process.platform): SuiteDef[] {
    return SUITES.filter((s) => s.group === group && suiteMatchesPlatform(s, platform));
}

/** Suites packed into the Modal GPU verify artifact.
 * Modal pack currently vendors `@dxo/core` only — non-core packages stay out until multi-package pack lands.
 */
export function selectGpuArtifactSuites(): SuiteDef[] {
    return SUITES.filter((s) => {
        if (s.includeInGpuArtifact === true) return true;
        if (!s.requiresGpu) return false;
        return s.packages.every((p) => p === '@dxo/core');
    });
}
