/**
 * Backend-neutral runtime contract vectors (Living `01` / Wave 1).
 * Run via runtime-contract-core and runtime-contract-lite verify gates.
 */

import assert from 'node:assert/strict';

/** Minimal tensor surface both @dxo/core and @dxo/lite must satisfy. */
export interface ContractTensor {
    readonly shape: readonly number[];
    add(other: ContractTensor): ContractTensor;
    matmul(other: ContractTensor): ContractTensor;
    ready(): Promise<void>;
    toArray(): Promise<number[]>;
    item(): Promise<number>;
}

export interface ContractRuntime {
    tensor(data: ArrayLike<number>, shape: readonly number[]): ContractTensor;
    zeros(shape: readonly number[]): ContractTensor;
    ones(shape: readonly number[]): ContractTensor;
    sync?(): Promise<void>;
    destroy?(): void;
}

export type ContractRuntimeFactory = () => Promise<ContractRuntime>;

function approx(a: number, b: number, tol = 1e-5): boolean {
    return Math.abs(a - b) <= tol;
}

function assertApproxArray(actual: number[], expected: number[], label: string): void {
    assert.equal(actual.length, expected.length, `${label}: length`);
    for (let i = 0; i < actual.length; i++) {
        assert.ok(approx(actual[i]!, expected[i]!), `${label}[${i}]: ${actual[i]} vs ${expected[i]}`);
    }
}

/** Shared scenarios for core/lite runtime contract gates. */
export async function runTensorRuntimeContract(createRuntime: ContractRuntimeFactory, label: string): Promise<void> {
    const rt = await createRuntime();

    // Factory returns synchronous handles.
    const x = rt.tensor([1, 2, 3, 4], [2, 2]);
    const w = rt.ones([2, 2]);
    assert.deepEqual([...x.shape], [2, 2]);
    assert.deepEqual([...w.shape], [2, 2]);

    const b = rt.tensor([5, 6, 7, 8], [2, 2]);
    // Op chain is synchronous; does not require per-op await.
    const y = x.matmul(b).add(x);
    assert.deepEqual([...y.shape], [2, 2]);

    // Barriers are async.
    await y.ready();
    assertApproxArray(await y.toArray(), [20, 24, 46, 54], `${label}: matmul+add`);

    const z = rt.zeros([1]);
    assertApproxArray(await z.toArray(), [0], `${label}: zeros`);

    const scalar = rt.tensor([3.5], [1]).add(rt.tensor([0.5], [1]));
    assert.ok(approx(await scalar.item(), 4), `${label}: item`);

    // Sync contract errors (shape) before barrier.
    assert.throws(() => x.matmul(rt.tensor([1], [1])), /matmul|dim|shape/i);

    rt.destroy?.();
    console.log(`${label} ok: sync handle ops + async barriers`);
}
