import { tensor, zeros, ones } from '@dxo/core';
import { type ContractRuntime, runTensorRuntimeContract } from '../contracts/tensor-runtime.js';

await runTensorRuntimeContract(async (): Promise<ContractRuntime> => {
    return {
        tensor: (data, shape) => tensor(data, shape),
        zeros: (shape) => zeros(shape),
        ones: (shape) => ones(shape),
    };
}, 'runtime-contract-core');
