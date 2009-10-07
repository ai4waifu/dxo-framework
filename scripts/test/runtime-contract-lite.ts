import { createRuntime } from '@dxo/lite';
import { type ContractRuntime, runTensorRuntimeContract } from '../contracts/tensor-runtime.js';

await runTensorRuntimeContract(async (): Promise<ContractRuntime> => {
    const rt = await createRuntime({ fallback: 'cpu' });
    return {
        tensor: (data, shape) => rt.tensor(data, shape),
        zeros: (shape) => rt.zeros(shape),
        ones: (shape) => rt.ones(shape),
        destroy: () => rt.destroy(),
    };
}, 'runtime-contract-lite');
