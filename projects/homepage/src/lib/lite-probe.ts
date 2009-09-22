import { Tensor, version } from '@dxo/lite';

export type LiteProbe = {
    version: string;
    tensorClass: string;
};

/** Runtime probe for @dxo/lite (WebGPU/WASM runtime; M0 stub). */
export function probeLite(): LiteProbe {
    const tensor = new Tensor();
    return {
        version: version(),
        tensorClass: tensor.constructor.name,
    };
}
