import { Tensor, version } from '@dxo/lite';

export type LiteProbe = {
    version: string;
    tensorClass: string;
};

/** Runtime probe for the lightweight CPU-only @dxo/lite package. */
export function probeLite(): LiteProbe {
    const tensor = new Tensor();
    return {
        version: version(),
        tensorClass: tensor.constructor.name,
    };
}
