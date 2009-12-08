import assert from 'node:assert/strict';
import { tensor } from '@dxo/core';
import {
    Classifier,
    compose,
    decodeClassification,
    defineLabelSpace,
    defineResNet,
    LinearHead,
    ResNet,
    type ResNetOptions,
    VisionError,
} from '@dxo/vision';

// Constructible ResNet + signature ports (no classes field).
{
    const backbone = new ResNet({ depth: 18 });
    assert.equal(backbone.depth, 18);
    assert.equal(backbone.signature.input.image.name, 'image');
    assert.equal(backbone.signature.output.features.name, 'features');
    assert.deepEqual([...backbone.features().shape], ['batch', 512]);
    backbone.freeze();
    assert.equal(backbone.trainable, false);
    backbone.unfreeze();
    assert.equal(backbone.trainable, true);

    const defined = defineResNet({ depth: 34, inChannels: 1 });
    assert.equal(defined.depth, 34);
    assert.deepEqual([...defined.features().shape], ['batch', 512]);
}

// Type-level: ResNetOptions must not accept classes / numClasses.
{
    const opts: ResNetOptions = { depth: 18 };
    assert.equal('classes' in opts, false);
    assert.equal('numClasses' in opts, false);
    // @ts-expect-error — classes is not part of ResNetOptions
    const bad: ResNetOptions = { depth: 18, classes: 1000 };
    void bad;
}

// compose + LinearHead → Classifier; logits path is wired without labels.
{
    const backbone = new ResNet({ depth: 18 });
    const head = new LinearHead({ output: 10 });
    const model = compose(backbone, head);
    assert.ok(model instanceof Classifier);
    assert.equal(head.inFeatures, 512);
    assert.equal(head.outFeatures, 10);

    const withPort = new Classifier({
        backbone,
        head: new LinearHead({ input: backbone.features(), output: 3 }),
    });
    assert.equal(withPort.head.inFeatures, 512);

    assert.throws(
        () => compose(backbone, new LinearHead({ input: 128, output: 10 })),
        (err: unknown) => err instanceof VisionError && err.code === 'HEAD_FEATURE_MISMATCH',
    );
}

// LabelSpace decode shape (external to model).
{
    const labels = defineLabelSpace({
        id: 'toy-3',
        labels: ['a', 'b', 'c'],
    });
    assert.equal(labels.size, 3);
    const logits = tensor([0.1, 2.5, 0.3], [1, 3]);
    const decoded = await decodeClassification(logits, { labels, topK: 2 });
    assert.equal(decoded.labelSpaceId, 'toy-3');
    assert.equal(decoded.topK.length, 2);
    assert.equal(decoded.topK[0]!.label, 'b');
    assert.equal(decoded.topK[0]!.index, 1);

    const zh = defineLabelSpace({ id: 'toy-3-zh', labels: ['甲', '乙', '丙'] });
    const zhDecoded = await decodeClassification(logits, { labels: zh, topK: 1 });
    assert.equal(zhDecoded.topK[0]!.label, '乙');
}

// Preview forward must not invent logits.
{
    const backbone = new ResNet({ depth: 18 });
    assert.throws(
        () => backbone.forward(tensor([0, 0, 0], [1, 3, 1, 1])),
        (err: unknown) => err instanceof VisionError && err.code === 'UNSUPPORTED',
    );
}

console.log('vision-compose-contract ok: ResNet/compose/LabelSpace');
