# 🖼️ @dxo/vision

Build image and video intelligence with TypeScript. Compose media inputs, preprocessing, tensor views, model execution, and clear predictions in one Node.js workflow.

## Capabilities

- Classification, detection, segmentation, and visual search.
- Image and video-frame batches.
- Explicit channels, layout, color space, and dtype handling.
- Efficient image-to-tensor views when storage can be shared.
- Reproducible preprocessing and model results.

```ts
const image = await loadImage('./cat.jpg');
const input = await image.toTensor({ layout: 'CHW', size: 224 });
const prediction = await classifier.predict(input);
console.log(prediction.label, prediction.score);
```

Use local artifacts or model providers for weights, and compose vision workflows with DXO training and UI packages.
