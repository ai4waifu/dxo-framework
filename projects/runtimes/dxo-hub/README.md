# 📦 @dxo/hub

Reliable model and dataset access for DXO applications.

Load artifacts from local storage or remote providers with caching, integrity checks, and offline-friendly workflows. Keep artifact retrieval separate from model execution so the same application can switch between local development, CI, and production storage.

```ts
const artifact = await hub.resolve('model://my-classifier');
const model = await loadModel(artifact);
```

Providers are composable, and applications retain control over credentials, storage, and deployment.
