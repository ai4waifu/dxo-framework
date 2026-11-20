# @dxo/train

Async CPU training loop with canonical safetensors checkpoints.

`Trainer.fitIter` / `run` yield epoch, batch, and checkpoint events. Checkpoint events contain the canonical `state`, encoded `bytes`, and `format: 'safetensors'`; callers decide where to persist the bytes.

```ts
for await (const event of trainer.fitIter()) {
  if (event.type === 'checkpoint') save(event.bytes);
}
```
