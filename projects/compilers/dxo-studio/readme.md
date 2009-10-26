# dxo-studio

Rust-side inspect run store reader and loopback HTTP API for `@dxo/studio`.

- **Not** a VMZ app — UI lives in `projects/runtimes/dxo-studio`.
- **Not** the append-only writer — trainers use `@dxo/inspect` (TS) with the same on-disk layout.
