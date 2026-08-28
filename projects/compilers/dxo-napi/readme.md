# 🔗 dxo-napi

Native Node.js binding layer for DXO.

This crate exposes the stable, type-erased runtime surface to JavaScript through N-API. It handles conversion, asynchronous completion, resource lifetime, and structured errors while keeping engine implementation details private.

## Contributing

Run `cargo test -p dxo-napi` and the repository smoke checks. N-API names are the source for generated TypeScript declarations.
