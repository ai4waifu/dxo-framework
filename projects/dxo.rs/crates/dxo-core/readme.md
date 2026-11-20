# 🦀 dxo-core

Rust execution core for DXO contributors.

This crate owns tensor storage, eager autograd, device abstractions, and backend facades used by the TypeScript runtime. It is intentionally independent of Node.js; JavaScript bindings live in the companion N-API crate.

## Contributing

Run `cargo test -p dxo-core` for focused tests and `cargo fmt --all -- --check` before opening a change. Keep public behavior and error semantics aligned with the TypeScript packages.
