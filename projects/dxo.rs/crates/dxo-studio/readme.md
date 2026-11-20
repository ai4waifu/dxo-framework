# dxo-studio

Rust service component for the DXO Studio runtime.

It reads recorded runs and exposes a local loopback API for Studio clients. The browser interface and event writer are separate packages so this crate can remain focused on secure, predictable read access.

## Contributing

Run `cargo test -p dxo-studio` and keep API responses backward-compatible with the public Studio client.
