//! Crate-level smoke checks.

use dxo_core::{DeviceKind, VERSION, backend_label, cpu_session};

#[test]
fn version_is_semver_like() {
    assert!(VERSION.starts_with("0."));
}

#[test]
fn titan_cpu_engine_wired() {
    let _ = cpu_session();
    assert_eq!(backend_label(), "cpu");
}

#[test]
fn device_parse() {
    assert_eq!(DeviceKind::parse("CPU").unwrap(), DeviceKind::Cpu);
    assert_eq!(DeviceKind::parse("cuda").unwrap(), DeviceKind::Cuda);
    assert!(DeviceKind::parse("metal").is_err());
}
