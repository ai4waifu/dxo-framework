//! Titan CPU session wiring.

use dxo_core::cpu_session;

#[test]
fn cpu_session_opens_once() {
    let a = cpu_session();
    let b = cpu_session();
    assert_eq!(a.device(), b.device());
}
