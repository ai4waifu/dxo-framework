//! Autograd scope flags.

use dxo_core::{is_grad_enabled, without_grad};

#[test]
fn without_grad_restores_flag() {
    assert!(is_grad_enabled());
    without_grad(|| {
        assert!(!is_grad_enabled());
    });
    assert!(is_grad_enabled());
}
