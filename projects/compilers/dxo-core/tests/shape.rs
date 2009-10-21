//! Shape and stride helpers.

use dxo_core::contiguous_strides;

#[test]
fn contiguous_strides_row_major() {
    assert_eq!(contiguous_strides(&[2, 3]), vec![3, 1]);
}
