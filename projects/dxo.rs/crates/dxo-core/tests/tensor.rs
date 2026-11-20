//! Tensor ops, views, and autograd finite-diff alignment.

use dxo_core::{Tensor, without_grad};

#[test]
fn matmul_2x2() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert!((c.data()[0] - 19.0).abs() < 1e-5);
    assert!((c.data()[3] - 50.0).abs() < 1e-5);
}

#[test]
fn add_bias_broadcast() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0], vec![2]).unwrap();
    let y = x.add(&b).unwrap();
    assert_eq!(y.data(), vec![11.0, 22.0, 13.0, 24.0]);
}

#[test]
fn reshape_is_zero_copy_view() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let v = t.reshape(&[4]).unwrap();
    assert!(t.shares_storage_with(&v));
    assert_eq!(v.shape(), &[4]);
}

#[test]
fn transpose_is_zero_copy_view() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let v = t.transpose().unwrap();
    assert!(t.shares_storage_with(&v));
    assert_eq!(v.shape(), &[2, 2]);
    assert_eq!(v.strides(), &[1, 2]);
}

#[test]
fn mul_broadcast() {
    let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
    let b = Tensor::from_vec(vec![10.0, 20.0], vec![2]).unwrap();
    let y = a.mul(&b).unwrap();
    assert_eq!(y.data(), vec![20.0, 60.0, 40.0, 100.0]);
}

#[test]
fn ones_and_randn_shapes() {
    let o = Tensor::ones(&[2, 3]);
    assert_eq!(o.shape(), &[2, 3]);
    assert!(o.data().iter().all(|&x| (x - 1.0).abs() < 1e-6));
    let r = Tensor::randn(&[4]);
    assert_eq!(r.shape(), &[4]);
    assert_eq!(r.numel(), 4);
}

#[test]
fn backward_matmul_sum_matches_fd() {
    let x = Tensor::from_vec_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true).unwrap();
    let w = Tensor::from_vec_grad(vec![0.5, 0.0, 0.0, 0.5], vec![2, 2], true).unwrap();
    let y = x.matmul(&w).unwrap().sum();
    y.backward().unwrap();
    let gx = x.grad().unwrap();
    let eps = 1e-3f32;
    let mut fd = [0.0f32; 4];
    for i in 0..4 {
        let mut xp = vec![1.0, 2.0, 3.0, 4.0];
        let mut xm = xp.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let yp = Tensor::from_vec(xp, vec![2, 2]).unwrap().matmul(&w.detach()).unwrap().sum().data()[0];
        let ym = Tensor::from_vec(xm, vec![2, 2]).unwrap().matmul(&w.detach()).unwrap().sum().data()[0];
        fd[i] = (yp - ym) / (2.0 * eps);
    }
    for i in 0..4 {
        assert!((gx[i] - fd[i]).abs() < 5e-2, "i={i} gx={} fd={}", gx[i], fd[i]);
    }
}

#[test]
fn without_grad_skips_tape() {
    let x = Tensor::from_vec_grad(vec![1.0, -1.0], vec![2], true).unwrap();
    let y = without_grad(|| x.relu());
    assert!(!y.requires_grad());
}
