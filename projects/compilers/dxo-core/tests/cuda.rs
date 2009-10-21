//! CUDA GEMM via the public tensor API.

use dxo_core::{DeviceKind, Tensor, cuda_available};

#[test]
fn gemm_matches_cpu_when_cuda_present() {
    if !cuda_available() {
        eprintln!("SKIP: CUDA Driver API unavailable");
        return;
    }
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    let cpu = a.matmul(&b).unwrap();
    let ac = a.to(DeviceKind::Cuda).unwrap();
    let bc = b.to(DeviceKind::Cuda).unwrap();
    let gpu = ac.matmul(&bc).unwrap().to(DeviceKind::Cpu).unwrap();
    for (i, (c, g)) in cpu.data().iter().zip(gpu.data().iter()).enumerate() {
        assert!((c - g).abs() < 1e-4, "index {i}: cpu={c} cuda={g}");
    }
}
