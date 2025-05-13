#![cfg(feature = "cuda")] // Only compile when CUDA feature is enabled

use rust_tensor_lib::{
    backend::{
        cpu::CpuBackend,
        cuda::{init_context, CudaBackend, CudaContextGuard, CudaTensor},
    },
    Backend, CpuTensor, Error, Tensor,
};
use serial_test::serial;

// Helper function to create a CUDA tensor from CPU tensor
fn cuda_tensor_from_cpu(cpu_tensor: &CpuTensor) -> CudaTensor {
    let shape = cpu_tensor.shape();
    // Get data from CPU tensor to host Vec
    let host_data =
        CpuBackend::copy_to_host(&*cpu_tensor.data()).expect("Failed to copy CPU data to host vec");

    // Create CUDA storage from host Vec
    let cuda_storage =
        CudaBackend::from_vec(host_data, &shape).expect("Failed to create CUDA storage from vec");

    // Create CUDA Tensor
    Tensor::<CudaBackend>::new(cuda_storage, false)
}

#[serial] // Ensure CUDA tests run sequentially
#[test]
#[cfg(feature = "cuda")] // Ensure it only runs when cuda feature is enabled
fn test_tensor_map_cuda() -> Result<(), Error> {
    println!("--- Running test_tensor_map_cuda ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    println!("CUDA Context Initialized for test_tensor_map_cuda.");

    // Case 1: Simple mapping function
    // Create a CPU tensor first, then convert, to ensure data consistency with CPU test
    let t1_cpu_data = vec![1.0, -2.0, 3.0];
    let t1_cpu_shape = &[3];
    let t1_cpu_for_cuda = Tensor::<CpuBackend>::from_vec(t1_cpu_data.clone(), t1_cpu_shape, true)?;
    let t1_cuda = cuda_tensor_from_cpu(&t1_cpu_for_cuda); // Use helper to convert

    let map_fn1 = |x: f32| x * 2.0 + 1.0;
    let mapped_t1_cuda = t1_cuda.map(map_fn1)?;

    assert_eq!(
        mapped_t1_cuda.shape(),
        &[3],
        "Shape mismatch for mapped_t1_cuda"
    );
    let expected_data1 = vec![3.0, -3.0, 7.0];
    let mapped_t1_host_data = CudaBackend::copy_to_host(&*mapped_t1_cuda.data())?;
    assert_eq!(
        mapped_t1_host_data, expected_data1,
        "Data mismatch for mapped_t1_cuda"
    );
    assert!(
        !mapped_t1_cuda.requires_grad(),
        "Output of map (mapped_t1_cuda) should not require_grad"
    );
    println!("CUDA Map Test Case 1 Passed: Simple map");

    // Case 2: Mapping an empty tensor
    let t2_cpu_empty_data: Vec<f32> = vec![];
    let t2_cpu_empty_shape = &[0, 2];
    let t2_cpu_empty_for_cuda =
        Tensor::<CpuBackend>::from_vec(t2_cpu_empty_data.clone(), t2_cpu_empty_shape, false)?;
    let t2_cuda_empty = cuda_tensor_from_cpu(&t2_cpu_empty_for_cuda);

    let map_fn2 = |x: f32| x.powi(2);
    let mapped_t2_cuda_empty = t2_cuda_empty.map(map_fn2)?;

    assert_eq!(
        mapped_t2_cuda_empty.shape(),
        &[0, 2],
        "Shape mismatch for mapped_t2_cuda_empty"
    );
    let mapped_t2_host_data = CudaBackend::copy_to_host(&*mapped_t2_cuda_empty.data())?;
    assert!(
        mapped_t2_host_data.is_empty(),
        "Data mismatch for mapped_t2_cuda_empty (should be empty)"
    );
    assert!(
        !mapped_t2_cuda_empty.requires_grad(),
        "Output of map (mapped_t2_cuda_empty) should not require_grad"
    );
    println!("CUDA Map Test Case 2 Passed: Empty tensor");

    // Case 3: Mapping a tensor with different shape
    let t3_cpu_matrix_data = vec![1.0, 2.0, 3.0, 4.0];
    let t3_cpu_matrix_shape = &[2, 2];
    let t3_cpu_matrix_for_cuda =
        Tensor::<CpuBackend>::from_vec(t3_cpu_matrix_data.clone(), t3_cpu_matrix_shape, true)?;
    let t3_cuda_matrix = cuda_tensor_from_cpu(&t3_cpu_matrix_for_cuda);

    let map_fn3 = |x: f32| x - 0.5;
    let mapped_t3_cuda_matrix = t3_cuda_matrix.map(map_fn3)?;

    assert_eq!(
        mapped_t3_cuda_matrix.shape(),
        &[2, 2],
        "Shape mismatch for mapped_t3_cuda_matrix"
    );
    let expected_data3 = vec![0.5, 1.5, 2.5, 3.5];
    let mapped_t3_host_data = CudaBackend::copy_to_host(&*mapped_t3_cuda_matrix.data())?;
    assert_eq!(
        mapped_t3_host_data, expected_data3,
        "Data mismatch for mapped_t3_cuda_matrix"
    );
    assert!(
        !mapped_t3_cuda_matrix.requires_grad(),
        "Output of map (mapped_t3_cuda_matrix) should not require_grad"
    );
    println!("CUDA Map Test Case 3 Passed: Matrix map");

    println!("--- test_tensor_map_cuda finished ---");
    Ok(())
}
