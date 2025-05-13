#![cfg(feature = "cuda")]

use rust_tensor_lib::{
    array::Array,
    backend::{
        cpu::CpuBackend,
        cuda::{init_context, CudaBackend, CudaContextGuard, CudaStorage},
    },
    error::Error,
    Backend,
};
use serial_test::serial;

#[serial]
#[test]
fn test_sgd_step() -> Result<(), Error> {
    // Initialize CUDA context FIRST
    init_context(0)?;
    let _guard = CudaContextGuard::new()?; // Create guard AFTER init

    use rust_tensor_lib::Backend;
    let param_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_data = vec![0.1f32, 0.2, 0.3, 0.4];
    let learning_rate = 0.1f32;
    let shape = &[4];

    // --- CPU Calculation (Using Storage Directly) ---
    let mut cpu_param_storage: Array = CpuBackend::from_vec(param_data.clone(), shape)?;
    let cpu_grad_storage: Array = CpuBackend::from_vec(grad_data.clone(), shape)?;
    CpuBackend::sgd_step(&mut cpu_param_storage, &cpu_grad_storage, learning_rate)?;
    // Get result from CPU storage (Array implements AsRef<[f32]>)
    let cpu_result_vec: Vec<f32> = cpu_param_storage.as_ref().to_vec();

    // --- CUDA Calculation (Using Storage Directly) ---
    let mut cuda_param_storage: CudaStorage = CudaBackend::from_vec(param_data.clone(), shape)?;
    let cuda_grad_storage: CudaStorage = CudaBackend::from_vec(grad_data.clone(), shape)?;
    // Directly test the backend's sgd_step function
    CudaBackend::sgd_step(&mut cuda_param_storage, &cuda_grad_storage, learning_rate)?;
    let cuda_result_vec = CudaBackend::copy_to_host(&cuda_param_storage)?;

    // Compare results
    const EPSILON: f32 = 1e-5;
    assert_eq!(cpu_result_vec.len(), cuda_result_vec.len());
    println!("CPU Result: {:?}", cpu_result_vec);
    println!("CUDA Result: {:?}", cuda_result_vec);
    for (cpu_val, cuda_val) in cpu_result_vec.iter().zip(cuda_result_vec.iter()) {
        assert!(
            (cpu_val - cuda_val).abs() < EPSILON,
            "CPU value {cpu_val} differs from CUDA value {cuda_val}"
        );
    }

    Ok(())
}

#[serial]
#[test]
fn test_sgd_step_zero_lr() -> Result<(), Error> {
    init_context(0)?; // Initialize context
    let _guard = CudaContextGuard::new()?; // Create guard

    let param_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_data = vec![0.1f32, 0.2, 0.3, 0.4];
    let learning_rate = 0.0f32;
    let shape = &[4];

    let mut cuda_param_storage = CudaBackend::from_vec(param_data.clone(), shape)?;
    let cuda_grad_storage = CudaBackend::from_vec(grad_data.clone(), shape)?;
    CudaBackend::sgd_step(&mut cuda_param_storage, &cuda_grad_storage, learning_rate)?;

    let result_vec = CudaBackend::copy_to_host(&cuda_param_storage)?;
    assert_eq!(
        param_data, result_vec,
        "Parameters changed with zero learning rate"
    );

    Ok(())
}

#[serial]
#[test]
fn test_sgd_step_large_lr() -> Result<(), Error> {
    init_context(0)?; // Initialize context
    let _guard = CudaContextGuard::new()?; // Create guard

    let param_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_data = vec![0.1f32, 0.2, 0.3, 0.4];
    let learning_rate = 10.0f32;
    let shape = &[4];

    let mut cuda_param_storage = CudaBackend::from_vec(param_data.clone(), shape)?;
    let cuda_grad_storage = CudaBackend::from_vec(grad_data.clone(), shape)?;
    CudaBackend::sgd_step(&mut cuda_param_storage, &cuda_grad_storage, learning_rate)?;

    let result_vec = CudaBackend::copy_to_host(&cuda_param_storage)?;
    const EPSILON: f32 = 1e-5;
    for i in 0..param_data.len() {
        let expected = param_data[i] - learning_rate * grad_data[i];
        assert!(
            (result_vec[i] - expected).abs() < EPSILON,
            "Unexpected result with large learning rate at index {i}: expected {expected}, got {}",
            result_vec[i]
        );
    }

    Ok(())
}
