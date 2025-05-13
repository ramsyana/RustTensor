#![cfg(feature = "cuda")] // Only compile when CUDA feature is enabled
#![allow(unused_imports)] // Temporarily allow unused imports during setup

use rust_tensor_lib::{
    backend::{
        cpu::CpuBackend,
        cuda::{
            get_global_context, init_context, CudaBackend, CudaContextGuard, CudaStorage,
            CudaTensor,
        },
    },
    ops,
    test_utils::{assert_storage_close, assert_storage_eq},
    Array, Backend, CpuTensor, Error, Tensor,
};
use serial_test::serial;
use std::rc::Rc;

use approx::assert_abs_diff_eq; // For floating-point comparisons

// --- Helper Functions ---

/// Creates a CPU Tensor for testing purposes.
fn cpu_tensor(data: Vec<f32>, shape: &[usize]) -> CpuTensor {
    Tensor::<CpuBackend>::new(
        CpuBackend::from_vec(data, shape).expect("Failed to create CPU storage"),
        false, // Gradients not needed for forward tests
    )
}

/// Creates a CUDA Tensor by copying data from a CPU Tensor.
/// Assumes CUDA context is initialized.
fn cuda_tensor_from_cpu(cpu_tensor: &CpuTensor) -> CudaTensor {
    println!("[test_cuda_tanh] Entered cuda_tensor_from_cpu");
    let shape = cpu_tensor.shape();
    // 1. Get data from CPU tensor to host Vec
    let host_data = CpuBackend::copy_to_host(&*cpu_tensor.data())
        .map_err(|e| Error::InternalLogicError(format!("Helper: Failed copy_to_host: {}", e)))
        .expect("Failed to copy CPU data to host vec");

    // 2. Create CUDA storage from host Vec
    let cuda_storage = CudaBackend::from_vec(host_data, &shape)
        .map_err(|e| Error::InternalLogicError(format!("Helper: Failed from_vec: {}", e)))
        .expect("Failed to create CUDA storage from vec");

    // 3. Create CUDA Tensor
    Tensor::<CudaBackend>::new(cuda_storage, false)
}

/// Asserts that the data in a CPU tensor and a CUDA tensor are element-wise close.
/// Copies CUDA data to host for comparison.
/// Assumes CUDA context is initialized.
fn assert_tensors_close(cpu_tensor: &CpuTensor, cuda_tensor: &CudaTensor, tolerance: f32) {
    // 1. Check shapes first
    let cpu_shape = cpu_tensor.shape();
    let cuda_shape = cuda_tensor.shape();
    assert_eq!(cpu_shape, cuda_shape, "Tensor shapes do not match");

    // 2. Get data from both tensors as host Vec<f32>
    let cpu_data = CpuBackend::copy_to_host(&*cpu_tensor.data())
        .expect("Failed to get CPU data for comparison");
    let cuda_data_host = CudaBackend::copy_to_host(&*cuda_tensor.data())
        .expect("Failed to copy CUDA data to host for comparison");

    // 3. Compare lengths (redundant due to shape check, but good practice)
    assert_eq!(
        cpu_data.len(),
        cuda_data_host.len(),
        "Data lengths do not match"
    );

    // 4. Compare elements element-wise with relative tolerance
    for (i, (cpu_val, cuda_val)) in cpu_data.iter().zip(cuda_data_host.iter()).enumerate() {
        let abs_diff = (cpu_val - cuda_val).abs();
        // For very small values, use absolute tolerance
        let rel_tolerance = if cpu_val.abs() < tolerance {
            tolerance
        } else {
            // For larger values, use relative tolerance
            cpu_val.abs() * tolerance
        };

        if abs_diff > rel_tolerance {
            panic!(
                "Values differ at index {}: CPU {} vs CUDA {} (abs_diff: {}, rel_tolerance: {})",
                i, cpu_val, cuda_val, abs_diff, rel_tolerance
            );
        }
    }
}

// Tolerance for floating point comparisons
const TOLERANCE: f32 = 1e-6;

// --- CUDA Op Tests ---

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_mse_loss() -> Result<(), Error> {
    use rust_tensor_lib::backend::cuda::{init_context, CudaContextGuard};
    use rust_tensor_lib::Reduction;
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let preds_cpu = cpu_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let targets_cpu = cpu_tensor(vec![1.5, 2.5, 3.5], &[3]);
    let preds_cuda = cuda_tensor_from_cpu(&preds_cpu);
    let targets_cuda = cuda_tensor_from_cpu(&targets_cpu);

    // None reduction
    let cpu_none = ops::mse_loss(&preds_cpu, &targets_cpu, Reduction::None)?;
    let cuda_none = ops::mse_loss(&preds_cuda, &targets_cuda, Reduction::None)?;
    assert_tensors_close(&cpu_none, &cuda_none, TOLERANCE);

    // Sum reduction
    let cpu_sum = ops::mse_loss(&preds_cpu, &targets_cpu, Reduction::Sum)?;
    let cuda_sum = ops::mse_loss(&preds_cuda, &targets_cuda, Reduction::Sum)?;
    let cpu_sum_val = cpu_sum.data().as_ref()[0];
    let cuda_sum_val =
        rust_tensor_lib::backend::cuda::CudaBackend::copy_to_host(&*cuda_sum.data())?[0];
    assert!(
        (cpu_sum_val - cuda_sum_val).abs() < TOLERANCE,
        "Sum reduction mismatch: {} vs {}",
        cpu_sum_val,
        cuda_sum_val
    );

    // Mean reduction
    let cpu_mean = ops::mse_loss(&preds_cpu, &targets_cpu, Reduction::Mean)?;
    let cuda_mean = ops::mse_loss(&preds_cuda, &targets_cuda, Reduction::Mean)?;
    let cpu_mean_val = cpu_mean.data().as_ref()[0];
    let cuda_mean_val =
        rust_tensor_lib::backend::cuda::CudaBackend::copy_to_host(&*cuda_mean.data())?[0];
    assert!(
        (cpu_mean_val - cuda_mean_val).abs() < TOLERANCE,
        "Mean reduction mismatch: {} vs {}",
        cpu_mean_val,
        cuda_mean_val
    );

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")] // Only run this test when the cuda feature is enabled
fn test_cuda_add() -> Result<(), Error> {
    init_context(0)?; // Initialize first
    let _guard = CudaContextGuard::new()?; // Then create guard

    // 1. Define Input Data & Shape
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let shape = &[2, 2];

    // 2. Create CPU Tensors (Inputs)
    let cpu_a = cpu_tensor(data_a.clone(), shape);
    let cpu_b = cpu_tensor(data_b.clone(), shape);

    // 3. Compute Expected Result on CPU
    let expected_cpu_result = ops::add(&cpu_a, &cpu_b)?;

    // 4. Create CUDA Tensors (Inputs) from CPU Tensors
    let cuda_a = cuda_tensor_from_cpu(&cpu_a);
    let cuda_b = cuda_tensor_from_cpu(&cpu_b);

    // 5. Compute Actual Result on GPU using the generic `ops::add`
    //    This will dispatch to `CudaBackend::add`.
    let actual_cuda_result = ops::add(&cuda_a, &cuda_b)?;

    // 6. Compare GPU result (copied to host) with CPU result
    assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);

    Ok(()) // Indicate test success
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_mul() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // 1. Define Input Data & Shape
    let data_a = vec![1.0, 2.0, 3.0, 4.0];
    let data_b = vec![5.0, 6.0, 7.0, 8.0];
    let shape = &[2, 2];

    // 2. Create CPU Tensors
    let cpu_a = cpu_tensor(data_a.clone(), shape);
    let cpu_b = cpu_tensor(data_b.clone(), shape);

    // 3. Compute Expected Result on CPU
    let expected_cpu_result = ops::mul(&cpu_a, &cpu_b)?;

    // 4. Create CUDA Tensors
    let cuda_a = cuda_tensor_from_cpu(&cpu_a);
    let cuda_b = cuda_tensor_from_cpu(&cpu_b);

    // 5. Compute Actual Result on GPU
    let actual_cuda_result = ops::mul(&cuda_a, &cuda_b)?;

    // 6. Compare
    assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_sub() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // 1. Define Input Data & Shape
    let data_a = vec![5.0, 7.0, 9.0, 11.0];
    let data_b = vec![1.0, 2.0, 3.0, 4.0];
    let shape = &[2, 2];

    // 2. Create CPU Tensors
    let cpu_a = cpu_tensor(data_a.clone(), shape);
    let cpu_b = cpu_tensor(data_b.clone(), shape);

    // 3. Compute Expected Result on CPU
    let expected_cpu_result = ops::sub(&cpu_a, &cpu_b)?;

    // 4. Create CUDA Tensors
    let cuda_a = cuda_tensor_from_cpu(&cpu_a);
    let cuda_b = cuda_tensor_from_cpu(&cpu_b);

    // 5. Compute Actual Result on GPU
    let actual_cuda_result = ops::sub(&cuda_a, &cuda_b)?;

    // 6. Compare
    assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_relu() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // 1. Define Input Data & Shape
    let data_a = vec![-1.0, 0.0, 1.0, 2.0, -5.5, 10.1];
    let shape = &[2, 3];

    // 2. Create CPU Tensor
    let cpu_a = cpu_tensor(data_a.clone(), shape);

    // 3. Compute Expected Result on CPU
    let expected_cpu_result = ops::relu(&cpu_a)?;

    // 4. Create CUDA Tensor
    let cuda_a = cuda_tensor_from_cpu(&cpu_a);

    // 5. Compute Actual Result on GPU
    let actual_cuda_result = ops::relu(&cuda_a)?;

    // 6. Compare
    assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_matmul() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // 1. Define Input Data & Shape (Example: 2x3 @ 3x2 => 2x2)
    let data_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape_a = &[2, 3];

    let data_b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let shape_b = &[3, 2];

    // 2. Create CPU Tensors
    let cpu_a = cpu_tensor(data_a.clone(), shape_a);
    let cpu_b = cpu_tensor(data_b.clone(), shape_b);

    // 3. Compute Expected Result on CPU
    let expected_cpu_result = ops::matmul(&cpu_a, &cpu_b)?;

    // 4. Create CUDA Tensors
    let cuda_a = cuda_tensor_from_cpu(&cpu_a);
    let cuda_b = cuda_tensor_from_cpu(&cpu_b);

    // 5. Compute Actual Result on GPU
    let actual_cuda_result = ops::matmul(&cuda_a, &cuda_b)?;

    // 6. Compare
    assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_transpose_accuracy() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create a simple non-square CPU tensor (2x3)
    let cpu = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let cuda = cuda_tensor_from_cpu(&cpu);

    // Transpose using CPU and CUDA backends
    let cpu_t = Tensor::new(CpuBackend::transpose(&*cpu.data())?, false);
    let cuda_t = Tensor::new(CudaBackend::transpose(&*cuda.data())?, false);

    // Compare the results
    assert_tensors_close(&cpu_t, &cuda_t, TOLERANCE);
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_context_threading() -> Result<(), Error> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    init_context(0)?; // Initialize main context

    let success = Arc::new(AtomicBool::new(true));
    let success_clone = success.clone();

    // Spawn thread that will use CUDA context
    let handle = thread::spawn(move || {
        match (|| {
            let _guard = CudaContextGuard::new()?; // Should acquire context in new thread

            // Perform some CUDA operations
            let data = vec![1.0, 2.0, 3.0, 4.0];
            let shape = &[2, 2];
            let cpu_tensor = cpu_tensor(data, shape);
            let cuda_tensor = cuda_tensor_from_cpu(&cpu_tensor);
            let _result = ops::relu(&cuda_tensor)?;

            // Verify result
            let cuda_result = ops::relu(&cuda_tensor)?;
            let cpu_result = ops::relu(&cpu_tensor)?;
            assert_tensors_close(&cpu_result, &cuda_result, TOLERANCE);

            Ok(()) as Result<(), Error>
        })() {
            Ok(_) => success_clone.store(true, Ordering::SeqCst),
            Err(e) => {
                eprintln!("Thread CUDA operation failed: {}", e);
                success_clone.store(false, Ordering::SeqCst);
            }
        }
    });

    // Wait for thread to complete
    handle.join().expect("Thread panicked");

    assert!(
        success.load(Ordering::SeqCst),
        "CUDA operations in thread failed"
    );
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_context_error_recovery() -> Result<(), Error> {
    // First initialize context normally
    init_context(0)?;

    // Try operations that should fail but not crash the context
    {
        let _guard = CudaContextGuard::new()?;

        // Try to create a tensor with mismatched dimensions
        let err = CudaBackend::from_vec(vec![1.0, 2.0], &[2, 2]);
        assert!(matches!(err, Err(Error::ShapeMismatch { .. })));

        // Try invalid matmul dimensions
        let a = cuda_tensor_from_cpu(&cpu_tensor(vec![1.0, 2.0], &[2]));
        let b = cuda_tensor_from_cpu(&cpu_tensor(vec![3.0, 4.0], &[2]));
        let err = ops::matmul(&a, &b);
        assert!(
            matches!(
                err,
                Err(Error::IncompatibleShapes { .. })
                    | Err(Error::InvalidOperation(_))
                    | Err(Error::CublasError(_))
                    | Err(Error::CudaError(_))
                    | Err(Error::DimensionMismatch(_, _))
            ),
            "unexpected error variant for invalid CUDA matmul: {:?}",
            err
        );
    }

    // Should still be able to use context for valid operations
    let _guard = CudaContextGuard::new()?;

    let data = vec![1.0, 2.0];
    let shape = &[2];
    let cpu_tensor = cpu_tensor(data, shape);
    let cuda_tensor = cuda_tensor_from_cpu(&cpu_tensor);

    // Verify we can still do operations
    let result = ops::relu(&cuda_tensor)?;
    let cpu_result = ops::relu(&cpu_tensor)?;
    assert_tensors_close(&cpu_result, &result, TOLERANCE);

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_context_concurrent_ops() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create input tensors
    let shape = &[2, 2];
    let a = cuda_tensor_from_cpu(&cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], shape));
    let b = cuda_tensor_from_cpu(&cpu_tensor(vec![5.0, 6.0, 7.0, 8.0], shape));

    // Launch multiple operations that will need to be synchronized
    let add_result = ops::add(&a, &b)?;
    let mul_result = ops::mul(&a, &b)?;
    let matmul_result = ops::matmul(&a, &b)?;

    // All results should be valid since operations are properly synchronized
    assert_eq!(add_result.shape(), shape);
    assert_eq!(mul_result.shape(), shape);
    assert_eq!(matmul_result.shape(), shape);

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_sum_along_axis() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: 2D -> 1D
    let cpu_in1 = cpu_tensor(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
    let cuda_in1 = cuda_tensor_from_cpu(&cpu_in1);
    // Axis 0
    let cpu_sum0 = Tensor::new(CpuBackend::sum_along_axis(&*cpu_in1.data(), 0)?, false);
    let cuda_sum0 = Tensor::new(CudaBackend::sum_along_axis(&*cuda_in1.data(), 0)?, false);
    assert_tensors_close(&cpu_sum0, &cuda_sum0, TOLERANCE);
    // Axis 1
    let cpu_sum1 = Tensor::new(CpuBackend::sum_along_axis(&*cpu_in1.data(), 1)?, false);
    let cuda_sum1 = Tensor::new(CudaBackend::sum_along_axis(&*cuda_in1.data(), 1)?, false);
    assert_tensors_close(&cpu_sum1, &cuda_sum1, TOLERANCE);

    // Case 2: 1D -> 0D (Scalar)
    let cpu_in2 = cpu_tensor(vec![10., 20., 30.], &[3]);
    let cuda_in2 = cuda_tensor_from_cpu(&cpu_in2);
    let cpu_sum_scalar = Tensor::new(CpuBackend::sum_along_axis(&*cpu_in2.data(), 0)?, false);
    let cuda_sum_scalar = Tensor::new(CudaBackend::sum_along_axis(&*cuda_in2.data(), 0)?, false);
    println!("CPU Sum Scalar Shape: {:?}", cpu_sum_scalar.shape());
    println!("CUDA Sum Scalar Shape: {:?}", cuda_sum_scalar.shape());
    println!(
        "CPU Sum Scalar Data: {:?}",
        CpuBackend::copy_to_host(&*cpu_sum_scalar.data())?
    );
    println!(
        "CUDA Sum Scalar Data: {:?}",
        CudaBackend::copy_to_host(&*cuda_sum_scalar.data())?
    );
    assert_tensors_close(&cpu_sum_scalar, &cuda_sum_scalar, TOLERANCE);

    // Case 3: Higher dimensions (e.g., 3D -> 2D)
    let cpu_in3 = cpu_tensor((1..=12).map(|x| x as f32).collect(), &[2, 3, 2]);
    let cuda_in3 = cuda_tensor_from_cpu(&cpu_in3);
    // Axis 1
    let cpu_sum_3d_1 = Tensor::new(CpuBackend::sum_along_axis(&*cpu_in3.data(), 1)?, false);
    let cuda_sum_3d_1 = Tensor::new(CudaBackend::sum_along_axis(&*cuda_in3.data(), 1)?, false);
    assert_tensors_close(&cpu_sum_3d_1, &cuda_sum_3d_1, TOLERANCE);

    println!("test_cuda_sum_along_axis PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_broadcast_to() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Scalar to Vector
    let cpu_in1 = cpu_tensor(vec![5.0], &[]);
    let cuda_in1 = cuda_tensor_from_cpu(&cpu_in1);
    let target_shape1 = &[4];
    let cpu_bcast1 = Tensor::new(
        CpuBackend::broadcast_to(&*cpu_in1.data(), target_shape1)?,
        false,
    );
    let cuda_bcast1 = Tensor::new(
        CudaBackend::broadcast_to(&*cuda_in1.data(), target_shape1)?,
        false,
    );
    assert_tensors_close(&cpu_bcast1, &cuda_bcast1, TOLERANCE);

    // Case 2: Vector to Matrix (broadcast row)
    let cpu_in2 = cpu_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let cuda_in2 = cuda_tensor_from_cpu(&cpu_in2);
    let target_shape2 = &[2, 3];
    let cpu_bcast2 = Tensor::new(
        CpuBackend::broadcast_to(&*cpu_in2.data(), target_shape2)?,
        false,
    );
    let cuda_bcast2 = Tensor::new(
        CudaBackend::broadcast_to(&*cuda_in2.data(), target_shape2)?,
        false,
    );
    assert_tensors_close(&cpu_bcast2, &cuda_bcast2, TOLERANCE);

    // Case 3: Vector to Matrix (broadcast column)
    let cpu_in3 = cpu_tensor(vec![10.0, 20.0], &[2, 1]);
    let cuda_in3 = cuda_tensor_from_cpu(&cpu_in3);
    let target_shape3 = &[2, 3];
    let cpu_bcast3 = Tensor::new(
        CpuBackend::broadcast_to(&*cpu_in3.data(), target_shape3)?,
        false,
    );
    let cuda_bcast3 = Tensor::new(
        CudaBackend::broadcast_to(&*cuda_in3.data(), target_shape3)?,
        false,
    );
    assert_tensors_close(&cpu_bcast3, &cuda_bcast3, TOLERANCE);

    println!("test_cuda_broadcast_to PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_mean_op() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let cpu_in = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.], &[2, 3]);
    let cuda_in = cuda_tensor_from_cpu(&cpu_in);

    let cpu_mean = ops::mean(&cpu_in, None)?;
    let cuda_mean = ops::mean(&cuda_in, None)?;

    assert_tensors_close(&cpu_mean, &cuda_mean, TOLERANCE);

    println!("test_cuda_mean_op PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_max_along_axis() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: 2D -> 1D
    let cpu_in1 = cpu_tensor(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
    let cuda_in1 = cuda_tensor_from_cpu(&cpu_in1);
    // Max along axis 0
    let cpu_max0 = Tensor::new(CpuBackend::max_along_axis(&*cpu_in1.data(), 0)?, false);
    let cuda_max0 = Tensor::new(CudaBackend::max_along_axis(&*cuda_in1.data(), 0)?, false);
    assert_tensors_close(&cpu_max0, &cuda_max0, TOLERANCE);
    // Max along axis 1
    let cpu_max1 = Tensor::new(CpuBackend::max_along_axis(&*cpu_in1.data(), 1)?, false);
    let cuda_max1 = Tensor::new(CudaBackend::max_along_axis(&*cuda_in1.data(), 1)?, false);
    assert_tensors_close(&cpu_max1, &cuda_max1, TOLERANCE);

    // Case 2: 1D -> 0D (Scalar)
    let cpu_in2 = cpu_tensor(vec![10., -5., 30.], &[3]);
    let cuda_in2 = cuda_tensor_from_cpu(&cpu_in2);
    let cpu_max_scalar = Tensor::new(CpuBackend::max_along_axis(&*cpu_in2.data(), 0)?, false);
    let cuda_max_scalar = Tensor::new(CudaBackend::max_along_axis(&*cuda_in2.data(), 0)?, false);
    assert_tensors_close(&cpu_max_scalar, &cuda_max_scalar, TOLERANCE);

    // Case 3: Higher dimensions (3D -> 2D)
    let cpu_in3 = cpu_tensor((1..=12).map(|x| x as f32).collect(), &[2, 3, 2]);
    let cuda_in3 = cuda_tensor_from_cpu(&cpu_in3);
    // Max along axis 1
    let cpu_max_3d_1 = Tensor::new(CpuBackend::max_along_axis(&*cpu_in3.data(), 1)?, false);
    let cuda_max_3d_1 = Tensor::new(CudaBackend::max_along_axis(&*cuda_in3.data(), 1)?, false);
    assert_tensors_close(&cpu_max_3d_1, &cuda_max_3d_1, TOLERANCE);

    println!("test_cuda_max_along_axis PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_sum_all() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Regular 2D tensor
    let cpu_in1 = cpu_tensor(vec![1., 2., 3., 4., 5., 6.], &[2, 3]);
    let cuda_in1 = cuda_tensor_from_cpu(&cpu_in1);
    let cpu_sum1 = CpuBackend::sum_all(&*cpu_in1.data())?;
    let cuda_sum1 = CudaBackend::sum_all(&*cuda_in1.data())?;
    assert!((cpu_sum1 - cuda_sum1).abs() < TOLERANCE);

    // Case 2: Higher dimensions (3D)
    let cpu_in2 = cpu_tensor((1..=12).map(|x| x as f32).collect(), &[2, 2, 3]);
    let cuda_in2 = cuda_tensor_from_cpu(&cpu_in2);
    let cpu_sum2 = CpuBackend::sum_all(&*cpu_in2.data())?;
    let cuda_sum2 = CudaBackend::sum_all(&*cuda_in2.data())?;
    assert!((cpu_sum2 - cuda_sum2).abs() < TOLERANCE);

    // Case 3: 1D tensor
    let cpu_in3 = cpu_tensor(vec![-1., 0., 1., 2.], &[4]);
    let cuda_in3 = cuda_tensor_from_cpu(&cpu_in3);
    let cpu_sum3 = CpuBackend::sum_all(&*cpu_in3.data())?;
    let cuda_sum3 = CudaBackend::sum_all(&*cuda_in3.data())?;
    assert!((cpu_sum3 - cuda_sum3).abs() < TOLERANCE);

    // Case 4: Scalar tensor (0D)
    let cpu_in4 = cpu_tensor(vec![42.], &[]);
    let cuda_in4 = cuda_tensor_from_cpu(&cpu_in4);
    let cpu_sum4 = CpuBackend::sum_all(&*cpu_in4.data())?;
    let cuda_sum4 = CudaBackend::sum_all(&*cuda_in4.data())?;
    assert!((cpu_sum4 - cuda_sum4).abs() < TOLERANCE);

    println!("test_cuda_sum_all PASSED");
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_div() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // 1. Define Input Data & Shape
    let data_a = vec![10.0, 15.0, 20.0, 25.0];
    let data_b = vec![2.0, 3.0, 4.0, 5.0];
    let shape = &[2, 2];

    // 2. Create CPU Tensors
    let cpu_a = cpu_tensor(data_a.clone(), shape);
    let cpu_b = cpu_tensor(data_b.clone(), shape);

    // 3. Compute Expected Result on CPU
    let expected_cpu_result = ops::div(&cpu_a, &cpu_b)?;

    // 4. Create CUDA Tensors
    let cuda_a = cuda_tensor_from_cpu(&cpu_a);
    let cuda_b = cuda_tensor_from_cpu(&cpu_b);

    // 5. Compute Actual Result on GPU
    let actual_cuda_result = ops::div(&cuda_a, &cuda_b)?;

    // 6. Compare
    assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_div_scalar() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test cases with different scalars
    let test_cases = vec![
        (vec![10.0, 15.0, 20.0, 25.0], &[2, 2], 2.0), // Normal division
        (vec![1.0, 2.0, 3.0, 4.0], &[2, 2], 0.5),     // Fraction scalar
        (vec![-8.0, -6.0, -4.0, -2.0], &[2, 2], -2.0), // Negative numbers
    ];

    for (data, shape, scalar) in test_cases {
        // Create CPU Tensor
        let cpu_a = cpu_tensor(data.clone(), shape);

        // Create CUDA Tensor
        let cuda_a = cuda_tensor_from_cpu(&cpu_a);

        // Compute Expected Result on CPU
        let expected_cpu_result = ops::div_scalar(&cpu_a, scalar)?;

        // Compute Actual Result on GPU
        let actual_cuda_result = ops::div_scalar(&cuda_a, scalar)?;

        // Compare
        assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);
    }

    // Test division by zero should return error
    let cpu_a = cpu_tensor(vec![1.0, 2.0], &[2]);
    let cuda_a = cuda_tensor_from_cpu(&cpu_a);
    assert!(matches!(
        ops::div_scalar(&cuda_a, 0.0),
        Err(Error::InvalidOperation(_))
    ));

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_exp() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test cases with different values
    let test_cases = vec![
        (vec![0.0, 1.0, -1.0], &[3]), // Basic cases
        (vec![-10.0, 10.0], &[2]),    // Large values
        (vec![0.5, -0.5, 0.1], &[3]), // Small values
        (vec![1.0], &[1]),            // Changed from &[] to &[1] for scalar
    ];

    // Use a larger tolerance for exp due to implementation differences between CPU and CUDA
    // CPU uses f32::exp() while CUDA uses expf()
    const EXP_TOLERANCE: f32 = 1e-3;

    for (data, shape) in test_cases {
        // Create CPU Tensor
        let cpu_a = cpu_tensor(data.clone(), shape);
        let cuda_a = cuda_tensor_from_cpu(&cpu_a);

        // Compute Expected Result on CPU
        let expected_cpu_result = ops::exp(&cpu_a)?;

        // Compute Actual Result on GPU
        let actual_cuda_result = ops::exp(&cuda_a)?;

        // Compare with larger tolerance for exp operation
        assert_tensors_close(&expected_cpu_result, &actual_cuda_result, EXP_TOLERANCE);
    }

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_ln() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test cases with different values
    let test_cases = vec![
        (vec![1.0, 2.0, 4.0], &[3]),       // Basic cases
        (vec![0.1, 0.01], &[2]),           // Small positive values
        (vec![100.0, 1000.0], &[2]),       // Large values
        (vec![std::f32::consts::E], &[1]), // Changed from hardcoded value to E constant
    ];

    for (data, shape) in test_cases {
        // Create CPU Tensor
        let cpu_a = cpu_tensor(data.clone(), shape);
        let cuda_a = cuda_tensor_from_cpu(&cpu_a);

        // Compute Expected Result on CPU
        let expected_cpu_result = ops::ln(&cpu_a)?;

        // Compute Actual Result on GPU
        let actual_cuda_result = ops::ln(&cuda_a)?;

        // Compare
        assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);
    }

    // Test non-positive input
    let cpu_a = cpu_tensor(vec![-1.0, 0.0], &[2]);
    let cuda_a = cuda_tensor_from_cpu(&cpu_a);
    let cuda_result = ops::ln(&cuda_a)?;
    let cpu_result = ops::ln(&cpu_a)?;
    // For non-positive inputs, ln should return -inf for 0 and NaN for negative
    let cuda_data = CudaBackend::copy_to_host(&*cuda_result.data())?;
    let cpu_data = CpuBackend::copy_to_host(&*cpu_result.data())?;
    assert!(cuda_data[0].is_nan() && cpu_data[0].is_nan()); // -1.0 -> NaN
    assert!(cuda_data[1].is_infinite() && cuda_data[1] < 0.0); // 0.0 -> -inf

    println!("test_cuda_ln PASSED");
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_log_softmax() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test cases
    let test_cases = vec![
        // 1D case (typical logits)
        (vec![1.0, 2.0, 3.0], vec![3], 0),
        // 2D case (batch of logits)
        (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 1),
        // Edge case with large values
        (vec![100.0, 100.1, 100.2], vec![3], 0),
        // Edge case with small values
        (vec![-100.0, -100.1, -100.2], vec![3], 0),
    ];

    for (data, shape, axis) in test_cases {
        let cpu_a = cpu_tensor(data.clone(), &shape);
        let cuda_a = cuda_tensor_from_cpu(&cpu_a);

        let expected_cpu_result = ops::log_softmax(&cpu_a, axis)?;
        let actual_cuda_result = ops::log_softmax(&cuda_a, axis)?;

        assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);
    }

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_log_softmax_edge_cases() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test edge cases
    let test_cases = vec![
        // Case 1: Very large numbers (potential overflow)
        (vec![1000.0, 1000.1, 1000.2], vec![3], 0),
        // Case 2: Very small numbers (potential underflow)
        (vec![-1000.0, -1000.1, -1000.2], vec![3], 0),
        // Case 3: Large differences (potential numeric instability)
        (vec![-100.0, 0.0, 100.0], vec![3], 0),
        // Case 4: All same values (test numerical stability)
        (vec![10.0, 10.0, 10.0], vec![3], 0),
        // Case 5: Single element (edge case)
        (vec![1.0], vec![1], 0),
        // Case 6: 3D tensor with different axes
        (
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
            2,
        ),
    ];

    for (data, shape, axis) in test_cases {
        let cpu_a = cpu_tensor(data.clone(), &shape);
        let cuda_a = cuda_tensor_from_cpu(&cpu_a);

        let expected_cpu_result = ops::log_softmax(&cpu_a, axis)?;
        let actual_cuda_result = ops::log_softmax(&cuda_a, axis)?;

        assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);
    }

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_broadcast_div() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test cases with broadcasting
    let test_cases = vec![
        // Case 1: Broadcasting scalar to vector
        (vec![2.0], vec![1], vec![2.0, 4.0, 6.0], vec![3]),
        // Case 2: Broadcasting row vector to matrix
        (
            vec![2.0, 4.0],
            vec![1, 2],
            vec![2.0, 4.0, 6.0, 8.0],
            vec![2, 2],
        ),
        // Case 3: Broadcasting column vector to matrix
        (
            vec![2.0, 4.0],
            vec![2, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ),
    ];

    for (data_a, shape_a, data_b, shape_b) in test_cases {
        let cpu_a = cpu_tensor(data_a.clone(), &shape_a);
        let cpu_b = cpu_tensor(data_b.clone(), &shape_b);

        let cuda_a = cuda_tensor_from_cpu(&cpu_a);
        let cuda_b = cuda_tensor_from_cpu(&cpu_b);

        let expected_cpu_result = ops::div(&cpu_a, &cpu_b)?;
        let actual_cuda_result = ops::div(&cuda_a, &cuda_b)?;

        assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);
    }

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_broadcast_sub() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test cases with broadcasting
    let test_cases = vec![
        // Case 1: Broadcasting scalar to vector
        (vec![10.0], vec![1], vec![1.0, 2.0, 3.0], vec![3]),
        // Case 2: Broadcasting row vector to matrix
        (
            vec![10.0, 20.0],
            vec![1, 2],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ),
        // Case 3: Broadcasting column vector to matrix
        (
            vec![10.0, 20.0],
            vec![2, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ),
    ];

    for (data_a, shape_a, data_b, shape_b) in test_cases {
        let cpu_a = cpu_tensor(data_a.clone(), &shape_a);
        let cpu_b = cpu_tensor(data_b.clone(), &shape_b);

        let cuda_a = cuda_tensor_from_cpu(&cpu_a);
        let cuda_b = cuda_tensor_from_cpu(&cpu_b);

        let expected_cpu_result = ops::sub(&cpu_a, &cpu_b)?;
        let actual_cuda_result = ops::sub(&cuda_a, &cuda_b)?;

        assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);
    }

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_transpose_verification() -> Result<(), Error> {
    println!("--- Running test_cuda_transpose_verification ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    println!("CUDA Context Initialized.");

    // Test Matrix 1 (Matches 'a' from matmul test)
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let shape1 = &[2, 2];
    let cpu_tensor1 = cpu_tensor(data1.clone(), shape1);
    let cuda_tensor1 = cuda_tensor_from_cpu(&cpu_tensor1);
    println!(
        "Testing transpose for tensor 1: shape={:?}, data={:?}",
        shape1, data1
    );

    // Perform CPU transpose
    let cpu_transposed_storage1 = CpuBackend::transpose(&*cpu_tensor1.data())?;
    let cpu_transposed1 = Tensor::new(cpu_transposed_storage1, false);
    println!(
        "CPU Transposed shape: {:?}, data: {:?}",
        cpu_transposed1.shape(),
        CpuBackend::copy_to_host(&*cpu_transposed1.data())?
    );

    // Perform CUDA transpose
    let cuda_transposed_storage1 = CudaBackend::transpose(&*cuda_tensor1.data())?;
    let cuda_transposed1 = Tensor::new(cuda_transposed_storage1, false);
    println!(
        "CUDA Transposed shape: {:?}, data (host): {:?}",
        cuda_transposed1.shape(),
        CudaBackend::copy_to_host(&*cuda_transposed1.data())?
    );

    // Compare
    assert_tensors_close(&cpu_transposed1, &cuda_transposed1, TOLERANCE);
    println!("Transpose for tensor 1 MATCHES.");

    // Test Matrix 2 (Matches 'b' from matmul test)
    let data2 = vec![5.0, 6.0, 7.0, 8.0];
    let shape2 = &[2, 2];
    let cpu_tensor2 = cpu_tensor(data2.clone(), shape2);
    let cuda_tensor2 = cuda_tensor_from_cpu(&cpu_tensor2);
    println!(
        "\nTesting transpose for tensor 2: shape={:?}, data={:?}",
        shape2, data2
    );

    // Perform CPU transpose
    let cpu_transposed_storage2 = CpuBackend::transpose(&*cpu_tensor2.data())?;
    let cpu_transposed2 = Tensor::new(cpu_transposed_storage2, false);
    println!(
        "CPU Transposed shape: {:?}, data: {:?}",
        cpu_transposed2.shape(),
        CpuBackend::copy_to_host(&*cpu_transposed2.data())?
    );

    // Perform CUDA transpose
    let cuda_transposed_storage2 = CudaBackend::transpose(&*cuda_tensor2.data())?;
    let cuda_transposed2 = Tensor::new(cuda_transposed_storage2, false);
    println!(
        "CUDA Transposed shape: {:?}, data (host): {:?}",
        cuda_transposed2.shape(),
        CudaBackend::copy_to_host(&*cuda_transposed2.data())?
    );

    // Compare
    assert_tensors_close(&cpu_transposed2, &cuda_transposed2, TOLERANCE);
    println!("Transpose for tensor 2 MATCHES.");

    // Test Matrix 3 (Non-square)
    let data3 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape3 = &[2, 3];
    let cpu_tensor3 = cpu_tensor(data3.clone(), shape3);
    let cuda_tensor3 = cuda_tensor_from_cpu(&cpu_tensor3);
    println!(
        "\nTesting transpose for tensor 3: shape={:?}, data={:?}",
        shape3, data3
    );

    // Perform CPU transpose
    let cpu_transposed_storage3 = CpuBackend::transpose(&*cpu_tensor3.data())?;
    let cpu_transposed3 = Tensor::new(cpu_transposed_storage3, false);
    println!(
        "CPU Transposed shape: {:?}, data: {:?}",
        cpu_transposed3.shape(),
        CpuBackend::copy_to_host(&*cpu_transposed3.data())?
    );

    // Perform CUDA transpose
    let cuda_transposed_storage3 = CudaBackend::transpose(&*cuda_tensor3.data())?;
    let cuda_transposed3 = Tensor::new(cuda_transposed_storage3, false);
    println!(
        "CUDA Transposed shape: {:?}, data (host): {:?}",
        cuda_transposed3.shape(),
        CudaBackend::copy_to_host(&*cuda_transposed3.data())?
    );

    // Compare
    assert_tensors_close(&cpu_transposed3, &cuda_transposed3, TOLERANCE);
    println!("Transpose for tensor 3 MATCHES.");

    println!("--- test_cuda_transpose_verification PASSED ---");
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_tanh() -> Result<(), Error> {
    println!("[test_cuda_tanh] Test entry");
    let _ = rust_tensor_lib::backend::cuda::init_context(0);
    let _context_guard = rust_tensor_lib::backend::cuda::CudaContextGuard::new()?;
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let cpu_tensor = cpu_tensor(data.clone(), &[5]);
    let cuda_tensor = cuda_tensor_from_cpu(&cpu_tensor);

    // Calculate expected results on CPU
    let cpu_result = ops::tanh(&cpu_tensor)?;

    // Calculate results on CUDA
    let cuda_result = ops::tanh(&cuda_tensor)?;

    // Compare results
    assert_tensors_close(&cpu_result, &cuda_result, 1e-5);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_softplus() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 10.1]; // Include a larger value
    let shape = [6];
    let cpu_input = cpu_tensor(data.clone(), &shape);
    let cpu_output = ops::softplus(&cpu_input)?; // Use ops::softplus

    let cuda_input = cuda_tensor_from_cpu(&cpu_input);
    let cuda_output = ops::softplus(&cuda_input)?; // Use ops::softplus

    // Use slightly higher tolerance if necessary due to potential exp/ln differences
    assert_tensors_close(&cpu_output, &cuda_output, 1e-5);
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_binary_cross_entropy_with_logits() -> Result<(), Error> {
    use crate::{assert_tensors_close, cpu_tensor, cuda_tensor_from_cpu, TOLERANCE};
    use approx::assert_abs_diff_eq;
    use rust_tensor_lib::{
        backend::cuda::{init_context, CudaBackend, CudaContextGuard},
        ops, CpuBackend, Error, Reduction, Tensor,
    };

    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let logits_data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let targets_data = vec![0.0, 1.0, 0.0, 1.0, 1.0];
    let shape = &[logits_data.len()];

    let cpu_logits = cpu_tensor(logits_data.clone(), shape);
    let cpu_targets = cpu_tensor(targets_data.clone(), shape);
    let cuda_logits = cuda_tensor_from_cpu(&cpu_logits);
    let cuda_targets = cuda_tensor_from_cpu(&cpu_targets);

    // Test Reduction::None
    let cpu_loss_none =
        ops::binary_cross_entropy_with_logits(&cpu_logits, &cpu_targets, Reduction::None)?;
    let cuda_loss_none =
        ops::binary_cross_entropy_with_logits(&cuda_logits, &cuda_targets, Reduction::None)?;
    assert_tensors_close(&cpu_loss_none, &cuda_loss_none, TOLERANCE);

    // Test Reduction::Mean
    let cpu_loss_mean =
        ops::binary_cross_entropy_with_logits(&cpu_logits, &cpu_targets, Reduction::Mean)?;
    let cuda_loss_mean =
        ops::binary_cross_entropy_with_logits(&cuda_logits, &cuda_targets, Reduction::Mean)?;
    let cpu_mean_val = cpu_loss_mean.data().as_ref()[0];
    let cuda_mean_val = CudaBackend::copy_to_host(&*cuda_loss_mean.data())?[0];
    assert_abs_diff_eq!(cpu_mean_val, cuda_mean_val, epsilon = TOLERANCE);

    // Test Reduction::Sum
    let cpu_loss_sum =
        ops::binary_cross_entropy_with_logits(&cpu_logits, &cpu_targets, Reduction::Sum)?;
    let cuda_loss_sum =
        ops::binary_cross_entropy_with_logits(&cuda_logits, &cuda_targets, Reduction::Sum)?;
    let cpu_sum_val = cpu_loss_sum.data().as_ref()[0];
    let cuda_sum_val = CudaBackend::copy_to_host(&*cuda_loss_sum.data())?[0];
    assert_abs_diff_eq!(cpu_sum_val, cuda_sum_val, epsilon = TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_min_reduction() -> Result<(), Error> {
    // Initialize CUDA context
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    println!("Starting test_cuda_min_reduction");

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    println!("Data: {:?}", data);

    let x_cpu = CpuBackend::from_vec(data.clone(), &[2, 3]).unwrap();
    let x_cuda = CudaBackend::from_vec(data, &[2, 3]).unwrap();

    println!("Testing global min");
    // Test global min
    let result_cpu = CpuBackend::min(&x_cpu, None).unwrap();
    println!(
        "CPU global min complete: {:?}",
        CpuBackend::copy_to_host(&result_cpu).unwrap()
    );

    let result_cuda = CudaBackend::min(&x_cuda, None).unwrap();
    println!(
        "CUDA global min complete: {:?}",
        CudaBackend::copy_to_host(&result_cuda).unwrap()
    );

    // Wrap storage back into Tensors for comparison using the correct helper
    let cpu_tensor_global = Tensor::new(result_cpu, false);
    let cuda_tensor_global = Tensor::new(result_cuda, false);
    assert_tensors_close(&cpu_tensor_global, &cuda_tensor_global, TOLERANCE);

    println!("Testing min along axis 0");
    // Test min along axis 0
    let result_cpu_ax0 = CpuBackend::min(&x_cpu, Some(0)).unwrap();
    println!(
        "CPU axis 0 min complete: {:?}",
        CpuBackend::copy_to_host(&result_cpu_ax0).unwrap()
    );

    let result_cuda_ax0 = CudaBackend::min(&x_cuda, Some(0)).unwrap();
    println!(
        "CUDA axis 0 min complete: {:?}",
        CudaBackend::copy_to_host(&result_cuda_ax0).unwrap()
    );

    // Wrap storage back into Tensors for comparison using the correct helper
    let cpu_tensor_ax0 = Tensor::new(result_cpu_ax0, false);
    let cuda_tensor_ax0 = Tensor::new(result_cuda_ax0, false);
    assert_tensors_close(&cpu_tensor_ax0, &cuda_tensor_ax0, TOLERANCE);

    println!("Testing min along axis 1");
    // Test min along axis 1
    let result_cpu_ax1 = CpuBackend::min(&x_cpu, Some(1)).unwrap();
    println!(
        "CPU axis 1 min complete: {:?}",
        CpuBackend::copy_to_host(&result_cpu_ax1).unwrap()
    );

    let result_cuda_ax1 = CudaBackend::min(&x_cuda, Some(1)).unwrap();
    println!(
        "CUDA axis 1 min complete: {:?}",
        CudaBackend::copy_to_host(&result_cuda_ax1).unwrap()
    );

    // Wrap storage back into Tensors for comparison using the correct helper
    let cpu_tensor_ax1 = Tensor::new(result_cpu_ax1, false);
    let cuda_tensor_ax1 = Tensor::new(result_cuda_ax1, false);
    assert_tensors_close(&cpu_tensor_ax1, &cuda_tensor_ax1, TOLERANCE);

    println!("Test completed successfully");
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_prod_reduction() -> Result<(), Error> {
    // Initialize CUDA context
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    println!("Starting test_cuda_prod_reduction");

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    println!("Data: {:?}", data);

    let x_cpu = CpuBackend::from_vec(data.clone(), &[2, 3]).unwrap();
    let x_cuda = CudaBackend::from_vec(data, &[2, 3]).unwrap();

    println!("Testing global product");
    // Test global product
    let result_cpu = CpuBackend::prod(&x_cpu, None).unwrap();
    println!(
        "CPU global prod complete: {:?}",
        CpuBackend::copy_to_host(&result_cpu).unwrap()
    );

    let result_cuda = CudaBackend::prod(&x_cuda, None).unwrap();
    println!(
        "CUDA global prod complete: {:?}",
        CudaBackend::copy_to_host(&result_cuda).unwrap()
    );

    // Wrap storage back into Tensors for comparison using the correct helper
    let cpu_tensor_global = Tensor::new(result_cpu, false);
    let cuda_tensor_global = Tensor::new(result_cuda, false);
    assert_tensors_close(&cpu_tensor_global, &cuda_tensor_global, TOLERANCE);

    println!("Testing prod along axis 0");
    // Test prod along axis 0
    let result_cpu_ax0 = CpuBackend::prod(&x_cpu, Some(0)).unwrap();
    println!(
        "CPU axis 0 prod complete: {:?}",
        CpuBackend::copy_to_host(&result_cpu_ax0).unwrap()
    );

    let result_cuda_ax0 = CudaBackend::prod(&x_cuda, Some(0)).unwrap();
    println!(
        "CUDA axis 0 prod complete: {:?}",
        CudaBackend::copy_to_host(&result_cuda_ax0).unwrap()
    );

    // Wrap storage back into Tensors for comparison using the correct helper
    let cpu_tensor_ax0 = Tensor::new(result_cpu_ax0, false);
    let cuda_tensor_ax0 = Tensor::new(result_cuda_ax0, false);
    assert_tensors_close(&cpu_tensor_ax0, &cuda_tensor_ax0, TOLERANCE);

    println!("Testing prod along axis 1");
    // Test prod along axis 1
    let result_cpu_ax1 = CpuBackend::prod(&x_cpu, Some(1)).unwrap();
    println!(
        "CPU axis 1 prod complete: {:?}",
        CpuBackend::copy_to_host(&result_cpu_ax1).unwrap()
    );

    let result_cuda_ax1 = CudaBackend::prod(&x_cuda, Some(1)).unwrap();
    println!(
        "CUDA axis 1 prod complete: {:?}",
        CudaBackend::copy_to_host(&result_cuda_ax1).unwrap()
    );

    // Wrap storage back into Tensors for comparison using the correct helper
    let cpu_tensor_ax1 = Tensor::new(result_cpu_ax1, false);
    let cuda_tensor_ax1 = Tensor::new(result_cuda_ax1, false);
    assert_tensors_close(&cpu_tensor_ax1, &cuda_tensor_ax1, TOLERANCE);

    println!("Test completed successfully");
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_logsumexp_reduction() -> Result<(), Error> {
    // Initialize CUDA context
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    println!("Starting test_cuda_logsumexp_reduction");

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    println!("Data: {:?}", data);

    let x_cpu = CpuBackend::from_vec(data.clone(), &[2, 3]).unwrap();
    let x_cuda = CudaBackend::from_vec(data, &[2, 3]).unwrap();

    println!("Testing global logsumexp");
    // Test global logsumexp
    let result_cpu = CpuBackend::logsumexp(&x_cpu, None).unwrap();
    println!(
        "CPU global logsumexp complete: {:?}",
        CpuBackend::copy_to_host(&result_cpu).unwrap()
    );

    let result_cuda = CudaBackend::logsumexp(&x_cuda, None).unwrap();
    println!(
        "CUDA global logsumexp complete: {:?}",
        CudaBackend::copy_to_host(&result_cuda).unwrap()
    );

    // Wrap storage back into Tensors for comparison using the correct helper
    let cpu_tensor_global = Tensor::new(result_cpu, false);
    let cuda_tensor_global = Tensor::new(result_cuda, false);
    assert_tensors_close(&cpu_tensor_global, &cuda_tensor_global, 1e-5);

    println!("Testing logsumexp along axis 0");
    // Test logsumexp along axis 0
    let result_cpu_ax0 = CpuBackend::logsumexp(&x_cpu, Some(0)).unwrap();
    println!(
        "CPU axis 0 logsumexp complete: {:?}",
        CpuBackend::copy_to_host(&result_cpu_ax0).unwrap()
    );

    let result_cuda_ax0 = CudaBackend::logsumexp(&x_cuda, Some(0)).unwrap();
    println!(
        "CUDA axis 0 logsumexp complete: {:?}",
        CudaBackend::copy_to_host(&result_cuda_ax0).unwrap()
    );

    // Wrap storage back into Tensors for comparison using the correct helper
    let cpu_tensor_ax0 = Tensor::new(result_cpu_ax0, false);
    let cuda_tensor_ax0 = Tensor::new(result_cuda_ax0, false);
    assert_tensors_close(&cpu_tensor_ax0, &cuda_tensor_ax0, 1e-5);

    println!("Testing logsumexp along axis 1");
    // Test logsumexp along axis 1
    let result_cpu_ax1 = CpuBackend::logsumexp(&x_cpu, Some(1)).unwrap();
    println!(
        "CPU axis 1 logsumexp complete: {:?}",
        CpuBackend::copy_to_host(&result_cpu_ax1).unwrap()
    );

    let result_cuda_ax1 = CudaBackend::logsumexp(&x_cuda, Some(1)).unwrap();
    println!(
        "CUDA axis 1 logsumexp complete: {:?}",
        CudaBackend::copy_to_host(&result_cuda_ax1).unwrap()
    );

    // Wrap storage back into Tensors for comparison using the correct helper
    let cpu_tensor_ax1 = Tensor::new(result_cpu_ax1, false);
    let cuda_tensor_ax1 = Tensor::new(result_cuda_ax1, false);
    assert_tensors_close(&cpu_tensor_ax1, &cuda_tensor_ax1, 1e-5);

    println!("Test completed successfully");
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_powf() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // 1. Test basic powf operation
    let cpu_a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let cpu_b = cpu_tensor(vec![2.0, 3.0, 0.5, 4.0], &[2, 2]);
    let expected_cpu_result = ops::powf(&cpu_a, &cpu_b)?;

    let cuda_a = cuda_tensor_from_cpu(&cpu_a);
    let cuda_b = cuda_tensor_from_cpu(&cpu_b);
    let actual_cuda_result = ops::powf(&cuda_a, &cuda_b)?;

    assert_tensors_close(&expected_cpu_result, &actual_cuda_result, 1e-5);

    // 2. Test broadcasting
    let cpu_a2 = cpu_tensor(vec![2.0, 3.0, 4.0], &[3]); // Base values
    let cpu_b2 = cpu_tensor(vec![0.5, 2.0, 3.0, 0.5, 2.0, 3.0], &[2, 3]); // Exponents
    let expected_cpu_result2 = ops::powf(&cpu_a2, &cpu_b2)?;

    let cuda_a2 = cuda_tensor_from_cpu(&cpu_a2);
    let cuda_b2 = cuda_tensor_from_cpu(&cpu_b2);
    let actual_cuda_result2 = ops::powf(&cuda_a2, &cuda_b2)?;

    assert_tensors_close(&expected_cpu_result2, &actual_cuda_result2, 1e-5);

    // 3. Test with scalar exponent (powf_scalar)
    let cpu_a3 = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let expected_cpu_result3 = ops::powf_scalar(&cpu_a3, 2.0)?;

    let cuda_a3 = cuda_tensor_from_cpu(&cpu_a3);
    let actual_cuda_result3 = ops::powf_scalar(&cuda_a3, 2.0)?;

    assert_tensors_close(&expected_cpu_result3, &actual_cuda_result3, 1e-5);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_square() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let cpu_a = cpu_tensor(vec![-2.0, 0.0, 3.0, 1.5], &[2, 2]);
    let expected_cpu_result = ops::square(&cpu_a)?;

    let cuda_a = cuda_tensor_from_cpu(&cpu_a);
    let actual_cuda_result = ops::square(&cuda_a)?;

    assert_tensors_close(&expected_cpu_result, &actual_cuda_result, TOLERANCE);
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_powf_gradient() -> Result<(), Error> {
    use rust_tensor_lib::backend::{CpuTensor, CudaTensor};
    use rust_tensor_lib::ops;
    println!("[test_cuda_powf_gradient] Testing powf gradient on CUDA");

    let _ = init_context(0);
    let _guard = CudaContextGuard::new()?;

    // CPU version
    let a_cpu = CpuTensor::from_vec(vec![2.0, 3.0], &[2], true).unwrap();
    let b_cpu = CpuTensor::from_vec(vec![3.0, 2.0], &[2], true).unwrap();
    let c_cpu = ops::powf(&a_cpu, &b_cpu).unwrap();
    // Create a scalar loss to call backward on
    let loss_cpu = ops::mean(&c_cpu, None).unwrap();

    // CUDA version
    // Create CUDA tensor directly with requires_grad=true
    let a_cpu_data = CpuBackend::copy_to_host(&*a_cpu.data()).unwrap();
    let a_cuda = CudaTensor::from_vec(a_cpu_data, a_cpu.shape().as_slice(), true)?;

    // Create CUDA tensor directly with requires_grad=true
    let b_cpu_data = CpuBackend::copy_to_host(&*b_cpu.data()).unwrap();
    let b_cuda = CudaTensor::from_vec(b_cpu_data, b_cpu.shape().as_slice(), true)?;

    let c_cuda = ops::powf(&a_cuda, &b_cuda).unwrap();
    // Create a scalar loss to call backward on
    let loss_cuda = ops::mean(&c_cuda, None).unwrap();

    // Compare forward results
    let c_cpu_data = c_cpu.data().as_ref().to_vec();
    let c_cuda_data = c_cuda.to_cpu().unwrap().data().as_ref().to_vec();

    for i in 0..2 {
        assert!(
            (c_cpu_data[i] - c_cuda_data[i]).abs() < 1e-5,
            "at index {}: CPU forward {} != CUDA forward {}",
            i,
            c_cpu_data[i],
            c_cuda_data[i]
        );
    }

    // Backward pass for both
    loss_cpu.backward().unwrap();
    loss_cuda.backward().unwrap();

    // Compare gradients for a
    let a_grad_cpu_vec = a_cpu.grad().unwrap().as_ref().to_vec();
    let a_grad_cuda_storage = a_cuda.grad().unwrap().clone();
    let a_grad_cuda_tensor = Tensor::<CudaBackend>::new(a_grad_cuda_storage, false);
    let a_grad_cuda_vec = a_grad_cuda_tensor
        .to_cpu()
        .unwrap()
        .data()
        .as_ref()
        .to_vec();

    for i in 0..a_grad_cpu_vec.len() {
        assert!(
            (a_grad_cpu_vec[i] - a_grad_cuda_vec[i]).abs() < 1e-5,
            "at index {}: CPU a_grad {} != CUDA a_grad {}",
            i,
            a_grad_cpu_vec[i],
            a_grad_cuda_vec[i]
        );
    }

    // Compare gradients for b
    let b_grad_cpu_vec = b_cpu.grad().unwrap().as_ref().to_vec();
    let b_grad_cuda_storage = b_cuda.grad().unwrap().clone();
    let b_grad_cuda_tensor = Tensor::<CudaBackend>::new(b_grad_cuda_storage, false);
    let b_grad_cuda_vec = b_grad_cuda_tensor
        .to_cpu()
        .unwrap()
        .data()
        .as_ref()
        .to_vec();

    for i in 0..b_grad_cpu_vec.len() {
        assert!(
            (b_grad_cpu_vec[i] - b_grad_cuda_vec[i]).abs() < 1e-5,
            "at index {}: CPU b_grad {} != CUDA b_grad {}",
            i,
            b_grad_cpu_vec[i],
            b_grad_cuda_vec[i]
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[serial]
fn test_maximum_forward_cuda() {
    init_context(0).unwrap();
    let _guard = CudaContextGuard::new();

    // Case 1: Tensors of the same shape
    let a_cpu = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], false).unwrap();
    let b_cpu = Tensor::<CpuBackend>::from_vec(vec![4.0, 2.0, 1.0, 5.0], &[2, 2], false).unwrap();
    let a_cuda = a_cpu.to_gpu(0).unwrap();
    let b_cuda = b_cpu.to_gpu(0).unwrap();

    let c_cpu = ops::maximum(&a_cpu, &b_cpu).unwrap();
    let c_cuda = ops::maximum(&a_cuda, &b_cuda).unwrap();

    assert_tensors_close(&c_cpu, &c_cuda, 1e-5);

    // Case 2: Scalar and vector broadcasting
    let scalar_cpu = Tensor::<CpuBackend>::from_vec(vec![3.0], &[], false).unwrap();
    let vector_cpu = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 2.0], &[3], false).unwrap();
    let scalar_cuda = scalar_cpu.to_gpu(0).unwrap();
    let vector_cuda = vector_cpu.to_gpu(0).unwrap();

    let result_cpu = ops::maximum(&scalar_cpu, &vector_cpu).unwrap();
    let result_cuda = ops::maximum(&scalar_cuda, &vector_cuda).unwrap();

    assert_tensors_close(&result_cpu, &result_cuda, 1e-5);

    // Case 3: Matrix and row vector broadcasting
    let matrix_cpu =
        Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let row_cpu = Tensor::<CpuBackend>::from_vec(vec![3.0, 1.0, 4.0], &[1, 3], false).unwrap();
    let matrix_cuda = matrix_cpu.to_gpu(0).unwrap();
    let row_cuda = row_cpu.to_gpu(0).unwrap();

    let result_cpu = ops::maximum(&matrix_cpu, &row_cpu).unwrap();
    let result_cuda = ops::maximum(&matrix_cuda, &row_cuda).unwrap();

    assert_tensors_close(&result_cpu, &result_cuda, 1e-5);
}

#[cfg(feature = "cuda")]
#[test]
#[serial]
fn test_minimum_forward_cuda() {
    init_context(0).unwrap();
    let _guard = CudaContextGuard::new();

    // Case 1: Tensors of the same shape
    let a_cpu = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], false).unwrap();
    let b_cpu = Tensor::<CpuBackend>::from_vec(vec![4.0, 2.0, 1.0, 5.0], &[2, 2], false).unwrap();
    let a_cuda = a_cpu.to_gpu(0).unwrap();
    let b_cuda = b_cpu.to_gpu(0).unwrap();

    let c_cpu = ops::minimum(&a_cpu, &b_cpu).unwrap();
    let c_cuda = ops::minimum(&a_cuda, &b_cuda).unwrap();

    assert_tensors_close(&c_cpu, &c_cuda, 1e-5);

    // Case 2: Scalar and vector broadcasting
    let scalar_cpu = Tensor::<CpuBackend>::from_vec(vec![3.0], &[], false).unwrap();
    let vector_cpu = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 2.0], &[3], false).unwrap();
    let scalar_cuda = scalar_cpu.to_gpu(0).unwrap();
    let vector_cuda = vector_cpu.to_gpu(0).unwrap();

    let result_cpu = ops::minimum(&scalar_cpu, &vector_cpu).unwrap();
    let result_cuda = ops::minimum(&scalar_cuda, &vector_cuda).unwrap();

    assert_tensors_close(&result_cpu, &result_cuda, 1e-5);

    // Case 3: Matrix and row vector broadcasting
    let matrix_cpu =
        Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let row_cpu = Tensor::<CpuBackend>::from_vec(vec![3.0, 1.0, 4.0], &[1, 3], false).unwrap();
    let matrix_cuda = matrix_cpu.to_gpu(0).unwrap();
    let row_cuda = row_cpu.to_gpu(0).unwrap();

    let result_cpu = ops::minimum(&matrix_cpu, &row_cpu).unwrap();
    let result_cuda = ops::minimum(&matrix_cuda, &row_cuda).unwrap();

    assert_tensors_close(&result_cpu, &result_cuda, 1e-5);
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_equal() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    println!("--- Running test_cuda_equal ---");

    // Case 1: Equal shapes, exact match
    let cpu_a1 = cpu_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let cpu_b1 = cpu_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let cuda_a1 = cuda_tensor_from_cpu(&cpu_a1);
    let cuda_b1 = cuda_tensor_from_cpu(&cpu_b1);
    let expected_cpu1 = Tensor::new(CpuBackend::equal(&*cpu_a1.data(), &*cpu_b1.data())?, false);
    let actual_cuda1 = Tensor::new(
        CudaBackend::equal(&*cuda_a1.data(), &*cuda_b1.data())?,
        false,
    );
    println!(
        "Case 1 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu1.data())?
    );
    println!(
        "Case 1 Actual (CUDA):  {:?}",
        CudaBackend::copy_to_host(&*actual_cuda1.data())?
    );
    assert_tensors_close(&expected_cpu1, &actual_cuda1, TOLERANCE);

    // Case 2: Equal shapes, some differences
    let cpu_a2 = cpu_tensor(vec![1.0, 2.1, 3.0], &[3]);
    let cpu_b2 = cpu_tensor(vec![1.0, 2.0, 3.1], &[3]);
    let cuda_a2 = cuda_tensor_from_cpu(&cpu_a2);
    let cuda_b2 = cuda_tensor_from_cpu(&cpu_b2);
    let expected_cpu2 = Tensor::new(CpuBackend::equal(&*cpu_a2.data(), &*cpu_b2.data())?, false);
    let actual_cuda2 = Tensor::new(
        CudaBackend::equal(&*cuda_a2.data(), &*cuda_b2.data())?,
        false,
    );
    println!(
        "Case 2 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu2.data())?
    );
    println!(
        "Case 2 Actual (CUDA):  {:?}",
        CudaBackend::copy_to_host(&*actual_cuda2.data())?
    );
    assert_tensors_close(&expected_cpu2, &actual_cuda2, TOLERANCE);

    // Case 3: Equal shapes, near equality (within tolerance)
    let tolerance = 1e-9;
    let val = 1.0;
    let cpu_a3 = cpu_tensor(vec![val], &[1]);
    let cpu_b3 = cpu_tensor(vec![val + tolerance * 0.5], &[1]); // Should be equal
    let cuda_a3 = cuda_tensor_from_cpu(&cpu_a3);
    let cuda_b3 = cuda_tensor_from_cpu(&cpu_b3);
    let expected_cpu3 = Tensor::new(CpuBackend::equal(&*cpu_a3.data(), &*cpu_b3.data())?, false);
    let actual_cuda3 = Tensor::new(
        CudaBackend::equal(&*cuda_a3.data(), &*cuda_b3.data())?,
        false,
    );
    println!(
        "Case 3 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu3.data())?
    );
    println!(
        "Case 3 Actual (CUDA):  {:?}",
        CudaBackend::copy_to_host(&*actual_cuda3.data())?
    );
    assert_tensors_close(&expected_cpu3, &actual_cuda3, TOLERANCE);

    // Case 4: Broadcasting (scalar and vector)
    let cpu_a4 = cpu_tensor(vec![2.0], &[1]);
    let cpu_b4 = cpu_tensor(vec![1.0, 2.0, 3.0, 2.0], &[4]);
    let cuda_a4 = cuda_tensor_from_cpu(&cpu_a4);
    let cuda_b4 = cuda_tensor_from_cpu(&cpu_b4);
    let expected_cpu4 = Tensor::new(CpuBackend::equal(&*cpu_a4.data(), &*cpu_b4.data())?, false);
    let actual_cuda4 = Tensor::new(
        CudaBackend::equal(&*cuda_a4.data(), &*cuda_b4.data())?,
        false,
    );
    println!(
        "Case 4 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu4.data())?
    );
    println!(
        "Case 4 Actual (CUDA):  {:?}",
        CudaBackend::copy_to_host(&*actual_cuda4.data())?
    );
    assert_tensors_close(&expected_cpu4, &actual_cuda4, TOLERANCE);

    // Case 5: Broadcasting (matrix and row vector)
    let cpu_a5 = cpu_tensor(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    let cpu_b5 = cpu_tensor(vec![1.0, 5.0, 3.0], &[1, 3]);
    let cuda_a5 = cuda_tensor_from_cpu(&cpu_a5);
    let cuda_b5 = cuda_tensor_from_cpu(&cpu_b5);
    let expected_cpu5 = Tensor::new(CpuBackend::equal(&*cpu_a5.data(), &*cpu_b5.data())?, false);
    let actual_cuda5 = Tensor::new(
        CudaBackend::equal(&*cuda_a5.data(), &*cuda_b5.data())?,
        false,
    );
    println!(
        "Case 5 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu5.data())?
    );
    println!(
        "Case 5 Actual (CUDA):  {:?}",
        CudaBackend::copy_to_host(&*actual_cuda5.data())?
    );
    assert_tensors_close(&expected_cpu5, &actual_cuda5, TOLERANCE);

    println!("--- test_cuda_equal PASSED ---");
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_greater() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Same shape
    let cpu_a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let cpu_b1 = cpu_tensor(vec![2.0, 3.0, 2.0], &[3]);
    let cuda_a1 = cuda_tensor_from_cpu(&cpu_a1);
    let cuda_b1 = cuda_tensor_from_cpu(&cpu_b1);
    let expected_cpu1 = ops::greater(&cpu_a1, &cpu_b1)?;
    let actual_cuda1 = ops::greater(&cuda_a1, &cuda_b1)?;
    assert_tensors_close(&expected_cpu1, &actual_cuda1, TOLERANCE);

    // Case 2: Broadcasting scalar
    let cpu_a2 = cpu_tensor(vec![3.0], &[]);
    let cpu_b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let cuda_a2 = cuda_tensor_from_cpu(&cpu_a2);
    let cuda_b2 = cuda_tensor_from_cpu(&cpu_b2);
    let expected_cpu2 = ops::greater(&cpu_a2, &cpu_b2)?;
    let actual_cuda2 = ops::greater(&cuda_a2, &cuda_b2)?;
    assert_tensors_close(&expected_cpu2, &actual_cuda2, TOLERANCE);

    // Case 3: Broadcasting matrix vs row
    let cpu_a3 = cpu_tensor(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
    let cpu_b3 = cpu_tensor(vec![3.0, 3.0, 3.0], &[1, 3]);
    let cuda_a3 = cuda_tensor_from_cpu(&cpu_a3);
    let cuda_b3 = cuda_tensor_from_cpu(&cpu_b3);
    let expected_cpu3 = ops::greater(&cpu_a3, &cpu_b3)?;
    let actual_cuda3 = ops::greater(&cuda_a3, &cuda_b3)?;
    assert_tensors_close(&expected_cpu3, &actual_cuda3, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_greater_equal() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Same shape including equals
    let cpu_a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let cpu_b1 = cpu_tensor(vec![2.0, 3.0, 2.0], &[3]);
    let cuda_a1 = cuda_tensor_from_cpu(&cpu_a1);
    let cuda_b1 = cuda_tensor_from_cpu(&cpu_b1);
    let expected_cpu1 = ops::greater_equal(&cpu_a1, &cpu_b1)?;
    let actual_cuda1 = ops::greater_equal(&cuda_a1, &cuda_b1)?;
    assert_tensors_close(&expected_cpu1, &actual_cuda1, TOLERANCE);

    // Case 2: Broadcasting scalar
    let cpu_a2 = cpu_tensor(vec![3.0], &[]);
    let cpu_b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let cuda_a2 = cuda_tensor_from_cpu(&cpu_a2);
    let cuda_b2 = cuda_tensor_from_cpu(&cpu_b2);
    let expected_cpu2 = ops::greater_equal(&cpu_a2, &cpu_b2)?;
    let actual_cuda2 = ops::greater_equal(&cuda_a2, &cuda_b2)?;
    assert_tensors_close(&expected_cpu2, &actual_cuda2, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_less() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Same shape
    let cpu_a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let cpu_b1 = cpu_tensor(vec![2.0, 3.0, 2.0], &[3]);
    let cuda_a1 = cuda_tensor_from_cpu(&cpu_a1);
    let cuda_b1 = cuda_tensor_from_cpu(&cpu_b1);
    let expected_cpu1 = ops::less(&cpu_a1, &cpu_b1)?;
    let actual_cuda1 = ops::less(&cuda_a1, &cuda_b1)?;
    assert_tensors_close(&expected_cpu1, &actual_cuda1, TOLERANCE);

    // Case 2: Broadcasting scalar
    let cpu_a2 = cpu_tensor(vec![3.0], &[]);
    let cpu_b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let cuda_a2 = cuda_tensor_from_cpu(&cpu_a2);
    let cuda_b2 = cuda_tensor_from_cpu(&cpu_b2);
    let expected_cpu2 = ops::less(&cpu_a2, &cpu_b2)?;
    let actual_cuda2 = ops::less(&cuda_a2, &cuda_b2)?;
    assert_tensors_close(&expected_cpu2, &actual_cuda2, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_less_equal() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Same shape including equals
    let cpu_a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let cpu_b1 = cpu_tensor(vec![2.0, 3.0, 2.0], &[3]);
    let cuda_a1 = cuda_tensor_from_cpu(&cpu_a1);
    let cuda_b1 = cuda_tensor_from_cpu(&cpu_b1);
    let expected_cpu1 = ops::less_equal(&cpu_a1, &cpu_b1)?;
    let actual_cuda1 = ops::less_equal(&cuda_a1, &cuda_b1)?;
    assert_tensors_close(&expected_cpu1, &actual_cuda1, TOLERANCE);

    // Case 2: Broadcasting
    let cpu_a2 = cpu_tensor(vec![3.0], &[]);
    let cpu_b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let cuda_a2 = cuda_tensor_from_cpu(&cpu_a2);
    let cuda_b2 = cuda_tensor_from_cpu(&cpu_b2);
    let expected_cpu2 = ops::less_equal(&cpu_a2, &cpu_b2)?;
    let actual_cuda2 = ops::less_equal(&cuda_a2, &cuda_b2)?;
    assert_tensors_close(&expected_cpu2, &actual_cuda2, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_not_equal() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Same shape with some equalities
    let cpu_a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let cpu_b1 = cpu_tensor(vec![2.0, 5.0, 3.0], &[3]);
    let cuda_a1 = cuda_tensor_from_cpu(&cpu_a1);
    let cuda_b1 = cuda_tensor_from_cpu(&cpu_b1);
    let expected_cpu1 = ops::not_equal(&cpu_a1, &cpu_b1)?;
    let actual_cuda1 = ops::not_equal(&cuda_a1, &cuda_b1)?;
    assert_tensors_close(&expected_cpu1, &actual_cuda1, TOLERANCE);

    // Case 2: Broadcasting with exact equality
    let cpu_a2 = cpu_tensor(vec![3.0], &[]);
    let cpu_b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let cuda_a2 = cuda_tensor_from_cpu(&cpu_a2);
    let cuda_b2 = cuda_tensor_from_cpu(&cpu_b2);
    let expected_cpu2 = ops::not_equal(&cpu_a2, &cpu_b2)?;
    let actual_cuda2 = ops::not_equal(&cuda_a2, &cuda_b2)?;
    assert_tensors_close(&expected_cpu2, &actual_cuda2, TOLERANCE);

    // Case 3: Near equality test (floating point tolerance)
    let tolerance = 1e-9;
    let val = 1.0;
    let cpu_a3 = cpu_tensor(vec![val], &[1]);
    let cpu_b3 = cpu_tensor(vec![val + tolerance * 0.9], &[1]); // Should NOT be equal
    let cuda_a3 = cuda_tensor_from_cpu(&cpu_a3);
    let cuda_b3 = cuda_tensor_from_cpu(&cpu_b3);
    let expected_cpu3 = ops::not_equal(&cpu_a3, &cpu_b3)?;
    let actual_cuda3 = ops::not_equal(&cuda_a3, &cuda_b3)?;
    assert_tensors_close(&expected_cpu3, &actual_cuda3, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_elu() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let data = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]; // Test with negative and positive values
    let shape = [7];
    let alpha = 1.0;

    let cpu_input = cpu_tensor(data.clone(), &shape);
    let cpu_output = ops::elu(&cpu_input, alpha)?; // Use ops::elu

    let cuda_input = cuda_tensor_from_cpu(&cpu_input);
    let cuda_output = ops::elu(&cuda_input, alpha)?; // Use ops::elu

    // Use slightly higher tolerance due to potential exp differences in implementation
    assert_tensors_close(&cpu_output, &cuda_output, 1e-5);

    // Test with a different alpha value
    let alpha2 = 2.0;
    let cpu_output2 = ops::elu(&cpu_input, alpha2)?;
    let cuda_output2 = ops::elu(&cuda_input, alpha2)?;

    assert_tensors_close(&cpu_output2, &cuda_output2, 1e-5);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_elu_gradient() -> Result<(), Error> {
    // Define constants similar to those in test_gradient_checker_cuda.rs
    const DEFAULT_EPSILON: f32 = 1e-4;

    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create a CUDA tensor with requires_grad=true
    // Use values different from 0 for numerical stability of finite difference
    let x_data = vec![-2.0, -0.5, 0.5, 2.0];
    let shape = &[4];
    let alpha = 1.0;

    // Create tensor with requires_grad
    let cuda_x = Tensor::<CudaBackend>::from_vec(x_data, shape, true)?;

    // Define a function that applies ELU and then mean reduction
    let elu_mean_fn = |inputs: &[CudaTensor]| -> Result<CudaTensor, Error> {
        let y = ops::elu(&inputs[0], alpha)?;
        ops::mean(&y, None) // Reduce to scalar for gradient checking
    };

    // Check gradient using finite differences
    rust_tensor_lib::test_utils::check_gradient(elu_mean_fn, &[cuda_x], 0, DEFAULT_EPSILON, 1e-2)?;

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_softplus_extended() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 10.0]; // Include a larger value
    let shape = [6];
    let cpu_input = cpu_tensor(data.clone(), &shape);
    let cpu_output = ops::softplus(&cpu_input)?; // Use ops::softplus

    let cuda_input = cuda_tensor_from_cpu(&cpu_input);
    let cuda_output = ops::softplus(&cuda_input)?; // Use ops::softplus

    // Use slightly higher tolerance if necessary due to potential exp/ln differences
    assert_tensors_close(&cpu_output, &cuda_output, 1e-5);
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_softmax_cross_entropy_forward() -> Result<(), Error> {
    use crate::{assert_tensors_close, cpu_tensor, cuda_tensor_from_cpu, TOLERANCE};
    use rust_tensor_lib::{
        backend::cuda::{init_context, CudaContextGuard},
        ops, Error, Reduction,
    };

    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // --- Test Data ---
    let logits_data = vec![1.0, 2.0, 3.0, 0.5, 0.5, 0.5];
    let targets_data = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let shape = &[2, 3];
    let axis = 1;

    let cpu_logits = cpu_tensor(logits_data.clone(), shape);
    let cpu_targets = cpu_tensor(targets_data.clone(), shape);
    let cuda_logits = cuda_tensor_from_cpu(&cpu_logits);
    let cuda_targets = cuda_tensor_from_cpu(&cpu_targets);

    // --- Test Reduction::None ---
    let expected_cpu_none =
        ops::softmax_cross_entropy(&cpu_logits, &cpu_targets, axis, Reduction::None)?;
    let actual_cuda_none =
        ops::softmax_cross_entropy(&cuda_logits, &cuda_targets, axis, Reduction::None)?;
    assert_tensors_close(&expected_cpu_none, &actual_cuda_none, TOLERANCE);

    // --- Test Reduction::Mean ---
    let expected_cpu_mean =
        ops::softmax_cross_entropy(&cpu_logits, &cpu_targets, axis, Reduction::Mean)?;
    let actual_cuda_mean =
        ops::softmax_cross_entropy(&cuda_logits, &cuda_targets, axis, Reduction::Mean)?;
    assert_tensors_close(&expected_cpu_mean, &actual_cuda_mean, TOLERANCE);

    // --- Test Reduction::Sum ---
    let expected_cpu_sum =
        ops::softmax_cross_entropy(&cpu_logits, &cpu_targets, axis, Reduction::Sum)?;
    let actual_cuda_sum =
        ops::softmax_cross_entropy(&cuda_logits, &cuda_targets, axis, Reduction::Sum)?;
    assert_tensors_close(&expected_cpu_sum, &actual_cuda_sum, TOLERANCE);

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_softmax_cross_entropy_gradient() -> Result<(), Error> {
    use rust_tensor_lib::{
        backend::cuda::{init_context, CudaBackend, CudaContextGuard},
        ops,
        test_utils::check_gradient,
        CudaTensor, Error, Reduction, Tensor,
    };

    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // --- Test Data ---
    let logits_data = vec![1.0, 2.0, 3.0, 0.5, 0.5, 0.5];
    let targets_data = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let shape = &[2, 3];
    let axis = 1;

    // Create CUDA tensors directly
    let logits = Tensor::<CudaBackend>::from_vec(logits_data, shape, true)?; // requires_grad = true
    let targets = Tensor::<CudaBackend>::from_vec(targets_data, shape, false)?; // requires_grad = false

    // Define the function for check_gradient
    let loss_fn = |inputs: &[CudaTensor]| -> Result<CudaTensor, Error> {
        ops::softmax_cross_entropy(&inputs[0], &targets, axis, Reduction::Mean)
    };

    // --- Gradient Check ---
    // Use slightly larger tolerance due to CUDA float precision
    check_gradient(loss_fn, &[logits], 0, 1e-3, 2e-2)?;

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_argmax() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    println!("--- Running test_cuda_argmax ---");

    // Case 1: Simple 1D
    let cpu_in1 = cpu_tensor(vec![1.0, 5.0, 2.0, 4.0], &[4]);
    let cuda_in1 = cuda_tensor_from_cpu(&cpu_in1);
    let axis1 = 0;
    // CPU Calculation
    let expected_cpu1 = ops::argmax(&cpu_in1, axis1)?;
    // CUDA Calculation
    let actual_cuda1 = ops::argmax(&cuda_in1, axis1)?;
    println!(
        "Case 1 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu1.data())?
    );
    println!(
        "Case 1 Actual (CUDA): {:?}",
        CudaBackend::copy_to_host(&*actual_cuda1.data())?
    );
    // Indices should be exactly equal as floats
    assert_tensors_close(&expected_cpu1, &actual_cuda1, 0.0);

    // Case 2: 2D along axis 0
    let cpu_in2 = cpu_tensor(vec![1.0, 5.0, 2.0, 4.0, 0.0, 6.0], &[2, 3]);
    let cuda_in2 = cuda_tensor_from_cpu(&cpu_in2);
    let axis2 = 0;
    let expected_cpu2 = ops::argmax(&cpu_in2, axis2)?;
    let actual_cuda2 = ops::argmax(&cuda_in2, axis2)?;
    println!(
        "Case 2 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu2.data())?
    );
    println!(
        "Case 2 Actual (CUDA): {:?}",
        CudaBackend::copy_to_host(&*actual_cuda2.data())?
    );
    assert_tensors_close(&expected_cpu2, &actual_cuda2, 0.0);

    // Case 3: 2D along axis 1
    let cpu_in3 = cpu_tensor(vec![1.0, 5.0, 2.0, 4.0, 0.0, 6.0], &[2, 3]);
    let cuda_in3 = cuda_tensor_from_cpu(&cpu_in3);
    let axis3 = 1;
    let expected_cpu3 = ops::argmax(&cpu_in3, axis3)?;
    let actual_cuda3 = ops::argmax(&cuda_in3, axis3)?;
    println!(
        "Case 3 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu3.data())?
    );
    println!(
        "Case 3 Actual (CUDA): {:?}",
        CudaBackend::copy_to_host(&*actual_cuda3.data())?
    );
    assert_tensors_close(&expected_cpu3, &actual_cuda3, 0.0);

    println!("--- test_cuda_argmax PASSED ---");
    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_argmin() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    println!("--- Running test_cuda_argmin ---");

    // Case 1: Simple 1D
    let cpu_in1 = cpu_tensor(vec![5.0, 1.0, 2.0, 0.5], &[4]);
    let cuda_in1 = cuda_tensor_from_cpu(&cpu_in1);
    let axis1 = 0;
    let expected_cpu1 = ops::argmin(&cpu_in1, axis1)?;
    let actual_cuda1 = ops::argmin(&cuda_in1, axis1)?;
    println!(
        "Case 1 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu1.data())?
    );
    println!(
        "Case 1 Actual (CUDA): {:?}",
        CudaBackend::copy_to_host(&*actual_cuda1.data())?
    );
    assert_tensors_close(&expected_cpu1, &actual_cuda1, 0.0);

    // Case 2: 2D along axis 0
    let cpu_in2 = cpu_tensor(vec![1.0, 5.0, 2.0, 0.5, 1.0, 6.0], &[2, 3]);
    let cuda_in2 = cuda_tensor_from_cpu(&cpu_in2);
    let axis2 = 0;
    let expected_cpu2 = ops::argmin(&cpu_in2, axis2)?;
    let actual_cuda2 = ops::argmin(&cuda_in2, axis2)?;
    println!(
        "Case 2 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu2.data())?
    );
    println!(
        "Case 2 Actual (CUDA): {:?}",
        CudaBackend::copy_to_host(&*actual_cuda2.data())?
    );
    assert_tensors_close(&expected_cpu2, &actual_cuda2, 0.0);

    // Case 3: 2D along axis 1
    let cpu_in3 = cpu_tensor(vec![1.0, 5.0, 0.5, 4.0, 1.0, 6.0], &[2, 3]);
    let cuda_in3 = cuda_tensor_from_cpu(&cpu_in3);
    let axis3 = 1;
    let expected_cpu3 = ops::argmin(&cpu_in3, axis3)?;
    let actual_cuda3 = ops::argmin(&cuda_in3, axis3)?;
    println!(
        "Case 3 Expected (CPU): {:?}",
        CpuBackend::copy_to_host(&*expected_cpu3.data())?
    );
    println!(
        "Case 3 Actual (CUDA): {:?}",
        CudaBackend::copy_to_host(&*actual_cuda3.data())?
    );
    assert_tensors_close(&expected_cpu3, &actual_cuda3, 0.0);

    println!("--- test_cuda_argmin PASSED ---");
    Ok(())
}

#[serial]
#[test]
fn test_cuda_max_pool2d_forward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let input_data = (1..=36).map(|x| x as f32).collect::<Vec<_>>(); // 1*1*6*6
    let input_cpu = cpu_tensor(input_data, &[1, 1, 6, 6]);
    let input_cuda = cuda_tensor_from_cpu(&input_cpu);

    let kernel_size = (2,2);
    let stride = (2,2);
    let padding = (0,0);

    let (expected_val_cpu, _expected_idx_cpu) = ops::cpu_ops::max_pool2d(&*input_cpu.data(), kernel_size, stride, padding)?;
    let expected_val_cpu_tensor = Tensor::new(expected_val_cpu, false);

    let (actual_val_cuda, _actual_idx_cuda) = CudaBackend::max_pool2d(&*input_cuda.data(), kernel_size, stride, padding)?;
    let actual_val_cuda_tensor = Tensor::new(actual_val_cuda, false);
    
    assert_tensors_close(&expected_val_cpu_tensor, &actual_val_cuda_tensor, TOLERANCE);
    Ok(())
}

#[serial]
#[test]
fn test_cuda_slice_forward() -> Result<(), Error> {
    println!("--- Running test_cuda_slice_forward ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test case 1: 2D tensor slicing
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = &[2, 3];
    let cpu_tensor = Tensor::<CpuBackend>::from_vec(data.clone(), shape, false)?;
    let cuda_tensor = cuda_tensor_from_cpu(&cpu_tensor);

    // Slice [:, 1:2] -> [[2], [5]]
    let ranges = &[0..2, 1..2];
    let cpu_result = ops::slice(&cpu_tensor, ranges)?;
    let cuda_result = ops::slice(&cuda_tensor, ranges)?;

    // Verify shapes and values
    assert_eq!(cpu_result.shape(), &[2, 1]);
    assert_eq!(cuda_result.shape(), &[2, 1]);
    assert_tensors_close(&cpu_result, &cuda_result, TOLERANCE);

    // Test case 2: 1D tensor slicing
    let data_1d = vec![1.0, 2.0, 3.0, 4.0];
    let shape_1d = &[4];
    let cpu_tensor_1d = Tensor::<CpuBackend>::from_vec(data_1d.clone(), shape_1d, false)?;
    let cuda_tensor_1d = cuda_tensor_from_cpu(&cpu_tensor_1d);

    // Slice [1:3] -> [2.0, 3.0]
    let ranges_1d = &[1..3];
    let cpu_result_1d = ops::slice(&cpu_tensor_1d, ranges_1d)?;
    let cuda_result_1d = ops::slice(&cuda_tensor_1d, ranges_1d)?;

    // Verify shapes and values
    assert_eq!(cpu_result_1d.shape(), &[2]);
    assert_eq!(cuda_result_1d.shape(), &[2]);
    assert_tensors_close(&cpu_result_1d, &cuda_result_1d, TOLERANCE);

    // Test case 3: 3D tensor slicing
    let data_3d: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let shape_3d = &[2, 3, 4];
    let cpu_tensor_3d = Tensor::<CpuBackend>::from_vec(data_3d.clone(), shape_3d, false)?;
    let cuda_tensor_3d = cuda_tensor_from_cpu(&cpu_tensor_3d);

    // Slice [0:1, 1:3, 2:4] -> [[[7, 8], [11, 12]]]
    let ranges_3d = &[0..1, 1..3, 2..4];
    let cpu_result_3d = ops::slice(&cpu_tensor_3d, ranges_3d)?;
    let cuda_result_3d = ops::slice(&cuda_tensor_3d, ranges_3d)?;

    // Verify shapes and values
    assert_eq!(cpu_result_3d.shape(), &[1, 2, 2]);
    assert_eq!(cuda_result_3d.shape(), &[1, 2, 2]);
    assert_tensors_close(&cpu_result_3d, &cuda_result_3d, TOLERANCE);

    // Test case 4: Empty slice result
    let ranges_empty = &[0..0, 0..3];
    let cpu_result_empty = ops::slice(&cpu_tensor, ranges_empty)?;
    let cuda_result_empty = ops::slice(&cuda_tensor, ranges_empty)?;

    // Verify shapes and values
    assert_eq!(cpu_result_empty.shape(), &[0, 3]);
    assert_eq!(cuda_result_empty.shape(), &[0, 3]);
    // Note: No need to check values for empty tensors

    println!("--- test_cuda_slice_forward PASSED ---");
    Ok(())
}

#[serial]
#[test]
fn test_cuda_slice_backward() -> Result<(), Error> {
    println!("--- Running test_cuda_slice_backward ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test case 1: 2D tensor slicing backward pass
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = &[2, 2];
    
    // Create tensors with requires_grad=true
    let cpu_tensor = Tensor::<CpuBackend>::from_vec(data.clone(), shape, true)?;
    let cuda_tensor = Tensor::<CudaBackend>::from_vec(data.clone(), shape, true)?;

    // Slice [0:1, :] -> [[1, 2]]
    let ranges = &[0..1, 0..2];
    let cpu_result = ops::slice(&cpu_tensor, ranges)?;
    let cuda_result = ops::slice(&cuda_tensor, ranges)?;

    // Create gradient for backward pass
    let grad_data = vec![10.0, 20.0];
    let grad_shape = &[1, 2];
    let cpu_grad = Tensor::<CpuBackend>::from_vec(grad_data.clone(), grad_shape, false)?;
    let cuda_grad = Tensor::<CudaBackend>::from_vec(grad_data.clone(), grad_shape, false)?;

    // Set the gradient for the output tensors
    cpu_result.set_grad(Some(cpu_grad.data().clone()));
    cuda_result.set_grad(Some(cuda_grad.data().clone()));
    
    // Perform backward pass
    cpu_result.backward()?;
    cuda_result.backward()?;

    // Get and compare gradients
    let cpu_input_grad = cpu_tensor.grad().unwrap();
    let cuda_input_grad = cuda_tensor.grad().unwrap();
    
    // Create tensors from the gradients for comparison
    let cpu_grad_tensor = Tensor::<CpuBackend>::new(cpu_input_grad.clone(), false);
    let cuda_grad_tensor = Tensor::<CudaBackend>::new(cuda_input_grad.clone(), false);
    
    // Expected gradient: [[10, 20], [0, 0]]
    assert_tensors_close(&cpu_grad_tensor, &cuda_grad_tensor, TOLERANCE);

    // Test case 2: 3D tensor slicing backward pass
    let data_3d: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let shape_3d = &[2, 2, 2];
    
    // Create tensors with requires_grad=true
    let cpu_tensor_3d = Tensor::<CpuBackend>::from_vec(data_3d.clone(), shape_3d, true)?;
    let cuda_tensor_3d = Tensor::<CudaBackend>::from_vec(data_3d.clone(), shape_3d, true)?;

    // Slice [0:1, :, 0:1] -> [[[1], [3]]]
    let ranges_3d = &[0..1, 0..2, 0..1];
    let cpu_result_3d = ops::slice(&cpu_tensor_3d, ranges_3d)?;
    let cuda_result_3d = ops::slice(&cuda_tensor_3d, ranges_3d)?;

    // Create gradient for backward pass
    let grad_data_3d = vec![30.0, 40.0];
    let grad_shape_3d = &[1, 2, 1];
    let cpu_grad_3d = Tensor::<CpuBackend>::from_vec(grad_data_3d.clone(), grad_shape_3d, false)?;
    let cuda_grad_3d = Tensor::<CudaBackend>::from_vec(grad_data_3d.clone(), grad_shape_3d, false)?;

    // Set the gradient for the output tensors
    cpu_result_3d.set_grad(Some(cpu_grad_3d.data().clone()));
    cuda_result_3d.set_grad(Some(cuda_grad_3d.data().clone()));
    
    // Perform backward pass
    cpu_result_3d.backward()?;
    cuda_result_3d.backward()?;

    // Get and compare gradients
    let cpu_input_grad_3d = cpu_tensor_3d.grad().unwrap();
    let cuda_input_grad_3d = cuda_tensor_3d.grad().unwrap();
    
    // Create tensors from the gradients for comparison
    let cpu_grad_tensor_3d = Tensor::<CpuBackend>::new(cpu_input_grad_3d.clone(), false);
    let cuda_grad_tensor_3d = Tensor::<CudaBackend>::new(cuda_input_grad_3d.clone(), false);
    
    assert_tensors_close(&cpu_grad_tensor_3d, &cuda_grad_tensor_3d, TOLERANCE);

    println!("--- test_cuda_slice_backward PASSED ---");
    Ok(())
}

#[serial]
#[test]
fn test_cuda_clip_forward() -> Result<(), Error> {
    println!("--- Running test_cuda_clip_forward ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test case 1: Basic clipping
    let data = vec![-1.0, 0.5, 2.0, 3.0];
    let shape = &[2, 2];
    
    // Create CPU and CUDA tensors
    let cpu_tensor = Tensor::<CpuBackend>::from_vec(data.clone(), shape, false)?;
    let cuda_tensor = Tensor::<CudaBackend>::from_vec(data.clone(), shape, false)?;

    // Clip values to [0.0, 2.0]
    let min_val = 0.0;
    let max_val = 2.0;
    let cpu_result = ops::clip(&cpu_tensor, min_val, max_val)?;
    let cuda_result = ops::clip(&cuda_tensor, min_val, max_val)?;

    // Verify shapes
    assert_eq!(cpu_result.shape(), shape);
    assert_eq!(cuda_result.shape(), shape);
    
    // Verify values
    assert_tensors_close(&cpu_result, &cuda_result, TOLERANCE);
    
    // Expected result: [0.0, 0.5, 2.0, 2.0]
    let expected_data = vec![0.0, 0.5, 2.0, 2.0];
    let expected_tensor = Tensor::<CpuBackend>::from_vec(expected_data, shape, false)?;
    
    // Compare CPU tensors directly
    let cpu_data_storage = cpu_result.data();
    let expected_data_storage = expected_tensor.data();
    let cpu_data = cpu_data_storage.get_data();
    let expected_data = expected_data_storage.get_data();
    assert_eq!(cpu_data.shape(), expected_data.shape());
    
    for (i, (a, b)) in cpu_data.iter().zip(expected_data.iter()).enumerate() {
        assert!((a - b).abs() < TOLERANCE, "Values at index {} differ: {} vs {}", i, a, b);
    }

    // Test case 2: Different min/max values
    let min_val2 = -0.5;
    let max_val2 = 1.0;
    let cpu_result2 = ops::clip(&cpu_tensor, min_val2, max_val2)?;
    let cuda_result2 = ops::clip(&cuda_tensor, min_val2, max_val2)?;

    // Verify shapes
    assert_eq!(cpu_result2.shape(), shape);
    assert_eq!(cuda_result2.shape(), shape);
    
    // Verify values
    assert_tensors_close(&cpu_result2, &cuda_result2, TOLERANCE);
    
    // Expected result: [-0.5, 0.5, 1.0, 1.0]
    let expected_data2 = vec![-0.5, 0.5, 1.0, 1.0];
    let expected_tensor2 = Tensor::<CpuBackend>::from_vec(expected_data2, shape, false)?;
    
    // Compare CPU tensors directly
    let cpu_data2_storage = cpu_result2.data();
    let expected_data2_storage = expected_tensor2.data();
    let cpu_data2 = cpu_data2_storage.get_data();
    let expected_data2 = expected_data2_storage.get_data();
    assert_eq!(cpu_data2.shape(), expected_data2.shape());
    
    for (i, (a, b)) in cpu_data2.iter().zip(expected_data2.iter()).enumerate() {
        assert!((a - b).abs() < TOLERANCE, "Values at index {} differ: {} vs {}", i, a, b);
    }

    // Test case 3: Empty tensor
    let empty_data: Vec<f32> = vec![];
    let empty_shape = &[0, 2];
    let cpu_empty = Tensor::<CpuBackend>::from_vec(empty_data.clone(), empty_shape, false)?;
    let cuda_empty = Tensor::<CudaBackend>::from_vec(empty_data.clone(), empty_shape, false)?;

    let cpu_empty_result = ops::clip(&cpu_empty, 0.0, 1.0)?;
    let cuda_empty_result = ops::clip(&cuda_empty, 0.0, 1.0)?;

    // Verify shapes for empty tensors
    assert_eq!(cpu_empty_result.shape(), empty_shape);
    assert_eq!(cuda_empty_result.shape(), empty_shape);

    println!("--- test_cuda_clip_forward PASSED ---");
    Ok(())
}

#[serial]
#[test]
fn test_cuda_clip_backward() -> Result<(), Error> {
    println!("--- Running test_cuda_clip_backward ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test case: Backward pass for clip operation
    let data = vec![-1.0, 0.5, 2.0, 3.0];
    let shape = &[2, 2];
    
    // Create tensors with requires_grad=true
    let cpu_tensor = Tensor::<CpuBackend>::from_vec(data.clone(), shape, true)?;
    let cuda_tensor = Tensor::<CudaBackend>::from_vec(data.clone(), shape, true)?;

    // Clip values to [0.0, 2.0]
    let min_val = 0.0;
    let max_val = 2.0;
    let cpu_result = ops::clip(&cpu_tensor, min_val, max_val)?;
    let cuda_result = ops::clip(&cuda_tensor, min_val, max_val)?;

    // Create gradient for backward pass (all ones)
    let grad_data = vec![1.0, 1.0, 1.0, 1.0];
    let cpu_grad = Tensor::<CpuBackend>::from_vec(grad_data.clone(), shape, false)?;
    let cuda_grad = Tensor::<CudaBackend>::from_vec(grad_data.clone(), shape, false)?;

    // Set the gradient for the output tensors
    cpu_result.set_grad(Some(cpu_grad.data().clone()));
    cuda_result.set_grad(Some(cuda_grad.data().clone()));
    
    // Perform backward pass
    cpu_result.backward()?;
    cuda_result.backward()?;

    // Get and compare gradients
    let cpu_input_grad = cpu_tensor.grad().unwrap();
    let cuda_input_grad = cuda_tensor.grad().unwrap();
    
    // Create tensors from the gradients for comparison
    let cpu_grad_tensor = Tensor::<CpuBackend>::new(cpu_input_grad.clone(), false);
    let cuda_grad_tensor = Tensor::<CudaBackend>::new(cuda_input_grad.clone(), false);
    
    // Expected gradient: [0.0, 1.0, 1.0, 0.0]
    // Only values within the clip range [0.0, 2.0] should have non-zero gradients
    assert_tensors_close(&cpu_grad_tensor, &cuda_grad_tensor, TOLERANCE);
    
    // Verify against expected values
    let expected_grad_data = vec![0.0, 1.0, 1.0, 0.0];
    let expected_grad_tensor = Tensor::<CpuBackend>::from_vec(expected_grad_data, shape, false)?;
    
    // Compare CPU tensors directly
    let cpu_grad_data_storage = cpu_grad_tensor.data();
    let expected_grad_data_storage = expected_grad_tensor.data();
    let cpu_grad_data = cpu_grad_data_storage.get_data();
    let expected_grad_data = expected_grad_data_storage.get_data();
    assert_eq!(cpu_grad_data.shape(), expected_grad_data.shape());
    
    for (i, (a, b)) in cpu_grad_data.iter().zip(expected_grad_data.iter()).enumerate() {
        assert!((a - b).abs() < TOLERANCE, "Gradient values at index {} differ: {} vs {}", i, a, b);
    }

    println!("--- test_cuda_clip_backward PASSED ---");
    Ok(())
}
