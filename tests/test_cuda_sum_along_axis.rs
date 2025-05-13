#![cfg(feature = "cuda")]
#![cfg(feature = "cuda")]
use rust_tensor_lib::backend::cuda::{init_context, CudaContextGuard};
use rust_tensor_lib::{Backend, CpuBackend, CudaBackend, Error, Tensor};
use serial_test::serial;

// Constant for comparing floating point values
const TOLERANCE: f32 = 1e-4;

// Helper functions
fn cpu_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<CpuBackend> {
    Tensor::from_vec(data, shape, false).unwrap()
}

fn cuda_tensor_from_cpu(cpu_tensor: &Tensor<CpuBackend>) -> Tensor<CudaBackend> {
    let data = CpuBackend::copy_to_host(&*cpu_tensor.data()).unwrap();
    let shape = cpu_tensor.shape();
    Tensor::from_vec(data, &shape, false).unwrap()
}

// Helper function to compare tensors
fn assert_tensors_close(
    actual: &Tensor<CudaBackend>,
    expected: &Tensor<CudaBackend>,
    tolerance: f32,
) -> Result<(), Error> {
    let actual_cpu = actual.to_cpu()?;
    let expected_cpu = expected.to_cpu()?;
    assert_eq!(
        actual_cpu.shape(),
        expected_cpu.shape(),
        "Tensor shapes don't match"
    );
    let actual_data = actual_cpu.data().as_ref().to_vec();
    let expected_data = expected_cpu.data().as_ref().to_vec();
    for (i, (a, e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (a - e).abs() < tolerance,
            "Values at index {i} differ: actual={a}, expected={e} (tolerance={tolerance})"
        );
    }
    Ok(())
}

#[test]
#[serial]
fn test_cuda_sum_along_axis_2d_to_1d_ax0() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create a 2x3 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = &[2, 3];
    let cpu_in = cpu_tensor(data, shape);
    let cuda_in = cuda_tensor_from_cpu(&cpu_in);

    // Sum along axis 0 (rows)
    // Expected CPU result
    let expected_cpu_storage = CpuBackend::sum_along_axis(&*cpu_in.data(), 0)?;
    let expected_cpu_tensor = Tensor::<CpuBackend>::new(expected_cpu_storage, false);
    // Actual CUDA result
    let actual_cuda_storage = CudaBackend::sum_along_axis(&*cuda_in.data(), 0)?;
    let actual_cuda_tensor = Tensor::<CudaBackend>::new(actual_cuda_storage, false);

    let expected_cuda_tensor = cuda_tensor_from_cpu(&expected_cpu_tensor);
    assert_tensors_close(&actual_cuda_tensor, &expected_cuda_tensor, TOLERANCE)?;

    Ok(())
}

#[test]
#[serial]
fn test_cuda_sum_along_axis_2d_to_1d_ax1() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create a 2x3 matrix
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = &[2, 3];
    let cpu_in = cpu_tensor(data, shape);
    let cuda_in = cuda_tensor_from_cpu(&cpu_in);

    // Sum along axis 1 (columns)
    // Expected CPU result
    let expected_cpu_storage = CpuBackend::sum_along_axis(&*cpu_in.data(), 1)?;
    let expected_cpu_tensor = Tensor::<CpuBackend>::new(expected_cpu_storage, false);
    // Actual CUDA result
    let actual_cuda_storage = CudaBackend::sum_along_axis(&*cuda_in.data(), 1)?;
    let actual_cuda_tensor = Tensor::<CudaBackend>::new(actual_cuda_storage, false);

    let expected_cuda_tensor = cuda_tensor_from_cpu(&expected_cpu_tensor);
    assert_tensors_close(&actual_cuda_tensor, &expected_cuda_tensor, TOLERANCE)?;

    Ok(())
}

#[test]
#[serial]
fn test_cuda_sum_along_axis_1d_to_scalar() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create a 1D vector
    let data = vec![1.0, 2.0, 3.0];
    let shape = &[3];
    let cpu_in = cpu_tensor(data, shape);
    let cuda_in = cuda_tensor_from_cpu(&cpu_in);

    // Sum along axis 0 (entire vector)
    // Expected CPU result
    let expected_cpu_storage = CpuBackend::sum_along_axis(&*cpu_in.data(), 0)?;
    let expected_cpu_tensor = Tensor::<CpuBackend>::new(expected_cpu_storage, false);
    // Actual CUDA result
    let actual_cuda_storage = CudaBackend::sum_along_axis(&*cuda_in.data(), 0)?;
    let actual_cuda_tensor = Tensor::<CudaBackend>::new(actual_cuda_storage, false);

    let expected_cuda_tensor = cuda_tensor_from_cpu(&expected_cpu_tensor);
    assert_tensors_close(&actual_cuda_tensor, &expected_cuda_tensor, TOLERANCE)?;

    Ok(())
}

#[test]
#[serial]
fn test_cuda_sum_along_axis_3d() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create a 2x3x2 tensor
    let data = (1..=12).map(|x| x as f32).collect();
    let shape = &[2, 3, 2];
    let cpu_in = cpu_tensor(data, shape);
    let cuda_in = cuda_tensor_from_cpu(&cpu_in);

    // Test summing along each axis
    for axis in 0..3 {
        let expected_cpu_storage = CpuBackend::sum_along_axis(&*cpu_in.data(), axis)?;
        let expected_cpu_tensor = Tensor::<CpuBackend>::new(expected_cpu_storage, false);
        let actual_cuda_storage = CudaBackend::sum_along_axis(&*cuda_in.data(), axis)?;
        let actual_cuda_tensor = Tensor::<CudaBackend>::new(actual_cuda_storage, false);

        let expected_cuda_tensor = cuda_tensor_from_cpu(&expected_cpu_tensor);
        assert_tensors_close(&actual_cuda_tensor, &expected_cuda_tensor, TOLERANCE)?;
    }

    Ok(())
}

#[test]
#[serial]
fn test_cuda_sum_along_axis_empty_tensor() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create an empty tensor with shape [0, 3]
    let data = vec![];
    let shape = &[0, 3];
    let cpu_in = cpu_tensor(data, shape);
    let cuda_in = cuda_tensor_from_cpu(&cpu_in);

    // Sum along axis 0
    // Expected CPU result
    let expected_cpu_storage = CpuBackend::sum_along_axis(&*cpu_in.data(), 0)?;
    let expected_cpu_tensor = Tensor::<CpuBackend>::new(expected_cpu_storage, false);
    // Actual CUDA result
    let actual_cuda_storage = CudaBackend::sum_along_axis(&*cuda_in.data(), 0)?;
    let actual_cuda_tensor = Tensor::<CudaBackend>::new(actual_cuda_storage, false);

    let expected_cuda_tensor = cuda_tensor_from_cpu(&expected_cpu_tensor);
    assert_tensors_close(&actual_cuda_tensor, &expected_cuda_tensor, TOLERANCE)?;

    Ok(())
}

#[test]
#[serial]
fn test_cuda_sum_along_axis_single_element() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create tensors with dimension size 1
    let data = vec![42.0];

    // Test 1D tensor [1]
    let shape_1d = &[1];
    let cpu_in_1d = cpu_tensor(data.clone(), shape_1d);
    let cuda_in_1d = cuda_tensor_from_cpu(&cpu_in_1d);

    let expected_cpu_storage_1d = CpuBackend::sum_along_axis(&*cpu_in_1d.data(), 0)?;
    let expected_cpu_tensor_1d = Tensor::<CpuBackend>::new(expected_cpu_storage_1d, false);

    let actual_cuda_storage_1d = CudaBackend::sum_along_axis(&*cuda_in_1d.data(), 0)?;
    let actual_cuda_tensor_1d = Tensor::<CudaBackend>::new(actual_cuda_storage_1d, false);

    let expected_cuda_tensor_1d = cuda_tensor_from_cpu(&expected_cpu_tensor_1d);
    assert_tensors_close(&actual_cuda_tensor_1d, &expected_cuda_tensor_1d, TOLERANCE)?;

    // Test 2D tensor [1, 1]
    let shape_2d = &[1, 1];
    let cpu_in_2d = cpu_tensor(data.clone(), shape_2d);
    let cuda_in_2d = cuda_tensor_from_cpu(&cpu_in_2d);

    for axis in 0..2 {
        let expected_cpu_storage_2d = CpuBackend::sum_along_axis(&*cpu_in_2d.data(), axis)?;
        let expected_cpu_tensor_2d = Tensor::<CpuBackend>::new(expected_cpu_storage_2d, false);
        let actual_cuda_storage_2d = CudaBackend::sum_along_axis(&*cuda_in_2d.data(), axis)?;
        let actual_cuda_tensor_2d = Tensor::<CudaBackend>::new(actual_cuda_storage_2d, false);

        let expected_cuda_tensor_2d = cuda_tensor_from_cpu(&expected_cpu_tensor_2d);
        assert_tensors_close(&actual_cuda_tensor_2d, &expected_cuda_tensor_2d, TOLERANCE)?;
    }

    Ok(())
}

#[test]
#[serial]
fn test_cuda_sum_along_axis_invalid_axis() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = &[2, 2];
    let cuda_in = Tensor::<CudaBackend>::from_vec(data, shape, false)?;

    // Test invalid axis (axis >= ndim)
    let result = CudaBackend::sum_along_axis(&*cuda_in.data(), 2);
    assert!(
        matches!(result, Err(Error::InvalidOperation(_))),
        "Expected InvalidOperation error for axis out of bounds"
    );

    // Test very large axis
    let result = CudaBackend::sum_along_axis(&*cuda_in.data(), usize::MAX);
    assert!(
        matches!(result, Err(Error::InvalidOperation(_))),
        "Expected InvalidOperation error for very large axis"
    );

    Ok(())
}
