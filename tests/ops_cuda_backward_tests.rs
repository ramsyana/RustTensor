#![cfg(feature = "cuda")]

use rust_tensor_lib::{
    backend::cuda::{init_context, CudaBackend, CudaContextGuard, CudaStorage},
    backend::Backend,
    graph::Op,
    graph::OpType,
    ops, CpuBackend, Error, Tensor,
};
use serial_test::serial;
use std::rc::Rc;

// Helper function to create a CPU tensor
fn cpu_tensor(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<CpuBackend>, Error> {
    Tensor::<CpuBackend>::from_vec(data, shape, false)
        .map_err(|e| Error::InternalLogicError(format!("Failed to create CPU tensor: {}", e)))
}

// Helper function to create a CUDA tensor
fn cuda_tensor(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<CudaBackend>, Error> {
    Tensor::<CudaBackend>::from_vec(data, shape, false)
}

// Helper function to create a CUDA tensor that requires gradients
fn cuda_tensor_req_grad(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<CudaBackend>, Error> {
    Tensor::<CudaBackend>::from_vec(data, shape, true)
}

// Helper function to convert CPU tensor to CUDA tensor
fn cuda_tensor_from_cpu(cpu_tensor: &Tensor<CpuBackend>) -> Result<Tensor<CudaBackend>, Error> {
    let shape = cpu_tensor.shape();
    let data = CpuBackend::copy_to_host(&*cpu_tensor.data()).map_err(|e| {
        Error::InternalLogicError(format!("Failed to copy CPU data to host vec: {}", e))
    })?;
    Tensor::<CudaBackend>::from_vec(data, &shape, cpu_tensor.requires_grad()).map_err(|e| {
        Error::InternalLogicError(format!("Failed to create CUDA tensor from vec: {}", e))
    })
}

// Constant for comparing floating point values
const TOLERANCE: f32 = 1e-4;

// Helper function to compare tensors
fn assert_tensors_close(
    actual: &Tensor<CudaBackend>,
    expected: &Tensor<CudaBackend>,
    tolerance: f32,
) -> Result<(), Error> {
    let actual_data = actual.to_cpu()?.data().as_ref().to_vec();
    let expected_data = expected.to_cpu()?.data().as_ref().to_vec();

    assert_eq!(
        actual.shape(),
        expected.shape(),
        "Tensor shapes don't match"
    );
    assert_eq!(
        actual_data.len(),
        expected_data.len(),
        "Tensor lengths don't match"
    );

    for (i, (x, y)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (x - y).abs() < tolerance,
            "Values at index {i} differ: actual={x}, expected={y} (tolerance={tolerance})"
        );
    }
    Ok(())
}

#[serial]
#[test]
fn test_cuda_max_backward_axis() -> Result<(), Error> {
    println!("--- Running test_cuda_max_backward_axis ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    println!("CUDA Context Initialized.");

    // Data with ties: max in row 0 is 5 (indices 1, 3), max in row 1 is 6 (index 2)
    let input_data = vec![1.0, 5.0, 2.0, 5.0, 3.0, 6.0];
    let input_shape = &[2, 3];
    let axis_to_test = Some(1); // Reduce along axis 1 (rows)

    println!("Creating CUDA input tensor (requires_grad=true)...");
    let input_cuda = Tensor::<CudaBackend>::from_vec(input_data.clone(), input_shape, true)?;
    println!(
        "Input CUDA Tensor (ID {}): shape={:?}",
        input_cuda.id(),
        input_cuda.shape()
    );

    println!("Performing forward CUDA operation (max)...");
    let forward_output_cuda = ops::max(&input_cuda, axis_to_test)?;
    println!(
        "Forward Output CUDA Tensor (ID {}): shape={:?}",
        forward_output_cuda.id(),
        forward_output_cuda.shape()
    );

    println!("Calculating scalar loss (mean)...");
    let loss_cuda = ops::mean(&forward_output_cuda, None)?; // Reduce to scalar: mean(5, 6) = 5.5
    println!(
        "Loss CUDA Tensor (ID {}): shape={:?}",
        loss_cuda.id(),
        loss_cuda.shape()
    );

    println!("Performing backward pass on CUDA...");
    loss_cuda.backward()?;
    println!("CUDA backward pass complete.");

    println!("Retrieving actual CUDA gradient...");
    let grad_cuda_storage = input_cuda.grad().ok_or(Error::NoGradientError)?.clone();
    let actual_grad_cuda = Tensor::<CudaBackend>::new(grad_cuda_storage, false);
    println!(
        "Actual CUDA Gradient Tensor (ID {}): shape={:?}",
        actual_grad_cuda.id(),
        actual_grad_cuda.shape()
    );

    // Calculate Expected Gradient using CPU Backend
    println!("Calculating expected gradient using CPU backend...");
    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cpu = ops::max(&input_cpu, axis_to_test)?;
    let loss_cpu = ops::mean(&forward_output_cpu, None)?;
    loss_cpu.backward()?;
    let grad_cpu_storage_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec = grad_cpu_storage_ref.as_ref().to_vec();
    let expected_grad_cuda =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec.clone(), input_shape, false)?;
    println!(
        "Expected CUDA Gradient Tensor (ID {}): shape={:?}",
        expected_grad_cuda.id(),
        expected_grad_cuda.shape()
    );
    println!("Expected Grad Data (from CPU): {:?}", grad_cpu_vec);

    println!("Comparing actual CUDA gradient with expected gradient...");
    assert_tensors_close(&expected_grad_cuda, &actual_grad_cuda, TOLERANCE)?;
    println!("Gradients match.");

    Ok(())
}

#[serial]
#[test]
fn test_cuda_max_backward_global() -> Result<(), Error> {
    println!("--- Running test_cuda_max_backward_global ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Example data with global max of 5.0 at indices 1 and 3
    let input_data = vec![1.0, 5.0, 2.0, 5.0, 3.0, 0.0];
    let input_shape = &[2, 3];
    let axis_to_test = None; // Global reduction

    println!("Creating CUDA input tensor (requires_grad=true)...");
    let input_cuda = Tensor::<CudaBackend>::from_vec(input_data.clone(), input_shape, true)?;

    println!("Performing forward CUDA operation (max)...");
    let forward_output_cuda = ops::max(&input_cuda, axis_to_test)?;
    let loss_cuda = ops::mean(&forward_output_cuda, None)?;

    println!("Performing backward pass on CUDA...");
    loss_cuda.backward()?;

    let grad_cuda_storage = input_cuda.grad().ok_or(Error::NoGradientError)?.clone();
    let actual_grad_cuda = Tensor::<CudaBackend>::new(grad_cuda_storage, false);

    // Calculate Expected Gradient using CPU Backend
    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cpu = ops::max(&input_cpu, axis_to_test)?;
    let loss_cpu = ops::mean(&forward_output_cpu, None)?;
    loss_cpu.backward()?;
    let grad_cpu_storage_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec = grad_cpu_storage_ref.as_ref().to_vec();
    let expected_grad_cuda =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec.clone(), input_shape, false)?;

    assert_tensors_close(&expected_grad_cuda, &actual_grad_cuda, TOLERANCE)?;

    Ok(())
}

#[serial]
#[test]
fn test_cuda_min_backward_axis() -> Result<(), Error> {
    println!("--- Running test_cuda_min_backward_axis ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Data with ties: min in row 0 is 1.0 (index 0), min in row 1 is 1.0 (index 1)
    let input_data = vec![1.0, 5.0, 2.0, 3.0, 1.0, 6.0];
    let input_shape = &[2, 3];
    let axis_to_test = Some(1); // Reduce along axis 1 (rows)

    let input_cuda = Tensor::<CudaBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cuda = ops::min(&input_cuda, axis_to_test)?;
    let loss_cuda = ops::mean(&forward_output_cuda, None)?;
    loss_cuda.backward()?;

    let grad_cuda_storage = input_cuda.grad().ok_or(Error::NoGradientError)?.clone();
    let actual_grad_cuda = Tensor::<CudaBackend>::new(grad_cuda_storage, false);

    // Calculate Expected Gradient using CPU Backend
    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cpu = ops::min(&input_cpu, axis_to_test)?;
    let loss_cpu = ops::mean(&forward_output_cpu, None)?;
    loss_cpu.backward()?;
    let grad_cpu_storage_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec = grad_cpu_storage_ref.as_ref().to_vec();
    let expected_grad_cuda =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec.clone(), input_shape, false)?;

    assert_tensors_close(&expected_grad_cuda, &actual_grad_cuda, TOLERANCE)?;

    Ok(())
}

#[serial]
#[test]
fn test_cuda_min_backward_global() -> Result<(), Error> {
    println!("--- Running test_cuda_min_backward_global ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Example data with global min of 1.0 at indices 0 and 4
    let input_data = vec![1.0, 5.0, 2.0, 3.0, 1.0, 6.0];
    let input_shape = &[2, 3];
    let axis_to_test = None; // Global reduction

    let input_cuda = Tensor::<CudaBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cuda = ops::min(&input_cuda, axis_to_test)?;
    let loss_cuda = ops::mean(&forward_output_cuda, None)?;
    loss_cuda.backward()?;

    let grad_cuda_storage = input_cuda.grad().ok_or(Error::NoGradientError)?.clone();
    let actual_grad_cuda = Tensor::<CudaBackend>::new(grad_cuda_storage, false);

    // Calculate Expected Gradient using CPU Backend
    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cpu = ops::min(&input_cpu, axis_to_test)?;
    let loss_cpu = ops::mean(&forward_output_cpu, None)?;
    loss_cpu.backward()?;
    let grad_cpu_storage_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec = grad_cpu_storage_ref.as_ref().to_vec();
    let expected_grad_cuda =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec.clone(), input_shape, false)?;

    assert_tensors_close(&expected_grad_cuda, &actual_grad_cuda, TOLERANCE)?;

    Ok(())
}

#[serial]
#[test]
fn test_cuda_prod_backward_axis() -> Result<(), Error> {
    println!("--- Running test_cuda_prod_backward_axis ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test case without zeros
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0];
    let input_shape = &[2, 3];
    let axis_to_test = Some(1); // Reduce along axis 1 (rows)

    let input_cuda = Tensor::<CudaBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cuda = ops::prod(&input_cuda, axis_to_test)?;
    let loss_cuda = ops::mean(&forward_output_cuda, None)?;
    loss_cuda.backward()?;

    let grad_cuda_storage = input_cuda.grad().ok_or(Error::NoGradientError)?.clone();
    let actual_grad_cuda = Tensor::<CudaBackend>::new(grad_cuda_storage, false);

    // Calculate Expected Gradient using CPU Backend
    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cpu = ops::prod(&input_cpu, axis_to_test)?;
    let loss_cpu = ops::mean(&forward_output_cpu, None)?;
    loss_cpu.backward()?;
    let grad_cpu_storage_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec = grad_cpu_storage_ref.as_ref().to_vec();
    let expected_grad_cuda =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec.clone(), input_shape, false)?;

    assert_tensors_close(&expected_grad_cuda, &actual_grad_cuda, TOLERANCE)?;

    // Test case with zeros
    let input_data_zeros = vec![1.0, 0.0, 3.0, 4.0, 1.0, 2.0];
    let input_cuda_zeros =
        Tensor::<CudaBackend>::from_vec(input_data_zeros.clone(), input_shape, true)?;
    let forward_output_cuda_zeros = ops::prod(&input_cuda_zeros, axis_to_test)?;
    let loss_cuda_zeros = ops::mean(&forward_output_cuda_zeros, None)?;
    loss_cuda_zeros.backward()?;

    let grad_cuda_storage_zeros = input_cuda_zeros
        .grad()
        .ok_or(Error::NoGradientError)?
        .clone();
    let actual_grad_cuda_zeros = Tensor::<CudaBackend>::new(grad_cuda_storage_zeros, false);

    // Calculate Expected Gradient using CPU Backend for zeros case
    let input_cpu_zeros =
        Tensor::<CpuBackend>::from_vec(input_data_zeros.clone(), input_shape, true)?;
    let forward_output_cpu_zeros = ops::prod(&input_cpu_zeros, axis_to_test)?;
    let loss_cpu_zeros = ops::mean(&forward_output_cpu_zeros, None)?;
    loss_cpu_zeros.backward()?;
    let grad_cpu_storage_ref_zeros = input_cpu_zeros.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec_zeros = grad_cpu_storage_ref_zeros.as_ref().to_vec();
    let expected_grad_cuda_zeros =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec_zeros.clone(), input_shape, false)?;

    assert_tensors_close(
        &expected_grad_cuda_zeros,
        &actual_grad_cuda_zeros,
        TOLERANCE,
    )?;

    Ok(())
}

#[serial]
#[test]
fn test_cuda_prod_backward_global() -> Result<(), Error> {
    println!("--- Running test_cuda_prod_backward_global ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test case without zeros
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0];
    let input_shape = &[2, 3];
    let axis_to_test = None; // Global reduction

    let input_cuda = Tensor::<CudaBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cuda = ops::prod(&input_cuda, axis_to_test)?;
    let loss_cuda = ops::mean(&forward_output_cuda, None)?;
    loss_cuda.backward()?;

    let grad_cuda_storage = input_cuda.grad().ok_or(Error::NoGradientError)?.clone();
    let actual_grad_cuda = Tensor::<CudaBackend>::new(grad_cuda_storage, false);

    // Calculate Expected Gradient using CPU Backend
    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cpu = ops::prod(&input_cpu, axis_to_test)?;
    let loss_cpu = ops::mean(&forward_output_cpu, None)?;
    loss_cpu.backward()?;
    let grad_cpu_storage_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec = grad_cpu_storage_ref.as_ref().to_vec();
    let expected_grad_cuda =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec.clone(), input_shape, false)?;

    assert_tensors_close(&expected_grad_cuda, &actual_grad_cuda, TOLERANCE)?;

    // Test case with zeros
    let input_data_zeros = vec![1.0, 0.0, 3.0, 4.0, 1.0, 2.0];
    let input_cuda_zeros =
        Tensor::<CudaBackend>::from_vec(input_data_zeros.clone(), input_shape, true)?;
    let forward_output_cuda_zeros = ops::prod(&input_cuda_zeros, axis_to_test)?;
    let loss_cuda_zeros = ops::mean(&forward_output_cuda_zeros, None)?;
    loss_cuda_zeros.backward()?;

    let grad_cuda_storage_zeros = input_cuda_zeros
        .grad()
        .ok_or(Error::NoGradientError)?
        .clone();
    let actual_grad_cuda_zeros = Tensor::<CudaBackend>::new(grad_cuda_storage_zeros, false);

    // Calculate Expected Gradient using CPU Backend for zeros case
    let input_cpu_zeros =
        Tensor::<CpuBackend>::from_vec(input_data_zeros.clone(), input_shape, true)?;
    let forward_output_cpu_zeros = ops::prod(&input_cpu_zeros, axis_to_test)?;
    let loss_cpu_zeros = ops::mean(&forward_output_cpu_zeros, None)?;
    loss_cpu_zeros.backward()?;
    let grad_cpu_storage_ref_zeros = input_cpu_zeros.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec_zeros = grad_cpu_storage_ref_zeros.as_ref().to_vec();
    let expected_grad_cuda_zeros =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec_zeros.clone(), input_shape, false)?;

    assert_tensors_close(
        &expected_grad_cuda_zeros,
        &actual_grad_cuda_zeros,
        TOLERANCE,
    )?;

    Ok(())
}

#[serial]
#[test]
fn test_cuda_logsumexp_backward_axis() -> Result<(), Error> {
    println!("--- Running test_cuda_logsumexp_backward_axis ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test data with large values for numerical stability
    let input_data = vec![1.0, 2.0, 3.0, 100.0, 101.0, 100.5];
    let input_shape = &[2, 3];
    let axis_to_test = Some(1); // Reduce along axis 1 (rows)

    let input_cuda = Tensor::<CudaBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cuda = ops::logsumexp(&input_cuda, axis_to_test)?;
    let loss_cuda = ops::mean(&forward_output_cuda, None)?;
    loss_cuda.backward()?;

    let grad_cuda_storage = input_cuda.grad().ok_or(Error::NoGradientError)?.clone();
    let actual_grad_cuda = Tensor::<CudaBackend>::new(grad_cuda_storage, false);

    // Calculate Expected Gradient using CPU Backend
    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cpu = ops::logsumexp(&input_cpu, axis_to_test)?;
    let loss_cpu = ops::mean(&forward_output_cpu, None)?;
    loss_cpu.backward()?;
    let grad_cpu_storage_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec = grad_cpu_storage_ref.as_ref().to_vec();
    let expected_grad_cuda =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec.clone(), input_shape, false)?;

    assert_tensors_close(&expected_grad_cuda, &actual_grad_cuda, TOLERANCE)?;

    Ok(())
}

#[serial]
#[test]
fn test_cuda_logsumexp_backward_global() -> Result<(), Error> {
    println!("--- Running test_cuda_logsumexp_backward_global ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test data with large values for numerical stability
    let input_data = vec![1.0, 2.0, 3.0, 100.0, 101.0, 100.5];
    let input_shape = &[2, 3];
    let axis_to_test = None; // Global reduction

    let input_cuda = Tensor::<CudaBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cuda = ops::logsumexp(&input_cuda, axis_to_test)?;
    let loss_cuda = ops::mean(&forward_output_cuda, None)?;
    loss_cuda.backward()?;

    let grad_cuda_storage = input_cuda.grad().ok_or(Error::NoGradientError)?.clone();
    let actual_grad_cuda = Tensor::<CudaBackend>::new(grad_cuda_storage, false);

    // Calculate Expected Gradient using CPU Backend
    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cpu = ops::logsumexp(&input_cpu, axis_to_test)?;
    let loss_cpu = ops::mean(&forward_output_cpu, None)?;
    loss_cpu.backward()?;
    let grad_cpu_storage_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_cpu_vec = grad_cpu_storage_ref.as_ref().to_vec();
    let expected_grad_cuda =
        Tensor::<CudaBackend>::from_vec(grad_cpu_vec.clone(), input_shape, false)?;

    assert_tensors_close(&expected_grad_cuda, &actual_grad_cuda, TOLERANCE)?;

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_mul_backward() -> Result<(), Error> {
    println!("--- Running test_cuda_mul_backward ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let a = cuda_tensor_req_grad(vec![1.0, 2.0], &[2])?;
    let b = cuda_tensor_req_grad(vec![3.0, 4.0], &[2])?;
    let c = ops::mul(&a, &b)?;
    let loss = ops::mean(&c, None)?; // Scalar loss

    loss.backward()?; // Backpropagate

    // Expected gradients: dA = B/N, dB = A/N where N is size
    let n = c.size() as f32;
    let expected_grad_a = vec![3.0 / n, 4.0 / n];
    let expected_grad_b = vec![1.0 / n, 2.0 / n];

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &cuda_tensor(expected_grad_a, &[2])?, TOLERANCE)?;
    assert_tensors_close(&grad_b, &cuda_tensor(expected_grad_b, &[2])?, TOLERANCE)?;

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_mul_backward_broadcast() -> Result<(), Error> {
    println!("--- Running test_cuda_mul_backward_broadcast ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Scalar a and vector b
    let a = cuda_tensor_req_grad(vec![2.0], &[])?;
    let b = cuda_tensor_req_grad(vec![3.0, 4.0], &[2])?;
    let c = ops::mul(&a, &b)?;
    let loss = ops::mean(&c, None)?;
    loss.backward()?;

    // Actual CUDA gradients
    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );

    // Expected gradients via CPU backend
    let a_cpu = Tensor::<CpuBackend>::from_vec(vec![2.0], &[], true)?;
    let b_cpu = Tensor::<CpuBackend>::from_vec(vec![3.0, 4.0], &[2], true)?;
    let c_cpu = ops::mul(&a_cpu, &b_cpu)?;
    let loss_cpu = ops::mean(&c_cpu, None)?;
    loss_cpu.backward()?;
    let grad_a_cpu_ref = a_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_a_cpu = grad_a_cpu_ref.as_ref().to_vec();
    let grad_b_cpu_ref = b_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_b_cpu = grad_b_cpu_ref.as_ref().to_vec();

    let expected_grad_a = cuda_tensor(grad_a_cpu, &[])?;
    let expected_grad_b = cuda_tensor(grad_b_cpu, &[2])?;

    assert_tensors_close(&grad_a, &expected_grad_a, TOLERANCE)?;
    assert_tensors_close(&grad_b, &expected_grad_b, TOLERANCE)?;

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_matmul_backward() -> Result<(), Error> {
    println!("--- Running test_cuda_matmul_backward ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    println!("CUDA Context initialized.");

    // Create input tensors with requires_grad=true
    let a = cuda_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let b = cuda_tensor_req_grad(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    println!("Input tensors created:");
    println!(
        "a (ID {}): shape={:?}, data={:?}",
        a.id(),
        a.shape(),
        CudaBackend::copy_to_host(&*a.data())?
    );
    println!(
        "b (ID {}): shape={:?}, data={:?}",
        b.id(),
        b.shape(),
        CudaBackend::copy_to_host(&*b.data())?
    );

    // Perform forward pass
    println!("\nPerforming forward pass (matmul)...");
    let c = ops::matmul(&a, &b)?;
    println!(
        "c (ID {}): shape={:?}, data={:?}",
        c.id(),
        c.shape(),
        CudaBackend::copy_to_host(&*c.data())?
    );

    // Get forward result for debugging
    let c_data = CudaBackend::copy_to_host(&*c.data())?;
    println!("Forward output data: {:?}", c_data);
    // Expected: [19.0, 22.0, 43.0, 50.0]

    // Create scalar loss
    println!("\nCalculating scalar loss (mean)...");
    let loss = ops::mean(&c, None)?;
    println!(
        "loss (ID {}): shape={:?}, data={:?}",
        loss.id(),
        loss.shape(),
        CudaBackend::copy_to_host(&*loss.data())?
    );
    let loss_data = CudaBackend::copy_to_host(&*loss.data())?;
    println!("Loss value: {:?}", loss_data);
    // Expected: [33.5] (mean of [19.0, 22.0, 43.0, 50.0])

    // Perform backward pass
    println!("\nPerforming backward pass...");
    // Log output_grad before backward (mean always produces scalar 1.0 grad, but log for completeness)
    println!("(Note: output_grad for mean should be 1.0 for scalar loss)");
    loss.backward()?;
    println!("Backward pass complete.");

    // Calculate expected gradients using CPU backend
    println!("\nCalculating expected gradients using CPU backend...");
    let a_cpu = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)?;
    let b_cpu = Tensor::<CpuBackend>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], true)?;
    let c_cpu = ops::matmul(&a_cpu, &b_cpu)?;
    let loss_cpu = ops::mean(&c_cpu, None)?;
    loss_cpu.backward()?;

    let grad_a_cpu = a_cpu.grad().ok_or(Error::NoGradientError)?;
    let grad_b_cpu = b_cpu.grad().ok_or(Error::NoGradientError)?;
    let expected_grad_a = grad_a_cpu.as_ref().to_vec();
    let expected_grad_b = grad_b_cpu.as_ref().to_vec();
    println!("Expected grad_a: {:?}", expected_grad_a);
    println!("Expected grad_b: {:?}", expected_grad_b);

    // Get actual CUDA gradients
    println!("\nRetrieving actual CUDA gradients...");
    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );

    // Print actual gradients for debugging
    let grad_a_data = CudaBackend::copy_to_host(&*grad_a.data())?;
    let grad_b_data = CudaBackend::copy_to_host(&*grad_b.data())?;
    println!("Actual grad_a: {:?}", grad_a_data);
    println!("Actual grad_b: {:?}", grad_b_data);

    // Compare gradients
    println!("\nComparing gradients...");
    let expected_grad_a_cuda = cuda_tensor(expected_grad_a, &[2, 2])?;
    let expected_grad_b_cuda = cuda_tensor(expected_grad_b, &[2, 2])?;

    assert_tensors_close(&grad_a, &expected_grad_a_cuda, TOLERANCE)?;
    assert_tensors_close(&grad_b, &expected_grad_b_cuda, TOLERANCE)?;
    println!("Gradients match within tolerance.");

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_relu_backward() -> Result<(), Error> {
    println!("--- Running test_cuda_relu_backward ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    println!("Context initialized.");

    let data_vec = vec![-1.0, 0.0, 1.0, 2.0];
    let shape = &[4];
    let x = cuda_tensor_req_grad(data_vec.clone(), shape)?;
    println!("Input tensor created: {:?}", x);

    let y = ops::relu(&x)?;
    println!("ReLU output tensor created: {:?}", y);

    // Set initial gradient directly instead of using mean
    let output_grad_data = vec![1.0; x.size()];
    let output_grad_storage = CudaBackend::from_vec(output_grad_data, shape)?;
    y.set_grad(Some(output_grad_storage));
    println!("Output gradient set for y.");

    println!("Calling backward on y...");
    y.backward()?;
    println!("Backward call finished.");

    let expected_grad_vec = vec![0.0, 0.0, 1.0, 1.0];
    let expected_tensor = cuda_tensor(expected_grad_vec, shape)?;
    println!("Expected gradient tensor created.");

    let grad_x_storage = x
        .grad()
        .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor x".to_string()))?
        .clone();
    let grad_x_tensor = Tensor::<CudaBackend>::new(grad_x_storage, false);
    println!("Actual gradient tensor created from storage.");

    assert_tensors_close(&grad_x_tensor, &expected_tensor, TOLERANCE)?;
    println!("--- test_cuda_relu_backward PASSED ---");

    Ok(())
}

#[serial]
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_log_softmax_backward() -> Result<(), Error> {
    println!("--- Running test_cuda_log_softmax_backward ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    println!("CUDA Context initialized.");

    // Create input tensor with requires_grad=true
    let x = cuda_tensor_req_grad(vec![1.0, 2.0, 3.0], &[3])?;
    println!("Input tensor x (ID {}): shape={:?}", x.id(), x.shape());
    let x_data = CudaBackend::copy_to_host(&*x.data())?;
    println!("Input data: {:?}", x_data);

    // Perform forward pass (log_softmax)
    println!("\nPerforming forward pass (log_softmax)...");
    let y = ops::log_softmax(&x, 0)?;
    println!("y (ID {}): shape={:?}", y.id(), y.shape());
    let y_data = CudaBackend::copy_to_host(&*y.data())?;
    println!("log_softmax output: {:?}", y_data);

    // Calculate softmax probabilities for verification
    let max_x = x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_x: Vec<f32> = x_data.iter().map(|&xi| (xi - max_x).exp()).collect();
    let sum_exp_x: f32 = exp_x.iter().sum();
    let p: Vec<f32> = exp_x.iter().map(|&e| e / sum_exp_x).collect();
    println!("Computed softmax probabilities: {:?}", p);

    // Create and compute scalar loss
    println!("\nCalculating scalar loss (mean)...");
    let loss = ops::mean(&y, None)?;
    println!(
        "loss (ID {}): shape={:?}, data={:?}",
        loss.id(),
        loss.shape(),
        CudaBackend::copy_to_host(&*loss.data())?
    );
    let loss_data = CudaBackend::copy_to_host(&*loss.data())?;
    println!("Loss value: {:?}", loss_data);

    // Perform backward pass
    println!("\nPerforming backward pass...");
    // Log output_grad before backward (mean always produces scalar 1.0 grad, but log for completeness)
    println!("(Note: output_grad for mean should be 1.0 for scalar loss)");
    loss.backward()?;
    println!("Backward pass complete.");

    // Calculate expected gradient
    println!("\nCalculating expected gradient...");
    let n = x.size() as f32;
    println!("n (size): {}", n);
    // dL/dx_i = dL/dy_i - p_i * sum(dL/dy) = (1/n) - p_i
    let expected_grad: Vec<f32> = p.iter().map(|&pi| (1.0 / n) - pi).collect();
    println!("Expected gradient: {:?}", expected_grad);

    // Get actual gradient
    println!("\nRetrieving actual gradient...");
    let grad_x = Tensor::<CudaBackend>::new(
        x.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor x".to_string()))?
            .clone(),
        false,
    );
    let grad_x_data = CudaBackend::copy_to_host(&*grad_x.data())?;
    println!("Actual gradient: {:?}", grad_x_data);

    // Compare gradients
    println!("\nComparing gradients...");
    let expected_grad_cuda = cuda_tensor(expected_grad, &[3])?;
    assert_tensors_close(&grad_x, &expected_grad_cuda, TOLERANCE)?;
    println!("Gradients match within tolerance.");

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_sum_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Global sum (None axis)
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let shape1 = &[2, 2];
    let cuda_tensor1: Tensor<CudaBackend> = cuda_tensor_req_grad(data1, shape1)?;

    // Forward pass: loss = sum(x, None)
    let sum1 = ops::sum(&cuda_tensor1, None)?;

    // Set initial gradient to 1.0 for the scalar
    let grad_data = vec![1.0];
    let grad_storage = CudaBackend::from_vec(grad_data, &[])?;
    sum1.set_grad(Some(grad_storage));
    sum1.backward()?;

    // Get gradient
    let actual_grad1_storage: CudaStorage = cuda_tensor1
        .grad()
        .ok_or_else(|| {
            Error::InternalLogicError("Missing gradient for tensor cuda_tensor1".to_string())
        })?
        .clone();
    let actual_grad1_tensor = Tensor::<CudaBackend>::new(actual_grad1_storage, false);

    // Expected gradient:
    // dLoss/dInput = 1.0 (broadcasted)
    let expected_grad1_data = vec![1.0; shape1.iter().product()];
    let expected_grad1 = cuda_tensor(expected_grad1_data, shape1)?;
    assert_tensors_close(&actual_grad1_tensor, &expected_grad1, TOLERANCE)?;

    // Case 2: Sum along axis
    let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape2 = &[2, 3];
    let cuda_tensor2: Tensor<CudaBackend> = cuda_tensor_req_grad(data2, shape2)?;

    // Forward pass with axis=1: loss = sum(x, Some(1))
    let sum2 = ops::sum(&cuda_tensor2, Some(1))?; // Shape [2]

    // Set initial gradient to 1.0 for each element
    let grad_data = vec![1.0; 2];
    let grad_storage = CudaBackend::from_vec(grad_data, &[2])?;
    sum2.set_grad(Some(grad_storage));
    sum2.backward()?;

    // Get gradient
    let actual_grad2_storage: CudaStorage = cuda_tensor2
        .grad()
        .ok_or_else(|| {
            Error::InternalLogicError("Missing gradient for tensor cuda_tensor2".to_string())
        })?
        .clone();
    let actual_grad2_tensor = Tensor::<CudaBackend>::new(actual_grad2_storage, false);

    // Expected gradient:
    // dLoss/dInput = 1.0 (broadcasted)
    let expected_grad2_data = vec![1.0; shape2.iter().product()];
    let expected_grad2 = cuda_tensor(expected_grad2_data, shape2)?;
    assert_tensors_close(&actual_grad2_tensor, &expected_grad2, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_mean_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Global mean
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = &[2, 2];
    let cpu_in = cpu_tensor(data.clone(), shape)?;
    let cuda_in = cuda_tensor_from_cpu(&cpu_in)?;
    let cuda_tensor: Tensor<CudaBackend> = Tensor::new(cuda_in.data().clone(), true);

    // Forward pass
    let mean = ops::mean(&cuda_tensor, None)?;
    mean.backward()?;

    // Get gradient
    let actual_grad_storage = cuda_tensor
        .grad()
        .ok_or_else(|| {
            Error::InternalLogicError("Missing gradient for tensor cuda_tensor".to_string())
        })?
        .clone();
    let actual_grad = Tensor::new(actual_grad_storage, false);

    // Expected gradient is 1/N for each element
    let n = shape.iter().product::<usize>() as f32;
    let expected_cpu_tensor = cpu_tensor(vec![1.0 / n; 4], shape)?;
    let expected_grad = cuda_tensor_from_cpu(&expected_cpu_tensor)?;
    assert_tensors_close(&expected_grad, &actual_grad, TOLERANCE)?;

    // Case 2: Mean along axis
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = &[2, 3];
    let cpu_in = cpu_tensor(data.clone(), shape)?;
    let cuda_in = cuda_tensor_from_cpu(&cpu_in)?;
    let cuda_tensor: Tensor<CudaBackend> = Tensor::new(cuda_in.data().clone(), true);

    // Forward pass with axis=1
    let mean = ops::mean(&cuda_tensor, Some(1))?;
    let loss = ops::mean(&mean, None)?; // Take mean again for scalar loss
    loss.backward()?;

    // Get gradient
    let actual_grad_storage = cuda_tensor
        .grad()
        .ok_or_else(|| {
            Error::InternalLogicError("Missing gradient for tensor cuda_tensor".to_string())
        })?
        .clone();
    let actual_grad = Tensor::new(actual_grad_storage, false);

    // CORRECTED: For loss = mean(mean(x, axis=1)), each element's gradient should be 1/N
    // where N is the total number of elements, because:
    // 1. mean(x, axis=1) divides each element by reduction_dim (3)
    // 2. final mean divides by the remaining size (2)
    // So each element contributes 1/(3*2) = 1/6 to the final result
    let total_elements = shape.iter().product::<usize>() as f32; // 2 * 3 = 6
    let expected_cpu_tensor = cpu_tensor(vec![1.0 / total_elements; 6], shape)?;
    let expected_grad = cuda_tensor_from_cpu(&expected_cpu_tensor)?;
    assert_tensors_close(&actual_grad, &expected_grad, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_div_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let a = cuda_tensor_req_grad(vec![6.0, 8.0], &[2])?;
    let b = cuda_tensor_req_grad(vec![2.0, 4.0], &[2])?;
    let c = ops::div(&a, &b)?;
    let loss = ops::mean(&c, None)?;

    loss.backward()?;

    // For division a/b:
    // da = dout * (1/b)
    // db = dout * (-a/b^2)
    let n = c.size() as f32;
    let expected_grad_a = vec![1.0 / (2.0 * n), 1.0 / (4.0 * n)];
    let expected_grad_b = vec![-6.0 / (4.0 * n), -8.0 / (16.0 * n)];

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &cuda_tensor(expected_grad_a, &[2])?, TOLERANCE)?;
    assert_tensors_close(&grad_b, &cuda_tensor(expected_grad_b, &[2])?, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_sub_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let a = cuda_tensor_req_grad(vec![3.0, 4.0], &[2])?;
    let b = cuda_tensor_req_grad(vec![1.0, 2.0], &[2])?;
    let c = ops::sub(&a, &b)?;
    let loss = ops::mean(&c, None)?;

    loss.backward()?;

    // For subtraction a-b:
    // da = dout
    // db = -dout
    let n = c.size() as f32;
    let expected_grad_a = vec![1.0 / n, 1.0 / n];
    let expected_grad_b = vec![-1.0 / n, -1.0 / n];

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &cuda_tensor(expected_grad_a, &[2])?, TOLERANCE)?;
    assert_tensors_close(&grad_b, &cuda_tensor(expected_grad_b, &[2])?, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_exp_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let x = cuda_tensor_req_grad(vec![0.0, 1.0], &[2])?;
    let y = ops::exp(&x)?;
    let loss = ops::mean(&y, None)?;

    loss.backward()?;

    // For exp(x):
    // dx = dout * exp(x)
    let n = y.size() as f32;
    let expected_grad = vec![1.0 / n, std::f32::consts::E / n];

    let grad_x = Tensor::<CudaBackend>::new(
        x.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor x".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_x, &cuda_tensor(expected_grad, &[2])?, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_ln_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let x = cuda_tensor_req_grad(vec![1.0, 2.0], &[2])?;
    let y = ops::ln(&x)?;
    let loss = ops::mean(&y, None)?;

    loss.backward()?;

    // For ln(x):
    // dx = dout * (1/x)
    let n = y.size() as f32;
    let expected_grad = vec![1.0 / (1.0 * n), 1.0 / (2.0 * n)];

    let grad_x = Tensor::<CudaBackend>::new(
        x.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor x".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_x, &cuda_tensor(expected_grad, &[2])?, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_broadcast_div_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Broadcasting scalar to vector
    let a = cuda_tensor_req_grad(vec![1.0], &[])?;
    let b = cuda_tensor_req_grad(vec![2.0, 4.0, 6.0], &[3])?;
    let c = ops::div(&a, &b)?;
    let loss = ops::mean(&c, None)?;
    loss.backward()?;

    // For broadcasting div:
    // da = sum(dout * (1/b))
    // db = dout * (-a/b^2)
    let n = c.size() as f32;
    let expected_grad_a = vec![1.0 / (2.0 * n) + 1.0 / (4.0 * n) + 1.0 / (6.0 * n)];
    let expected_grad_b = vec![-1.0 / (4.0 * n), -1.0 / (16.0 * n), -1.0 / (36.0 * n)];

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &cuda_tensor(expected_grad_a, &[])?, TOLERANCE)?;
    assert_tensors_close(&grad_b, &cuda_tensor(expected_grad_b, &[3])?, TOLERANCE)?;

    // Case 2: Broadcasting row vector to matrix
    let a = cuda_tensor_req_grad(vec![2.0, 4.0], &[1, 2])?;
    let b = cuda_tensor_req_grad(vec![2.0, 4.0, 6.0, 8.0], &[2, 2])?;
    let c = ops::div(&a, &b)?;
    let loss = ops::mean(&c, None)?;
    loss.backward()?;

    // Expected gradients for row vector broadcasting
    let n = c.size() as f32;
    let expected_grad_a = vec![
        1.0 / (2.0 * n) + 1.0 / (6.0 * n),
        1.0 / (4.0 * n) + 1.0 / (8.0 * n),
    ];
    let expected_grad_b = vec![
        -2.0 / (4.0 * n),
        -4.0 / (16.0 * n),
        -2.0 / (36.0 * n),
        -4.0 / (64.0 * n),
    ];

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &cuda_tensor(expected_grad_a, &[1, 2])?, TOLERANCE)?;
    assert_tensors_close(&grad_b, &cuda_tensor(expected_grad_b, &[2, 2])?, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_broadcast_sub_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: Broadcasting scalar to vector
    let a = cuda_tensor_req_grad(vec![10.0], &[])?;
    let b = cuda_tensor_req_grad(vec![1.0, 2.0, 3.0], &[3])?;
    let c = ops::sub(&a, &b)?;
    let loss = ops::mean(&c, None)?;
    loss.backward()?;

    // For broadcasting subtraction:
    // da = sum(dout)
    // db = -dout
    let n = c.size() as f32;
    let expected_grad_a = vec![1.0 / n + 1.0 / n + 1.0 / n];
    let expected_grad_b = vec![-1.0 / n, -1.0 / n, -1.0 / n];

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &cuda_tensor(expected_grad_a, &[])?, TOLERANCE)?;
    assert_tensors_close(&grad_b, &cuda_tensor(expected_grad_b, &[3])?, TOLERANCE)?;

    // Case 2: Broadcasting row vector to matrix
    let a = cuda_tensor_req_grad(vec![10.0, 20.0], &[1, 2])?;
    let b = cuda_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let c = ops::sub(&a, &b)?;
    let loss = ops::mean(&c, None)?;
    loss.backward()?;

    // Expected gradients for row vector broadcasting
    let n = c.size() as f32;
    let expected_grad_a = vec![1.0 / n + 1.0 / n, 1.0 / n + 1.0 / n];
    let expected_grad_b = vec![-1.0 / n, -1.0 / n, -1.0 / n, -1.0 / n];

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &cuda_tensor(expected_grad_a, &[1, 2])?, TOLERANCE)?;
    assert_tensors_close(&grad_b, &cuda_tensor(expected_grad_b, &[2, 2])?, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_broadcast_log_softmax_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: 2D tensor with axis=1 (along rows)
    let x = cuda_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let y = ops::log_softmax(&x, 1)?;
    let loss = ops::mean(&y, None)?;

    loss.backward()?;

    // Expected gradients for each row should sum to zero due to softmax constraint
    let grad =
        CudaBackend::copy_to_host(&*x.grad().ok_or_else(|| {
            Error::InternalLogicError("Missing gradient for tensor x".to_string())
        })?)?;

    // Check row sums are approximately zero (softmax constraint)
    let row1_sum: f32 = grad[0..3].iter().sum();
    let row2_sum: f32 = grad[3..6].iter().sum();
    assert!(
        row1_sum.abs() < TOLERANCE,
        "Row 1 gradient sum should be ~0"
    );
    assert!(
        row2_sum.abs() < TOLERANCE,
        "Row 2 gradient sum should be ~0"
    );

    // Case 2: 3D tensor with different axes
    let x = cuda_tensor_req_grad(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
    )?;

    // Test log_softmax along last axis
    let y = ops::log_softmax(&x, 2)?;
    let loss = ops::mean(&y, None)?;
    loss.backward()?;

    // Check gradient properties
    let grad =
        CudaBackend::copy_to_host(&*x.grad().ok_or_else(|| {
            Error::InternalLogicError("Missing gradient for tensor x".to_string())
        })?)?;

    // Each 3-element group should sum to ~0
    for chunk in grad.chunks(3) {
        let sum: f32 = chunk.iter().sum();
        assert!(sum.abs() < TOLERANCE, "Gradient chunk sum should be ~0");
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_broadcast_add_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test scalar-vector broadcasting
    let a = cuda_tensor_req_grad(vec![2.0], &[])?; // scalar
    let b = cuda_tensor_req_grad(vec![1.0, 2.0, 3.0], &[3])?; // vector
    let c = ops::add(&a, &b)?;
    let loss = ops::mean(&c, None)?;

    loss.backward()?;

    let n = c.size() as f32;
    let expected_grad_a = cuda_tensor(vec![1.0 / n + 1.0 / n + 1.0 / n], &[])?;
    let expected_grad_b = cuda_tensor(vec![1.0 / n, 1.0 / n, 1.0 / n], &[3])?;

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &expected_grad_a, TOLERANCE)?;
    assert_tensors_close(&grad_b, &expected_grad_b, TOLERANCE)?;

    // Test matrix-vector broadcasting
    let a = cuda_tensor_req_grad(vec![1.0, 2.0], &[2, 1])?; // column vector
    let b = cuda_tensor_req_grad(vec![3.0, 4.0, 5.0], &[1, 3])?; // row vector
    let c = ops::add(&a, &b)?; // results in 2x3 matrix
    let loss = ops::mean(&c, None)?;

    loss.backward()?;

    let n = c.size() as f32;
    let expected_grad_a = cuda_tensor(
        vec![1.0 / n + 1.0 / n + 1.0 / n, 1.0 / n + 1.0 / n + 1.0 / n],
        &[2, 1],
    )?;
    let expected_grad_b = cuda_tensor(
        vec![1.0 / n + 1.0 / n, 1.0 / n + 1.0 / n, 1.0 / n + 1.0 / n],
        &[1, 3],
    )?;

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_b = Tensor::<CudaBackend>::new(
        b.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor b".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &expected_grad_a, TOLERANCE)?;
    assert_tensors_close(&grad_b, &expected_grad_b, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_reduction_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test sum reduction with axis
    let a = cuda_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let sum_col = ops::sum(&a, Some(0))?; // sum along rows
    let loss = ops::mean(&sum_col, None)?;

    loss.backward()?;

    let n = sum_col.size() as f32;
    let expected_grad = cuda_tensor(vec![1.0 / n, 1.0 / n, 1.0 / n, 1.0 / n], &[2, 2])?;

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &expected_grad, TOLERANCE)?;

    // Test mean reduction with axis
    let a = cuda_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let mean_row = ops::mean(&a, Some(1))?; // mean along columns
    let loss = ops::mean(&mean_row, None)?;

    loss.backward()?;

    let n = mean_row.size() as f32;
    let m = 3.0; // size of reduction dimension
    let expected_grad = cuda_tensor(
        vec![
            1.0 / (n * m),
            1.0 / (n * m),
            1.0 / (n * m),
            1.0 / (n * m),
            1.0 / (n * m),
            1.0 / (n * m),
        ],
        &[2, 3],
    )?;

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    assert_tensors_close(&grad_a, &expected_grad, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_numerical_stability() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Test log_softmax with large numbers
    let a = cuda_tensor_req_grad(vec![1000.0, 1000.1, 1000.2], &[3])?;
    let b = ops::log_softmax(&a, 0)?;
    let loss = ops::mean(&b, None)?;

    loss.backward()?;

    let grad_a = Tensor::<CudaBackend>::new(
        a.grad()
            .ok_or_else(|| Error::InternalLogicError("Missing gradient for tensor a".to_string()))?
            .clone(),
        false,
    );
    let grad_data = grad_a.to_cpu()?.data().as_ref().to_vec();

    // Check that gradients are finite and sum to ~0
    let grad_sum: f32 = grad_data.iter().sum();
    assert!(
        grad_sum.abs() < TOLERANCE,
        "log_softmax gradients should sum to 0"
    );
    grad_data.iter().for_each(|&x| {
        assert!(!x.is_nan(), "gradient should not be NaN");
    });

    // Test exp/ln stability with small/large numbers
    let small = cuda_tensor_req_grad(vec![1e-30], &[])?;

    // Test ln with small number
    let ln_small = ops::ln(&small)?;
    let loss = ops::mean(&ln_small, None)?;
    loss.backward()?;
    let grad_small = Tensor::<CudaBackend>::new(
        small
            .grad()
            .ok_or_else(|| {
                Error::InternalLogicError("Missing gradient for tensor small".to_string())
            })?
            .clone(),
        false,
    );
    let grad_small_data = grad_small.to_cpu()?.data().as_ref().to_vec();
    assert!(
        !grad_small_data[0].is_nan() && !grad_small_data[0].is_infinite(),
        "ln gradient should be finite for small input"
    );

    // Test exp with large number
    let large = cuda_tensor_req_grad(vec![1e30], &[])?;
    let exp_large = ops::exp(&large)?;
    let loss = ops::mean(&exp_large, None)?;
    loss.backward()?;

    if let Some(grad_ref) = large.grad() {
        let grad_large_tensor = Tensor::<CudaBackend>::new(grad_ref.clone(), false);
        let grad_large_data = grad_large_tensor.to_cpu()?.data().as_ref().to_vec();
        let grad_val = grad_large_data[0];

        // Check that gradient is positive infinity but not NaN for large exp input
        assert!(
            !grad_val.is_nan(),
            "exp gradient should not be NaN for large input, got: {}",
            grad_val
        );
        assert!(
            grad_val.is_infinite() && grad_val.is_sign_positive(),
            "exp gradient should be positive infinity for large input, got: {}",
            grad_val
        );
    } else {
        panic!("No gradient calculated for large exp input");
    }

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_chained_ops_backward() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Case 1: mean(sum(x)) - Global operations
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let shape1 = &[2, 2];
    let cuda_tensor1: Tensor<CudaBackend> = cuda_tensor_req_grad(data1, shape1)?;

    let sum1 = ops::sum(&cuda_tensor1, None)?; // Shape []
    let loss1 = ops::mean(&sum1, None)?; // Shape []

    // Set initial gradient to 1.0 for the scalar output
    let grad_data = vec![1.0];
    let grad_storage = CudaBackend::from_vec(grad_data, &[])?;
    loss1.set_grad(Some(grad_storage));
    loss1.backward()?;

    // Get gradient
    let actual_grad1_storage: CudaStorage = cuda_tensor1
        .grad()
        .ok_or_else(|| {
            Error::InternalLogicError("Missing gradient for tensor cuda_tensor1".to_string())
        })?
        .clone();
    let actual_grad1_tensor = Tensor::<CudaBackend>::new(actual_grad1_storage, false);

    // Expected gradient:
    // dLoss/dSum = 1/1 = 1.0 (mean of scalar)
    // dSum/dInput = 1.0 (broadcasted)
    // Total gradient = 1.0 everywhere
    let expected_grad1_data = vec![1.0; shape1.iter().product()];
    let expected_grad1 = cuda_tensor(expected_grad1_data, shape1)?;
    assert_tensors_close(&actual_grad1_tensor, &expected_grad1, TOLERANCE)?;

    // Case 2: mean(sum(x, axis=1)) - Axis reduction then global mean
    let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape2 = &[2, 3];
    let cuda_tensor2: Tensor<CudaBackend> = cuda_tensor_req_grad(data2, shape2)?;

    let sum2 = ops::sum(&cuda_tensor2, Some(1))?; // Shape [2]
    let loss2 = ops::mean(&sum2, None)?; // Shape []

    // Set initial gradient to 1.0 for the scalar output
    let grad_data = vec![1.0];
    let grad_storage = cuda_tensor(grad_data, &[])?.data().clone();
    loss2.set_grad(Some(grad_storage));
    loss2.backward()?;

    // Get gradient
    let actual_grad2_storage: CudaStorage = cuda_tensor2
        .grad()
        .ok_or_else(|| {
            Error::InternalLogicError("Missing gradient for tensor cuda_tensor2".to_string())
        })?
        .clone();
    let actual_grad2_tensor = Tensor::<CudaBackend>::new(actual_grad2_storage, false);

    // Expected gradient:
    // dLoss/dSum = 1/2 (mean of 2-element vector)
    // dSum/dInput = 1.0 (broadcasted)
    // Total gradient = 0.5 everywhere
    let expected_grad2_data = vec![0.5; shape2.iter().product()];
    let expected_grad2 = cuda_tensor(expected_grad2_data, shape2)?;
    assert_tensors_close(&actual_grad2_tensor, &expected_grad2, TOLERANCE)?;

    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_transpose_verification() -> Result<(), Error> {
    println!("--- Running test_cuda_transpose_verification ---");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    println!("CUDA Context initialized.");

    // Test matrices (same as used in matmul test)
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    let shape = [2, 2];

    // Create CPU tensors
    let a_cpu = Tensor::<CpuBackend>::from_vec(a_data.clone(), &shape, false)?;
    let b_cpu = Tensor::<CpuBackend>::from_vec(b_data.clone(), &shape, false)?;
    // Create CUDA tensors
    let a_cuda = cuda_tensor(a_data.clone(), &shape)?;
    let b_cuda = cuda_tensor(b_data.clone(), &shape)?;

    // Transpose CPU
    let a_cpu_t = CpuBackend::transpose(&a_cpu.data())?;
    let b_cpu_t = CpuBackend::transpose(&b_cpu.data())?;
    let a_cpu_t_vec = a_cpu_t.as_ref().to_vec();
    let b_cpu_t_vec = b_cpu_t.as_ref().to_vec();

    // Transpose CUDA
    let a_cuda_t = CudaBackend::transpose(&a_cuda.data())?;
    let b_cuda_t = CudaBackend::transpose(&b_cuda.data())?;
    let a_cuda_t_vec = CudaBackend::copy_to_host(&a_cuda_t)?;
    let b_cuda_t_vec = CudaBackend::copy_to_host(&b_cuda_t)?;

    println!("CPU transpose a: {:?}", a_cpu_t_vec);
    println!("CUDA transpose a: {:?}", a_cuda_t_vec);
    println!("CPU transpose b: {:?}", b_cpu_t_vec);
    println!("CUDA transpose b: {:?}", b_cuda_t_vec);

    // Compare elementwise
    assert_eq!(a_cpu_t_vec.len(), a_cuda_t_vec.len());
    assert_eq!(b_cpu_t_vec.len(), b_cuda_t_vec.len());
    for (i, (cpu, cuda)) in a_cpu_t_vec.iter().zip(a_cuda_t_vec.iter()).enumerate() {
        assert!(
            (cpu - cuda).abs() < 1e-6,
            "Mismatch at index {} for a: CPU={} CUDA={}",
            i,
            cpu,
            cuda
        );
    }
    for (i, (cpu, cuda)) in b_cpu_t_vec.iter().zip(b_cuda_t_vec.iter()).enumerate() {
        assert!(
            (cpu - cuda).abs() < 1e-6,
            "Mismatch at index {} for b: CPU={} CUDA={}",
            i,
            cpu,
            cuda
        );
    }

    println!("Transpose verification passed for both a and b.");
    Ok(())
}

#[test]
fn test_cuda_min_backward() {
    // Initialize CUDA context at the beginning
    use rust_tensor_lib::backend::cuda::{init_context, CudaContextGuard};
    init_context(0).unwrap();
    let _guard = CudaContextGuard::new().unwrap();

    let data = vec![1.0, 5.0, 2.0, 3.0, 1.0, 6.0];
    let shape = &[2, 3];

    // Create CPU and CUDA tensors
    let x_cpu_storage = CpuBackend::from_vec(data.clone(), shape).unwrap();
    let x_cuda_storage = CudaBackend::from_vec(data, shape).unwrap();
    let x_cpu = Tensor::<CpuBackend>::new(x_cpu_storage, false);
    let x_cuda = Tensor::<CudaBackend>::new(x_cuda_storage, false);

    // Gradient for global reduction (to scalar) - use empty shape for scalar
    let grad_output_cpu = CpuBackend::ones(&[]).unwrap();
    let grad_output_cuda = CudaBackend::ones(&[]).unwrap();

    let op_cpu = Op {
        op_type: OpType::Min(None),
        inputs: vec![x_cpu.clone()], // Clone here
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };
    let op_cuda = Op {
        op_type: OpType::Min(None),
        inputs: vec![x_cuda.clone()], // Clone here
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };

    let grad_input_cpu = CpuBackend::min_backward(&op_cpu, &grad_output_cpu).unwrap();
    let grad_input_cuda = CudaBackend::min_backward(&op_cuda, &grad_output_cuda).unwrap();

    // Convert CUDA output to CPU vec for comparison
    let cuda_vec = CudaBackend::copy_to_host(&grad_input_cuda).unwrap();
    let cpu_vec = CpuBackend::copy_to_host(&grad_input_cpu).unwrap();
    assert_eq!(cuda_vec, cpu_vec);

    // Test axis reduction backward
    let grad_output_cpu = CpuBackend::ones(&[3]).unwrap();
    let grad_output_cuda = CudaBackend::ones(&[3]).unwrap();
    let op_cpu = Op {
        op_type: OpType::Min(Some(0)),
        inputs: vec![x_cpu], // No need to clone here as it's the last use
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };
    let op_cuda = Op {
        op_type: OpType::Min(Some(0)),
        inputs: vec![x_cuda], // No need to clone here as it's the last use
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };

    let grad_input_cpu = CpuBackend::min_backward(&op_cpu, &grad_output_cpu).unwrap();
    let grad_input_cuda = CudaBackend::min_backward(&op_cuda, &grad_output_cuda).unwrap();

    // Convert CUDA output to CPU vec for comparison
    let cuda_vec = CudaBackend::copy_to_host(&grad_input_cuda).unwrap();
    let cpu_vec = CpuBackend::copy_to_host(&grad_input_cpu).unwrap();
    assert_eq!(cuda_vec, cpu_vec);
}

#[test]
fn test_cuda_prod_backward() {
    // Initialize CUDA context at the beginning
    use rust_tensor_lib::backend::cuda::{init_context, CudaContextGuard};
    init_context(0).unwrap();
    let _guard = CudaContextGuard::new().unwrap();

    let data = vec![2.0, 3.0, 4.0];
    // Create Tensor objects, not just storage
    let x_cpu_storage = CpuBackend::from_vec(data.clone(), &[3]).unwrap();
    let x_cuda_storage = CudaBackend::from_vec(data, &[3]).unwrap();
    let x_cpu = Tensor::<CpuBackend>::new(x_cpu_storage, false);
    let x_cuda = Tensor::<CudaBackend>::new(x_cuda_storage, false);

    // Use empty shape for scalar gradient
    let grad_output_cpu = CpuBackend::ones(&[]).unwrap();
    let grad_output_cuda = CudaBackend::ones(&[]).unwrap();

    let op_cpu = Op {
        op_type: OpType::Prod(None),
        inputs: vec![x_cpu.clone()], // Clone here to keep ownership of x_cpu
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };
    let op_cuda = Op {
        op_type: OpType::Prod(None),
        inputs: vec![x_cuda.clone()], // Clone here to keep ownership of x_cuda
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };

    let grad_input_cpu = CpuBackend::prod_backward(&op_cpu, &grad_output_cpu).unwrap();
    let grad_input_cuda = CudaBackend::prod_backward(&op_cuda, &grad_output_cuda).unwrap();

    // Convert CUDA output to CPU vec for comparison
    let cuda_vec = CudaBackend::copy_to_host(&grad_input_cuda).unwrap();
    let cpu_vec = CpuBackend::copy_to_host(&grad_input_cpu).unwrap();

    // Compare with tolerance
    assert_eq!(cuda_vec.len(), cpu_vec.len());
    for (i, (cpu_val, cuda_val)) in cpu_vec.iter().zip(cuda_vec.iter()).enumerate() {
        assert!(
            (cpu_val - cuda_val).abs() < 1e-5,
            "Values at index {i} differ: CPU={cpu_val}, CUDA={cuda_val} (tolerance=1e-5)"
        );
    }
}

#[test]
fn test_cuda_logsumexp_backward() {
    // Initialize CUDA context at the beginning
    use rust_tensor_lib::backend::cuda::{init_context, CudaContextGuard};
    init_context(0).unwrap();
    let _guard = CudaContextGuard::new().unwrap();

    let data = vec![1.0, 2.0, 3.0];
    // Create Tensor objects, not just storage
    let x_cpu_storage = CpuBackend::from_vec(data.clone(), &[3]).unwrap();
    let x_cuda_storage = CudaBackend::from_vec(data, &[3]).unwrap();
    let x_cpu = Tensor::<CpuBackend>::new(x_cpu_storage, false);
    let x_cuda = Tensor::<CudaBackend>::new(x_cuda_storage, false);

    // Use empty shape for scalar gradient
    let grad_output_cpu = CpuBackend::ones(&[]).unwrap();
    let grad_output_cuda = CudaBackend::ones(&[]).unwrap();

    let op_cpu = Op {
        op_type: OpType::LogSumExp(None),
        inputs: vec![x_cpu.clone()], // Clone here to keep ownership of x_cpu
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };
    let op_cuda = Op {
        op_type: OpType::LogSumExp(None),
        inputs: vec![x_cuda.clone()], // Clone here to keep ownership of x_cuda
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };

    let grad_input_cpu = CpuBackend::logsumexp_backward(&op_cpu, &grad_output_cpu).unwrap();
    let grad_input_cuda = CudaBackend::logsumexp_backward(&op_cuda, &grad_output_cuda).unwrap();

    // Convert CUDA output to CPU vec for comparison
    let cuda_vec = CudaBackend::copy_to_host(&grad_input_cuda).unwrap();
    let cpu_vec = CpuBackend::copy_to_host(&grad_input_cpu).unwrap();

    // Compare with tolerance
    assert_eq!(cuda_vec.len(), cpu_vec.len());
    for (i, (cpu_val, cuda_val)) in cpu_vec.iter().zip(cuda_vec.iter()).enumerate() {
        assert!(
            (cpu_val - cuda_val).abs() < 1e-5,
            "Values at index {i} differ: CPU={cpu_val}, CUDA={cuda_val} (tolerance=1e-5)"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[serial]
fn test_cuda_maximum_backward() {
    init_context(0).unwrap();
    let _guard = CudaContextGuard::new();

    // Create input tensors
    let a = Tensor::<CudaBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], true).unwrap();
    let b = Tensor::<CudaBackend>::from_vec(vec![4.0, 2.0, 1.0, 5.0], &[2, 2], true).unwrap();

    // Forward pass
    let c = ops::maximum(&a, &b).unwrap();

    // Create gradient tensor for backward pass
    let grad_output =
        Tensor::<CudaBackend>::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2], false).unwrap();

    // Set the gradient for c before calling backward
    c.set_grad(Some(grad_output.data().clone()));

    // Backward pass
    c.backward().unwrap();

    // Get computed gradients
    // Convert to CPU for comparison
    let a_grad_cuda = a.grad().unwrap();
    let b_grad_cuda = b.grad().unwrap();

    // Copy to host for comparison
    let a_grad = CudaBackend::copy_to_host(&a_grad_cuda).unwrap();
    let b_grad = CudaBackend::copy_to_host(&b_grad_cuda).unwrap();

    // Expected gradients:
    // a >= b: [1.0, 5.0, 3.0, 2.0] >= [4.0, 2.0, 1.0, 5.0] = [false, true, true, false]
    // b > a: [4.0, 2.0, 1.0, 5.0] > [1.0, 5.0, 3.0, 2.0] = [true, false, false, true]
    // grad_a = grad_output * (a >= b) = [0.0, 1.0, 1.0, 0.0]
    // grad_b = grad_output * (b > a) = [1.0, 0.0, 0.0, 1.0]

    assert_eq!(a_grad, vec![0.0, 1.0, 1.0, 0.0]);
    assert_eq!(b_grad, vec![1.0, 0.0, 0.0, 1.0]);
}

#[cfg(feature = "cuda")]
#[test]
#[serial]
fn test_cuda_maximum_backward_broadcasting() {
    init_context(0).unwrap();
    let _guard = CudaContextGuard::new();

    // Create input tensors with broadcasting
    let a = Tensor::<CudaBackend>::from_vec(vec![2.0], &[], true).unwrap(); // Scalar
    let b = Tensor::<CudaBackend>::from_vec(vec![1.0, 3.0, 2.0], &[3], true).unwrap(); // Vector

    // Forward pass
    let c = ops::maximum(&a, &b).unwrap();

    // Create gradient tensor for backward pass
    let grad_output = Tensor::<CudaBackend>::from_vec(vec![1.0, 1.0, 1.0], &[3], false).unwrap();

    // Set the gradient for c before calling backward
    c.set_grad(Some(grad_output.data().clone()));

    // Backward pass
    c.backward().unwrap();

    // Get computed gradients
    // Convert to CPU for comparison
    let a_grad_cuda = a.grad().unwrap();
    let b_grad_cuda = b.grad().unwrap();

    // Copy to host for comparison
    let a_grad = CudaBackend::copy_to_host(&a_grad_cuda).unwrap();
    let b_grad = CudaBackend::copy_to_host(&b_grad_cuda).unwrap();

    // Expected gradients:
    // a >= b: [2.0] >= [1.0, 3.0, 2.0] (after broadcasting) = [true, false, true]
    // b > a: [1.0, 3.0, 2.0] > [2.0, 2.0, 2.0] (after broadcasting) = [false, true, false]
    // grad_a = sum(grad_output * (a >= b)) = sum([1.0, 0.0, 1.0]) = 2.0
    // grad_b = grad_output * (b > a) = [0.0, 1.0, 0.0]

    assert_eq!(a_grad, vec![2.0]);
    assert_eq!(b_grad, vec![0.0, 1.0, 0.0]);
}

#[cfg(feature = "cuda")]
#[test]
#[serial]
fn test_cuda_minimum_backward() {
    init_context(0).unwrap();
    let _guard = CudaContextGuard::new();

    // Create input tensors
    let a = Tensor::<CudaBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], true).unwrap();
    let b = Tensor::<CudaBackend>::from_vec(vec![4.0, 2.0, 1.0, 5.0], &[2, 2], true).unwrap();

    // Forward pass
    let c = ops::minimum(&a, &b).unwrap();

    // Create gradient tensor for backward pass
    let grad_output =
        Tensor::<CudaBackend>::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[2, 2], false).unwrap();

    // Set the gradient for c before calling backward
    c.set_grad(Some(grad_output.data().clone()));

    // Backward pass
    c.backward().unwrap();

    // Get computed gradients
    // Convert to CPU for comparison
    let a_grad_cuda = a.grad().unwrap();
    let b_grad_cuda = b.grad().unwrap();

    // Copy to host for comparison
    let a_grad = CudaBackend::copy_to_host(&a_grad_cuda).unwrap();
    let b_grad = CudaBackend::copy_to_host(&b_grad_cuda).unwrap();

    // Expected gradients:
    // a <= b: [1.0, 5.0, 3.0, 2.0] <= [4.0, 2.0, 1.0, 5.0] = [true, false, false, true]
    // b < a: [4.0, 2.0, 1.0, 5.0] < [1.0, 5.0, 3.0, 2.0] = [false, true, false, false]
    // grad_a = grad_output * (a <= b) = [1.0, 0.0, 0.0, 1.0]
    // grad_b = grad_output * (b < a) = [0.0, 1.0, 1.0, 0.0]

    assert_eq!(a_grad, vec![1.0, 0.0, 0.0, 1.0]);
    assert_eq!(b_grad, vec![0.0, 1.0, 1.0, 0.0]);
}

#[cfg(feature = "cuda")]
#[test]
#[serial]
fn test_cuda_minimum_backward_broadcasting() {
    init_context(0).unwrap();
    let _guard = CudaContextGuard::new();

    // Create input tensors with broadcasting
    let a = Tensor::<CudaBackend>::from_vec(vec![2.0], &[], true).unwrap(); // Scalar
    let b = Tensor::<CudaBackend>::from_vec(vec![1.0, 3.0, 2.0], &[3], true).unwrap(); // Vector

    // Forward pass
    let c = ops::minimum(&a, &b).unwrap();

    // Create gradient tensor for backward pass
    let grad_output = Tensor::<CudaBackend>::from_vec(vec![1.0, 1.0, 1.0], &[3], false).unwrap();

    // Set the gradient for c before calling backward
    c.set_grad(Some(grad_output.data().clone()));

    // Backward pass
    c.backward().unwrap();

    // Get computed gradients
    // Convert to CPU for comparison
    let a_grad_cuda = a.grad().unwrap();
    let b_grad_cuda = b.grad().unwrap();

    // Copy to host for comparison
    let a_grad = CudaBackend::copy_to_host(&a_grad_cuda).unwrap();
    let b_grad = CudaBackend::copy_to_host(&b_grad_cuda).unwrap();

    // Expected gradients:
    // a <= b: [2.0] <= [1.0, 3.0, 2.0] (after broadcasting) = [false, true, true]
    // b < a: [1.0, 3.0, 2.0] < [2.0, 2.0, 2.0] (after broadcasting) = [true, false, false]
    // grad_a = sum(grad_output * (a <= b)) = sum([0.0, 1.0, 1.0]) = 2.0
    // grad_b = grad_output * (b < a) = [1.0, 0.0, 0.0]

    assert_eq!(a_grad, vec![2.0]);
    assert_eq!(b_grad, vec![1.0, 0.0, 0.0]);
}

#[cfg(feature = "cuda")]
#[test]
#[serial]
fn test_cuda_minimum_gradient() -> Result<(), Error> {
    use rust_tensor_lib::test_utils::check_gradient;
    use rust_tensor_lib::CudaTensor;

    // Apply the DEFAULT_EPSILON constant from other tests
    const DEFAULT_EPSILON: f32 = 1e-3;

    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // --- Case 1: No Broadcasting ---
    // Use values with clear min to avoid gradient discontinuities near ties during check
    let a1 = cuda_tensor_req_grad(vec![1.0, 4.0, 2.0], &[3])?;
    let b1 = cuda_tensor_req_grad(vec![3.0, 2.0, 1.0], &[3])?;

    let minimum_mean_fn1 = |inputs: &[CudaTensor]| -> Result<CudaTensor, Error> {
        let result = ops::minimum(&inputs[0], &inputs[1])?;
        ops::mean(&result, None) // Reduce to scalar
    };

    let inputs1 = vec![a1, b1];
    check_gradient(minimum_mean_fn1, &inputs1, 0, DEFAULT_EPSILON, 3e-2)?; // Check grad wrt a
    check_gradient(minimum_mean_fn1, &inputs1, 1, DEFAULT_EPSILON, 3e-2)?; // Check grad wrt b

    // --- Case 2: Broadcasting Scalar ---
    let a2 = cuda_tensor_req_grad(vec![2.5], &[])?; // scalar - fixed by removing extra comma
    let b2 = cuda_tensor_req_grad(vec![1.0, 3.0, 2.0], &[3])?; // vector

    let minimum_mean_fn2 = |inputs: &[CudaTensor]| -> Result<CudaTensor, Error> {
        let result = ops::minimum(&inputs[0], &inputs[1])?;
        ops::mean(&result, None) // Reduce to scalar
    };

    let inputs2 = vec![a2, b2];
    check_gradient(minimum_mean_fn2, &inputs2, 0, DEFAULT_EPSILON, 3e-2)?; // Check grad wrt a (scalar)
    check_gradient(minimum_mean_fn2, &inputs2, 1, DEFAULT_EPSILON, 3e-2)?; // Check grad wrt b (vector)

    Ok(())
}
