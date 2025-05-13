// tests/optim_tests.rs
use approx::assert_abs_diff_eq;
use rust_tensor_lib::{optim::*, Backend, CpuBackend, Error, Tensor}; // For float comparisons

#[cfg(feature = "cuda")]
use rust_tensor_lib::backend::cuda::{init_context, CudaBackend, CudaContextGuard};
#[cfg(feature = "cuda")]
use serial_test::serial;

#[test]
fn test_momentum_sgd_cpu() {
    // --- Setup ---
    let initial_params = vec![1.0, 2.0];
    let gradients = vec![0.1, -0.2];
    let shape = &[2];
    let lr = 0.1;
    let momentum = 0.9;

    let param_tensor = Tensor::<CpuBackend>::from_vec(initial_params.clone(), shape, true).unwrap();
    let grad_storage = CpuBackend::from_vec(gradients.clone(), shape).unwrap();
    param_tensor.set_grad(Some(grad_storage));

    let mut optimizer =
        MomentumSGD::<CpuBackend>::new(vec![param_tensor.clone()], lr, momentum).unwrap();

    // --- Step 1 ---
    optimizer.step().unwrap();

    // --- Calculate Expected Values (Step 1) ---
    let mut expected_velocity = [0.0; 2]; // Initial velocity is 0
    let mut expected_param = initial_params.clone();
    for i in 0..2 {
        expected_velocity[i] = momentum * 0.0 + gradients[i];
        expected_param[i] -= lr * expected_velocity[i];
    }

    // --- Assertions (Step 1) ---
    let actual_param_data = CpuBackend::copy_to_host(&*optimizer.parameters[0].data()).unwrap();
    for i in 0..2 {
        assert_abs_diff_eq!(actual_param_data[i], expected_param[i], epsilon = 1e-6);
    }
    let actual_velocity_data = CpuBackend::copy_to_host(&optimizer.velocities[0]).unwrap();
    for i in 0..2 {
        assert_abs_diff_eq!(
            actual_velocity_data[i],
            expected_velocity[i],
            epsilon = 1e-6
        );
    }
}

#[test]
fn test_adam_cpu() -> Result<(), Error> {
    // --- Setup ---
    let initial_params = vec![1.0, 2.0];
    let gradients = vec![0.1, -0.2];
    let shape = &[2];
    let lr = 0.1;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;

    println!("CPU Test: Initializing tensors...");
    let param_tensor = Tensor::<CpuBackend>::from_vec(initial_params.clone(), shape, true)?;
    let grad_storage = CpuBackend::from_vec(gradients.clone(), shape)?;
    param_tensor.set_grad(Some(grad_storage)); // Set the gradient for the parameter

    println!("CPU Test: Creating Adam optimizer...");
    let mut optimizer =
        Adam::<CpuBackend>::new(vec![param_tensor.clone()], lr, beta1, beta2, epsilon)?;

    // --- Step 1 ---
    println!("CPU Test: Performing optimizer step 1...");
    optimizer.step()?;
    println!("CPU Test: Optimizer step 1 finished.");

    // --- Calculate Expected Values (Step 1, t=1) ---
    println!("CPU Test: Calculating expected values for step 1...");
    let t = 1;
    let mut expected_m = [0.0; 2];
    let mut expected_v = [0.0; 2];
    let mut expected_param = initial_params.clone();

    // Adam formulas applied manually for verification
    let beta1_t = beta1.powi(t); // beta1^t
    let beta2_t = beta2.powi(t); // beta2^t

    for i in 0..shape[0] {
        // Iterate through each element
        let g = gradients[i];

        // Update biased moments (m_0 = 0, v_0 = 0)
        expected_m[i] = (1.0 - beta1) * g;
        expected_v[i] = (1.0 - beta2) * (g * g);

        // Bias correction
        let m_hat = expected_m[i] / (1.0 - beta1_t);
        let v_hat = expected_v[i] / (1.0 - beta2_t);

        // Parameter update
        expected_param[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
    }
    println!("CPU Test: Expected Param = {:?}", expected_param);
    println!("CPU Test: Expected M     = {:?}", expected_m);
    println!("CPU Test: Expected V     = {:?}", expected_v);

    // --- Assertions (Step 1) ---
    println!("CPU Test: Performing assertions for step 1...");
    assert_eq!(optimizer.t, 1, "Timestep should be 1 after one step");

    // Compare parameters
    let actual_param_data = CpuBackend::copy_to_host(&*optimizer.parameters[0].data())?;
    println!("CPU Test: Actual Param   = {:?}", actual_param_data);
    for i in 0..shape[0] {
        assert_abs_diff_eq!(actual_param_data[i], expected_param[i], epsilon = 1e-5);
    }

    // Compare m_states
    let actual_m_data = CpuBackend::copy_to_host(&optimizer.m_states[0])?;
    println!("CPU Test: Actual M       = {:?}", actual_m_data);
    for i in 0..shape[0] {
        assert_abs_diff_eq!(actual_m_data[i], expected_m[i], epsilon = 1e-5);
    }

    // Compare v_states
    let actual_v_data = CpuBackend::copy_to_host(&optimizer.v_states[0])?;
    println!("CPU Test: Actual V       = {:?}", actual_v_data);
    for i in 0..shape[0] {
        assert_abs_diff_eq!(actual_v_data[i], expected_v[i], epsilon = 1e-5);
    }

    println!("CPU Test: Adam CPU test finished successfully.");
    Ok(())
}

#[cfg(feature = "cuda")]
#[serial]
#[test]
fn test_momentum_sgd_cuda() -> Result<(), Error> {
    // --- Setup ---
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let initial_params = vec![1.0, 2.0];
    let gradients = vec![0.1, -0.2];
    let shape = &[2];
    let lr = 0.1;
    let momentum = 0.9;

    let param_tensor =
        Tensor::<CudaBackend>::from_vec(initial_params.clone(), shape, true).unwrap();
    let grad_storage = CudaBackend::from_vec(gradients.clone(), shape).unwrap();
    param_tensor.set_grad(Some(grad_storage));

    let mut optimizer =
        MomentumSGD::<CudaBackend>::new(vec![param_tensor.clone()], lr, momentum).unwrap();

    // --- Step 1 ---
    optimizer.step().unwrap();

    // --- Calculate Expected Values (Step 1) ---
    let mut expected_velocity = [0.0; 2];
    let mut expected_param = initial_params.clone();
    for i in 0..2 {
        expected_velocity[i] = momentum * 0.0 + gradients[i];
        expected_param[i] -= lr * expected_velocity[i];
    }

    // --- Assertions (Step 1) ---
    let actual_param_data = CudaBackend::copy_to_host(&*optimizer.parameters[0].data())?;
    for i in 0..2 {
        assert_abs_diff_eq!(actual_param_data[i], expected_param[i], epsilon = 1e-6);
    }
    let actual_velocity_data = CudaBackend::copy_to_host(&optimizer.velocities[0])?;
    for i in 0..2 {
        assert_abs_diff_eq!(
            actual_velocity_data[i],
            expected_velocity[i],
            epsilon = 1e-6
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[serial] // Ensures CUDA tests run one after another
#[test]
fn test_adam_cuda() -> Result<(), Error> {
    // --- CUDA Setup ---
    println!("CUDA Test: Initializing CUDA context...");
    init_context(0)?;
    let _guard = CudaContextGuard::new()?; // Ensures context is active
    println!("CUDA Test: CUDA context initialized.");

    // --- Setup (Identical to CPU test) ---
    let initial_params = vec![1.0, 2.0];
    let gradients = vec![0.1, -0.2];
    let shape = &[2];
    let lr = 0.1;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;

    println!("CUDA Test: Initializing tensors...");
    let param_tensor = Tensor::<CudaBackend>::from_vec(initial_params.clone(), shape, true)?;
    let grad_storage = CudaBackend::from_vec(gradients.clone(), shape)?;
    param_tensor.set_grad(Some(grad_storage)); // Set the gradient for the parameter

    println!("CUDA Test: Creating Adam optimizer...");
    let mut optimizer =
        Adam::<CudaBackend>::new(vec![param_tensor.clone()], lr, beta1, beta2, epsilon)?;

    // --- Step 1 ---
    println!("CUDA Test: Performing optimizer step 1...");
    optimizer.step()?;
    println!("CUDA Test: Optimizer step 1 finished.");

    // --- Calculate Expected Values (Step 1, t=1) ---
    println!("CUDA Test: Calculating expected values for step 1...");
    let t = 1;
    let mut expected_m = [0.0; 2];
    let mut expected_v = [0.0; 2];
    let mut expected_param = initial_params.clone();
    let beta1_t = beta1.powi(t);
    let beta2_t = beta2.powi(t);
    for i in 0..shape[0] {
        let g = gradients[i];
        expected_m[i] = (1.0 - beta1) * g;
        expected_v[i] = (1.0 - beta2) * (g * g);
        let m_hat = expected_m[i] / (1.0 - beta1_t);
        let v_hat = expected_v[i] / (1.0 - beta2_t);
        expected_param[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
    }
    println!("CUDA Test: Expected Param = {:?}", expected_param);
    println!("CUDA Test: Expected M     = {:?}", expected_m);
    println!("CUDA Test: Expected V     = {:?}", expected_v);

    // --- Assertions (Step 1) ---
    println!("CUDA Test: Performing assertions for step 1...");
    assert_eq!(optimizer.t, 1, "Timestep should be 1 after one step");

    // Compare parameters (copy from GPU to host first)
    let actual_param_data = CudaBackend::copy_to_host(&*optimizer.parameters[0].data())?;
    println!("CUDA Test: Actual Param   = {:?}", actual_param_data);
    for i in 0..shape[0] {
        assert_abs_diff_eq!(actual_param_data[i], expected_param[i], epsilon = 1e-5);
    }

    // Compare m_states (copy from GPU to host first)
    let actual_m_data = CudaBackend::copy_to_host(&optimizer.m_states[0])?;
    println!("CUDA Test: Actual M       = {:?}", actual_m_data);
    for i in 0..shape[0] {
        assert_abs_diff_eq!(actual_m_data[i], expected_m[i], epsilon = 1e-5);
    }

    // Compare v_states (copy from GPU to host first)
    let actual_v_data = CudaBackend::copy_to_host(&optimizer.v_states[0])?;
    println!("CUDA Test: Actual V       = {:?}", actual_v_data);
    for i in 0..shape[0] {
        assert_abs_diff_eq!(actual_v_data[i], expected_v[i], epsilon = 1e-5);
    }

    println!("CUDA Test: Adam CUDA test finished successfully.");
    Ok(())
}

#[cfg(feature = "cuda")]
#[serial] // Ensure sequential execution for CUDA tests
#[test]
fn test_adagrad_cuda() -> Result<(), Error> {
    // --- CUDA Setup ---
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // --- Setup (Identical to CPU test) ---
    let initial_params = vec![1.0, 2.0];
    let gradients = vec![0.1, -0.2];
    let shape = &[2];
    let lr = 0.1;
    let epsilon = 1e-8;

    // --- Create Tensors and Optimizer ---
    let param_tensor = Tensor::<CudaBackend>::from_vec(initial_params.clone(), shape, true)?;
    let grad_storage = CudaBackend::from_vec(gradients.clone(), shape)?;
    param_tensor.set_grad(Some(grad_storage));

    let mut optimizer = AdaGrad::<CudaBackend>::new(vec![param_tensor.clone()], lr, epsilon)?;

    // --- Step 1 ---
    optimizer.step()?;

    // --- Calculate Expected Values ---
    let mut expected_accum_sq_grad = [0.0; 2];
    let mut expected_param = initial_params.clone();
    for i in 0..shape[0] {
        let g = gradients[i];
        expected_accum_sq_grad[i] += g * g;
        expected_param[i] -= lr * g / (expected_accum_sq_grad[i].sqrt() + epsilon);
    }

    // --- Assertions ---
    let actual_param_data = CudaBackend::copy_to_host(&*optimizer.parameters[0].data())?;
    for i in 0..shape[0] {
        assert_abs_diff_eq!(actual_param_data[i], expected_param[i], epsilon = 1e-5);
    }

    let actual_accum_sq_grad_data = CudaBackend::copy_to_host(&optimizer.accumulated_sq_grad[0])?;
    for i in 0..shape[0] {
        assert_abs_diff_eq!(
            actual_accum_sq_grad_data[i],
            expected_accum_sq_grad[i],
            epsilon = 1e-5
        );
    }

    Ok(())
}

// --- AdaGrad CPU Test ---
#[test]
fn test_adagrad_cpu() -> Result<(), Error> {
    // --- Setup ---
    let initial_params = vec![1.0, 2.0];
    let gradients = vec![0.1, -0.2];
    let shape = &[2];
    let lr = 0.1;
    let epsilon = 1e-8;

    let param_tensor = Tensor::<CpuBackend>::from_vec(initial_params.clone(), shape, true)?;
    let grad_storage = CpuBackend::from_vec(gradients.clone(), shape)?;
    param_tensor.set_grad(Some(grad_storage));

    let mut optimizer = AdaGrad::<CpuBackend>::new(vec![param_tensor.clone()], lr, epsilon)?;

    // --- Step 1 ---
    optimizer.step()?;

    // --- Calculate Expected Values (Step 1) ---
    let mut expected_accum_sq_grad = [0.0; 2]; // Initial state is 0
    let mut expected_param = initial_params.clone();
    for i in 0..shape[0] {
        let g = gradients[i];
        expected_accum_sq_grad[i] += g * g; // Update accumulator state
        expected_param[i] -= lr * g / (expected_accum_sq_grad[i].sqrt() + epsilon);
        // Update parameter
    }

    // --- Assertions (Step 1) ---
    // Compare parameters
    let actual_param_data = CpuBackend::copy_to_host(&*optimizer.parameters[0].data())?;
    for i in 0..shape[0] {
        assert_abs_diff_eq!(actual_param_data[i], expected_param[i], epsilon = 1e-5);
    }

    // Compare accumulated_sq_grad state
    let actual_accum_sq_grad_data = CpuBackend::copy_to_host(&optimizer.accumulated_sq_grad[0])?;
    for i in 0..shape[0] {
        assert_abs_diff_eq!(
            actual_accum_sq_grad_data[i],
            expected_accum_sq_grad[i],
            epsilon = 1e-5
        );
    }

    Ok(())
}
