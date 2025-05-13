use rust_tensor_lib::{ops, test_utils::check_gradient, CpuBackend, Error, Reduction, Tensor};

#[test]
fn test_mse_loss_backward_with_grad_check() -> Result<(), Error> {
    use rust_tensor_lib::Reduction;
    let preds = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true).unwrap();
    let targets = Tensor::<CpuBackend>::from_vec(vec![1.5, 2.5, 3.5], &[3], false).unwrap();
    let mse_loss_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        ops::mse_loss(&inputs[0], &targets, Reduction::Mean)
    };
    let epsilon = 1e-4;
    let tolerance = 2e-2;
    check_gradient(mse_loss_fn, &[preds], 0, epsilon, tolerance)?;
    Ok(())
}

#[test]
fn test_add_backward_with_grad_check() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], true).unwrap();
    let b = Tensor::<CpuBackend>::from_vec(vec![3.0, 4.0], &[2], true).unwrap();

    // Define the function to check (must return a scalar Tensor)
    let add_mean_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let res = ops::add(&inputs[0], &inputs[1])?;
        ops::mean(&res, None) // Return scalar mean
    };

    let inputs = vec![a, b];
    let epsilon = 1e-4;
    let tolerance = 2e-2;

    // Check gradient w.r.t input 'a' (index 0)
    check_gradient(add_mean_fn, &inputs, 0, epsilon, tolerance)?;

    // Check gradient w.r.t input 'b' (index 1)
    check_gradient(add_mean_fn, &inputs, 1, epsilon, tolerance)?;

    Ok(())
}

#[test]
fn test_mul_backward_with_grad_check() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], true).unwrap();
    let b = Tensor::<CpuBackend>::from_vec(vec![3.0, 4.0], &[2], true).unwrap();

    let mul_mean_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let res = ops::mul(&inputs[0], &inputs[1])?;
        ops::mean(&res, None)
    };

    let inputs = vec![a, b];
    let epsilon = 1e-4;
    let tolerance = 2e-2;

    check_gradient(mul_mean_fn, &inputs, 0, epsilon, tolerance)?;
    check_gradient(mul_mean_fn, &inputs, 1, epsilon, tolerance)?;

    Ok(())
}

#[test]
fn test_matmul_backward_with_grad_check() -> Result<(), Error> {
    // Create 2x2 matrices for testing
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();
    let b = Tensor::<CpuBackend>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], true).unwrap();

    let matmul_mean_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let res = ops::matmul(&inputs[0], &inputs[1])?;
        ops::mean(&res, None)
    };

    let inputs = vec![a, b];
    let epsilon = 1e-4;
    let tolerance = 2e-2;

    check_gradient(matmul_mean_fn, &inputs, 0, epsilon, tolerance)?;
    check_gradient(matmul_mean_fn, &inputs, 1, epsilon, tolerance)?;

    Ok(())
}

#[test]
fn test_relu_backward_with_grad_check() -> Result<(), Error> {
    // Test ReLU with positive and negative values
    // FIX: Move the test point further from the non-differentiable point (0.0)
    // Use a value more negative than the epsilon used for checking (1e-4)
    let neg_val_near_zero = -2e-4; // e.g., -2 * epsilon_for_check
    let x = Tensor::<CpuBackend>::from_vec(vec![-1.0, neg_val_near_zero, 1.0, 2.0], &[4], true)
        .unwrap();

    let relu_mean_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let res = ops::relu(&inputs[0])?;
        ops::mean(&res, None)
    };

    let inputs = vec![x];
    let epsilon_for_check = 1e-4; // Epsilon for finite difference
    let tolerance = 2e-2;

    check_gradient(relu_mean_fn, &inputs, 0, epsilon_for_check, tolerance)?;

    Ok(())
}

#[test]
fn test_log_softmax_backward_with_grad_check() -> Result<(), Error> {
    let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true).unwrap();

    let log_softmax_mean_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let res = ops::log_softmax(&inputs[0], 0)?;
        ops::mean(&res, None)
    };

    let inputs = vec![x];
    let epsilon = 1e-4;
    let tolerance = 2e-2;

    check_gradient(log_softmax_mean_fn, &inputs, 0, epsilon, tolerance)?;

    Ok(())
}

#[test]
fn test_tanh_gradient() -> Result<(), Error> {
    let input = Tensor::<CpuBackend>::from_vec(vec![-1.0, 0.0, 1.0], &[3], true).unwrap();
    let tanh_mean_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let y = ops::tanh(&inputs[0])?;
        ops::mean(&y, None)
    };
    let epsilon = 1e-4;
    let tolerance = 2e-2;
    check_gradient(tanh_mean_fn, &[input], 0, epsilon, tolerance)?;
    Ok(())
}

#[test]
fn test_softplus_gradient() -> Result<(), Error> {
    // Use moderate values to avoid potential precision issues with the simple formula
    let input = Tensor::<CpuBackend>::from_vec(vec![-2.0, 0.0, 2.0, 5.0], &[4], true)?;
    let softplus_mean_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let y = ops::softplus(&inputs[0])?;
        ops::mean(&y, None)
    };
    let epsilon = 1e-4;
    let tolerance = 2e-2;
    check_gradient(softplus_mean_fn, &[input], 0, epsilon, tolerance)?;
    Ok(())
}

#[test]
fn test_cpu_bce_with_logits_gradient() -> Result<(), Error> {
    let logits = Tensor::<CpuBackend>::from_vec(vec![-1.0, 0.5, 2.0], &[3], true).unwrap();
    // Targets should NOT require grad for this loss function's gradient check w.r.t logits
    let targets = Tensor::<CpuBackend>::from_vec(vec![0.0, 1.0, 1.0], &[3], false).unwrap();

    let bce_loss_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        // Index 0 is logits, targets is captured
        ops::binary_cross_entropy_with_logits(&inputs[0], &targets, Reduction::Mean)
    };

    let epsilon = 1e-4;
    let tolerance = 2e-2;

    // Check gradient only w.r.t. logits (input at index 0)
    check_gradient(bce_loss_fn, &[logits], 0, epsilon, tolerance)?;

    Ok(())
}

#[test]
fn test_min_gradient() -> Result<(), Error> {
    let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();

    // Test global min
    let min_global_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let min = ops::min(&inputs[0], None)?;
        ops::mean(&min, None) // Convert to scalar for gradient checking
    };

    check_gradient(min_global_fn, &[x.clone()], 0, 1e-3, 2e-2)?;

    // Test axis min
    let min_axis_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let min = ops::min(&inputs[0], Some(0))?;
        ops::mean(&min, None) // Convert to scalar for gradient checking
    };

    check_gradient(min_axis_fn, &[x.clone()], 0, 1e-3, 2e-2)?;

    Ok(())
}

#[test]
fn test_prod_gradient() -> Result<(), Error> {
    let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();

    // Test global product
    let prod_global_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let prod = ops::prod(&inputs[0], None)?;
        ops::mean(&prod, None) // Convert to scalar for gradient checking
    };

    check_gradient(prod_global_fn, &[x.clone()], 0, 1e-3, 2e-2)?;

    // Test axis product
    let prod_axis_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let prod = ops::prod(&inputs[0], Some(0))?;
        ops::mean(&prod, None) // Convert to scalar for gradient checking
    };

    check_gradient(prod_axis_fn, &[x.clone()], 0, 1e-3, 2e-2)?;

    Ok(())
}

#[test]
fn test_logsumexp_gradient() -> Result<(), Error> {
    let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();

    // Test global logsumexp
    let logsumexp_global_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let lse = ops::logsumexp(&inputs[0], None)?;
        ops::mean(&lse, None) // Convert to scalar for gradient checking
    };

    check_gradient(logsumexp_global_fn, &[x.clone()], 0, 1e-3, 2e-2)?;

    // Test axis logsumexp
    let logsumexp_axis_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let lse = ops::logsumexp(&inputs[0], Some(0))?;
        ops::mean(&lse, None) // Convert to scalar for gradient checking
    };

    check_gradient(logsumexp_axis_fn, &[x.clone()], 0, 1e-3, 2e-2)?;

    Ok(())
}

#[test]
fn test_square_gradient() -> Result<(), Error> {
    let input = Tensor::<CpuBackend>::from_vec(vec![-2.0, 0.5, 3.0], &[3], true)?;
    let square_mean_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        let y = ops::square(&inputs[0])?;
        ops::mean(&y, None) // Reduce to scalar
    };
    check_gradient(square_mean_fn, &[input], 0, 1e-4, 2e-2)?;
    Ok(())
}
