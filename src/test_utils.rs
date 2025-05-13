use crate::{Backend, Error, Tensor};
use std::cell::Ref;
use std::fmt::Debug;

/// Checks the gradient of a function with respect to a specific input tensor.
///
/// # Arguments
/// * `func`: A closure that takes a slice of input tensors and returns a scalar Tensor (the loss).
/// * `inputs`: A slice of input tensors. At least one must `require_grad`.
/// * `input_idx_to_check`: The index in the `inputs` slice for which to check the gradient.
/// * `epsilon`: A small value for finite difference perturbation (e.g., 1e-4).
/// * `tolerance`: The maximum allowed relative or absolute difference between analytical and numerical gradients.
///
/// # Returns
/// * `Ok(())` if the gradients match within the tolerance.
/// * `Err(String)` describing the mismatch otherwise.
pub fn check_gradient<B, F>(
    func: F,
    inputs: &[Tensor<B>],
    _input_idx_to_check: usize,
    epsilon: f32,
    tolerance: f32,
) -> Result<(), Error>
where
    B: Backend + 'static,
    F: Fn(&[Tensor<B>]) -> Result<Tensor<B>, Error>,
    B::Storage: Clone + Debug,
{
    if _input_idx_to_check >= inputs.len() {
        return Err(Error::InvalidOperation(format!(
            "input_idx_to_check ({}) is out of bounds for inputs slice (len {})",
            _input_idx_to_check,
            inputs.len()
        )));
    }

    let target_input = &inputs[_input_idx_to_check];
    if !target_input.requires_grad() {
        println!(
            "Skipping gradient check for input {} (ID {}) as it does not require grad.",
            _input_idx_to_check,
            target_input.id()
        );
        return Ok(());
    }

    // --- Step 2: Compute Analytical Gradient ---
    let analytical_grad_vec = compute_analytical_gradient(&func, inputs, _input_idx_to_check)?;

    // --- Step 3: Compute Numerical Gradient ---
    let numerical_grad_vec =
        compute_numerical_gradient(&func, inputs, _input_idx_to_check, epsilon)?;

    // --- Step 4: Compare Gradients ---
    compare_gradients(
        &analytical_grad_vec,
        &numerical_grad_vec,
        tolerance,
        _input_idx_to_check,
    )
}

fn compute_analytical_gradient<B, F>(
    func: &F,
    inputs: &[Tensor<B>],
    _input_idx_to_check: usize,
) -> Result<Vec<f32>, Error>
where
    B: Backend + 'static,
    F: Fn(&[Tensor<B>]) -> Result<Tensor<B>, Error>,
    B::Storage: Clone,
{
    // Ensure requires_grad is true for at least one input
    if !inputs.iter().any(|t| t.requires_grad()) {
        return Err(Error::InvalidOperation(
            "Cannot compute analytical gradient: No input tensor requires grad.".to_string(),
        ));
    }

    // Zero out any existing gradients on inputs that require grad
    for input in inputs.iter().filter(|t| t.requires_grad()) {
        input.zero_grad();
    }

    // --- Forward Pass ---
    let loss = func(inputs)?;

    // Ensure loss is scalar
    if loss.shape().iter().product::<usize>() != 1 {
        return Err(Error::InvalidOperation(format!(
            "Function must return a scalar tensor for gradient checking, got shape {:?}",
            loss.shape()
        )));
    }

    // --- Backward Pass ---
    loss.backward()?;

    // --- Extract Gradient ---
    let target_input = &inputs[_input_idx_to_check];
    let analytical_grad_opt: Option<Ref<B::Storage>> = target_input.grad();

    match analytical_grad_opt {
        Some(grad_ref) => {
            // Copy gradient data to host Vec<f32>
            B::copy_to_host(&*grad_ref)
        }
        None => {
            // If the tensor requires grad but has no gradient after backward,
            // it implies the gradient is zero.
            if target_input.requires_grad() {
                println!(
                    "Analytical gradient for input {} (ID {}) is None, assuming zero gradient.",
                    _input_idx_to_check,
                    target_input.id()
                );
                Ok(vec![0.0; target_input.size().max(1)]) // Use max(1) for scalar
            } else {
                Err(Error::InvalidOperation(format!(
                    "Input tensor {} (ID {}) does not require grad, cannot get analytical gradient.",
                    _input_idx_to_check,
                    target_input.id()
                )))
            }
        }
    }
}

fn compute_numerical_gradient<B, F>(
    func: &F,
    original_inputs: &[Tensor<B>],
    _input_idx_to_check: usize,
    epsilon: f32,
) -> Result<Vec<f32>, Error>
where
    B: Backend + 'static,
    F: Fn(&[Tensor<B>]) -> Result<Tensor<B>, Error>,
    B::Storage: Clone,
{
    let target_input = &original_inputs[_input_idx_to_check];
    let input_size = target_input.size().max(1); // Use max(1) for scalar
    let mut numerical_grad_vec = vec![0.0; input_size];

    // Clone inputs since we'll be modifying them
    let mut inputs_plus = original_inputs.to_vec();
    let mut inputs_minus = original_inputs.to_vec();

    // Get original data as host Vec<f32>
    let original_data = B::copy_to_host(&*target_input.data())?;

    // Check if original_data is empty which can happen for 0-sized tensors
    if original_data.is_empty() && input_size > 0 {
        return Err(Error::InternalLogicError(format!(
            "Inconsistency: Input size is {} but host data vector is empty.",
            input_size
        )));
    }

    for i in 0..input_size {
        // Ensure index is valid for original_data
        if i >= original_data.len() {
            return Err(Error::InternalLogicError(format!(
                "Index {} out of bounds for original_data (len {})",
                i,
                original_data.len()
            )));
        }

        // --- Perturb +epsilon ---
        {
            let mut data_plus = original_data.clone();
            data_plus[i] += epsilon;
            let perturbed_plus = B::from_vec(data_plus, target_input.shape().as_slice())?;
            inputs_plus[_input_idx_to_check] =
                Tensor::new(perturbed_plus, target_input.requires_grad());
        }

        let loss_plus = func(&inputs_plus)?;
        let loss_plus_val = B::copy_to_host(&*loss_plus.data())?;
        // Ensure loss_plus_val is not empty before indexing
        if loss_plus_val.is_empty() {
            return Err(Error::InternalLogicError(
                "Loss tensor (+eps) copied to host resulted in an empty vector.".to_string(),
            ));
        }

        // --- Perturb -epsilon ---
        {
            let mut data_minus = original_data.clone();
            // Ensure index is valid again (belt and suspenders)
            if i >= data_minus.len() {
                return Err(Error::InternalLogicError(format!(
                    "Index {} out of bounds for data_minus (len {})",
                    i,
                    data_minus.len()
                )));
            }
            data_minus[i] -= epsilon;
            let perturbed_minus = B::from_vec(data_minus, target_input.shape().as_slice())?;
            inputs_minus[_input_idx_to_check] =
                Tensor::new(perturbed_minus, target_input.requires_grad());
        }

        let loss_minus = func(&inputs_minus)?;
        let loss_minus_val = B::copy_to_host(&*loss_minus.data())?;
        // Ensure loss_minus_val is not empty before indexing
        if loss_minus_val.is_empty() {
            return Err(Error::InternalLogicError(
                "Loss tensor (-eps) copied to host resulted in an empty vector.".to_string(),
            ));
        }

        // Central difference formula
        let grad = (loss_plus_val[0] - loss_minus_val[0]) / (2.0 * epsilon);
        numerical_grad_vec[i] = grad;
    }

    Ok(numerical_grad_vec)
}

fn compare_gradients(
    analytical: &[f32],
    numerical: &[f32],
    tolerance: f32,
    _input_idx: usize,
) -> Result<(), Error> {
    if analytical.len() != numerical.len() {
        return Err(Error::InternalLogicError(format!(
            "Gradient size mismatch: analytical size={}, numerical size={}",
            analytical.len(),
            numerical.len()
        )));
    }

    let mut max_rel_err = 0.0;
    let mut max_abs_err = 0.0;
    let mut max_err_idx = 0;

    for (i, (a, n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let abs_err = (a - n).abs();
        let rel_err = if a.abs() > 1e-8 && n.abs() > 1e-8 {
            abs_err / a.abs().max(n.abs())
        } else {
            abs_err
        };

        if rel_err > max_rel_err {
            max_rel_err = rel_err;
            max_abs_err = abs_err;
            max_err_idx = i;
        }
    }

    if max_rel_err <= tolerance {
        Ok(())
    } else {
        Err(Error::GradientCheckError {
            analytical: analytical.to_vec(),
            numerical: numerical.to_vec(),
            max_rel_error: max_rel_err,
            max_abs_error: max_abs_err,
            at_index: max_err_idx,
        })
    }
}

pub fn assert_storage_eq<B: Backend>(a: &B::Storage, b: &B::Storage) {
    let a_data = B::copy_to_host(a).unwrap();
    let b_data = B::copy_to_host(b).unwrap();
    assert_eq!(a_data.len(), b_data.len(), "Storage lengths don't match");
    for (i, (a_val, b_val)) in a_data.iter().zip(b_data.iter()).enumerate() {
        assert_eq!(
            *a_val, *b_val,
            "Values at index {i} don't match: a={a_val}, b={b_val}"
        );
    }
}

pub fn assert_storage_close<B: Backend>(a: &B::Storage, b: &B::Storage, tol: f32) {
    let a_data = B::copy_to_host(a).unwrap();
    let b_data = B::copy_to_host(b).unwrap();
    assert_eq!(a_data.len(), b_data.len(), "Storage lengths don't match");
    for (i, (a_val, b_val)) in a_data.iter().zip(b_data.iter()).enumerate() {
        assert!(
            (a_val - b_val).abs() < tol,
            "Values at index {i} aren't close enough: a={a_val}, b={b_val}, diff={}, tol={tol}",
            (a_val - b_val).abs()
        );
    }
}
