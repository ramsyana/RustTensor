//! Generic tensor operations that dispatch to backend implementations
//! and construct the computation graph when gradients are required.

// Thresholds (elements) where CUDA might be faster. Adjust based on benchmarks.
/// Number of elements above which `add` will use CUDA if available.
#[allow(dead_code)]
pub(crate) const ADD_CUDA_THRESHOLD: usize = 1024 * 1024; // 1 million elements
/// Inner dimension size above which `matmul` will use CUDA if available.
#[allow(dead_code)]
pub(crate) const MATMUL_CUDA_THRESHOLD_K: usize = 256;
/// Number of elements above which `relu` will use CUDA if available.
#[allow(dead_code)]
pub(crate) const RELU_CUDA_THRESHOLD: usize = 1024 * 1024; // 1 million elements

use crate::backend::Backend;
use crate::error::{Error, Reduction};
use crate::graph::{Op, OpType};
use crate::tensor::Tensor;

// Declare CPU-specific implementation modules (used by CpuBackend)
pub mod cpu_backward;
pub mod cpu_ops;



// --- Forward Operations ---

/// Creates a tensor with values sampled from a normal (Gaussian) distribution N(mean, std_dev^2).
///
/// This operation is NOT differentiable.
///
/// # Arguments
/// * `shape`: The desired shape of the output tensor.
/// * `mean`: The mean (center) of the normal distribution.
/// * `std_dev`: The standard deviation (spread or "width") of the normal distribution.
///
/// # Returns
/// A `Result` containing the new tensor or an error.
///
/// # Errors
/// Returns an error if:
/// * `std_dev` is negative
/// * The shape contains invalid dimensions
pub fn random_normal<B: Backend>(
    shape: &[usize],
    mean: f32,
    std_dev: f32,
) -> Result<Tensor<B>, Error> {
    if std_dev < 0.0 {
        return Err(Error::InvalidOperation(format!(
            "Standard deviation ({}) must be non-negative for normal distribution",
            std_dev
        )));
    }
    let storage = B::random_normal(shape, mean, std_dev)?;
    // Random generation ops are not differentiable
    let output = Tensor::new(storage, false);
    output.run_hooks();
    Ok(output)
}

/// Creates a tensor with values sampled from a uniform distribution U(low, high).
///
/// This operation is NOT differentiable.
///
/// # Arguments
/// * `shape`: The desired shape of the output tensor.
/// * `low`: The lower bound (inclusive) of the uniform distribution.
/// * `high`: The upper bound (exclusive) of the uniform distribution.
///
/// # Returns
/// A `Result` containing the new tensor or an error.
///
/// # Errors
/// Returns an error if:
/// * `high` <= `low`
/// * The shape contains invalid dimensions
pub fn random_uniform<B: Backend>(
    shape: &[usize],
    low: f32,
    high: f32,
) -> Result<Tensor<B>, Error> {
    if high <= low {
        return Err(Error::InvalidOperation(format!(
            "Upper bound ({}) must be greater than lower bound ({}) for uniform distribution",
            high, low
        )));
    }
    let storage = B::random_uniform(shape, low, high)?;
    // Random generation ops are not differentiable
    let output = Tensor::new(storage, false);
    output.run_hooks();
    Ok(output)
}

/// Creates a tensor with values sampled from a Bernoulli distribution B(p).
/// Each element will be 1.0 with probability `p` and 0.0 with probability `1-p`.
///
/// This operation is NOT differentiable.
///
/// # Arguments
/// * `shape`: The desired shape of the output tensor.
/// * `p`: The probability of sampling 1.0. Must be between 0.0 and 1.0 inclusive.
///
/// # Returns
/// A `Result` containing the new tensor or an error.
///
/// # Errors
/// Returns an error if:
/// * `p` is not between 0.0 and 1.0 inclusive
/// * The shape contains invalid dimensions
pub fn bernoulli<B: Backend>(shape: &[usize], p: f32) -> Result<Tensor<B>, Error> {
    if !(0.0..=1.0).contains(&p) {
        return Err(Error::InvalidOperation(format!(
            "Probability ({}) must be between 0.0 and 1.0 for Bernoulli distribution",
            p
        )));
    }
    let storage = B::bernoulli(shape, p)?;
    // Random generation ops are not differentiable
    let output = Tensor::new(storage, false);
    output.run_hooks();
    Ok(output)
}

/// Matrix multiplication that dispatches to the backend implementation
pub fn matmul<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(Error::DimensionMismatch(
            2,
            a_shape.len().max(b_shape.len()),
        ));
    }
    let (_a_rows, a_cols) = (a_shape[0], a_shape[1]);
    let (b_rows, _b_cols) = (b_shape[0], b_shape[1]);
    if a_cols != b_rows {
        return Err(Error::IncompatibleShapes {
            op: "matmul".to_string(),
            shape_a: a_shape,
            shape_b: b_shape,
        });
    }
    
    // Create clones of the data to avoid temporary reference issues
    let a_data_clone = a.data().clone();
    let b_data_clone = b.data().clone();
    
    // Dispatch to backend implementation
    let output_data = B::matmul(&a_data_clone, &b_data_clone)?;
    
    // Set up gradient tracking if needed
    let requires_grad = a.requires_grad() || b.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);
    
    if requires_grad {
        let op = Op::new(
            OpType::Matmul,
            vec![a.clone(), b.clone()],
            |op_ctx, grad_output| {
                let a = &op_ctx.inputs[0];
                let b = &op_ctx.inputs[1];
                
                // Calculate gradients for a and b
                // grad_a = grad_output @ b.T
                // grad_b = a.T @ grad_output
                let b_t = B::transpose(&*b.data())?;
                let a_t = B::transpose(&*a.data())?;
                
                let grad_a = B::matmul(grad_output, &b_t)?;
                let grad_b = B::matmul(&a_t, grad_output)?;
                
                Ok(vec![grad_a, grad_b])
            },
        );
        output_tensor.set_op(op);
    }
    
    output_tensor.run_hooks();
    Ok(output_tensor)
}
/// 2D convolution operation (NCHW, like PyTorch).
///
/// - input: [N, C_in, H_in, W_in]
/// - weights: [C_out, C_in, K_h, K_w]
/// - bias: [C_out] or None
/// - stride: (usize, usize)
/// - padding: (usize, usize)
pub fn conv2d<B: Backend>(
    input: &Tensor<B>,
    weights: &Tensor<B>,
    bias: Option<&Tensor<B>>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor<B>, Error> {
    // Create clones of the data to avoid temporary reference issues
    let input_data_clone = input.data().clone();
    let weights_data_clone = weights.data().clone();
    // Clone the bias data if it exists
    let bias_data_clone = match bias {
        Some(b) => Some(b.data().clone()),
        None => None,
    };
    
    // Dispatch to backend implementation
    let output_data = B::conv2d(
        &input_data_clone,
        &weights_data_clone,
        bias_data_clone.as_ref(),
        stride,
        padding,
    )?;
    
    // Set up gradient tracking if needed
    let requires_grad = input.requires_grad() || weights.requires_grad() || bias.map_or(false, |b| b.requires_grad());
    let output_tensor = Tensor::new(output_data, requires_grad);
    
    if requires_grad {
        let op = Op::new(
            OpType::Conv2d { stride, padding },
            vec![
                input.clone(),
                weights.clone(),
                bias.cloned().unwrap_or_else(|| {
                    let c_out = weights.shape()[0]; // First dimension is output channels
                    Tensor::new(B::zeros(&[c_out]).unwrap(), false)
                }),
            ],
            |op_ctx, grad_output| {
                // Delegate to the backend implementation
                let input = &op_ctx.inputs[0];
                let weights = &op_ctx.inputs[1];
                let input_data = &*input.data();
                let weights_data = &*weights.data();
                
                // Extract stride and padding from op_type
                let (stride, padding) = match op_ctx.op_type {
                    OpType::Conv2d { stride, padding } => (stride, padding),
                    _ => return Err(Error::InternalLogicError("Expected Conv2d op type".to_string())),
                };
                
                // Call the backend's conv2d_backward
                let (grad_input, grad_weights, grad_bias) = B::conv2d_backward(
                    input_data,
                    weights_data,
                    grad_output,
                    stride,
                    padding,
                )?;
                
                // Return gradients for all inputs
                Ok(vec![grad_input, grad_weights, grad_bias.unwrap_or_else(|| B::zeros(&[0]).unwrap())])
            },
        );
        output_tensor.set_op(op);
    }
    
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// 2D transpose convolution operation (a.k.a. deconvolution, NCHW, like PyTorch).
///
/// - input: [N, C_in, H_in, W_in]
/// - weights: [C_in, C_out, K_h, K_w]
/// - bias: [C_out] or None
/// - stride: (usize, usize)
/// - padding: (usize, usize)
///
/// # Returns
/// Output tensor of shape [N, C_out, H_out, W_out]
///
/// # Errors
/// Returns an error if backend operation fails or shapes are incompatible.
pub fn conv2d_transpose<B: Backend>(
    input: &Tensor<B>,
    weights: &Tensor<B>,
    bias: Option<&Tensor<B>>,
    stride: (usize, usize),
    padding: (usize, usize),
    // output_padding is sometimes needed to resolve ambiguity in output size
    // For now, we can try to calculate output size or require it. Let's start simpler.
    // output_padding: (usize, usize),
) -> Result<Tensor<B>, Error> {
    let output_data = B::conv2d_transpose(
        &*input.data(),
        &*weights.data(),
        bias.map(|b| b.data()).as_deref(),
        stride,
        padding,
        // output_padding,
    )?;
    let requires_grad = input.requires_grad() || weights.requires_grad() || bias.map_or(false, |b| b.requires_grad());
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::Conv2DTranspose { stride, padding, output_padding: (0,0) /* TODO: Handle output_padding */ },
            vec![input.clone(), weights.clone(), bias.cloned().unwrap_or_else(|| {
                // Dummy bias if None, C_out is weights.shape[1] for ConvTranspose
                Tensor::new(B::zeros(&[weights.shape()[1]]).unwrap(), false)
            })],
            |op_ctx, grad_output| {
                let input_t = &op_ctx.inputs[0];
                let weights_t = &op_ctx.inputs[1];
                // Extract params from op_type
                let (s, p) = match op_ctx.op_type {
                    OpType::Conv2DTranspose { stride, padding, .. } => (stride, padding),
                    _ => return Err(Error::InternalLogicError("Incorrect OpType".into())),
                };
                let (grad_input, grad_weights, grad_bias_opt) = B::conv2d_transpose_backward(
                    &*input_t.data(), &*weights_t.data(), grad_output, s, p
                )?;
                Ok(vec![grad_input, grad_weights, grad_bias_opt.unwrap_or_else(|| B::zeros(&[0]).unwrap())])
            }
        );
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Explicit backward for conv2d (for testing)
pub fn conv2d_backward<B: Backend>(
    input: &Tensor<B>,
    weights: &Tensor<B>,
    grad_output: &Tensor<B>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<(Tensor<B>, Tensor<B>, Option<Tensor<B>>), Error> {
    let (gi, gw, gb) = B::conv2d_backward(&*input.data(), &*weights.data(), &*grad_output.data(), stride, padding)?;
    Ok((Tensor::new(gi, false), Tensor::new(gw, false), gb.map(|b| Tensor::new(b, false))))
}

/// MaxPool2D operation (public API)
pub fn max_pool2d<B: Backend>(
    input: &Tensor<B>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor<B>, Error> {
    let (output_storage, indices_storage) =
        B::max_pool2d(&*input.data(), kernel_size, stride, padding)?;

    let requires_grad = input.requires_grad();
    let output_tensor = Tensor::new(output_storage, requires_grad);

    if requires_grad {
        // Create a tensor for indices (not requiring grad itself, but needed for grad computation of input)
        let indices_tensor = Tensor::new(indices_storage, false); // Indices don't need grad

        let op = Op::new(
            OpType::MaxPool2D { kernel_size, stride, padding },
            vec![input.clone(), indices_tensor.clone()], // Pass input AND indices to Op
            |op_ctx, grad_output| {
                // B::max_pool2d_backward will use op_ctx.inputs[0] (original input)
                // and op_ctx.inputs[1] (indices)
                B::max_pool2d_backward(op_ctx, grad_output)
                    .map(|grad_input| {
                        // Return grad for input, and a zeroed tensor for indices (no grad needed)
                        let indices_shape = op_ctx.inputs[1].shape();
                        let zero_indices_grad = B::zeros(&indices_shape).unwrap();
                        vec![grad_input, zero_indices_grad]
                    }) // Both are B::Storage, matching contract
            },
        );
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

// This is a helper function for element-wise multiplication
pub fn element_wise_mul<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    // Create clones of the data to avoid temporary reference issues
    let a_data_clone = a.data().clone();
    let b_data_clone = b.data().clone();
    
    // Dispatch to backend implementation
    let output_data = B::mul(&a_data_clone, &b_data_clone)?;
    
    // Set up gradient tracking if needed
    let requires_grad = a.requires_grad() || b.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);
    
    if requires_grad {
        let op = Op::new(
            OpType::Mul,
            vec![a.clone(), b.clone()],
            |op_ctx, grad_output| {
                let a = &op_ctx.inputs[0];
                let b = &op_ctx.inputs[1];
                
                // For element-wise multiplication, the gradients are:
                // grad_a = grad_output * b
                // grad_b = grad_output * a
                let grad_a = B::mul(grad_output, &*b.data())?;
                let grad_b = B::mul(grad_output, &*a.data())?;
                
                Ok(vec![grad_a, grad_b])
            },
        );
        output_tensor.set_op(op);
    }
    
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Element-wise multiplication that dispatches to the backend implementation
pub fn mul<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::mul(&*a.data(), &*b.data())?;
    let requires_grad = a.requires_grad() || b.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::Mul,
            vec![a.clone(), b.clone()],
            |op_ctx, grad_output| {
                let (grad_a, grad_b) = B::mul_backward(op_ctx, grad_output)?;

                Ok(vec![grad_a, grad_b])
            },
        );
        output_tensor.set_op(op);
    }

    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Element-wise addition that dispatches to the backend implementation
pub fn add<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::add(&*a.data(), &*b.data())?;
    let requires_grad = a.requires_grad() || b.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::Add,
            vec![a.clone(), b.clone()],
            |op_ctx, grad_output| {
                let (grad_a, grad_b) = B::add_backward(op_ctx, grad_output)?;
                Ok(vec![grad_a, grad_b])
            },
        );
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// ReLU activation that dispatches to the backend implementation
pub fn relu<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::relu(&*x.data())?;
    let requires_grad = x.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);
    if requires_grad {
        let op = Op::new(OpType::Relu, vec![x.clone()], |_op_ctx, grad_output| {
            let grad_x = B::relu_backward(_op_ctx, grad_output)?;
            Ok(vec![grad_x])
        });
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// ELU (Exponential Linear Unit) activation that dispatches to the backend implementation
///
/// Formula:
/// - `alpha * (exp(x) - 1)` if `x < 0`
/// - `x` if `x >= 0`
///
/// # Arguments
/// * `x` - The input tensor
/// * `alpha` - The alpha value for ELU (controls the value for negative inputs). Typically 1.0.
///
/// # Returns
/// A new tensor containing the ELU activation results
pub fn elu<B: Backend>(x: &Tensor<B>, alpha: f32) -> Result<Tensor<B>, Error> {
    let output_data = B::elu(&*x.data(), alpha)?;
    let requires_grad = x.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::Elu(alpha),
            vec![x.clone()],
            |op_ctx, grad_output| {
                let grad_x = B::elu_backward(op_ctx, grad_output)?;
                Ok(vec![grad_x]) // elu_backward returns single grad
            },
        );
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Sigmoid activation that dispatches to the backend implementation
pub fn sigmoid<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::sigmoid(&*x.data())?;
    let requires_grad = x.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(OpType::Sigmoid, vec![x.clone()], |op_ctx, grad_output| {
            let grad_x = B::sigmoid_backward(op_ctx, grad_output)?;
            Ok(vec![grad_x])
        });
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Log-softmax activation that dispatches to the backend implementation
pub fn log_softmax<B: Backend>(x: &Tensor<B>, axis: usize) -> Result<Tensor<B>, Error> {
    let output_data = B::log_softmax(&*x.data(), axis)?;

    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new(output_data.clone(), requires_grad);
        if requires_grad {
            let mut op = Op::new(
                OpType::LogSoftmax(axis), // Store axis in OpType
                vec![x.clone()],
                |op_ctx, grad_output| {
                    let grad_x = B::log_softmax_backward(op_ctx, grad_output)?;
                    Ok(vec![grad_x]) // log_softmax_backward returns single grad
                },
            );
            // Store the output data in the op for reuse during backward pass
            op.cached_outputs = Some(output_data);
            output.set_op(op);
        }
        output.run_hooks();
        output
    };

    Ok(output_tensor)
}

// --- Core Mathematical Operations ---

/// Computes the mean squared error between predictions and targets.
///
/// Loss = (preds - targets)^2
/// The result is then reduced according to the `reduction` parameter.
pub fn mse_loss<B: Backend>(
    preds: &Tensor<B>,
    targets: &Tensor<B>,
    reduction: Reduction,
) -> Result<Tensor<B>, Error> {
    if preds.shape() != targets.shape() {
        return Err(Error::IncompatibleShapes {
            op: "mse_loss".to_string(),
            shape_a: preds.shape(),
            shape_b: targets.shape(),
        });
    }

    // Warn if targets require grad (conventionally not needed)
    if targets.requires_grad() {
        println!("Warning: Targets tensor in mse_loss requires grad, but gradients are typically not computed w.r.t targets.");
    }

    // 1. Calculate difference: diff = preds - targets
    let diff = sub(preds, targets)?;

    // 2. Square the difference: squared_diff = diff^2
    // Use ops::mul as square isn't implemented yet
    let squared_diff = mul(&diff, &diff)?;

    // 3. Apply reduction
    let final_loss = match reduction {
        Reduction::None => {
            // Return the loss for each sample in the batch
            squared_diff
        }
        Reduction::Sum => {
            // Sum the per-sample losses
            sum(&squared_diff, None)? // Global sum
        }
        Reduction::Mean => {
            // Average the per-sample losses
            mean(&squared_diff, None)? // Global mean
        }
    };

    Ok(final_loss)
}

/// Computes element-wise absolute value: |x|
pub fn abs<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::abs(&*x.data())?;
    let requires_grad = x.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);
    if requires_grad {
        let op = Op::new(OpType::Abs, vec![x.clone()], |op_ctx, grad_output| {
            let grad_x = B::abs_backward(op_ctx, grad_output)?;
            Ok(vec![grad_x])
        });
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Computes the mean reduction along a specified axis or globally if axis=None
pub fn mean<B: Backend>(x: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, Error> {
    let output_data = B::mean(&*x.data(), axis)?;
    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            output_data,
            requires_grad,
            if requires_grad {
                Some(OpType::Mean(axis))
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Sums elements along a specified axis or globally.
pub fn sum<B: Backend>(x: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, Error> {
    let requires_grad = x.requires_grad();
    let output_tensor = if let Some(axis) = axis {
        let sum_result = B::sum_along_axis(&*x.data(), axis)?;
        {
            let output = Tensor::new_with_op(
                sum_result,
                requires_grad,
                if requires_grad {
                    Some(OpType::Sum(Some(axis)))
                } else {
                    None
                },
                vec![x.clone()],
            );
            output.run_hooks();
            output
        }
    } else {
        let total_sum = B::sum_all(&*x.data())?;
        let scalar_storage = B::from_vec(vec![total_sum], &[])?;
        {
            let output = Tensor::new_with_op(
                scalar_storage,
                requires_grad,
                if requires_grad {
                    Some(OpType::Sum(None))
                } else {
                    None
                },
                vec![x.clone()],
            );
            output.run_hooks();
            output
        }
    };
    Ok(output_tensor)
}

/// Element-wise subtraction: a - b
pub fn sub<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let result = B::sub(&*a.data(), &*b.data())?;
    let requires_grad = a.requires_grad() || b.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            result,
            requires_grad,
            if requires_grad {
                Some(OpType::Sub)
            } else {
                None
            },
            vec![a.clone(), b.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Computes element-wise division: a / b
pub fn div<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let result = B::div(&*a.data(), &*b.data())?;
    let requires_grad = a.requires_grad() || b.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            result,
            requires_grad,
            if requires_grad {
                Some(OpType::Div)
            } else {
                None
            },
            vec![a.clone(), b.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Computes element-wise exponential: exp(x)
pub fn exp<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let result = B::exp(&*x.data())?;
    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            result,
            requires_grad,
            if requires_grad {
                Some(OpType::Exp)
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Computes element-wise natural logarithm: ln(x)
pub fn ln<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let result = B::ln(&*x.data())?;
    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            result,
            requires_grad,
            if requires_grad {
                Some(OpType::Ln)
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Computes element-wise square root: sqrt(x)
pub fn sqrt<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let result = B::sqrt(&*x.data())?;
    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            result,
            requires_grad,
            if requires_grad {
                Some(OpType::Sqrt)
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Element-wise division by a scalar
pub fn div_scalar<B: Backend>(x: &Tensor<B>, scalar: f32) -> Result<Tensor<B>, Error> {
    let output_data = B::div_scalar(&*x.data(), scalar)?;

    let requires_grad = x.requires_grad();
    let output = {
        let output = Tensor::new(output_data, requires_grad);
        if requires_grad {
            // Need div_scalar_backward implemented
            return Err(Error::InvalidOperation(
                "Backward pass for div_scalar operation not implemented yet".to_string(),
            ));
            // let op = Op::new(OpType::DivScalar(scalar), vec![x.clone()], B::div_scalar_backward);
            // output.set_op(op);
        }
        output.run_hooks();
        output
    };
    Ok(output)
}

/// Element-wise multiplication by a scalar
pub fn mul_scalar<B: Backend>(x: &Tensor<B>, scalar: f32) -> Result<Tensor<B>, Error> {
    let output_data = B::mul_scalar(&*x.data(), scalar)?;

    let requires_grad = x.requires_grad();
    let output = {
        let output = Tensor::new(output_data, requires_grad);
        if requires_grad {
            // For multiplication by scalar, the gradient is just the scalar itself
            let op = Op::new(OpType::MulScalar(scalar), vec![x.clone()], |op, grad| {
                // Gradient for x is just grad * scalar
                let scalar = match op.op_type {
                    OpType::MulScalar(s) => s,
                    _ => return Err(Error::InvalidOperation("Expected MulScalar op type".to_string())),
                };
                let grad_x = B::mul_scalar(grad, scalar)?;
                Ok(vec![grad_x])
            });
            output.set_op(op);
        }
        output.run_hooks();
        output
    };
    Ok(output)
}

/// Broadcasts a tensor to a new shape that is compatible with broadcasting rules.
pub fn broadcast_to<B: Backend>(x: &Tensor<B>, shape: &[usize]) -> Result<Tensor<B>, Error> {
    let output_data = B::broadcast_to(&*x.data(), shape)?;
    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            output_data,
            requires_grad,
            if requires_grad {
                Some(OpType::Broadcast)
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Reshapes a tensor to a new shape without changing its data.
/// The new shape must have the same total number of elements as the original shape.
pub fn view<B: Backend>(x: &Tensor<B>, shape: &[usize]) -> Result<Tensor<B>, Error> {
    // Check that the total number of elements is the same
    let old_size = x.shape().iter().product::<usize>();
    let new_size = shape.iter().product::<usize>();
    if old_size != new_size {
        return Err(Error::IncompatibleShapes {
            op: "view".to_string(),
            shape_a: x.shape().to_vec(),
            shape_b: shape.to_vec(),
        });
    }

    // Create a new tensor with the same data but different shape
    let mut output_data = x.data().clone();
    B::set_shape(&mut output_data, shape)?;

    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            output_data,
            requires_grad,
            if requires_grad {
                Some(OpType::View)
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Computes the maximum value along the specified axis.
/// If axis is None, computes the global maximum.
pub fn max<B: Backend>(x: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, Error> {
    let output_data = B::max(&*x.data(), axis)?;
    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            output_data,
            requires_grad,
            if requires_grad {
                Some(OpType::Max(axis))
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Computes the minimum value along the specified axis.
/// If axis is None, computes the global minimum.
pub fn min<B: Backend>(x: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, Error> {
    let output_data = B::min(&*x.data(), axis)?;
    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            output_data,
            requires_grad,
            if requires_grad {
                Some(OpType::Min(axis))
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Computes the product of all elements along the specified axis.
/// If axis is None, computes the global product.
pub fn prod<B: Backend>(x: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, Error> {
    let output_data = B::prod(&*x.data(), axis)?;
    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            output_data,
            requires_grad,
            if requires_grad {
                Some(OpType::Prod(axis))
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Computes the log-sum-exp along the specified axis.
/// If axis is None, computes the global log-sum-exp.
/// Uses the max-trick for numerical stability.
pub fn logsumexp<B: Backend>(x: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, Error> {
    let output_data = B::logsumexp(&*x.data(), axis)?;
    let requires_grad = x.requires_grad();
    let output_tensor = {
        let output = Tensor::new_with_op(
            output_data,
            requires_grad,
            if requires_grad {
                Some(OpType::LogSumExp(axis))
            } else {
                None
            },
            vec![x.clone()],
        );
        output.run_hooks();
        output
    };
    Ok(output_tensor)
}

/// Returns the indices of maximum values along the specified axis.
/// The indices are returned as f32 values.
/// This operation is NOT differentiable.
pub fn argmax<B: Backend>(x: &Tensor<B>, axis: usize) -> Result<Tensor<B>, Error> {
    let output_data = B::argmax(&*x.data(), axis)?;
    // ArgMax is not differentiable, so requires_grad is always false
    {
        let output = Tensor::new(output_data, false);
        output.run_hooks();
        Ok(output)
    }
}

/// Returns the indices of minimum values along the specified axis.
/// The indices are returned as f32 values.
/// This operation is NOT differentiable.
pub fn argmin<B: Backend>(x: &Tensor<B>, axis: usize) -> Result<Tensor<B>, Error> {
    let output_data = B::argmin(&*x.data(), axis)?;
    // ArgMin is not differentiable, so requires_grad is always false
    {
        let output = Tensor::new(output_data, false);
        output.run_hooks();
        Ok(output)
    }
}

/// Computes the element-wise hyperbolic tangent: tanh(x)
pub fn tanh<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::tanh(&*x.data())?;
    let requires_grad = x.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);
    if requires_grad {
        let op = Op::new(OpType::Tanh, vec![x.clone()], |op_ctx, grad_output| {
            let grad_x = B::tanh_backward(op_ctx, grad_output)?;
            Ok(vec![grad_x])
        });
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Computes the element-wise softplus activation: log(1 + exp(x))
pub fn softplus<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::softplus(&*x.data())?; // Call backend forward impl
    let requires_grad = x.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad); // Create output tensor

    if requires_grad {
        // Create the Op for autograd if needed
        let op = Op::new(
            OpType::Softplus, // Use the Softplus OpType
            vec![x.clone()],  // Input tensor
            |op_ctx, grad_output| {
                // This closure calls the backend's backward implementation
                // It receives the Op context (op_ctx.inputs[0] is x) and the incoming gradient (grad_output)
                let grad_x = B::softplus_backward(op_ctx, grad_output)?;
                Ok(vec![grad_x]) // softplus_backward returns a single gradient
            },
        );
        // Associate the Op with the output tensor for autograd
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Computes the binary cross-entropy loss between logits and targets.
///
/// This version is numerically stable and expects logits (raw scores), NOT probabilities.
/// It combines a sigmoid layer and the BCELoss in one function.
///
/// Formula (numerically stable):
/// loss = max(logits, 0) - logits * targets + log(1 + exp(-abs(logits)))
///      = relu(logits) - logits * targets + log(1 + exp(-abs(logits)))
///
/// The result is then reduced according to the `reduction` parameter.
///
/// # Arguments
/// * `logits`: The predicted logits (raw scores) - Tensor requiring gradient.
/// * `targets`: The ground truth labels (0 or 1) - Tensor NOT requiring gradient.
/// * `reduction`: Specifies the reduction to apply to the output (`None`, `Mean`, `Sum`).
///
/// # Returns
/// A `Result` containing the loss tensor (scalar if reduction is Mean/Sum) or an error.
pub fn binary_cross_entropy_with_logits<B: Backend>(
    logits: &Tensor<B>,
    targets: &Tensor<B>,
    reduction: Reduction,
) -> Result<Tensor<B>, Error> {
    // Input validation
    if logits.shape() != targets.shape() {
        return Err(Error::IncompatibleShapes {
            op: "binary_cross_entropy_with_logits".to_string(),
            shape_a: logits.shape(),
            shape_b: targets.shape(),
        });
    }
    if targets.requires_grad() {
        // Consider using a more robust logging mechanism if available
        println!("Warning: Targets tensor in binary_cross_entropy_with_logits requires grad. This is unusual.");
    }

    // Create scalar tensors needed for calculations ON THE CORRECT BACKEND
    let scalar_minus_one = Tensor::<B>::from_vec(vec![-1.0], &[], false)?;
    let scalar_one = Tensor::<B>::from_vec(vec![1.0], &[], false)?;

    // Term 1: max(logits, 0) -> use relu
    let term1 = relu(logits)?; // relu(logits)

    // Term 2: logits * targets
    let term2 = mul(logits, targets)?; // logits * targets

    // Term 3: log(1 + exp(-abs(logits)))
    let abs_logits = abs(logits)?; // abs(logits)
    let neg_abs_logits = mul(&abs_logits, &scalar_minus_one)?; // -abs(logits)
    let exp_neg_abs = exp(&neg_abs_logits)?; // exp(-abs(logits))
    let one_plus_exp = add(&scalar_one, &exp_neg_abs)?; // 1 + exp(-abs(logits))
    let term3 = ln(&one_plus_exp)?; // ln(1 + exp(-abs(logits)))

    // Combine terms: term1 - term2 + term3
    let intermediate_loss = sub(&term1, &term2)?;
    let loss_elements = add(&intermediate_loss, &term3)?; // Element-wise loss

    // Apply reduction
    let final_loss = match reduction {
        Reduction::None => {
            // Return the loss for each sample in the batch
            loss_elements
        }
        Reduction::Sum => {
            // Sum the per-sample losses
            sum(&loss_elements, None)? // Global sum
        }
        Reduction::Mean => {
            // Average the per-sample losses
            mean(&loss_elements, None)? // Global mean
        }
    };

    Ok(final_loss)
}

/// Element-wise power function, raising each element in tensor `a` to the power of the corresponding element in tensor `b`.
/// Supports broadcasting.
///
/// # Arguments
/// * `a` - The base tensor.
/// * `b` - The exponent tensor.
///
/// # Returns
/// A `Result` containing the tensor with each element of `a` raised to the power of the corresponding element in `b`,
/// or an error if the operation fails (e.g., due to incompatible shapes for broadcasting).
pub fn powf<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::powf(&*a.data(), &*b.data())?;
    let requires_grad = a.requires_grad() || b.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::Powf,
            vec![a.clone(), b.clone()],
            |op_ctx, grad_output| {
                let (grad_a, grad_b) = B::powf_backward(op_ctx, grad_output)?;
                Ok(vec![grad_a, grad_b])
            },
        );
        output_tensor.set_op(op);
    }

    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Element-wise powf of a scalar that dispatches to the backend implementation
/// via a temporary tensor created for the scalar.
pub fn powf_scalar<B: Backend>(a: &Tensor<B>, b: f32) -> Result<Tensor<B>, Error> {
    // Create a tensor filled with the scalar value `b`
    let b_shape = vec![1]; // Scalar is a 1-element 1D tensor
    let b_data = B::from_vec(vec![b], &b_shape)?;

    // Create a tensor from the scalar that can be broadcast
    let b_tensor = Tensor::new(b_data, false); // scalar is not differentiable

    // Compute a ^ b
    powf(a, &b_tensor)
}

/// Computes the element-wise square of a tensor: x^2
pub fn square<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    // Call the backend's forward implementation
    let output_data = B::square(&*x.data())?; // Deref Ref<Storage> to &Storage

    // Determine if the output requires gradients
    let requires_grad = x.requires_grad();

    // Create the output tensor
    let output_tensor = Tensor::new(output_data, requires_grad);

    // If gradients are needed, create and set the backward operation
    if requires_grad {
        // The backward closure captures the input tensor `x`
        let op = Op::new(
            OpType::Square,  // Use the new OpType
            vec![x.clone()], // The input to the square operation
            |op_ctx, grad_output| {
                // This closure calls the backend's backward implementation
                // It receives the Op context (op_ctx.inputs[0] is x) and the incoming gradient (grad_output)
                let grad_x = B::square_backward(op_ctx, grad_output)?;
                Ok(vec![grad_x]) // square_backward returns a single gradient for x
            },
        );
        // Associate the operation with the output tensor for autograd
        output_tensor.set_op(op);
    }

    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Computes the element-wise maximum of two tensors (`max(a, b)`).
/// Supports broadcasting.
///
/// # Arguments
/// * `a` - The first input tensor.
/// * `b` - The second input tensor.
///
/// # Returns
/// A `Result` containing the tensor with element-wise maximum values.
pub fn maximum<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    // Perform the forward pass using the backend implementation
    let output_data = B::maximum(&*a.data(), &*b.data())?; // Call trait method

    // Determine if gradients are required for the output
    let requires_grad = a.requires_grad() || b.requires_grad();

    // Create the output tensor
    let output_tensor = Tensor::new(output_data, requires_grad);

    // If gradients are required, set up the backward operation
    if requires_grad {
        let op = Op::new(
            OpType::Maximum,            // Use the new OpType
            vec![a.clone(), b.clone()], // Inputs to the operation
            |_op_ctx, grad_output| {
                // Call the backend's specific backward implementation
                let (grad_a, grad_b) = B::maximum_backward(_op_ctx, grad_output)?;
                // Return the computed gradients in the expected order
                Ok(vec![grad_a, grad_b])
            },
        );
        // Associate the operation with the output tensor
        output_tensor.set_op(op);
    }

    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Computes the element-wise minimum of two tensors (`min(a, b)`).
/// Supports broadcasting.
///
/// # Arguments
/// * `a` - The first input tensor.
/// * `b` - The second input tensor.
///
/// # Returns
/// A `Result` containing the tensor with element-wise minimum values.
pub fn minimum<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    // Perform the forward pass using the backend implementation
    let output_data = B::minimum(&*a.data(), &*b.data())?; // Call trait method

    // Determine if gradients are required for the output
    let requires_grad = a.requires_grad() || b.requires_grad();

    // Create the output tensor
    let output_tensor = Tensor::new(output_data, requires_grad);

    // If gradients are required, set up the backward operation
    if requires_grad {
        let op = Op::new(
            OpType::Minimum,            // Use the new OpType
            vec![a.clone(), b.clone()], // Inputs to the operation
            |op_ctx, grad_output| {
                // Call the backend's specific backward implementation
                let (grad_a, grad_b) = B::minimum_backward(op_ctx, grad_output)?;
                // Return the computed gradients in the expected order
                Ok(vec![grad_a, grad_b])
            },
        );
        // Associate the operation with the output tensor
        output_tensor.set_op(op);
    }

    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Computes element-wise greater comparison: `a > b`.
/// Supports broadcasting. Returns 1.0 for true, 0.0 for false.
/// This operation is NOT differentiable.
pub fn greater<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    // Call the backend implementation
    let output_data = B::greater(&*a.data(), &*b.data())?;
    // Comparison ops are not differentiable
    {
        let output = Tensor::new(output_data, false);
        output.run_hooks();
        Ok(output)
    }
}

/// Computes element-wise greater or equal comparison: `a >= b`.
/// Supports broadcasting. Returns 1.0 for true, 0.0 for false.
/// This operation is NOT differentiable.
pub fn greater_equal<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::greater_equal(&*a.data(), &*b.data())?;
    {
        let output = Tensor::new(output_data, false);
        output.run_hooks();
        Ok(output)
    }
}

/// Computes element-wise less comparison: `a < b`.
/// Supports broadcasting. Returns 1.0 for true, 0.0 for false.
/// This operation is NOT differentiable.
pub fn less<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::less(&*a.data(), &*b.data())?;
    {
        let output = Tensor::new(output_data, false);
        output.run_hooks();
        Ok(output)
    }
}

/// Computes element-wise less or equal comparison: `a <= b`.
/// Supports broadcasting. Returns 1.0 for true, 0.0 for false.
/// This operation is NOT differentiable.
pub fn less_equal<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::less_equal(&*a.data(), &*b.data())?;
    {
        let output = Tensor::new(output_data, false);
        output.run_hooks();
        Ok(output)
    }
}

/// Computes element-wise not equal comparison: `a != b`.
/// Supports broadcasting. Returns 1.0 for true, 0.0 for false.
/// This operation is NOT differentiable.
pub fn not_equal<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::not_equal(&*a.data(), &*b.data())?;
    {
        let output = Tensor::new(output_data, false);
        output.run_hooks();
        Ok(output)
    }
}

/// Computes the softmax cross-entropy loss between logits and one-hot encoded targets.
///
/// This function combines `log_softmax` and a negative log likelihood calculation
/// for numerical stability and efficiency. It's commonly used as the final loss
/// layer in multi-class classification problems.
///
/// **Note:** The `targets` tensor **must** be one-hot encoded.
///
/// # Arguments
/// * `logits`: The raw, unnormalized scores output by the model (Tensor requiring grad).
///             Shape: `[batch_size, ..., num_classes]`
/// * `targets`: The ground truth labels in one-hot format (Tensor *not* requiring grad).
///              Shape must match `logits`.
/// * `axis`: The dimension corresponding to the classes.
/// * `reduction`: Specifies how to aggregate the loss across the batch (`None`, `Sum`, `Mean`).
///
/// # Returns
/// A `Result` containing the loss tensor. The shape depends on the `reduction`:
/// * `None`: Shape `[batch_size, ...]`, same as input excluding the class `axis`.
/// * `Sum` or `Mean`: Scalar tensor `[]`.
///
/// # Errors
/// Returns an error if shapes are incompatible, the axis is invalid, or an
/// underlying operation fails.
pub fn softmax_cross_entropy<B: Backend>(
    logits: &Tensor<B>,
    targets: &Tensor<B>,
    axis: usize,
    reduction: Reduction,
) -> Result<Tensor<B>, Error> {
    // --- Input Validation ---
    let logits_shape = logits.shape();
    let targets_shape = targets.shape();

    if logits_shape != targets_shape {
        return Err(Error::IncompatibleShapes {
            op: "softmax_cross_entropy".to_string(),
            shape_a: logits_shape,
            shape_b: targets_shape,
        });
    }
    if axis >= logits_shape.len() {
        return Err(Error::InvalidIndex(vec![axis]));
    }
    if targets.requires_grad() {
        // Log a warning, gradients are usually not needed for targets in cross-entropy
        println!(
            "Warning: Targets tensor in softmax_cross_entropy requires grad. This is unusual."
        );
        // Proceed anyway, but the gradient w.r.t targets might not be meaningful
        // depending on how the `mul` gradient handles it.
    }

    // --- Calculation Steps ---

    // 1. Apply log_softmax to logits along the class axis
    // log_probs shape: same as logits
    let log_probs = log_softmax(logits, axis)?;

    // 2. Calculate element-wise product: target * log_softmax(logits)
    // Note: Since targets are one-hot, this effectively selects the log_prob
    // of the true class for each sample.
    // elementwise_product shape: same as logits
    let elementwise_product = mul(&log_probs, targets)?;

    // 3. Sum the result along the class axis.
    // This calculates -log(p_true_class) for each sample.
    // sum_neg_log_probs shape: [batch_size, ...] (axis dimension removed)
    let sum_neg_log_probs = sum(&elementwise_product, Some(axis))?;

    // 4. Negate the sum to get the positive cross-entropy loss per sample.
    // Create a scalar tensor with value -1.0 for negation via multiplication
    let scalar_minus_one = Tensor::<B>::from_vec(vec![-1.0], &[], false)?;
    let per_sample_loss = mul(&sum_neg_log_probs, &scalar_minus_one)?;

    // 5. Apply reduction across the batch/remaining dimensions
    let final_loss = match reduction {
        Reduction::None => {
            // Return the loss for each sample in the batch
            per_sample_loss
        }
        Reduction::Sum => {
            // Sum the per-sample losses
            sum(&per_sample_loss, None)? // Global sum
        }
        Reduction::Mean => {
            // Average the per-sample losses
            mean(&per_sample_loss, None)? // Global mean
        }
    };

    Ok(final_loss)
}

/// Matrix transpose operation that dispatches to the backend implementation
pub fn transpose<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::transpose(&*x.data())?;
    let requires_grad = x.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::Transpose,
            vec![x.clone()],
            |_op_ctx, grad_output| {
                // Transpose is self-adjoint: gradient is just another transpose
                let grad_x = B::transpose(grad_output)?;
                Ok(vec![grad_x])
            },
        );
        output_tensor.set_op(op);
    }

    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Concatenates multiple tensors along a specified axis.
///
/// # Arguments
/// * `tensors` - A slice of references to tensors to concatenate
/// * `axis` - The axis along which to concatenate the tensors
///
/// # Returns
/// A new tensor containing the concatenated data
///
/// # Examples
/// ```
/// use rust_tensor_lib::{CpuTensor, ops, backend::Backend};
/// 
/// let a = CpuTensor::from_vec(vec![1.0, 2.0], &[2], false).unwrap();
/// let b = CpuTensor::from_vec(vec![3.0, 4.0], &[2], false).unwrap();
/// let c = ops::concat(&[&a, &b], 0).unwrap();
/// assert_eq!(c.shape(), &[4]);
/// let c_data = rust_tensor_lib::backend::cpu::CpuBackend::copy_to_host(&*c.data()).unwrap();
/// assert_eq!(c_data, vec![1.0, 2.0, 3.0, 4.0]);
/// 
/// // Concatenate along a different axis
/// let a = CpuTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false).unwrap();
/// let b = CpuTensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2], false).unwrap();
/// let c = ops::concat(&[&a, &b], 1).unwrap();
/// assert_eq!(c.shape(), &[2, 4]);
/// let c_data = rust_tensor_lib::backend::cpu::CpuBackend::copy_to_host(&*c.data()).unwrap();
/// assert_eq!(c_data, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
/// ```
pub fn concat<B: Backend>(tensors: &[&Tensor<B>], axis: usize) -> Result<Tensor<B>, Error> {
    if tensors.is_empty() {
        return Err(Error::InvalidOperation("Cannot concat empty list of tensors".to_string()));
    }
    
    // Validate shapes: all tensors must have same ndim and same shape except along `axis`.
    let first_shape = tensors[0].shape();
    let ndim = first_shape.len();
    
    if axis >= ndim {
        return Err(Error::InvalidIndex(vec![axis]));
    }
    
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.shape().len() != ndim {
            return Err(Error::IncompatibleShapes { 
                op: "concat".to_string(), 
                shape_a: first_shape.to_vec(), 
                shape_b: t.shape().to_vec() 
            });
        }
        
        for (d, (s1, s2)) in first_shape.iter().zip(t.shape().iter()).enumerate() {
            if d != axis && s1 != s2 {
                return Err(Error::IncompatibleShapes { 
                    op: format!("concat dim {} of tensor {}", d, i), 
                    shape_a: first_shape.to_vec(), 
                    shape_b: t.shape().to_vec() 
                });
            }
        }
    }

    // Create a Vec of references to the tensor data
    // Create a vector to store tensor data references
    let mut tensors_data_refs: Vec<&B::Storage> = Vec::with_capacity(tensors.len());
    
    // Keep the Ref<'_, B::Storage> objects alive until the end of the function
    let data_refs: Vec<_> = tensors.iter().map(|t| t.data()).collect();
    
    // Create references to the tensor data
    for data_ref in &data_refs {
        tensors_data_refs.push(&**data_ref);
    }
    
    let output_data = B::concat(&tensors_data_refs, axis)?;

    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let input_shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape().to_vec()).collect();
        let inputs: Vec<Tensor<B>> = tensors.iter().map(|&t| t.clone()).collect();
        
        let op = Op::new(
            OpType::Concat { axis, input_shapes },
            inputs,
            |op_ctx, grad_output| {
                // This closure calls the backend's backward implementation
                B::concat_backward(op_ctx, grad_output) // Returns Vec<B::Storage>
            },
        );
        output_tensor.set_op(op);
    }
    
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Extracts a slice from a tensor along specified dimensions.
///
/// # Arguments
/// * `input` - The input tensor
/// * `ranges` - A slice of ranges, one for each dimension, specifying the slice to extract.
///              Each range is in the form `start..end` where `start` is inclusive and `end` is exclusive.
///
/// # Returns
/// A new tensor containing the sliced data
///
/// # Errors
/// Returns an error if:
/// * The number of ranges doesn't match the number of dimensions in the input tensor
/// * Any range is out of bounds for its corresponding dimension
/// * Any range has start > end
///
/// # Examples
/// ```
/// use rust_tensor_lib::{CpuTensor, ops, backend::Backend};
/// 
/// let tensor = CpuTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
/// // Extract the first row
/// let slice = ops::slice(&tensor, &[0..1, 0..3]).unwrap();
/// assert_eq!(slice.shape(), &[1, 3]);
/// let slice_data = rust_tensor_lib::backend::cpu::CpuBackend::copy_to_host(&*slice.data()).unwrap();
/// assert_eq!(slice_data, vec![1.0, 2.0, 3.0]);
/// ```
pub fn slice<B: Backend>(input: &Tensor<B>, ranges: &[std::ops::Range<usize>]) -> Result<Tensor<B>, Error> {
    // Validate ranges against input.shape()
    if ranges.len() != input.shape().len() {
        return Err(Error::InvalidOperation(format!(
            "Slice ranges length {} must match input dimensionality {}",
            ranges.len(), input.shape().len()
        )));
    }
    
    for (i, range) in ranges.iter().enumerate() {
        if range.start > range.end {
            return Err(Error::InvalidOperation(format!(
                "Invalid slice range at dimension {}: start ({}) > end ({})",
                i, range.start, range.end
            )));
        }
        if range.end > input.shape()[i] {
            return Err(Error::InvalidIndex(vec![i]));
        }
    }

    let output_data = B::slice(&*input.data(), ranges)?;
    let requires_grad = input.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::Slice {
                input_shape: input.shape().to_vec(),
                ranges: ranges.to_vec(),
            },
            vec![input.clone()],
            |op_ctx, grad_output| {
                let grad_input = B::slice_backward(op_ctx, grad_output)?;
                Ok(vec![grad_input])
            },
        );
        output_tensor.set_op(op);
    }
    
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Inserts a new dimension of size 1 at the specified axis.
///
/// # Arguments
/// * `input` - The input tensor
/// * `axis` - The axis at which to insert the new dimension
///
/// # Returns
/// A new tensor with an additional dimension of size 1 at the specified axis
///
/// # Errors
/// Returns an error if the axis is invalid (greater than input.shape().len())
///
/// # Examples
/// ```
/// use rust_tensor_lib::{CpuTensor, ops, backend::Backend};
/// 
/// let tensor = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], false).unwrap();
/// // Insert a new dimension at axis 0
/// let expanded = ops::expand_dims(&tensor, 0).unwrap();
/// assert_eq!(expanded.shape(), &[1, 3]);
/// 
/// // Insert a new dimension at axis 1
/// let expanded = ops::expand_dims(&tensor, 1).unwrap();
/// assert_eq!(expanded.shape(), &[3, 1]);
/// ```
pub fn expand_dims<B: Backend>(input: &Tensor<B>, axis: usize) -> Result<Tensor<B>, Error> {
    if axis > input.shape().len() { // Allow axis == input.shape().len() for appending
        return Err(Error::InvalidIndex(vec![axis]));
    }
    let output_data = B::expand_dims(&*input.data(), axis)?;
    let requires_grad = input.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::ExpandDims { axis },
            vec![input.clone()],
            |op_ctx, grad_output| {
                B::expand_dims_backward(op_ctx, grad_output).map(|g| vec![g])
            },
        );
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Removes dimensions of size 1 from the tensor.
///
/// # Arguments
/// * `input` - The input tensor
/// * `axis` - The axis to squeeze out. If None, all dimensions of size 1 are removed.
///
/// # Returns
/// A new tensor with the specified dimension(s) of size 1 removed
///
/// # Errors
/// Returns an error if the axis is invalid or not of size 1
///
/// # Examples
/// ```
/// use rust_tensor_lib::{CpuTensor, ops, backend::Backend};
/// 
/// let tensor = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3, 1], false).unwrap();
/// // Squeeze specific axis
/// let squeezed = ops::squeeze(&tensor, Some(0)).unwrap();
/// assert_eq!(squeezed.shape(), &[3, 1]);
/// 
/// // Squeeze all dimensions of size 1
/// let squeezed = ops::squeeze(&tensor, None).unwrap();
/// assert_eq!(squeezed.shape(), &[3]);
/// ```
pub fn squeeze<B: Backend>(input: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, Error> {
    let original_input_shape = input.shape().to_vec(); // Capture before potential modification
    
    // Validate axis if Some
    if let Some(ax) = axis {
        if ax >= original_input_shape.len() {
            return Err(Error::InvalidIndex(vec![ax]));
        }
        if original_input_shape[ax] != 1 {
            return Err(Error::InvalidOperation(format!(
                "Cannot squeeze axis {} of shape {:?} (not size 1)", 
                ax, original_input_shape
            )));
        }
    }

    let output_data = B::squeeze(&*input.data(), axis)?;
    let requires_grad = input.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::Squeeze { axis, original_input_shape }, // Store original shape
            vec![input.clone()],
            |op_ctx, grad_output| {
                B::squeeze_backward(op_ctx, grad_output).map(|g| vec![g])
            },
        );
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Computes the sine of each element in the input tensor.
///
/// # Arguments
/// * `x` - The input tensor
///
/// # Returns
/// A new tensor with the sine of each element in the input tensor
///
/// # Errors
/// Returns an error if the backend fails to compute the sine
pub fn sin<B: Backend>(x: &Tensor<B>) -> Result<Tensor<B>, Error> {
    let output_data = B::sin(&*x.data())?;
    let requires_grad = x.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);
    if requires_grad {
        let op = Op::new(OpType::Sin, vec![x.clone()], |op_ctx, grad_output| {
            B::sin_backward(op_ctx, grad_output).map(|g| vec![g])
        });
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}

/// Clips the values of a tensor to be within [min_val, max_val].
///
/// # Arguments
/// * `input` - The input tensor
/// * `min_val` - The minimum value to clip to
/// * `max_val` - The maximum value to clip to
///
/// # Returns
/// A new tensor with all values clipped to the range [min_val, max_val]
///
/// # Errors
/// Returns an error if min_val > max_val or if the backend fails to compute the clip
///
/// # Examples
/// ```
/// use rust_tensor_lib::{CpuTensor, ops, backend::Backend, backend::cpu::CpuBackend};
/// 
/// let tensor = CpuTensor::from_vec(vec![-1.0, 0.5, 2.0], &[3], false).unwrap();
/// let clipped = ops::clip(&tensor, 0.0, 1.0).unwrap();
/// 
/// // Values are clipped to [0.0, 1.0]
/// let expected = CpuTensor::from_vec(vec![0.0, 0.5, 1.0], &[3], false).unwrap();
/// 
/// // Compare the values in the tensors
/// let clipped_storage = clipped.data();
/// let expected_storage = expected.data();
/// let clipped_data = clipped_storage.get_data();
/// let expected_data = expected_storage.get_data();
/// assert_eq!(clipped_data.shape(), expected_data.shape());
/// 
/// for (a, b) in clipped_data.iter().zip(expected_data.iter()) {
///     assert!((a - b).abs() < 1e-5);
/// }
/// ```
pub fn clip<B: Backend>(input: &Tensor<B>, min_val: f32, max_val: f32) -> Result<Tensor<B>, Error> {
    if min_val > max_val {
        return Err(Error::InvalidOperation("Clip min_val cannot be greater than max_val".to_string()));
    }
    let output_data = B::clip(&*input.data(), min_val, max_val)?;
    let requires_grad = input.requires_grad();
    let output_tensor = Tensor::new(output_data, requires_grad);

    if requires_grad {
        let op = Op::new(
            OpType::Clip { min_val, max_val }, // Store min_val, max_val
            vec![input.clone()],
            |op_ctx, grad_output| {
                B::clip_backward(op_ctx, grad_output).map(|g| vec![g])
            },
        );
        output_tensor.set_op(op);
    }
    output_tensor.run_hooks();
    Ok(output_tensor)
}
