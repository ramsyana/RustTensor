// src/ops/cpu_backward.rs
//! CPU implementations for backward operations, generic over the Backend trait.

use crate::array::Array;
use crate::error::Error;
use ndarray::ArrayD;

/// 2D max pooling backward (NCHW)
pub fn max_pool2d_backward(
    grad_output: &Array,           // [N, C, H_out, W_out]
    indices_storage: &Array,       // [N, C, H_out, W_out] (f32 indices)
    input_shape: &[usize],         // [N, C, H, W]
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Array, Error> {
    let (n, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let (k_h, k_w) = kernel_size;
    let (s_h, s_w) = stride;
    let (p_h, p_w) = padding;
    let h_out = (h + 2 * p_h - k_h) / s_h + 1;
    let w_out = (w + 2 * p_w - k_w) / s_w + 1;
    let mut grad_input = ArrayD::<f32>::zeros(vec![n, c, h, w]);
    let grad_output_nd = grad_output.get_data();
    let indices_nd = indices_storage.get_data();
    for b in 0..n {
        for ch in 0..c {
            for y_out in 0..h_out {
                for x_out in 0..w_out {
                    let g_out = grad_output_nd.get([b, ch, y_out, x_out]).copied().unwrap_or(0.0);
                    let max_flat_idx = indices_nd.get([b, ch, y_out, x_out]).copied().unwrap_or(0.0).round() as usize;
                    let ky = max_flat_idx / k_w;
                    let kx = max_flat_idx % k_w;
                    let in_y = y_out * s_h + ky;
                    let in_x = x_out * s_w + kx;
                    let in_y_pad = in_y as isize - p_h as isize;
                    let in_x_pad = in_x as isize - p_w as isize;
                    if in_y_pad >= 0 && in_y_pad < h as isize && in_x_pad >= 0 && in_x_pad < w as isize {
                        grad_input[&[b, ch, in_y_pad as usize, in_x_pad as usize][..]] += g_out;
                    }
                }
            }
        }
    }
    Ok(Array::new(grad_input))
}

use crate::backend::Backend;
use crate::graph::{Op, OpType};

// Note: These functions are now generic over B: Backend
// They return Vec<B::Storage> as expected by the Op definition

/// Compute gradients for sum reduction (y = sum(x, axis))
#[allow(dead_code)] // Function is only used through the Backend trait
pub(crate) fn sum_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage, // Gradient of the sum result (dL/dy)
) -> Result<Vec<B::Storage>, Error> {
    // Return Vec for consistency with Op struct
    // --- Input Validation ---
    if op.inputs.len() != 1 {
        return Err(Error::InvalidOperation(format!(
            "Sum operation backward expected 1 input, found {}",
            op.inputs.len()
        )));
    }
    let input_tensor = &op.inputs[0];
    let input_data_ref = input_tensor.data(); // Keep Ref alive
    let input_shape = B::shape(&*input_data_ref).to_vec(); // Clone shape for later use

    // Get the axis from the OpType
    let grad_input = match op.op_type {
        OpType::Sum(None) => {
            // Global sum: broadcast scalar gradient to input shape
            B::broadcast_to(output_grad, &input_shape)?
        }
        OpType::Sum(Some(axis)) => {
            // Axis sum: broadcast gradient along the reduced axis
            let output_shape = B::shape(output_grad);
            let mut target_reshape = output_shape.to_vec();

            // Insert size-1 axis at the reduced dimension's position
            if axis > target_reshape.len() {
                // Use > because insert happens at index
                return Err(Error::InvalidIndex(vec![axis]));
            }
            target_reshape.insert(axis, 1);

            // Reshape the gradient to insert the reduced axis with size 1
            let mut reshaped_grad = output_grad.clone();
            // Use B::set_shape to modify the storage's metadata
            B::set_shape(&mut reshaped_grad, &target_reshape)?;

            // Broadcast the reshaped gradient to the original input shape
            B::broadcast_to(&reshaped_grad, &input_shape)?
        }
        _ => {
            return Err(Error::InternalLogicError(format!(
                "Incorrect OpType ({}) passed to sum_backward. Expected Sum.",
                op.op_type
            )));
        }
    };

    Ok(vec![grad_input])
}

/// Compute gradients for matrix multiplication (C = A @ B)
pub(crate) fn matmul_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.len() != 2 {
        return Err(Error::InvalidOperation(
            "Matmul requires 2 inputs".to_string(),
        ));
    }
    let a_data_ref = op.inputs[0].data(); // Keep Ref alive
    let b_data_ref = op.inputs[1].data(); // Keep Ref alive
    let a = &*a_data_ref;
    let b = &*b_data_ref;

    // dA = dC @ B.T
    let b_t = B::transpose(b)?; // Call trait method
    let grad_a = B::matmul(output_grad, &b_t)?; // Call trait method

    // dB = A.T @ dC
    let a_t = B::transpose(a)?; // Call trait method
    let grad_b = B::matmul(&a_t, output_grad)?; // Call trait method

    Ok(vec![grad_a, grad_b])
}

/// Compute gradients for element-wise multiplication (C = A * B) - Handles Broadcasting
pub(crate) fn mul_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.len() != 2 {
        return Err(Error::InvalidOperation("Mul requires 2 inputs".to_string()));
    }
    let a_data_ref = op.inputs[0].data(); // Keep Ref alive
    let b_data_ref = op.inputs[1].data(); // Keep Ref alive
    let a = &*a_data_ref;
    let b = &*b_data_ref;
    let a_shape = B::shape(a); // Use B::shape
    let b_shape = B::shape(b); // Use B::shape

    // dA = dC * B
    let mut grad_a = B::mul(output_grad, b)?; // Use B::mul
                                              // dB = dC * A
    let mut grad_b = B::mul(output_grad, a)?; // Use B::mul

    // Handle broadcasting: Sum gradients along broadcasted dimensions
    grad_a = unbroadcast::<B>(grad_a, a_shape)?;
    grad_b = unbroadcast::<B>(grad_b, b_shape)?;

    Ok(vec![grad_a, grad_b])
}

/// Compute gradients for element-wise addition (C = A + B) - Handles Broadcasting
pub(crate) fn add_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.len() != 2 {
        return Err(Error::InvalidOperation("Add requires 2 inputs".to_string()));
    }
    let a_data_ref = op.inputs[0].data(); // Keep Ref alive
    let b_data_ref = op.inputs[1].data(); // Keep Ref alive
    let a_shape = B::shape(&*a_data_ref);
    let b_shape = B::shape(&*b_data_ref);

    // dA = dC, dB = dC initially
    let mut grad_a = output_grad.clone();
    let mut grad_b = output_grad.clone();

    // Handle broadcasting: Sum gradients along broadcasted dimensions
    grad_a = unbroadcast::<B>(grad_a, a_shape)?;
    grad_b = unbroadcast::<B>(grad_b, b_shape)?;

    Ok(vec![grad_a, grad_b])
}

/// Compute gradients for max reduction (y = max(x, axis))
#[allow(dead_code)]
pub(crate) fn max_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.len() != 1 {
        return Err(Error::InvalidOperation(format!(
            "Max operation backward expected 1 input, found {}",
            op.inputs.len()
        )));
    }

    let input_tensor = &op.inputs[0];
    let input_data_ref = input_tensor.data();
    let input_shape = B::shape(&*input_data_ref).to_vec();

    // Recompute max values to create the mask
    let axis = match op.op_type {
        OpType::Max(axis) => axis,
        _ => {
            return Err(Error::InternalLogicError(
                "Incorrect OpType in max_backward".to_string(),
            ))
        }
    };

    // Recompute the max values
    let max_values = B::max(&*input_data_ref, axis)?;

    // Create a mask where input equals max (after broadcasting)
    let max_broadcast = match axis {
        None => B::broadcast_to(&max_values, &input_shape)?,
        Some(ax) => {
            let mut expanded_shape = B::shape(&max_values).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_max = max_values;
            B::set_shape(&mut reshaped_max, &expanded_shape)?;
            B::broadcast_to(&reshaped_max, &input_shape)?
        }
    };

    // Create mask: 1.0 where x == max_val, 0.0 elsewhere
    let mask = B::equal(&*input_data_ref, &max_broadcast)?;

    // Count number of elements that achieved the max (for handling ties)
    let count = match axis {
        None => B::broadcast_to(&B::sum_along_axis(&mask, 0)?, &input_shape)?,
        Some(ax) => {
            let sum = B::sum_along_axis(&mask, ax)?;
            let mut expanded_shape = B::shape(&sum).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_sum = sum;
            B::set_shape(&mut reshaped_sum, &expanded_shape)?;
            B::broadcast_to(&reshaped_sum, &input_shape)?
        }
    };

    // Broadcast output_grad if needed
    let grad_broadcast = match axis {
        None => B::broadcast_to(output_grad, &input_shape)?,
        Some(ax) => {
            let mut expanded_shape = B::shape(output_grad).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_grad = output_grad.clone();
            B::set_shape(&mut reshaped_grad, &expanded_shape)?;
            B::broadcast_to(&reshaped_grad, &input_shape)?
        }
    };

    // Compute gradient: grad_input = output_grad * mask / count
    let grad_input = B::div(&B::mul(&grad_broadcast, &mask)?, &count)?;

    Ok(vec![grad_input])
}

/// Compute gradients for min reduction (y = min(x, axis))
#[allow(dead_code)]
pub(crate) fn min_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.len() != 1 {
        return Err(Error::InvalidOperation(format!(
            "Min operation backward expected 1 input, found {}",
            op.inputs.len()
        )));
    }

    let input_tensor = &op.inputs[0];
    let input_data_ref = input_tensor.data();
    let input_shape = B::shape(&*input_data_ref).to_vec();

    // Recompute min values to create the mask
    let axis = match op.op_type {
        OpType::Min(axis) => axis,
        _ => {
            return Err(Error::InternalLogicError(
                "Incorrect OpType in min_backward".to_string(),
            ))
        }
    };

    // Recompute the min values
    let min_values = B::min(&*input_data_ref, axis)?;

    // Create a mask where input equals min (after broadcasting)
    let min_broadcast = match axis {
        None => B::broadcast_to(&min_values, &input_shape)?,
        Some(ax) => {
            let mut expanded_shape = B::shape(&min_values).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_min = min_values;
            B::set_shape(&mut reshaped_min, &expanded_shape)?;
            B::broadcast_to(&reshaped_min, &input_shape)?
        }
    };

    // Create mask: 1.0 where x == min_val, 0.0 elsewhere
    let mask = B::equal(&*input_data_ref, &min_broadcast)?;

    // Count number of elements that achieved the min (for handling ties)
    let count = match axis {
        None => B::broadcast_to(&B::sum_along_axis(&mask, 0)?, &input_shape)?,
        Some(ax) => {
            let sum = B::sum_along_axis(&mask, ax)?;
            let mut expanded_shape = B::shape(&sum).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_sum = sum;
            B::set_shape(&mut reshaped_sum, &expanded_shape)?;
            B::broadcast_to(&reshaped_sum, &input_shape)?
        }
    };

    // Broadcast output_grad if needed
    let grad_broadcast = match axis {
        None => B::broadcast_to(output_grad, &input_shape)?,
        Some(ax) => {
            let mut expanded_shape = B::shape(output_grad).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_grad = output_grad.clone();
            B::set_shape(&mut reshaped_grad, &expanded_shape)?;
            B::broadcast_to(&reshaped_grad, &input_shape)?
        }
    };

    // Compute gradient: grad_input = output_grad * mask / count
    let grad_input = B::div(&B::mul(&grad_broadcast, &mask)?, &count)?;

    Ok(vec![grad_input])
}

/// Compute gradients for product reduction (y = prod(x, axis))
#[allow(dead_code)]
pub(crate) fn prod_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.len() != 1 {
        return Err(Error::InvalidOperation(format!(
            "Product operation backward expected 1 input, found {}",
            op.inputs.len()
        )));
    }

    let input_tensor = &op.inputs[0];
    let input_data_ref = input_tensor.data();
    let input_shape = B::shape(&*input_data_ref).to_vec();

    // Get the axis from the OpType
    let axis = match op.op_type {
        OpType::Prod(axis) => axis,
        _ => {
            return Err(Error::InternalLogicError(
                "Incorrect OpType in prod_backward".to_string(),
            ))
        }
    };

    // Recompute the product
    let prod_values = B::prod(&*input_data_ref, axis)?;

    // Broadcast the product if needed
    let prod_broadcast = match axis {
        None => B::broadcast_to(&prod_values, &input_shape)?,
        Some(ax) => {
            let mut expanded_shape = B::shape(&prod_values).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_prod = prod_values;
            B::set_shape(&mut reshaped_prod, &expanded_shape)?;
            B::broadcast_to(&reshaped_prod, &input_shape)?
        }
    };

    // Broadcast output_grad if needed
    let grad_broadcast = match axis {
        None => B::broadcast_to(output_grad, &input_shape)?,
        Some(ax) => {
            let mut expanded_shape = B::shape(output_grad).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_grad = output_grad.clone();
            B::set_shape(&mut reshaped_grad, &expanded_shape)?;
            B::broadcast_to(&reshaped_grad, &input_shape)?
        }
    };

    // Compute gradient: grad_input = output_grad * prod / x
    // Handle zeros carefully to avoid division by zero
    let epsilon = 1e-10; // Small constant to avoid division by zero
    let safe_input = B::add(&*input_data_ref, &B::from_vec(vec![epsilon], &[1])?)?;
    let grad_input = B::div(&B::mul(&grad_broadcast, &prod_broadcast)?, &safe_input)?;

    Ok(vec![grad_input])
}

/// Compute gradients for logsumexp reduction (y = logsumexp(x, axis))
#[allow(dead_code)]
pub(crate) fn logsumexp_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.len() != 1 {
        return Err(Error::InvalidOperation(format!(
            "LogSumExp operation backward expected 1 input, found {}",
            op.inputs.len()
        )));
    }

    let input_tensor = &op.inputs[0];
    let input_data_ref = input_tensor.data();
    let input_shape = B::shape(&*input_data_ref).to_vec();

    // Get the axis from the OpType
    let axis = match op.op_type {
        OpType::LogSumExp(axis) => axis,
        _ => {
            return Err(Error::InternalLogicError(
                "Incorrect OpType in logsumexp_backward".to_string(),
            ))
        }
    };

    // Recompute logsumexp
    let lse = B::logsumexp(&*input_data_ref, axis)?;

    // Broadcast logsumexp if needed
    let lse_broadcast = match axis {
        None => B::broadcast_to(&lse, &input_shape)?,
        Some(ax) => {
            let mut expanded_shape = B::shape(&lse).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_lse = lse;
            B::set_shape(&mut reshaped_lse, &expanded_shape)?;
            B::broadcast_to(&reshaped_lse, &input_shape)?
        }
    };

    // Broadcast output_grad if needed
    let grad_broadcast = match axis {
        None => B::broadcast_to(output_grad, &input_shape)?,
        Some(ax) => {
            let mut expanded_shape = B::shape(output_grad).to_vec();
            expanded_shape.insert(ax, 1);
            let mut reshaped_grad = output_grad.clone();
            B::set_shape(&mut reshaped_grad, &expanded_shape)?;
            B::broadcast_to(&reshaped_grad, &input_shape)?
        }
    };

    // Compute softmax: exp(x - logsumexp)
    let shifted = B::sub(&*input_data_ref, &lse_broadcast)?;
    let softmax = B::exp(&shifted)?;

    // Compute gradient: grad_input = output_grad * softmax
    let grad_input = B::mul(&grad_broadcast, &softmax)?;

    Ok(vec![grad_input])
}

/// Compute gradients for element-wise power function (y = a^b)
#[allow(dead_code)] // Function is only used through the Backend trait
pub(crate) fn powf_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<(B::Storage, B::Storage), Error> {
    if op.inputs.len() != 2 {
        return Err(Error::InvalidOperation(format!(
            "powf_backward expects 2 inputs, got {}",
            op.inputs.len()
        )));
    }

    let a_data_ref = op.inputs[0].data();
    let b_data_ref = op.inputs[1].data();
    let a = &*a_data_ref;
    let b = &*b_data_ref;

    // Keep original shapes for unbroadcasting
    let a_shape = B::shape(a);
    let b_shape = B::shape(b);

    // Calculate a^b for use in the gradient computation
    let a_pow_b = B::powf(a, b)?;

    // Calculate gradient for a: dL/da = dL/dy * b * a^(b-1)
    // First, compute b - 1
    let ones = B::from_vec(vec![1.0f32], &[])?;
    let ones_broadcast = B::broadcast_to(&ones, b_shape)?;
    let b_minus_one = B::sub(b, &ones_broadcast)?;

    // Then compute a^(b-1)
    let a_pow_b_minus_one = B::powf(a, &b_minus_one)?;

    // Then multiply by b
    let b_times_a_pow_b_minus_one = B::mul(b, &a_pow_b_minus_one)?;

    // Finally multiply by output_grad
    let grad_a_potentially_broadcasted = B::mul(output_grad, &b_times_a_pow_b_minus_one)?;

    // Calculate gradient for b: dL/db = dL/dy * a^b * ln(a)
    // First calculate ln(a)
    let ln_a = B::ln(a)?;

    // Multiply ln(a) by a^b
    let a_pow_b_times_ln_a = B::mul(&a_pow_b, &ln_a)?;

    // Finally multiply by output_grad
    let grad_b_potentially_broadcasted = B::mul(output_grad, &a_pow_b_times_ln_a)?;

    // Apply unbroadcast to correctly handle broadcasting
    let grad_a = unbroadcast::<B>(grad_a_potentially_broadcasted, a_shape)?;
    let grad_b = unbroadcast::<B>(grad_b_potentially_broadcasted, b_shape)?;

    Ok((grad_a, grad_b))
}

/// Helper function to reverse broadcasting by summing along axes.
pub(crate) fn unbroadcast<B: Backend>(
    grad: B::Storage,
    target_shape: &[usize],
) -> Result<B::Storage, Error> {
    // Get shape directly from the owned storage before modifying it
    let grad_shape = B::shape(&grad).to_vec(); // Clone the shape

    println!(
        "[DEBUG unbroadcast] Initial grad shape: {:?}, target_shape: {:?}",
        grad_shape, target_shape
    );
    println!("[DEBUG unbroadcast] Initial grad data: {:?}", &grad);

    if grad_shape.as_slice() == target_shape {
        println!("[DEBUG unbroadcast] Shapes already match, returning original grad");
        return Ok(grad);
    }

    let mut current_grad = grad; // Work with the owned storage
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();
    let mut current_shape_vec = grad_shape; // Use the cloned shape

    // --- Sum leading dimensions ---
    if grad_ndim > target_ndim {
        let axes_to_sum: Vec<usize> = (0..(grad_ndim - target_ndim)).collect();
        println!(
            "[DEBUG unbroadcast] Summing leading dimensions: {:?}",
            axes_to_sum
        );
        for &axis in axes_to_sum.iter().rev() {
            // Pass owned current_grad by reference, get new owned storage back
            current_grad = B::sum_along_axis(&current_grad, axis)?;
            // Update logical shape
            if axis < current_shape_vec.len() {
                current_shape_vec.remove(axis);
            }
        }
        println!(
            "[DEBUG unbroadcast] After summing leading dimensions - shape: {:?}, data: {:?}",
            B::shape(&current_grad),
            &current_grad
        );
    }

    // --- Sum dimensions where target is 1 ---
    let current_ndim = current_shape_vec.len();
    let effective_target_ndim = target_shape.len();
    let mut axes_to_sum_for_size_1 = Vec::new();

    let pad_len = effective_target_ndim.saturating_sub(current_ndim);
    for i in 0..effective_target_ndim {
        let target_dim = target_shape[i];
        let current_dim = if i < pad_len {
            1
        } else {
            current_shape_vec[i - pad_len]
        };

        if target_dim == 1 && current_dim != 1 {
            let current_axis_index = i.saturating_sub(pad_len);
            if current_axis_index < current_ndim {
                axes_to_sum_for_size_1.push(current_axis_index);
            } else { /* error */
            }
        } else if target_dim != current_dim && current_dim != 1 { /* error */
        }
    }

    println!(
        "[DEBUG unbroadcast] Axes to sum for size 1: {:?}",
        axes_to_sum_for_size_1
    );
    axes_to_sum_for_size_1.sort_unstable_by(|a, b| b.cmp(a));
    axes_to_sum_for_size_1.dedup();

    // Perform the summing
    for axis in axes_to_sum_for_size_1 {
        // The axis index refers to the shape *before* this loop started summing size-1 dims.
        // We need to track the current shape *during* the loop or adjust the axis.
        // Let's track the shape:
        let current_loop_shape = B::shape(&current_grad).to_vec(); // Get shape in loop
        println!(
            "[DEBUG unbroadcast] Current shape before summing axis {}: {:?}",
            axis, current_loop_shape
        );
        if axis < current_loop_shape.len() {
            current_grad = B::sum_along_axis(&current_grad, axis)?;
            println!(
                "[DEBUG unbroadcast] After summing axis {} - new shape: {:?}, data: {:?}",
                axis,
                B::shape(&current_grad),
                &current_grad
            );
        } else {
            // This means the axis index from the *original* dimension list
            // is now invalid because previous sums removed dimensions *before* it.
            // We need to adjust the axis based on how many dimensions before it were removed.
            // This gets complicated. A simpler way might be to always sum from highest axis down.
            // Since we sorted descending, this *should* handle it, but let's add a safety check/warning.
            println!(
                "Warning: Skipping unbroadcast sum axis {} (likely already removed for shape {:?})",
                axis, current_loop_shape
            );
        }
    }

    // --- Final reshape if necessary ---
    // Get the shape *before* attempting the mutable borrow
    let final_shape_vec = B::shape(&current_grad).to_vec();
    let final_size = final_shape_vec.iter().product::<usize>().max(1);
    let target_size = target_shape.iter().product::<usize>().max(1);

    println!(
        "[DEBUG unbroadcast] Final shape before reshape: {:?}, target: {:?}",
        final_shape_vec, target_shape
    );
    println!(
        "[DEBUG unbroadcast] Final size: {}, target size: {}",
        final_size, target_size
    );

    if final_shape_vec.as_slice() == target_shape {
        println!("[DEBUG unbroadcast] Shapes match, returning current grad");
        Ok(current_grad)
    } else if final_size == target_size {
        // Sizes match, attempt reshape using Backend::set_shape
        println!("[DEBUG unbroadcast] Sizes match, reshaping to target shape");
        match B::set_shape(&mut current_grad, target_shape) {
            // Mutable borrow here is okay
            Ok(()) => {
                println!(
                    "[DEBUG unbroadcast] Reshape successful, final data: {:?}",
                    &current_grad
                );
                Ok(current_grad)
            }
            Err(e) => Err(Error::InternalLogicError(format!(
                "unbroadcast: Failed final reshape from {:?} to {:?} (size match): {}",
                final_shape_vec,
                target_shape,
                e // Use the captured shape vec here
            ))),
        }
    } else {
        println!(
            "[DEBUG unbroadcast] Size mismatch error: {:?} vs {:?}",
            final_shape_vec, target_shape
        );
        Err(Error::IncompatibleShapes {
            op: "unbroadcast (final shape/size)".to_string(),
            shape_a: final_shape_vec, // Use the captured shape vec here
            shape_b: target_shape.to_vec(),
        })
    }
}

/// Compute gradients for mean reduction (y = mean(x) over axis or globally)
pub(crate) fn mean_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.len() != 1 {
        return Err(Error::InvalidOperation("Mean requires 1 input".to_string()));
    }
    let input_data_ref = op.inputs[0].data();
    let input_storage = &*input_data_ref;
    let input_shape = B::shape(input_storage).to_vec(); // Clone shape

    println!("[DEBUG mean_backward] Op: {:?}", op.op_type);
    println!("[DEBUG mean_backward] Input shape: {:?}", input_shape);
    println!(
        "[DEBUG mean_backward] Output grad shape: {:?}",
        B::shape(output_grad)
    );
    println!("[DEBUG mean_backward] Output grad data: {:?}", output_grad);

    // Get the axis from the OpType
    let grad_input = match op.op_type {
        OpType::Mean(None) => {
            // Global mean: broadcast scalar gradient to input shape and divide by total size
            let size = B::size(input_storage).max(1) as f32; // Use max(1) for scalar case
            println!("[DEBUG mean_backward] Global mean with size: {}", size);

            if size == 0.0 {
                return Ok(vec![B::zeros(&input_shape)?]);
            }
            let grad_broadcast = B::broadcast_to(output_grad, &input_shape)?;
            println!(
                "[DEBUG mean_backward] Broadcast grad shape: {:?}",
                B::shape(&grad_broadcast)
            );

            let result = B::div_scalar(&grad_broadcast, size)?;
            println!(
                "[DEBUG mean_backward] Final grad shape: {:?}, data: {:?}",
                B::shape(&result),
                result
            );
            result
        }
        OpType::Mean(Some(axis)) => {
            // Axis mean: broadcast gradient along the reduced axis and divide by axis size
            if axis >= input_shape.len() {
                return Err(Error::InvalidIndex(vec![axis]));
            }
            let axis_size = input_shape[axis].max(1) as f32; // Use max(1) for dim size 0
            println!(
                "[DEBUG mean_backward] Axis mean (axis: {}) with axis_size: {}",
                axis, axis_size
            );

            if axis_size == 0.0 {
                return Ok(vec![B::zeros(&input_shape)?]);
            }

            // Reshape the gradient to insert the reduced axis with size 1
            let output_shape = B::shape(output_grad);
            let mut target_reshape = output_shape.to_vec();
            if axis > target_reshape.len() {
                // Use > because insert happens at index
                return Err(Error::InvalidIndex(vec![axis]));
            }
            target_reshape.insert(axis, 1);
            println!("[DEBUG mean_backward] Target reshape: {:?}", target_reshape);

            let mut reshaped_grad = output_grad.clone();
            B::set_shape(&mut reshaped_grad, &target_reshape)?;
            println!(
                "[DEBUG mean_backward] Reshaped grad shape: {:?}",
                B::shape(&reshaped_grad)
            );

            // Broadcast the reshaped gradient to the original input shape
            let broadcasted = B::broadcast_to(&reshaped_grad, &input_shape)?;
            println!(
                "[DEBUG mean_backward] Broadcasted grad shape: {:?}",
                B::shape(&broadcasted)
            );

            // Divide by the size of the reduced dimension
            let result = B::div_scalar(&broadcasted, axis_size)?;
            println!(
                "[DEBUG mean_backward] Final result shape: {:?}",
                B::shape(&result)
            );
            result
        }
        _ => {
            return Err(Error::InternalLogicError(format!(
                "Incorrect OpType ({}) passed to mean_backward. Expected Mean.",
                op.op_type
            )));
        }
    };

    Ok(vec![grad_input])
}

/// Compute gradients for ReLU activation (y = max(0, x))
pub(crate) fn relu_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.is_empty() {
        return Err(Error::InvalidOperation("Relu requires 1 input".to_string()));
    }
    let input_data_ref = op.inputs[0].data(); // Keep Ref alive
    let input = &*input_data_ref;

    // dx = dC * (1 if x > 0 else 0)
    let mask = B::map(input, |x| if x > 0.0 { 1.0 } else { 0.0 })?; // Use B::map
    let grad_input = B::mul(output_grad, &mask)?; // Use B::mul

    Ok(vec![grad_input])
}

/// Compute gradients for log-softmax (y = log_softmax(x))
pub(crate) fn log_softmax_backward<B: Backend>(
    op: &Op<B>,
    output_grad: &B::Storage,
) -> Result<Vec<B::Storage>, Error> {
    if op.inputs.is_empty() {
        return Err(Error::InvalidOperation(
            "LogSoftmax requires 1 input".to_string(),
        ));
    }
    let input_data_ref = op.inputs[0].data(); // Keep Ref alive
    let input_data = &*input_data_ref;
    let input_shape = B::shape(input_data);
    let axis = op.axis()?;

    // Recompute LogSoftmax output 'y' needed for the gradient formula
    let output = B::log_softmax(input_data, axis)?;

    // dx = dC - exp(y) * sum(dC along axis)
    // where exp(y) is softmax(x) = p
    let p = B::exp(&output)?;

    // Calculate sum(dC) along the specified axis
    let sum_grad = B::sum_along_axis(output_grad, axis)?; // Shape is now reduced

    // Reshape sum_grad to be compatible for broadcasting (insert singleton dim at axis)
    let mut intermediate_shape = B::shape(&sum_grad).to_vec(); // Shape of the reduced tensor
    if axis < input_shape.len() {
        // Check axis validity again
        intermediate_shape.insert(axis, 1); // Insert singleton dimension
    } else {
        return Err(Error::InvalidIndex(vec![axis]));
    }

    let mut sum_grad_reshaped = sum_grad; // Take ownership
    B::set_shape(&mut sum_grad_reshaped, &intermediate_shape)?; // Reshape metadata

    // Broadcast the *reshaped* sum_grad back to the original input shape
    let sum_grad_expanded = B::broadcast_to(&sum_grad_reshaped, input_shape)?;

    // grad_input = output_grad - p * sum_grad_expanded
    let term2 = B::mul(&p, &sum_grad_expanded)?;
    let grad_input = B::sub(output_grad, &term2)?;

    Ok(vec![grad_input])
}

// Helper to get axis from Op safely (already defined in the target code)
impl<B: Backend> Op<B> {
    pub(crate) fn axis(&self) -> Result<usize, Error> {
        // Make pub(crate)
        match self.op_type {
            OpType::LogSoftmax(axis) => Ok(axis),
            // Add other axis-based ops like Mean(axis) here if needed
            _ => Err(Error::InvalidOperation(format!(
                "Op {} does not have an axis",
                self.op_type
            ))),
        }
    }
}
