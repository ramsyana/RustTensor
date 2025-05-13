use crate::array::Array;
use crate::error::Error;
use ndarray::{ArrayD, Axis};
use ndarray::s;

/// 2D max pooling (NCHW). Returns (output, indices)
pub fn max_pool2d(
    input: &Array,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<(Array, Array), Error> {
    let input_shape = input.shape(); // [N, C, H, W]
    if input_shape.len() != 4 {
        return Err(Error::InvalidOperation("max_pool2d expects 4D input (NCHW)".to_string()));
    }
    let (n, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let (k_h, k_w) = kernel_size;
    let (s_h, s_w) = stride;
    let (p_h, p_w) = padding;
    let h_out = (h + 2 * p_h - k_h) / s_h + 1;
    let w_out = (w + 2 * p_w - k_w) / s_w + 1;
    let mut output = ndarray::ArrayD::<f32>::zeros(vec![n, c, h_out, w_out]);
    let mut indices = ndarray::ArrayD::<f32>::zeros(vec![n, c, h_out, w_out]);
    let input_nd = input.get_data();
    for b in 0..n {
        for ch in 0..c {
            for y_out in 0..h_out {
                for x_out in 0..w_out {
                    let y_start = y_out * s_h as usize;
                    let x_start = x_out * s_w as usize;
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;
                    let mut idx_in_window = 0;
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let in_y = y_start + ky as usize;
                            let in_x = x_start + kx as usize;
                            let in_y_pad = in_y as isize - p_h as isize;
                            let in_x_pad = in_x as isize - p_w as isize;
                            let val = if in_y_pad >= 0 && in_y_pad < h as isize && in_x_pad >= 0 && in_x_pad < w as isize {
                                input_nd[&[b, ch, in_y_pad as usize, in_x_pad as usize][..]]
                            } else {
                                f32::NEG_INFINITY
                            };
                            if val > max_val {
                                max_val = val;
                                max_idx = idx_in_window;
                            }
                            idx_in_window += 1;
                        }
                    }
                    let idx = vec![b, ch, y_out, x_out];
                    output[idx.as_slice()] = max_val;
                    indices[idx.as_slice()] = max_idx as f32;
                }
            }
        }
    }
    Ok((Array::new(output), Array::new(indices)))
}

/// Matrix multiplication of two tensors
pub fn matmul(a: &Array, b: &Array) -> Result<Array, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    // Ensure inputs are 2D - Use into_dimensionality which returns Result
    let a_2d = a_data.view().into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| Error::ShapeError(format!("Input 'a' is not 2D: {}", e)))?;
    let b_2d = b_data.view().into_dimensionality::<ndarray::Ix2>()
        .map_err(|e| Error::ShapeError(format!("Input 'b' is not 2D: {}", e)))?;

    // Check shape compatibility for matmul: (m, k) x (k, n)
    if a_2d.shape()[1] != b_2d.shape()[0] {
        return Err(Error::IncompatibleShapes {
            op: "matmul".to_string(),
            shape_a: a_2d.shape().to_vec(),
            shape_b: b_2d.shape().to_vec(),
        });
    }

    let result_2d = a_2d.dot(&b_2d);
    Ok(Array::new(result_2d.into_dyn()))
}

/// 2D convolution (NCHW, im2col+matmul implementation).
pub fn conv2d(
    input: &Array,
    weights: &Array,
    bias: Option<&Array>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Array, Error> {
    // Shapes
    let input_shape = input.shape(); // [N, C_in, H_in, W_in]
    let weight_shape = weights.shape(); // [C_out, C_in, K_h, K_w]
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return Err(Error::InvalidOperation("conv2d expects 4D input and weights".to_string()));
    }
    let (n, c_in, h_in, w_in) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let (c_out, c_in_w, k_h, k_w) = (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]);
    if c_in != c_in_w {
        return Err(Error::InvalidOperation("conv2d: input channels != weight channels".to_string()));
    }
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
    // im2col: output shape [N, C_in * K_h * K_w, H_out * W_out]
    let mut cols = ndarray::Array3::<f32>::zeros((n, c_in * k_h * k_w, h_out * w_out));
    let input_nd = input.get_data();
    for b in 0..n {
        let img = input_nd.slice(s![b, .., .., ..]);
        let mut col = cols.slice_mut(s![b, .., ..]);
        let mut col_idx = 0;
        for y in 0..h_out {
            for x in 0..w_out {
                let y_start = y as isize * stride_h as isize - pad_h as isize;
                let x_start = x as isize * stride_w as isize - pad_w as isize;
                let mut patch = Vec::with_capacity(c_in * k_h * k_w);
                for c in 0..c_in {
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let in_y = y_start + ky as isize;
                            let in_x = x_start + kx as isize;
                            let val = if in_y >= 0 && (in_y as usize) < h_in && in_x >= 0 && (in_x as usize) < w_in {
                                img[(c, in_y as usize, in_x as usize)]
                            } else {
                                0.0
                            };
                            patch.push(val);
                        }
                    }
                }
                for (i, v) in patch.into_iter().enumerate() {
                    col[(i, col_idx)] = v;
                }
                col_idx += 1;
            }
        }
    }
    // Reshape weights to [C_out, C_in * K_h * K_w]
    let weights_2d = weights.get_data().clone().into_shape_with_order((c_out, c_in * k_h * k_w)).unwrap();
    // Output: [N, C_out, H_out * W_out]
    let mut out = ndarray::Array3::<f32>::zeros((n, c_out, h_out * w_out));
    for b in 0..n {
        let col = cols.slice(s![b, .., ..]);
        let result = weights_2d.dot(&col);
        out.slice_mut(s![b, .., ..]).assign(&result);
    }
    // Add bias if present
    if let Some(bias_arr) = bias {
        let bias_data = bias_arr.get_data();
        if bias_data.len() != c_out {
            return Err(Error::InvalidOperation("conv2d: bias shape mismatch".to_string()));
        }
        for b in 0..n {
            for co in 0..c_out {
                for idx in 0..(h_out * w_out) {
                    out[(b, co, idx)] += bias_data[co];
                }
            }
        }
    }
    // Reshape output to [N, C_out, H_out, W_out]
    let mut out4d = ndarray::Array4::<f32>::zeros((n, c_out, h_out, w_out));
    for b in 0..n {
        for co in 0..c_out {
            for idx in 0..(h_out * w_out) {
                let y = idx / w_out;
                let x = idx % w_out;
                out4d[(b, co, y, x)] = out[(b, co, idx)];
            }
        }
    }
    Ok(Array::new(out4d.into_dyn()))
}

/// Backward for conv2d (NCHW, im2col-based implementation).
pub fn conv2d_backward(
    input: &Array,
    weights: &Array,
    grad_output: &Array,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<(Array, Array, Option<Array>), Error> {
    // Shapes
    let input_shape = input.shape(); // [N, C_in, H_in, W_in]
    let weight_shape = weights.shape(); // [C_out, C_in, K_h, K_w]
    let _grad_out_shape = grad_output.shape(); // [N, C_out, H_out, W_out]
    let (n, c_in, h_in, w_in) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
    let (c_out, _, k_h, k_w) = (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]);
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let h_out = (h_in + 2 * pad_h - k_h) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - k_w) / stride_w + 1;
    // im2col for input
    let mut cols = ndarray::Array3::<f32>::zeros((n, c_in * k_h * k_w, h_out * w_out));
    let input_nd = input.get_data();
    for b in 0..n {
        let img = input_nd.slice(s![b, .., .., ..]);
        let mut col = cols.slice_mut(s![b, .., ..]);
        let mut col_idx = 0;
        for y in 0..h_out {
            for x in 0..w_out {
                let y_start = y as isize * stride_h as isize - pad_h as isize;
                let x_start = x as isize * stride_w as isize - pad_w as isize;
                let mut patch = Vec::with_capacity(c_in * k_h * k_w);
                for c in 0..c_in {
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let in_y = y_start + ky as isize;
                            let in_x = x_start + kx as isize;
                            let val = if in_y >= 0 && (in_y as usize) < h_in && in_x >= 0 && (in_x as usize) < w_in {
                                img[(c, in_y as usize, in_x as usize)]
                            } else {
                                0.0
                            };
                            patch.push(val);
                        }
                    }
                }
                for (i, v) in patch.into_iter().enumerate() {
                    col[(i, col_idx)] = v;
                }
                col_idx += 1;
            }
        }
    }
    // Reshape grad_output to [N, C_out, H_out*W_out]
    let grad_out_nd = grad_output.get_data();
    let mut grad_out_3d = ndarray::Array3::<f32>::zeros((n, c_out, h_out * w_out));
    for b in 0..n {
        for co in 0..c_out {
            for idx in 0..(h_out * w_out) {
                let y = idx / w_out;
                let x = idx % w_out;
                grad_out_3d[(b, co, idx)] = grad_out_nd[[b, co, y, x]];
            }
        }
    }
    // dW: [C_out, C_in * K_h * K_w]
    let mut grad_w = ndarray::Array2::<f32>::zeros((c_out, c_in * k_h * k_w));
    for b in 0..n {
        let col = cols.slice(s![b, .., ..]);
        let grad_out = grad_out_3d.slice(s![b, .., ..]);
        grad_w = &grad_w + &grad_out.dot(&col.t());
    }
    let grad_w = grad_w.into_shape_with_order((c_out, c_in, k_h, k_w)).unwrap();
    // db: [C_out]
    let mut grad_b = ndarray::Array1::<f32>::zeros(c_out);
    for b in 0..n {
        for co in 0..c_out {
            grad_b[co] += grad_out_3d.slice(s![b, co, ..]).sum();
        }
    }
    // dInput: full convolution (col2im)
    let mut grad_input = ndarray::Array4::<f32>::zeros((n, c_in, h_in, w_in));
    let weights_2d = weights.get_data().clone().into_shape_with_order((c_out, c_in * k_h * k_w)).unwrap();
    for b in 0..n {
        let grad_out = grad_out_3d.slice(s![b, .., ..]);
        let _grad_out_t = grad_out.t(); // [H_out*W_out, C_out]
        let grad_cols = weights_2d.t().dot(&grad_out);
        // col2im
        let col_idx = 0;
        for y in 0..h_out {
            for x in 0..w_out {
                let y_start = y as isize * stride_h as isize - pad_h as isize;
                let x_start = x as isize * stride_w as isize - pad_w as isize;
                let col_slice = grad_cols.column(col_idx);
                let mut idx = 0;
                for c in 0..c_in {
                    for ky in 0..k_h {
                        for kx in 0..k_w {
                            let in_y = y_start + ky as isize;
                            let in_x = x_start + kx as isize;
                            if in_y >= 0 && in_y < h_in as isize && in_x >= 0 && in_x < w_in as isize {
                                grad_input[(b, c, in_y as usize, in_x as usize)] += col_slice[idx];
                            }
                            idx += 1;
                        }
                    }
                }
            }
        }
    }
    Ok((
        Array::new(grad_input.into_dyn()),
        Array::new(grad_w.into_dyn()),
        Some(Array::new(grad_b.into_dyn())),
    ))
}

/// Element-wise multiplication of two tensors with broadcasting support
pub fn mul(a: &Array, b: &Array) -> Result<Array, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();


    // Get maximum dimensionality
    let a_shape = a_data.shape();
    let b_shape = b_data.shape();
    let ndim = a_shape.len().max(b_shape.len());

    // Adjust arrays to have same number of dimensions
    let a_view = a_data.view();
    let mut a_adjusted = a_view.clone();
    for _ in 0..(ndim - a_view.ndim()) {
        a_adjusted = a_adjusted.insert_axis(Axis(0));
    }

    let b_view = b_data.view();
    let mut b_adjusted = b_view.clone();
    for _ in 0..(ndim - b_view.ndim()) {
        b_adjusted = b_adjusted.insert_axis(Axis(0));
    }

    // Calculate broadcast shape
    let mut broadcast_shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let a_dim = a_adjusted.shape()[i];
        let b_dim = b_adjusted.shape()[i];
        if a_dim == b_dim {
            broadcast_shape.push(a_dim);
        } else if a_dim == 1 || b_dim == 1 {
            broadcast_shape.push(a_dim.max(b_dim));
        } else {
            return Err(Error::IncompatibleShapes {
                op: String::from("mul"),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            });
        }
    }

    // Attempt to broadcast arrays
    let a_broadcast =
        a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or(Error::IncompatibleShapes {
                op: String::from("mul"),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            })?;
    let b_broadcast = b_adjusted
        .broadcast(broadcast_shape)
        .ok_or(Error::IncompatibleShapes {
            op: String::from("mul"),
            shape_a: a_data.shape().to_vec(),
            shape_b: b_data.shape().to_vec(),
        })?;

    Ok(Array::new(&a_broadcast * &b_broadcast))
}

/// Element-wise addition of two tensors with broadcasting support
pub fn add(a: &Array, b: &Array) -> Result<Array, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    // Get maximum dimensionality
    let a_shape = a_data.shape();
    let b_shape = b_data.shape();
    let ndim = a_shape.len().max(b_shape.len());

    // Adjust arrays to have same number of dimensions
    let a_view = a_data.view();
    let mut a_adjusted = a_view.clone();
    for _ in 0..(ndim - a_view.ndim()) {
        a_adjusted = a_adjusted.insert_axis(Axis(0));
    }

    let b_view = b_data.view();
    let mut b_adjusted = b_view.clone();
    for _ in 0..(ndim - b_view.ndim()) {
        b_adjusted = b_adjusted.insert_axis(Axis(0));
    }

    // Calculate broadcast shape
    let mut broadcast_shape = vec![0; ndim];
    for i in 0..ndim {
        let a_dim = if i < a_shape.len() { a_shape[i] } else { 1 };
        let b_dim = if i < b_shape.len() { b_shape[i] } else { 1 };
        if a_dim == 1 || b_dim == 1 {
            broadcast_shape[i] = a_dim.max(b_dim);
        } else if a_dim == b_dim {
            broadcast_shape[i] = a_dim;
        } else {
            return Err(Error::IncompatibleShapes {
                op: String::from("add"),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            });
        }
    }

    // Broadcast both arrays to the target shape
    let a_broadcast =
        a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or(Error::IncompatibleShapes {
                op: String::from("add"),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            })?;
    let b_broadcast = b_adjusted
        .broadcast(broadcast_shape)
        .ok_or(Error::IncompatibleShapes {
            op: String::from("add"),
            shape_a: a_data.shape().to_vec(),
            shape_b: b_data.shape().to_vec(),
        })?;

    Ok(Array::new(&a_broadcast + &b_broadcast))
}

/// Mean of tensor along specified axis
pub fn mean(a: &Array, axis: Option<usize>) -> Result<Array, Error> {
    let a_data = a.get_data();

    match axis {
        Some(ax) => {
            if ax >= a_data.ndim() {
                return Err(Error::InvalidIndex(vec![ax]));
            }
            // mean_axis returns Option<Array> - handle None case (e.g., axis size 0)
            a_data
                .mean_axis(Axis(ax))
                .map(|arr| Array::new(arr.into_dyn()))
                .ok_or_else(|| Error::InvalidOperation(format!("Mean failed along axis {}", ax)))
        }
        None => {
            // mean returns Option<f32> - handle None case (e.g., empty tensor)
            a_data
                .mean()
                .map(|mean_val| Array::new(ArrayD::from_elem(vec![], mean_val)))
                .ok_or(Error::EmptyTensor)
        }
    }
}

/// ReLU activation function
pub fn relu(a: &Array) -> Result<Array, Error> {
    let a_data = a.get_data();
    if a_data.is_empty() {
        return Ok(Array::zeros(a.shape())); // Return empty array matching shape
    }
    Ok(Array::new(a_data.mapv(|x| x.max(0.0))))
}

/// Sigmoid activation function
pub fn sigmoid(a: &Array) -> Result<Array, Error> {
    let a_data = a.get_data();
    if a_data.is_empty() {
        return Ok(Array::zeros(a.shape()));
    }
    Ok(Array::new(a_data.mapv(|x| 1.0 / (1.0 + (-x).exp()))))
}

/// Log-softmax function with numerical stability
pub fn log_softmax(a: &Array, axis: usize) -> Result<Array, Error> {
    let a_data = a.get_data();

    if axis >= a_data.ndim() {
        return Err(Error::InvalidIndex(vec![axis]));
    }

    // Find max along axis for numerical stability
    let max = a_data.fold_axis(Axis(axis), f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let shifted = a_data - &max.insert_axis(Axis(axis));

    // Compute exp and sum
    let exp = shifted.mapv(|x| x.exp());
    let sum = exp.sum_axis(Axis(axis));

    // Compute log_softmax using the log-sum-exp trick
    let log_sum_exp = sum.mapv(|x| x.ln());
    let log_softmax = shifted - &log_sum_exp.insert_axis(Axis(axis));

    Ok(Array::new(log_softmax))
}

/// Computes the maximum value along the specified axis.
/// If axis is None, computes the global maximum.
pub fn max(a: &Array, axis: Option<usize>) -> Result<Array, Error> {
    let a_data = a.get_data();
    if a_data.is_empty() {
        return Ok(Array::zeros(a.shape())); // Return empty array matching shape
    }

    match axis {
        Some(ax) => {
            if ax >= a_data.ndim() {
                return Err(Error::InvalidIndex(vec![ax]));
            }
            // Use fold_axis to compute max along specified axis
            let max = a_data.fold_axis(Axis(ax), f32::NEG_INFINITY, |acc, &x| acc.max(x));
            Ok(Array::new(max.into_dyn()))
        }
        None => {
            // Global max: fold over all elements
            let max_val = a_data.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            Ok(Array::new(ArrayD::from_elem(vec![], max_val)))
        }
    }
}

/// Computes the minimum value along the specified axis.
/// If axis is None, computes the global minimum.
pub fn min(a: &Array, axis: Option<usize>) -> Result<Array, Error> {
    let a_data = a.get_data();
    if a_data.is_empty() {
        return Ok(Array::zeros(a.shape())); // Return empty array matching shape
    }

    match axis {
        Some(ax) => {
            if ax >= a_data.ndim() {
                return Err(Error::InvalidIndex(vec![ax]));
            }
            // Use fold_axis to compute min along specified axis
            let min = a_data.fold_axis(Axis(ax), f32::INFINITY, |acc, &x| acc.min(x));
            Ok(Array::new(min.into_dyn()))
        }
        None => {
            // Global min: fold over all elements
            let min_val = a_data.fold(f32::INFINITY, |acc, &x| acc.min(x));
            // Return shape [] instead of [1] for global reduction
            Ok(Array::new(ArrayD::from_elem(vec![], min_val)))
        }
    }
}

/// Computes the product of all elements along the specified axis.
/// If axis is None, computes the global product.
pub fn prod(a: &Array, axis: Option<usize>) -> Result<Array, Error> {
    let a_data = a.get_data();
    if a_data.is_empty() {
        return Ok(Array::zeros(a.shape())); // Return empty array matching shape
    }

    match axis {
        Some(ax) => {
            if ax >= a_data.ndim() {
                return Err(Error::InvalidIndex(vec![ax]));
            }
            // Use fold_axis to compute product along specified axis
            let prod = a_data.fold_axis(Axis(ax), 1.0, |acc, &x| acc * x);
            Ok(Array::new(prod.into_dyn()))
        }
        None => {
            // Global product: fold over all elements
            let prod_val = a_data.fold(1.0, |acc, &x| acc * x);
            // Return shape [] instead of [1] for global reduction
            Ok(Array::new(ArrayD::from_elem(vec![], prod_val)))
        }
    }
}

/// Computes the log-sum-exp along the specified axis.
/// If axis is None, computes the global log-sum-exp.
/// Uses the max-trick for numerical stability.
pub fn logsumexp(a: &Array, axis: Option<usize>) -> Result<Array, Error> {
    let a_data = a.get_data();
    if a_data.is_empty() {
        return Ok(Array::zeros(a.shape())); // Return empty array matching shape
    }

    match axis {
        Some(ax) => {
            if ax >= a_data.ndim() {
                return Err(Error::InvalidIndex(vec![ax]));
            }
            // Find max along axis for numerical stability
            let max = a_data.fold_axis(Axis(ax), f32::NEG_INFINITY, |acc, &x| acc.max(x));
            let shifted = a_data - &max.clone().insert_axis(Axis(ax));

            // Compute exp and sum
            let exp = shifted.mapv(|x| x.exp());
            let sum = exp.sum_axis(Axis(ax));

            // Compute final result: max + ln(sum(exp(x - max)))
            let result = max + sum.mapv(|x| x.ln());
            Ok(Array::new(result.into_dyn()))
        }
        None => {
            // Global logsumexp
            let max_val = a_data.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            let shifted = a_data.mapv(|x| x - max_val);
            let exp_sum = shifted.mapv(|x| x.exp()).sum();
            let result = max_val + exp_sum.ln();
            // Return shape [] instead of [1] for global reduction
            Ok(Array::new(ArrayD::from_elem(vec![], result)))
        }
    }
}

/// Returns the indices of maximum values along the specified axis.
/// The indices are returned as f32 values.
pub fn argmax(a: &Array, axis: usize) -> Result<Array, Error> {
    let a_data = a.get_data();
    if a_data.is_empty() {
        return Ok(Array::zeros(a.shape())); // Return empty array matching shape
    }

    if axis >= a_data.ndim() {
        return Err(Error::InvalidIndex(vec![axis]));
    }

    // Create output array with reduced dimension
    let mut output_shape = a_data.shape().to_vec();
    output_shape.remove(axis);
    let output_dim = ndarray::IxDyn(&output_shape);

    // Create output array
    let mut output = ArrayD::zeros(output_dim);

    // Create a view with the reduction axis as the last axis for easier iteration
    let axis_len = a_data.shape()[axis];

    // For 1D case, just find the index of the max value
    if a_data.ndim() == 1 {
        let (max_idx, _) = a_data.iter().enumerate().fold(
            (0, f32::NEG_INFINITY),
            |(idx_max, val_max), (idx, &val)| {
                if val > val_max {
                    (idx, val)
                } else {
                    (idx_max, val_max)
                }
            },
        );

        // Return scalar result
        return Ok(Array::new(ArrayD::from_elem(vec![], max_idx as f32)));
    }

    // For multi-dimensional arrays, iterate over the output array and find max in each slice
    for (out_idx, out_val) in output.indexed_iter_mut() {
        // Construct indices to get a slice along the reduction axis
        let mut idx_vec = Vec::with_capacity(a_data.ndim());
        let mut out_dim = 0;

        for ax in 0..a_data.ndim() {
            if ax == axis {
                // This is the reduction axis, we'll iterate over it
                idx_vec.push(0);
            } else {
                // For other dimensions, use the current output index
                idx_vec.push(out_idx[out_dim]);
                out_dim += 1;
            }
        }

        // Find max value and its index along the axis
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for i in 0..axis_len {
            // Update the index for the reduction axis
            idx_vec[axis] = i;

            // Get the value
            let val = a_data[&idx_vec[..]];

            // Update max if needed
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Store the index of the max value
        *out_val = max_idx as f32;
    }

    Ok(Array::new(output))
}

/// Returns the indices of minimum values along the specified axis.
/// The indices are returned as f32 values.
pub fn argmin(a: &Array, axis: usize) -> Result<Array, Error> {
    let a_data = a.get_data();
    if a_data.is_empty() {
        return Ok(Array::zeros(a.shape())); // Return empty array matching shape
    }

    if axis >= a_data.ndim() {
        return Err(Error::InvalidIndex(vec![axis]));
    }

    // Create output array with reduced dimension
    let mut output_shape = a_data.shape().to_vec();
    output_shape.remove(axis);
    let output_dim = ndarray::IxDyn(&output_shape);

    // Create output array
    let mut output = ArrayD::zeros(output_dim);

    // Create a view with the reduction axis as the last axis for easier iteration
    let axis_len = a_data.shape()[axis];

    // For 1D case, just find the index of the min value
    if a_data.ndim() == 1 {
        let (min_idx, _) = a_data.iter().enumerate().fold(
            (0, f32::INFINITY),
            |(idx_min, val_min), (idx, &val)| {
                if val < val_min {
                    (idx, val)
                } else {
                    (idx_min, val_min)
                }
            },
        );

        // Return scalar result
        return Ok(Array::new(ArrayD::from_elem(vec![], min_idx as f32)));
    }

    // For multi-dimensional arrays, iterate over the output array and find min in each slice
    for (out_idx, out_val) in output.indexed_iter_mut() {
        // Construct indices to get a slice along the reduction axis
        let mut idx_vec = Vec::with_capacity(a_data.ndim());
        let mut out_dim = 0;

        for ax in 0..a_data.ndim() {
            if ax == axis {
                // This is the reduction axis, we'll iterate over it
                idx_vec.push(0);
            } else {
                // For other dimensions, use the current output index
                idx_vec.push(out_idx[out_dim]);
                out_dim += 1;
            }
        }

        // Find min value and its index along the axis
        let mut min_idx = 0;
        let mut min_val = f32::INFINITY;

        for i in 0..axis_len {
            // Update the index for the reduction axis
            idx_vec[axis] = i;

            // Get the value
            let val = a_data[&idx_vec[..]];

            // Update min if needed
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }

        // Store the index of the min value
        *out_val = min_idx as f32;
    }

    Ok(Array::new(output))
}

/// Element-wise power function. Raises each element in a to the power of the corresponding element in b.
/// Supports broadcasting.
pub fn powf(a: &Array, b: &Array) -> Result<Array, Error> {
    let a_data = a.get_data();
    let b_data = b.get_data();

    // Get maximum dimensionality
    let a_shape = a_data.shape();
    let b_shape = b_data.shape();
    let ndim = a_shape.len().max(b_shape.len());

    // Adjust arrays to have same number of dimensions
    let a_view = a_data.view();
    let mut a_adjusted = a_view.clone();
    for _ in 0..(ndim - a_view.ndim()) {
        a_adjusted = a_adjusted.insert_axis(Axis(0));
    }

    let b_view = b_data.view();
    let mut b_adjusted = b_view.clone();
    for _ in 0..(ndim - b_view.ndim()) {
        b_adjusted = b_adjusted.insert_axis(Axis(0));
    }

    // Calculate broadcast shape
    let mut broadcast_shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let a_dim = a_adjusted.shape()[i];
        let b_dim = b_adjusted.shape()[i];
        if a_dim == b_dim {
            broadcast_shape.push(a_dim);
        } else if a_dim == 1 || b_dim == 1 {
            broadcast_shape.push(a_dim.max(b_dim));
        } else {
            return Err(Error::IncompatibleShapes {
                op: String::from("powf"),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            });
        }
    }

    // Attempt to broadcast arrays
    let a_broadcast =
        a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or(Error::IncompatibleShapes {
                op: String::from("powf"),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            })?;
    let b_broadcast = b_adjusted
        .broadcast(broadcast_shape)
        .ok_or(Error::IncompatibleShapes {
            op: String::from("powf"),
            shape_a: a_data.shape().to_vec(),
            shape_b: b_data.shape().to_vec(),
        })?;

    // Use Zip to apply the powf operation elementwise
    let mut result = ndarray::ArrayD::<f32>::zeros(a_broadcast.raw_dim());
    ndarray::Zip::from(&mut result)
        .and(&a_broadcast)
        .and(&b_broadcast)
        .for_each(|r, &a, &b| *r = a.powf(b));

    Ok(Array::new(result))
}

/// Element-wise square root function. Returns a new Array with the square root of each element in the input.
pub fn sqrt(a: &Array) -> Result<Array, Error> {
    let a_data = a.get_data();
    // Check for negative values
    if a_data.iter().any(|&x| x < 0.0) {
        return Err(Error::InvalidOperation("sqrt: input contains negative values".to_string()));
    }
    let result = a_data.mapv(|x| x.sqrt());
    Ok(Array::new(result))
}

