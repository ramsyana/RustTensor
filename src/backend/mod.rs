//! Backend trait definition and module structure.

use crate::error::Error;
use std::fmt::{Debug, Display};

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;

pub type CpuTensor = crate::tensor::Tensor<cpu::CpuBackend>;
#[cfg(feature = "cuda")]
pub type CudaTensor = crate::tensor::Tensor<cuda::CudaBackend>;

pub trait Backend: Sized + Debug + Clone {
    /// Performs MomentumSGD update step in-place.
    /// velocity = momentum * velocity + grad
    /// param -= lr * velocity
    fn momentum_sgd_step(
        param: &mut Self::Storage,
        grad: &Self::Storage,
        velocity: &mut Self::Storage,
        lr: f32,
        momentum: f32,
    ) -> Result<(), Error>;

    /// Performs AdaGrad update step in-place.
    /// accum_sq_grad += grad^2
    /// param -= lr * grad / (sqrt(accum_sq_grad) + epsilon)
    /// Updates accum_sq_grad and param states.
    fn adagrad_step(
        param: &mut Self::Storage,         // Parameter tensor data (theta)
        grad: &Self::Storage,              // Gradient (g_t)
        accum_sq_grad: &mut Self::Storage, // Accumulated squared gradient state (G_t)
        lr: f32,                           // Learning rate (alpha)
        epsilon: f32,                      // Small term for numerical stability
    ) -> Result<(), Error>;

    // NOTE: Removed AsRef<[f32]> + AsMut<[f32]> bounds
    type Storage: Clone + Debug + Display;

    // --- Factory Methods (Creating Storage) ---

    /// Creates new storage filled with zeros.
    fn zeros(shape: &[usize]) -> Result<Self::Storage, Error>;
    /// Creates new storage filled with ones.
    fn ones(shape: &[usize]) -> Result<Self::Storage, Error>;
    /// Creates new storage from a flat vector and a shape.
    fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self::Storage, Error>;
    
    #[cfg(feature = "serialization")]
    /// Creates new storage from a host vector, shape, and device.
    /// This is used for deserialization to create storage on the appropriate device.
    fn from_host_vec(data: Vec<f32>, shape: &[usize], _device: crate::Device) -> Result<Self::Storage, Error> {
        // Default implementation just uses from_vec
        // Backend implementations can override this if they need special handling
        Self::from_vec(data, shape)
    }
    /// Creates new storage with Kaiming uniform initialization.
    fn kaiming_uniform(fan_in: usize, shape: &[usize]) -> Result<Self::Storage, Error>;

    // --- Random Generation Methods ---
    /// Creates new storage filled with values from a uniform distribution U(low, high).
    fn random_uniform(shape: &[usize], low: f32, high: f32) -> Result<Self::Storage, Error>;

    /// Creates new storage filled with values from a normal distribution N(mean, std_dev^2).
    fn random_normal(shape: &[usize], mean: f32, std_dev: f32) -> Result<Self::Storage, Error>;

    /// Creates new storage filled with values from a Bernoulli distribution B(p).
    fn bernoulli(shape: &[usize], p: f32) -> Result<Self::Storage, Error>;

    // --- Shape/Data Access ---

    /// Returns the device of the storage.
    fn device(storage: &Self::Storage) -> crate::Device;

    /// Returns the shape of the storage.
    fn shape(storage: &Self::Storage) -> &[usize];
    /// Returns the total number of elements in the storage.
    fn size(storage: &Self::Storage) -> usize;
    /// Consumes the storage and returns its data as a flat `Vec<f32>`.
    /// May involve device-to-host transfer for GPU storage.
    fn into_raw_vec(storage: Self::Storage) -> Result<Vec<f32>, Error>;
    /// Sets the data of `storage` by consuming `data`. Shapes must match.
    /// This might involve copying data on the device.
    fn set_data(storage: &mut Self::Storage, data: Self::Storage) -> Result<(), Error>;
    /// Sets the shape of the storage without changing its data.
    /// The new shape must have the same total number of elements.
    fn set_shape(storage: &mut Self::Storage, shape: &[usize]) -> Result<(), Error>;

    // --- Core Mathematical Operations ---

    /// Matrix multiplication (typically requires 2D storage).
    fn matmul(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    /// 2D Convolution (NCHW format).
    /// input: [N, C_in, H_in, W_in]
    /// weights: [C_out, C_in, K_h, K_w]
    /// bias: [C_out] or None
    /// stride: (usize, usize)
    /// padding: (usize, usize)
    fn conv2d(
        input: &Self::Storage,
        weights: &Self::Storage,
        bias: Option<&Self::Storage>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self::Storage, Error>;

    /// Backward pass for conv2d.
    /// Returns (grad_input, grad_weights, grad_bias)
    fn conv2d_backward(
        input: &Self::Storage,
        weights: &Self::Storage,
        grad_output: &Self::Storage,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Self::Storage, Self::Storage, Option<Self::Storage>), Error>;
    
    /// 2D transpose convolution (NCHW format).
    /// input: [N, C_in, H_in, W_in]
    /// weights: [C_in, C_out, K_h, K_w]
    /// bias: [C_out] or None
    /// stride: (usize, usize)
    /// padding: (usize, usize)
    fn conv2d_transpose(
        input: &Self::Storage,
        weights: &Self::Storage,
        bias: Option<&Self::Storage>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self::Storage, Error>;
    
    /// Backward pass for 2D transpose convolution.
    /// Returns (grad_input, grad_weights, grad_bias)
    fn conv2d_transpose_backward(
        input: &Self::Storage,
        weights: &Self::Storage,
        grad_output: &Self::Storage,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Self::Storage, Self::Storage, Option<Self::Storage>), Error>;

    // Returns (output_values, output_indices_f32)
    fn max_pool2d(
        input: &Self::Storage,    // [N, C, H_in, W_in]
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Self::Storage, Self::Storage), Error>; // Output, Indices (as f32)

    fn max_pool2d_backward(
        op_ctx: &crate::graph::Op<Self>,          // Contains original input and indices tensor
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error>; // grad_input

    // --- Reduction Operations ---

    /// Computes the maximum value along the specified axis.
    /// If axis is None, computes the global maximum.
    fn max(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error>;

    /// Computes the minimum value along the specified axis.
    /// If axis is None, computes the global minimum.
    fn min(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error>;

    /// Computes the product of all elements along the specified axis.
    /// If axis is None, computes the global product.
    fn prod(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error>;

    /// Computes the log-sum-exp along the specified axis.
    /// If axis is None, computes the global log-sum-exp.
    /// Uses the max-trick for numerical stability.
    fn logsumexp(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error>;

    /// Returns the indices of maximum values along the specified axis.
    /// The indices are returned as f32 values.
    fn argmax(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error>;

    /// Returns the indices of minimum values along the specified axis.
    /// The indices are returned as f32 values.
    fn argmin(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error>;

    /// Element-wise equality comparison (returns 1.0 for true, 0.0 for false).
    /// Supports broadcasting.
    fn equal(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Element-wise greater comparison (a > b). Supports broadcasting.
    /// Returns 1.0f32 for true, 0.0f32 for false.
    fn greater(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Element-wise greater or equal comparison (a >= b). Supports broadcasting.
    /// Returns 1.0f32 for true, 0.0f32 for false.
    fn greater_equal(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Element-wise less comparison (a < b). Supports broadcasting.
    /// Returns 1.0f32 for true, 0.0f32 for false.
    fn less(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Element-wise less or equal comparison (a <= b). Supports broadcasting.
    /// Returns 1.0f32 for true, 0.0f32 for false.
    fn less_equal(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Element-wise not equal comparison (a != b). Supports broadcasting.
    /// Returns 1.0f32 for true, 0.0f32 for false.
    fn not_equal(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    // --- Backward Operations for Reductions ---

    /// Computes the gradient for the max operation.
    fn max_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Computes the gradient for the min operation.
    fn min_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Computes the gradient for the prod operation.
    fn prod_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Computes the gradient for the logsumexp operation.
    fn logsumexp_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Element-wise multiplication (supports broadcasting).
    fn mul(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Element-wise addition (supports broadcasting).
    fn add(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Element-wise subtraction (supports broadcasting).
    fn sub(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Element-wise division (supports broadcasting).
    fn div(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Element-wise division by a scalar.
    fn div_scalar(x: &Self::Storage, scalar: f32) -> Result<Self::Storage, Error>;

    // --- GPU-Specific Data Movement Methods ---

    /// Copies data from device storage to a host vector.
    /// For CPU backend, this is equivalent to into_raw_vec.
    fn copy_to_host(storage: &Self::Storage) -> Result<Vec<f32>, Error>;

    /// Updates device storage from host vector.
    /// For CPU backend, this is equivalent to from_vec.
    fn update_from_host(storage: &mut Self::Storage, data: &[f32]) -> Result<(), Error>;

    /// Performs SGD update step in-place: w -= lr * dw
    /// This is optimized for each backend (e.g., using cuBLAS axpy for GPU)
    fn sgd_step(w: &mut Self::Storage, dw: &Self::Storage, lr: f32) -> Result<(), Error>;

    /// Performs Adam update step in-place.
    /// param -= lr * m_hat / (sqrt(v_hat) + epsilon)
    /// Updates m and v states as well.
    #[allow(clippy::too_many_arguments)]
    fn adam_step(
        param: &mut Self::Storage, // Parameter tensor data (theta)
        grad: &Self::Storage,      // Gradient (g_t)
        m: &mut Self::Storage,     // First moment estimate (m_t)
        v: &mut Self::Storage,     // Second raw moment estimate (v_t)
        lr: f32,                   // Learning rate (alpha)
        beta1: f32,                // Exponential decay rate for m_t
        beta2: f32,                // Exponential decay rate for v_t
        epsilon: f32,              // Small term for numerical stability
        t: usize,                  // Current timestep
    ) -> Result<(), Error>;

    /// Transposes the storage (behavior depends on dimensionality, typically for 2D).
    fn transpose(x: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Broadcasts the storage to a new compatible shape.
    fn broadcast_to(x: &Self::Storage, shape: &[usize]) -> Result<Self::Storage, Error>;
    /// Applies the exponential function element-wise.
    fn exp(x: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Applies the natural logarithm element-wise.
    fn ln(x: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Applies a function element-wise.
    /// The function `F` needs `Send + Sync + 'static` for potential multi-threading.
    fn map<F>(x: &Self::Storage, f: F) -> Result<Self::Storage, Error>
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static;

    // --- Reduction Operations ---

    /// Sums elements along a specified axis.
    fn sum_along_axis(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error>;
    /// Sums all elements in the storage, returning a scalar f32.
    fn sum_all(x: &Self::Storage) -> Result<f32, Error>;
    /// Finds the maximum element along a specified axis.
    fn max_along_axis(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error>;
    /// Calculates the mean along a specified axis (or globally if `axis` is `None`).
    /// Returns storage (even for global mean, resulting in a 0-dim storage).
    fn mean(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error>;

    // --- Neural Network Specific Operations ---

    /// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
    fn relu(x: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Applies the sigmoid activation function element-wise.
    fn sigmoid(x: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Applies the log-softmax function along a specified axis.
    fn log_softmax(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error>;

    /// Applies the Exponential Linear Unit (ELU) activation function element-wise.
    /// Formula: alpha * (exp(x) - 1) if x < 0, x if x >= 0
    fn elu(x: &Self::Storage, alpha: f32) -> Result<Self::Storage, Error>;

    /// Backward pass for sigmoid activation.
    fn sigmoid_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    // --- Backward Operations for Autograd ---
    // These functions compute the gradients with respect to the *inputs* of the forward operation.
    // They take the `Op<Self>` context (containing original inputs) and the output gradient.

    /// Backward pass for matrix multiplication. Returns gradients for `a` and `b`.
    fn matmul_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error>;
    /// Backward pass for element-wise multiplication. Returns gradients for `a` and `b`.
    fn mul_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error>;
    /// Backward pass for element-wise addition. Returns gradients for `a` and `b`.
    fn add_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error>;
    /// Backward pass for mean reduction. Returns gradient for the input `x`.
    fn mean_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;
    /// Backward pass for ReLU activation. Returns gradient for the input `x`.
    fn relu_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;
    /// Backward pass for log-softmax activation. Returns gradient for the input `x`.
    fn log_softmax_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Backward pass for ELU activation. Returns gradient for the input `x`.
    /// The derivative is alpha * exp(x) if x < 0, 1 if x >= 0
    fn elu_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Backward pass for sum reduction. Returns gradient for the input `x`.
    /// The `Op` context contains axis information via `OpType::Sum(Option<usize>)`.
    fn sum_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Backward pass for element-wise division. Returns gradients for `a` and `b`.
    fn div_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error>;

    /// Backward pass for element-wise subtraction. Returns gradients for `a` and `b`.
    fn sub_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error>;

    /// Backward pass for exponential operation. Returns gradient for the input `x`.
    fn exp_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Backward pass for natural logarithm. Returns gradient for the input `x`.
    fn ln_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    // Add more backward operations (sub, div, transpose, etc.) as needed.

    /// Applies the absolute value function element-wise.
    fn abs(x: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Backward pass for absolute value.
    fn abs_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Computes the element-wise square root: sqrt(x)
    fn sqrt(x: &Self::Storage) -> Result<Self::Storage, Error>;
    /// Computes the gradient for the sqrt operation.
    fn sqrt_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Applies the hyperbolic tangent (tanh) activation function element-wise.
    ///
    /// The hyperbolic tangent function is defined as tanh(x) = 2 / (1 + exp(-2x)) - 1.
    /// It maps any real-valued number to a value between -1 and 1.
    fn tanh(x: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Backward pass for tanh activation.
    ///
    /// The derivative of the tanh function is given by d(tanh(x))/dx = 1 - tanh(x)^2.
    /// This function computes the gradient of the tanh activation with respect to its input.
    fn tanh_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Applies the softplus activation function element-wise: log(1 + exp(x)).
    fn softplus(x: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Backward pass for softplus activation.
    fn softplus_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Element-wise power function, raising each element in 'a' to the power of the corresponding element in 'b'.
    fn powf(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Backward pass for powf operation. Returns gradients for both 'a' and 'b'.
    fn powf_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error>;

    /// Computes the element-wise square: x^2
    fn square(x: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Backward pass for element-wise square.
    /// `op` contains the original input `x` (the first element of `op.inputs`).
    /// `output_grad` is the gradient flowing back (dL/dy where y = x^2).
    /// Returns the gradient w.r.t input x (dL/dx = dL/dy * 2x).
    fn square_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    /// Element-wise maximum of two tensors (supports broadcasting).
    fn maximum(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Backward pass for element-wise maximum. Returns gradients for `a` and `b`.
    fn maximum_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error>;

    /// Element-wise minimum of two tensors (supports broadcasting).
    fn minimum(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Backward pass for element-wise minimum. Returns gradients for `a` and `b`.
    fn minimum_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error>;

    /// Applies the sine function element-wise: sin(x).
    fn sin(x: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Backward pass for sine activation.
    ///
    /// The derivative of the sine function is given by d(sin(x))/dx = cos(x).
    /// This function computes the gradient of the sine activation with respect to its input.
    fn sin_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error>;
    
    /// Applies the cosine function element-wise: cos(x).
    fn cos(x: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Backward pass for cosine activation.
    ///
    /// The derivative of the cosine function is given by d(cos(x))/dx = -sin(x).
    /// This function computes the gradient of the cosine activation with respect to its input.
    fn cos_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error>;
    
    /// Applies the tangent function element-wise: tan(x).
    fn tan(x: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Backward pass for tangent activation.
    ///
    /// The derivative of the tangent function is given by d(tan(x))/dx = 1 + tan²(x) = 1/cos²(x).
    /// This function computes the gradient of the tangent activation with respect to its input.
    fn tan_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error>;

    // --- Array Operations ---

    /// Extracts a slice from a tensor along specified dimensions.
    ///
    /// # Arguments
    /// * `x` - The input tensor storage
    /// * `ranges` - A slice of ranges, one for each dimension, specifying the slice to extract
    ///
    /// # Returns
    /// A new storage containing the sliced data
    ///
    /// # Errors
    /// Returns an error if the ranges are invalid or out of bounds
    fn slice(x: &Self::Storage, ranges: &[std::ops::Range<usize>]) -> Result<Self::Storage, Error>;

    /// Computes the gradient for the slice operation.
    ///
    /// # Arguments
    /// * `op_ctx` - The operation context containing the original input tensor and the slice ranges
    /// * `grad_output` - The gradient flowing back from the output of the slice operation
    ///
    /// # Returns
    /// The gradient with respect to the input tensor
    ///
    /// # Errors
    /// Returns an error if the gradient computation fails
    fn slice_backward(op_ctx: &crate::graph::Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Concatenates multiple tensors along a specified axis.
    ///
    /// # Arguments
    /// * `tensors_data` - A slice of references to tensor storages to concatenate
    /// * `axis` - The axis along which to concatenate the tensors
    ///
    /// # Returns
    /// A new storage containing the concatenated data
    ///
    /// # Errors
    /// Returns an error if:
    /// * The tensors have incompatible shapes (all dimensions except the concat dimension must match)
    /// * The axis is out of bounds for any tensor
    fn concat(tensors_data: &[&Self::Storage], axis: usize) -> Result<Self::Storage, Error>;

    /// Computes the gradient for the concat operation.
    ///
    /// # Arguments
    /// * `op_ctx` - The operation context containing the original input tensors and the concat axis
    /// * `grad_output` - The gradient flowing back from the output of the concat operation
    ///
    /// # Returns
    /// A vector of gradients, one for each input tensor
    ///
    /// # Errors
    /// Returns an error if the gradient computation fails
    fn concat_backward(op_ctx: &crate::graph::Op<Self>, grad_output: &Self::Storage) -> Result<Vec<Self::Storage>, Error>;

    /// Inserts a new dimension of size 1 at the specified axis.
    ///
    /// # Arguments
    /// * `x` - The input tensor storage
    /// * `axis` - The axis at which to insert the new dimension
    ///
    /// # Returns
    /// A new storage with an additional dimension of size 1 at the specified axis
    ///
    /// # Errors
    /// Returns an error if the axis is invalid
    fn expand_dims(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error>;

    /// Computes the gradient for the expand_dims operation.
    ///
    /// # Arguments
    /// * `op_ctx` - The operation context containing the original input tensor and the axis
    /// * `grad_output` - The gradient flowing back from the output of the expand_dims operation
    ///
    /// # Returns
    /// The gradient with respect to the input tensor
    ///
    /// # Errors
    /// Returns an error if the gradient computation fails
    fn expand_dims_backward(op_ctx: &crate::graph::Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Removes dimensions of size 1 from the tensor.
    ///
    /// # Arguments
    /// * `x` - The input tensor storage
    /// * `axis` - The axis to squeeze out. If None, all dimensions of size 1 are removed.
    ///
    /// # Returns
    /// A new storage with the specified dimension(s) of size 1 removed
    ///
    /// # Errors
    /// Returns an error if the axis is invalid or not of size 1
    fn squeeze(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error>;

    /// Computes the gradient for the squeeze operation.
    ///
    /// # Arguments
    /// * `op_ctx` - The operation context containing the original input tensor and the axis
    /// * `grad_output` - The gradient flowing back from the output of the squeeze operation
    ///
    /// # Returns
    /// The gradient with respect to the input tensor
    ///
    /// # Errors
    /// Returns an error if the gradient computation fails
    fn squeeze_backward(op_ctx: &crate::graph::Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error>;

    /// Clips the values of a tensor to be within [min_val, max_val].
    ///
    /// # Arguments
    /// * `x` - The input tensor storage
    /// * `min_val` - The minimum value to clip to
    /// * `max_val` - The maximum value to clip to
    ///
    /// # Returns
    /// A new storage with all values clipped to the range [min_val, max_val]
    ///
    /// # Errors
    /// Returns an error if the operation fails
    fn clip(x: &Self::Storage, min_val: f32, max_val: f32) -> Result<Self::Storage, Error>;

    /// Computes the gradient for the clip operation.
    ///
    /// # Arguments
    /// * `op_ctx` - The operation context containing the original input tensor and the min/max values
    /// * `grad_output` - The gradient flowing back from the output of the clip operation
    ///
    /// # Returns
    /// The gradient with respect to the input tensor
    ///
    /// # Errors
    /// Returns an error if the gradient computation fails
    fn clip_backward(op_ctx: &crate::graph::Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error>;
    
    /// Element-wise multiplication by a scalar.
    ///
    /// # Arguments
    /// * `x` - The input tensor storage
    /// * `scalar` - The scalar value to multiply by
    ///
    /// # Returns
    /// A new storage with all values multiplied by the scalar
    ///
    /// # Errors
    /// Returns an error if the operation fails
    fn mul_scalar(x: &Self::Storage, scalar: f32) -> Result<Self::Storage, Error>;
}
