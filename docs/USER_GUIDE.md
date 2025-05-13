# Rust Tensor Library User Guide

This guide provides a step-by-step introduction to using the Rust Tensor Library, covering installation, basic operations, automatic differentiation, and more advanced features.

## Table of Contents

1. [Installation](#installation)
2. [Creating Tensors](#creating-tensors)
3. [Basic Operations](#basic-operations)
4. [Automatic Differentiation](#automatic-differentiation)
5. [Hooks](#hooks)
6. [Working with Different Backends](#working-with-different-backends)
7. [Neural Network Operations](#neural-network-operations)
8. [Optimizers](#optimizers)
9. [Serialization](#serialization)
10. [Examples](#examples)
11. [Debugging Tips](#debugging-tips)

## Installation

Add the Rust Tensor Library to your project's `Cargo.toml`:

```toml
[dependencies]
# CPU only:
rust_tensor_library = "0.1.0"

# With CUDA support:
# rust_tensor_library = { version = "0.1.0", features = ["cuda"] }

# With serialization support:
# rust_tensor_library = { version = "0.1.0", features = ["serialization"] }

# With both CUDA and serialization support:
# rust_tensor_library = { version = "0.1.0", features = ["cuda", "serialization"] }
```

### CUDA Setup (Optional)

If you want to use the CUDA backend, you'll need:

1. NVIDIA CUDA Toolkit (11.0 or later)
2. A CUDA-capable GPU
3. Up-to-date NVIDIA drivers

Set up your environment variables:

```bash
# Linux/macOS
export CUDA_PATH=/usr/local/cuda-11.8  # Adjust for your CUDA version
export CUBLAS_LIB_DIR=$CUDA_PATH/lib64
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}

# Windows (in PowerShell)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"  # Adjust path
$env:CUBLAS_LIB_DIR = "$env:CUDA_PATH\lib\x64"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
```

## Creating Tensors

There are multiple ways to create tensors:

### From Vectors

```rust
use rust_tensor_library::{Tensor, CpuBackend};

// Create a 1D tensor from a vector (with gradient tracking)
let tensor_1d = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;

// Create a 2D tensor (2x3 matrix)
let tensor_2d = Tensor::<CpuBackend>::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 
    &[2, 3], 
    true
)?;
```

### Using Initialization Functions

```rust
use rust_tensor_library::{Tensor, CpuBackend};

// Create a tensor filled with zeros
let zeros = Tensor::<CpuBackend>::zeros(&[2, 3], true)?;

// Create a tensor filled with ones
let ones = Tensor::<CpuBackend>::ones(&[2, 3], true)?;

// Create a tensor with random values (uniform distribution)
let random = Tensor::<CpuBackend>::rand(&[2, 3], true)?;

// Create a tensor with random values (normal distribution)
let random_normal = Tensor::<CpuBackend>::randn(&[2, 3], true)?;
```

### Kaiming Initialization (for Neural Networks)

```rust
use rust_tensor_library::{Tensor, CpuBackend, kaiming_uniform};

// Initialize weights for a neural network layer
let fan_in = 784; // Input features
let fan_out = 128; // Output features
let weights_data = kaiming_uniform(&[fan_out, fan_in])?;
let weights = Tensor::<CpuBackend>::from_array(weights_data, true)?;
```

## Basic Operations

The library supports multiple ways to perform operations on tensors: using operator overloading, calling operations as methods on tensors, or using the ops module functions.

### Arithmetic Operations

```rust
// Element-wise addition
let c = &a + &b;                // Using operator overloading
let c = a.add(&b)?;             // Using tensor method
let c = ops::add(&a, &b)?;      // Using ops module function

// Element-wise subtraction
let c = &a - &b;                // Using operator overloading
let c = a.sub(&b)?;             // Using tensor method
let c = ops::sub(&a, &b)?;      // Using ops module function

// Element-wise multiplication
let c = &a * &b;                // Using operator overloading
let c = a.mul(&b)?;             // Using tensor method
let c = ops::mul(&a, &b)?;      // Using ops module function

// Element-wise division
let c = &a / &b;                // Using operator overloading
let c = a.div(&b)?;             // Using tensor method
let c = ops::div(&a, &b)?;      // Using ops module function

// Matrix multiplication
let c = a.matmul(&b)?;          // Using tensor method
let c = ops::matmul(&a, &b)?;   // Using ops module function

// Scalar operations
let d = &a * 2.0;               // Multiply by scalar (operator overloading)
let d = a.mul_scalar(2.0)?;     // Using tensor method
let d = ops::mul_scalar(&a, 2.0)?; // Using ops module function
```

### Activation Functions

```rust
use rust_tensor_library::ops;

// ReLU activation
let activated = x.relu()?;                // Using tensor method
let activated = ops::relu(&x)?;           // Using ops module function

// Sigmoid activation
let activated = x.sigmoid()?;             // Using tensor method
let activated = ops::sigmoid(&x)?;        // Using ops module function

// Tanh activation
let activated = x.tanh()?;                // Using tensor method
let activated = ops::tanh(&x)?;           // Using ops module function

// ELU activation
let activated = x.elu(1.0)?;              // Using tensor method with alpha=1.0
let activated = ops::elu(&x, 1.0)?;       // Using ops module function

// Softplus activation
let activated = x.softplus()?;            // Using tensor method
let activated = ops::softplus(&x)?;       // Using ops module function

// Softmax (log-softmax is often used for numerical stability)
let log_probs = x.log_softmax(1)?;        // Using tensor method (axis=1)
let log_probs = ops::log_softmax(&x, 1)?; // Using ops module function (axis=1)
```

### Reduction Operations

```rust
// Mean of all elements
let mean_all = x.mean(None)?;             // Using tensor method
let mean_all = ops::mean(&x, None)?;      // Using ops module function

// Sum along a specific axis (e.g., axis 0)
let sum_axis0 = x.sum(Some(0))?;          // Using tensor method
let sum_axis0 = ops::sum(&x, Some(0))?;   // Using ops module function

// Max along an axis
let max_axis1 = x.max(Some(1))?;          // Using tensor method
let max_axis1 = ops::max(&x, Some(1))?;   // Using ops module function

// Min along an axis
let min_axis1 = x.min(Some(1))?;          // Using tensor method
let min_axis1 = ops::min(&x, Some(1))?;   // Using ops module function

// Product along an axis
let prod_axis0 = x.prod(Some(0))?;        // Using tensor method
let prod_axis0 = ops::prod(&x, Some(0))?; // Using ops module function

// LogSumExp (numerically stable way to compute log(sum(exp(x))))
let lse = x.logsumexp(Some(1))?;          // Using tensor method
let lse = ops::logsumexp(&x, Some(1))?;   // Using ops module function
```

### Shape Manipulation

```rust
// Reshape a tensor
let reshaped = x.view(&[new_dim1, new_dim2])?;       // Using tensor method
let reshaped = ops::view(&x, &[new_dim1, new_dim2])?; // Using ops module function

// Transpose a tensor
let transposed = x.transpose(&[1, 0])?;              // Using tensor method
let transposed = ops::transpose(&x, &[1, 0])?;        // Using ops module function

// Concatenate tensors along an axis
// For concat, we typically use the ops function since it takes multiple tensors
let concatenated = ops::concat(&[&tensor1, &tensor2], 0)?;

// Expand dimensions
let expanded = x.expand_dims(0)?;                    // Using tensor method
let expanded = ops::expand_dims(&x, 0)?;             // Using ops module function

// Squeeze a dimension
let squeezed = x.squeeze(0)?;                        // Using tensor method
let squeezed = ops::squeeze(&x, 0)?;                 // Using ops module function

// Slice a tensor
let sliced = x.slice(&[(0, 2), (1, 3)])?;           // Using tensor method to get x[0:2, 1:3]
let sliced = ops::slice(&x, &[(0, 2), (1, 3)])?;     // Using ops module function

// Clone a tensor (creates a new tensor with the same data)
let cloned = x.clone()?;                             // Using tensor method
```

### Custom Element-wise Operations with map

⚠️ **IMPORTANT: BREAKS AUTOGRAD GRAPH** ⚠️

The `map` method allows you to apply a custom function to each element of a tensor. However, it's important to understand that this operation breaks the automatic differentiation graph. The resulting tensor will not have gradient tracking, and gradients cannot flow through this operation during backpropagation.

```rust
use rust_tensor_library::{Tensor, CpuBackend};

// Create a tensor with gradient tracking
let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 4.0, 9.0], &[3], true)?;

// Apply custom function (square root) to each element
let y = x.map(|v| v.sqrt())?;

// Note: y will have requires_grad = false, regardless of x's setting
assert_eq!(y.requires_grad(), false);

// The computation graph is broken between x and y
// If we compute backward through y, no gradients will flow to x

// However, x can still receive gradients through other paths
let z = x.mul_scalar(2.0)?;
z.backward()?;

// x has gradients from z, but not from y
assert!(x.grad().is_some());
```

#### Alternatives to map

If you need gradient tracking, use built-in operations instead of `map`. For example:

```rust
// Instead of: let y = x.map(|v| v.sqrt())?;

// Use the built-in sqrt operation
let y = x.sqrt()?;  // or ops::sqrt(&x)?

// Now y.requires_grad() == true and gradients will flow properly
```

## Automatic Differentiation

The library uses dynamic automatic differentiation to compute gradients:

```rust
use rust_tensor_library::{Tensor, CpuBackend, ops};

// Create tensors with gradient tracking
let x = Tensor::<CpuBackend>::from_vec(vec![2.0], &[1], true)?;

// Forward pass: compute y = x^2
let y = ops::square(&x)?;

// Backward pass: compute gradients
y.backward()?;

// Access gradients (dy/dx = 2x = 4.0 at x=2.0)
if let Some(grad_ref) = x.grad() {
    let grad_data = CpuBackend::copy_to_host(&*grad_ref)?;
    println!("Gradient of x: {:?}", grad_data);  // Should be [4.0]
}
```

### Computing Gradients with Respect to Multiple Inputs

```rust
// Create tensors with gradient tracking
let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], true)?;
let b = Tensor::<CpuBackend>::from_vec(vec![3.0, 4.0], &[2], true)?;

// Forward pass
let c = &a + &b;
let d = ops::mean(&c, None)?;

// Backward pass
d.backward()?;

// Access gradients
// For d = mean(a+b), gradient of both a and b is [0.5, 0.5]
if let Some(grad_a_ref) = a.grad() {
    let grad_a_data = CpuBackend::copy_to_host(&*grad_a_ref)?;
    println!("Gradient of a: {:?}", grad_a_data);
}

if let Some(grad_b_ref) = b.grad() {
    let grad_b_data = CpuBackend::copy_to_host(&*grad_b_ref)?;
    println!("Gradient of b: {:?}", grad_b_data);
}
```

## Hooks

Hooks provide a powerful mechanism to customize tensor behavior during forward and backward passes. They can be used for debugging, monitoring, or modifying tensor operations.

### Adding Hooks

```rust
use rust_tensor_library::{Tensor, CpuBackend, hooks::{Hook, FnHook}};

// Create a tensor
let mut x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;

// Add a hook that will be called during the forward pass
let hook_id = x.register_forward_hook(FnHook::new(|tensor, _input, output| {
    println!("Forward pass - Tensor ID: {}, Output shape: {:?}", tensor.id(), output.shape());
    Ok(())
}))?;

// Perform some operations
let y = x.relu()?;

// Output: "Forward pass - Tensor ID: 1, Output shape: [3]"

// Remove the hook when no longer needed
x.remove_hook(hook_id);
```

### Backward Hooks

```rust
use rust_tensor_library::{Tensor, CpuBackend, hooks::{Hook, FnHook}};

// Create a tensor
let mut x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;

// Add a hook that will be called during the backward pass
let hook_id = x.register_backward_hook(FnHook::new(|tensor, grad_input, grad_output| {
    println!("Backward pass - Tensor ID: {}, Grad output shape: {:?}", 
             tensor.id(), grad_output.shape());
    // You can also modify the gradient if needed
    Ok(())
}))?;

// Perform operations and backward pass
let y = x.square()?;
let z = y.mean(None)?;
z.backward()?;

// Output: "Backward pass - Tensor ID: 1, Grad output shape: [3]"

// Remove the hook when no longer needed
x.remove_hook(hook_id);
```

### Use Cases for Hooks

1. **Debugging**: Monitor intermediate values during forward and backward passes
2. **Gradient Clipping**: Modify gradients to prevent exploding gradients
3. **Feature Visualization**: Capture activations in neural networks
4. **Custom Regularization**: Apply custom regularization during the backward pass
5. **Gradient Flow Analysis**: Track how gradients flow through your network

## Working with Different Backends

### CPU Backend

```rust
use rust_tensor_library::{Tensor, CpuBackend, ops};

// Create a CPU tensor
let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
let y = ops::exp(&x)?;
```

### CUDA Backend (GPU)

```rust
use rust_tensor_library::{Tensor, CudaBackend, ops};

// Create a CUDA tensor directly
let x = Tensor::<CudaBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
let y = ops::exp(&x)?;

// Or convert a CPU tensor to CUDA
let x_cpu = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
let x_cuda = x_cpu.to_gpu(0)?;  // 0 is the GPU device ID
```

### Converting Between Backends

```rust
// CPU to GPU
let x_cpu = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
let x_cuda = x_cpu.to_gpu(0)?;

// GPU to CPU
let x_cpu_again = x_cuda.to_cpu()?;
```

## Neural Network Operations

### Loss Functions

```rust
use rust_tensor_library::{Tensor, CpuBackend, ops, Reduction};

// Mean Squared Error (MSE) Loss
let mse = predictions.mse_loss(&targets, Reduction::Mean)?;        // Using tensor method
let mse = ops::mse_loss(&predictions, &targets, Reduction::Mean)?; // Using ops module function

// Binary Cross Entropy with Logits Loss
let bce = logits.binary_cross_entropy_with_logits(&targets, Reduction::Mean)?;        // Using tensor method
let bce = ops::binary_cross_entropy_with_logits(&logits, &targets, Reduction::Mean)?; // Using ops module function

// Cross Entropy Loss (combines log_softmax and NLL loss)
let loss = logits.softmax_cross_entropy(&targets, 1, Reduction::Mean)?;        // Using tensor method
let loss = ops::softmax_cross_entropy(&logits, &targets, 1, Reduction::Mean)?; // Using ops module function

// L1 Loss (Mean Absolute Error)
let l1_loss = predictions.l1_loss(&targets, Reduction::Mean)?;        // Using tensor method
let l1_loss = ops::l1_loss(&predictions, &targets, Reduction::Mean)?; // Using ops module function
```

### Convolution Operations

```rust
// 2D Convolution
// Input: [batch_size, in_channels, height, width]
// Weights: [out_channels, in_channels, kernel_height, kernel_width]
let output = input.conv2d(&weights, None, (stride_h, stride_w), (padding_h, padding_w))?;        // Using tensor method
let output = ops::conv2d(&input, &weights, (stride_h, stride_w), (padding_h, padding_w))?;      // Using ops module function

// 2D Convolution with bias
let output = input.conv2d(&weights, Some(&bias), (stride_h, stride_w), (padding_h, padding_w))?; // Using tensor method with bias

// 2D Transposed Convolution (for upsampling)
let output = input.conv2d_transpose(&weights, None, (stride_h, stride_w), (padding_h, padding_w))?;        // Using tensor method
let output = ops::conv2d_transpose(&input, &weights, (stride_h, stride_w), (padding_h, padding_w))?;      // Using ops module function

// Max Pooling
let output = input.max_pool2d((kernel_h, kernel_w), (stride_h, stride_w), (padding_h, padding_w))?;        // Using tensor method
let output = ops::max_pool2d(&input, (kernel_h, kernel_w), (stride_h, stride_w), (padding_h, padding_w))?; // Using ops module function

// Batch Normalization
let output = input.batch_norm(&gamma, &beta, &running_mean, &running_var, epsilon, momentum, training)?;        // Using tensor method
let output = ops::batch_norm(&input, &gamma, &beta, &running_mean, &running_var, epsilon, momentum, training)?; // Using ops module function
```

### Building a Simple Neural Network

```rust
use rust_tensor_library::{Tensor, CpuBackend, ops, kaiming_uniform};

// Define a simple MLP model
struct MlpModel<B: Backend> {
    fc1_weight: Tensor<B>,
    fc1_bias: Tensor<B>,
    fc2_weight: Tensor<B>,
    fc2_bias: Tensor<B>,
}

impl<B: Backend> MlpModel<B> {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self, Error> {
        // Initialize weights with Kaiming initialization
        let fc1_weight_data = kaiming_uniform(&[hidden_dim, input_dim])?;
        let fc1_weight = Tensor::<B>::from_array(fc1_weight_data, true)?;
        let fc1_bias = Tensor::<B>::zeros(&[hidden_dim], true)?;
        
        let fc2_weight_data = kaiming_uniform(&[output_dim, hidden_dim])?;
        let fc2_weight = Tensor::<B>::from_array(fc2_weight_data, true)?;
        let fc2_bias = Tensor::<B>::zeros(&[output_dim], true)?;
        
        Ok(Self {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        })
    }
    
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>, Error> {
        // First layer: Linear + ReLU
        let fc1 = ops::add(&ops::matmul(x, &ops::transpose(&self.fc1_weight, &[1, 0])?)?, &self.fc1_bias)?;
        let hidden = ops::relu(&fc1)?;
        
        // Second layer: Linear
        let output = ops::add(&ops::matmul(&hidden, &ops::transpose(&self.fc2_weight, &[1, 0])?)?, &self.fc2_bias)?;
        
        Ok(output)
    }
    
    fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.fc1_weight, &self.fc1_bias, &self.fc2_weight, &self.fc2_bias]
    }
}
```

## Optimizers

The library provides several optimizers for training neural networks:

### SGD (Stochastic Gradient Descent)

```rust
use rust_tensor_library::{Tensor, CpuBackend, optim::SGD};

// Create an SGD optimizer with learning rate 0.01
let mut optimizer = SGD::new(model.parameters(), 0.01)?;

// With momentum and weight decay
let mut optimizer_with_momentum = SGD::new_with_config(
    model.parameters(),
    0.01,    // learning rate
    0.9,     // momentum
    0.0005   // weight decay
)?;

// Training loop
for epoch in 0..num_epochs {
    // Forward pass
    let output = model.forward(&input)?;
    let loss = compute_loss(&output, &target)?;
    
    // Backward pass
    loss.backward()?;
    
    // Update parameters
    optimizer.step()?;
    
    // Zero gradients for next iteration
    optimizer.zero_grad()?;
}
```

### Adam Optimizer

```rust
use rust_tensor_library::{Tensor, CpuBackend, optim::Adam};

// Create an Adam optimizer
let mut optimizer = Adam::new(
    model.parameters(),
    0.001,  // learning rate
    0.9,    // beta1
    0.999,  // beta2
    1e-8    // epsilon
)?;

// Training loop (same as SGD example)
```

### AdaGrad Optimizer

```rust
use rust_tensor_library::{Tensor, CpuBackend, optim::AdaGrad};

// Create an AdaGrad optimizer
let mut optimizer = AdaGrad::new(
    model.parameters(),
    0.01,   // learning rate
    1e-10   // epsilon
)?;

// Training loop (same as SGD example)
```

### RMSProp Optimizer

```rust
use rust_tensor_library::{Tensor, CpuBackend, optim::RMSProp};

// Create an RMSProp optimizer
let mut optimizer = RMSProp::new(
    model.parameters(),
    0.001,  // learning rate
    0.99,   // alpha (decay rate)
    1e-8    // epsilon
)?;

// Training loop (same as SGD example)
```

### Learning Rate Scheduling

You can manually adjust learning rates between epochs:

```rust
use rust_tensor_library::{Tensor, CpuBackend, optim::SGD};

// Create an SGD optimizer
let mut optimizer = SGD::new(model.parameters(), 0.1)?;

// Training loop with learning rate decay
for epoch in 0..num_epochs {
    // Decay learning rate every 30 epochs
    if epoch > 0 && epoch % 30 == 0 {
        optimizer.set_learning_rate(optimizer.learning_rate() * 0.1);
        println!("Epoch {}: Adjusted learning rate to {}", epoch, optimizer.learning_rate());
    }
    
    // Regular training loop
    // ...
}
```

## Serialization

With the `serialization` feature enabled, you can save and load tensors:

```rust
use rust_tensor_library::{Tensor, CpuBackend};

// Create a tensor
let tensor = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)?;

// Save the tensor to a file
tensor.save_to_file("my_tensor.json")?;

// Load the tensor from a file
let loaded_tensor = Tensor::<CpuBackend>::load_from_file("my_tensor.json")?;
```

### Cross-Device Serialization

You can save a tensor from one device and load it to another:

```rust
// Save a CUDA tensor
let cuda_tensor = Tensor::<CudaBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
cuda_tensor.save_to_file("cuda_tensor.json")?;

// Load it to CPU
let cpu_tensor = Tensor::<CpuBackend>::load_from_file("cuda_tensor.json")?;

// Or vice versa
let cpu_tensor = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
cpu_tensor.save_to_file("cpu_tensor.json")?;
let cuda_tensor = Tensor::<CudaBackend>::load_from_file("cpu_tensor.json")?;
```

## Examples

### Modern API Usage Example

This example demonstrates the modern API with operator overloading and tensor methods:

```rust
use rust_tensor_library::{Tensor, CpuBackend, optim::Adam, Reduction};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Create tensors with gradient tracking
    let x = Tensor::<CpuBackend>::from_vec(vec![0.5, 0.2, 0.1], &[3], true)?;
    let y = Tensor::<CpuBackend>::from_vec(vec![0.8, 0.4, 0.2], &[3], true)?;
    
    // Create model parameters
    let w = Tensor::<CpuBackend>::from_vec(vec![0.1], &[1], true)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![0.0], &[1], true)?;
    
    // Create optimizer
    let mut optimizer = Adam::new(vec![&w, &b], 0.1, 0.9, 0.999, 1e-8)?;
    
    // Training loop
    for epoch in 0..100 {
        // Forward pass using operator overloading and tensor methods
        let pred = &x * &w + &b;  // Using operator overloading
        
        // Compute loss using tensor method
        let loss = pred.mse_loss(&y, Reduction::Mean)?;
        
        // Backward pass
        loss.backward()?;
        
        // Update parameters
        optimizer.step()?;
        optimizer.zero_grad()?;
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.item());
        }
    }
    
    // Final prediction using tensor methods
    let final_pred = x.mul(&w)?.add(&b)?;
    println!("Final parameters: w = {}, b = {}", w.item(), b.item());
    println!("Predictions: {:?}", final_pred.to_vec()?);
    println!("Targets: {:?}", y.to_vec()?);
    
    Ok(())
}
```

The library includes several example applications that demonstrate different use cases:

### MNIST Training (MLP)

```bash
# CPU version
cargo run --example train_mnist_cpu

# GPU version
cargo run --features cuda --example train_mnist_gpu
```

### MNIST Training (CNN)

```bash
# CPU version
cargo run --example train_mnist_cnn_cpu

# GPU version
cargo run --features cuda --example train_mnist_cnn_gpu
```

### Sine Wave Regression

```bash
cargo run --example sine_regression_cpu
```

### Character-Level LSTM RNN

```bash
cargo run --example lstm_char_rnn_cpu
```

### Tensor Serialization

```bash
cargo run --features serialization --example tensor_serialization
```

## Debugging Tips

### Tensor Inspection

```rust
// Print tensor shape and a sample of its data
tensor.show("my_tensor");

// Print only the tensor shape
tensor.show_shape("my_tensor");

// Convert tensor to Vec for inspection
let data = tensor.to_vec()?;
println!("Tensor data: {:?}", data);
```

### Enabling Debug Logs

Enable the `debug_logs` feature to see detailed diagnostic information:

```toml
[dependencies]
rust_tensor_library = { version = "0.1.0", features = ["debug_logs"] }
```

With this feature enabled, the library will print detailed information about tensor operations, which is useful for debugging. The debug logs include:

- Tensor creation and shape information
- Operation execution details
- CUDA kernel launches and synchronization (when using the CUDA backend)
- Gradient computation during backward passes
- Memory management information

Example of debug logs for a CUDA operation:

```
[DEBUG rust_tensor_lib::backend::cuda::ops] sum_along_axis: input_shape=[2, 3], axis=1
[DEBUG rust_tensor_lib::backend::cuda::ops] sum_along_axis: output_shape=[2]
[DEBUG rust_tensor_lib::backend::cuda::ops] Launching kernel with grid_size=1, block_size=256
```

All debug prints in the library use the `debug_println!` macro, which is conditionally compiled based on the `debug_logs` feature flag. This ensures that there is no performance overhead when the feature is disabled.

### Gradient Checking

For complex models, you can use numerical gradient checking to verify your gradients:

```rust
use rust_tensor_library::{test_utils::check_gradients, ops};

// Function to check
let f = |x: &Tensor<CpuBackend>| -> Result<Tensor<CpuBackend>, Error> {
    ops::square(x)
};

// Check gradients
let x = Tensor::<CpuBackend>::from_vec(vec![2.0], &[1], true)?;
let (passed, max_diff) = check_gradients(f, &x, 1e-5)?;

println!("Gradient check passed: {}, max difference: {}", passed, max_diff);
```

## Next Steps

- Explore the [Architecture Overview](ARCHITECTURE.md) to understand the library's design.
- Check the [Performance Guide](PERFORMANCE.md) for optimization tips.
- Try running and modifying the examples to build your own models.
- Contribute to the library by following the guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md).
