# RustTensor Library

**A learning-focused, high-performance tensor computation library built from scratch in Rust, featuring automatic differentiation and CPU/CUDA backends.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Vision & Goals

This library is primarily an **educational exploration** into building the core components of a modern deep learning framework. Key goals include:

*   **Deep Understanding:** Gain insight into how Tensors, automatic differentiation (Autograd), backend abstractions (CPU/GPU), and optimizers function internally by implementing them directly.
*   **Performance with Rust & CUDA:** Leverage Rust's safety and performance alongside custom CUDA kernels and cuBLAS integration for efficient GPU acceleration, complementing a solid `ndarray`-based CPU backend.
*   **Rust ML Foundation:** Provide a growing set of building blocks (Tensors, a comprehensive suite of Ops, Autograd, multiple Optimizers, and foundational NN Layers) for defining, training, and experimenting with custom machine learning models, including CNNs and sequence models, entirely within the Rust ecosystem.

## Documentation

* [**User Guide**](docs/USER_GUIDE.md): Step-by-step guide to using the library, from installation to advanced features.
* [**Architecture Overview**](docs/ARCHITECTURE.md): Detailed explanation of the library's design and components.
* [**Performance Guide**](docs/PERFORMANCE.md): Benchmarking, profiling, and optimization information.

## Project Status

**Status:** This library is under active development. While core features like CPU/CUDA backends, autograd, and foundational operations are implemented and tested (sufficient for training MLPs like the MNIST example), it currently serves educational and experimental purposes best.

*   **Strengths:** Clear backend abstraction, working CUDA integration with custom kernels, functional dynamic autograd, extensive set of mathematical and array manipulation operations with CPU/CUDA backends, support for foundational CNN layers (Conv2D, MaxPool2D, Conv2DTranspose), multiple standard optimizers (SGD, Adam, Adagrad, MomentumSGD), and demonstrated capability to build and train MLPs, CNNs, and even character-level LSTMs (from fundamental ops).
*   **Limitations:** While foundational layers like Conv2D, MaxPool2D, and Conv2DTranspose are implemented, more advanced/specialized layers (e.g., optimized RNN/LSTM cells, Attention mechanisms) are future work. API is stabilizing but may still see minor evolutionary changes.

Contributions and feedback are highly welcome!

## Features

*   **Operator Overloading & Ergonomic API:**
    *   Use standard Rust operators (`+`, `-`, `*`, `/`) for arithmetic on tensors.
    *   Intuitive methods like `.mean()`, `.backward()`, `.matmul()`, and more for common operations.
    *   Cleaner, more readable code for model building and experimentation.

### Debugging and Introspection
*   `.show("label")`: Prints the tensor's ID, shape, and a sample of its data.
*   `.show_shape("label")`: Prints the tensor's ID and shape.

*   **CPU & CUDA Backends:**
    *   CPU backend using [`ndarray`](https://crates.io/crates/ndarray) for host computation.
    *   Supports optional integration with system BLAS libraries (like OpenBLAS) for potentially accelerated `matmul` via feature flags (see below).
*   CUDA backend leveraging custom kernels and cuBLAS (via `cust` and `cublas-sys`) for GPU acceleration (requires `cuda` feature).
*   **Dynamic Autograd:**
    *   Constructs computation graphs on-the-fly.
    *   Computes gradients automatically via reverse-mode differentiation.
*   **Comprehensive Operations Suite:**
    *   **Arithmetic:** Add, Subtract, Multiply, Divide (with broadcasting).
    *   **Matrix:** Matmul (CPU/cuBLAS), Transpose.
    *   **Trigonometric:** Sin, Cos, Tan.
    *   **Exponential & Power:** Exp, Log (ln), Pow, Sqrt, Square.
    *   **Activation Functions:** ReLU, Sigmoid, Tanh, Softplus, ELU, LogSoftmax.
    *   **Reduction Operations:** Sum, Mean, Max, Min, Prod, LogSumExp (global or along axes).
    *   **Indexing & Manipulation:** Slice, Concat, ExpandDims, Squeeze, View/Reshape.
    *   **Comparison & Clipping:** Equal, Greater, Less (and variants), Clip.
    *   **Normalization-related:** LogSoftmax.
*   **Optimizers:**
    *   Stochastic Gradient Descent (SGD)
    *   Adam
    *   Adagrad
    *   MomentumSGD
    *   (All optimizers support both CPU and CUDA backends with custom kernels for GPU acceleration where applicable).
*   **Serialization:**
    *   Save and load tensors to/from files with the `serialization` feature.
    *   Seamless cross-device serialization (save from GPU, load to CPU and vice versa).
    *   Preserves tensor data, shape, gradient (if present), and metadata.
*   **Neural Network Layers:**
    *   **Convolutional:** `Conv2D` (with CPU and CUDA im2col/col2im + matmul implementations).
    *   **Pooling:** `MaxPool2D` (with CPU and CUDA implementations, including index tracking for backward pass).
    *   **Transposed Convolution:** `Conv2DTranspose` (implemented for CPU and CUDA).

*   **Rich Examples Suite:**
    *   **MLP for MNIST:** Trains a Multi-Layer Perceptron on the MNIST dataset (CPU: `train_mnist_cpu.rs`, GPU: `train_mnist_gpu.rs`).
    *   **CNN for MNIST:** Demonstrates Convolutional Neural Network training on MNIST, utilizing `Conv2D` and `MaxPool2D` layers (CPU: `train_mnist_cnn_cpu.rs`, GPU: `train_mnist_cnn_gpu.rs`).
    *   **Sine Wave Regression:** A simple MLP model learns to fit a noisy sine wave, showcasing basic regression and optimization (CPU: `sine_regression_cpu.rs`).
    *   **Character-Level LSTM RNN:** A more advanced example building an LSTM cell from fundamental tensor operations to perform character-level text generation, demonstrating the flexibility of the autograd system (CPU: `lstm_char_rnn_cpu.rs`).
*   **Built in Rust:** Aims to provide a memory-safe and performant implementation.

## Requirements

### Basic Setup
*   Rust **1.70** or later (check `Cargo.toml` for specific MSRV if set).
*   Cargo (Rust's package manager).

### Dataset Requirement (MNIST)
*   **Before running or testing any MNIST examples, you must obtain the dataset files:**
    *   `mnist_train.csv`
    *   `mnist_test.csv`
*   Place both files inside a `data/` directory at the project root (i.e., `./data/mnist_train.csv`).
*   These files are commonly available online—please search for "mnist_train.csv" and "mnist_test.csv" to find sources. (Direct links are not provided here.)

### CUDA Support (Optional)
To enable and use the CUDA backend (`--features cuda`):
*   **NVIDIA CUDA Toolkit:** Version **11.0 or later** recommended. This includes the `nvcc` compiler, runtime libraries (like `cudart`), and development libraries (like `cublas`).
*   **NVIDIA GPU:** A CUDA-capable GPU (Compute Capability 3.5+ recommended, check `cust` crate compatibility).
*   **NVIDIA Driver:** An up-to-date driver compatible with your GPU and CUDA Toolkit version.
*   **Environment Variables:** Crucial for both building and running:
    *   `CUDA_PATH`: (**Build & Runtime**) Set to the root directory of your CUDA Toolkit installation (e.g., `/usr/local/cuda-11.8`, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`). Needed to find `nvcc` and headers/libs.
    *   `CUBLAS_LIB_DIR`: (**Build Time**) Path to the cuBLAS library file (e.g., `$CUDA_PATH/lib64`, `%CUDA_PATH%\lib\x64`). Used by `build.rs` to link against cuBLAS.
    *   `LD_LIBRARY_PATH` (Linux/macOS) or `PATH` (Windows): (**Runtime**) Must include the directory containing CUDA runtime libraries (`libcudart.so`, `libcublas.so`, `.dll` equivalents) so the executable can find them. Often this is `$CUDA_PATH/lib64` on Linux or `%CUDA_PATH%\bin` on Windows.

    **Example (Linux/macOS):**
    ```bash
    # Adjust version/path as needed
    export CUDA_PATH=/usr/local/cuda-11.8
    export CUBLAS_LIB_DIR=$CUDA_PATH/lib64
    # Add CUDA libs to runtime linker path
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}
    ```

## Installation

Add this crate to your project's `Cargo.toml`:

```toml
[dependencies]
# CPU only:
rust_tensor_library = "0.1.0"

# --- OR ---

# With CUDA support (ensure environment variables are set *before* building!):
# rust_tensor_library = { version = "0.1.0", features = ["cuda"] }

# With serialization support:
# rust_tensor_library = { version = "0.1.0", features = ["serialization"] }

# With both CUDA and serialization support:
# rust_tensor_library = { version = "0.1.0", features = ["cuda", "serialization"] }
```

## Quick Start

```rust
use rust_tensor_library::{Tensor, CpuBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors that require gradient tracking
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![4.0, 5.0, 6.0], &[3], true)?;
    
    // Perform operations
    let c = &a + &b; // Element-wise addition
    let d = c.mean(None)?; // Global mean reduction
    
    // Print results
    println!("a: {:?}", a.to_vec()?);
    println!("b: {:?}", b.to_vec()?);
    println!("c = a + b: {:?}", c.to_vec()?);
    println!("d = mean(c): {:?}", d.to_vec()?);
    
    // Compute gradients
    d.backward()?;
    
    // Access and print gradients
    if let Some(grad_a_ref) = a.grad() {
        let grad_a_data = CpuBackend::copy_to_host(&*grad_a_ref)?;
        println!("Gradient of a: {:?}", grad_a_data);
        // For d = mean(a+b), and a = [a1, a2, a3], b = [b1, b2, b3]
        // d = ((a1+b1) + (a2+b2) + (a3+b3)) / 3
        // d(d)/da_i = 1/3. So grad_a should be [1/3, 1/3, 1/3]
        // Expected: [0.333..., 0.333..., 0.333...]
    }
    
    if let Some(grad_b_ref) = b.grad() {
        let grad_b_data = CpuBackend::copy_to_host(&*grad_b_ref)?;
        println!("Gradient of b: {:?}", grad_b_data);
        // Similarly, d(d)/db_i = 1/3. So grad_b should be [1/3, 1/3, 1/3]
        // Expected: [0.333..., 0.333..., 0.333...]
    }

    Ok(())
}
```

## Running Examples

*(**Note:** The MNIST examples require `mnist_train.csv` and `mnist_test.csv` files. Place them in a `data/` directory in the project root **before running or testing**. You can search for these files online; they are widely available as CSV exports of the standard MNIST dataset.)*

### CPU Example
Trains a simple MLP on MNIST using the CPU backend.
```bash
cargo run --example train_mnist_cpu
```

### GPU Example
Trains the same MLP on MNIST using the CUDA backend. Requires the `cuda` feature and proper CUDA environment setup.
```bash
cargo run --features cuda --example train_mnist_gpu
```

### CNN Example
Trains a CNN on MNIST using the CPU backend.
```bash
cargo run --example train_mnist_cnn_cpu
```

### CNN GPU Example
Trains a CNN on MNIST using the CUDA backend. Requires the `cuda` feature and proper CUDA environment setup.
```bash
cargo run --features cuda --example train_mnist_cnn_gpu
```

### Sine Wave Regression Example
A simple MLP model learns to fit a noisy sine wave, showcasing basic regression and optimization.
```bash
cargo run --example sine_regression_cpu
```

### Character-Level LSTM RNN Example
A more advanced example building an LSTM cell from fundamental tensor operations to perform character-level text generation, demonstrating the flexibility of the autograd system.
```bash
cargo run --example lstm_char_rnn_cpu
```

### Tensor Serialization Example
Demonstrates how to save and load tensors to/from files using the serialization feature.

```bash
cargo run --features serialization --example tensor_serialization
```

#### Serialization Usage

To use the serialization feature in your own code:

```rust
use rust_tensor_lib::{Tensor, CpuBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor
    let tensor = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)?;
    
    // Save the tensor to a file
    tensor.save_to_file("my_tensor.json")?;
    
    // Load the tensor from a file
    let loaded_tensor = Tensor::<CpuBackend>::load_from_file("my_tensor.json")?;
    
    // Verify the tensors are the same
    assert_eq!(tensor.shape(), loaded_tensor.shape());
    
    Ok(())
}
```

The serialization feature preserves:
- Tensor data values
- Tensor shape
- `requires_grad` flag
- Device information

This allows for seamless saving and loading of tensors during model development and deployment.

### CUDA Serialization Example
Shows cross-device serialization between CPU and GPU backends.
```bash
cargo run --features "cuda,serialization" --example cuda_serialization
```

## Running Tests

Run tests using the default features (CPU backend only):
```bash
cargo test
```

Run tests including the CUDA backend tests (requires `cuda` feature and environment setup):
```bash
cargo test --features cuda
```

Run specific test suites for targeted testing:
```bash
# CPU Ops & Backward Tests
cargo test --test ops_cpu_tests

# CUDA Forward Ops Tests
cargo test --features cuda --test ops_cuda_tests

# CUDA Backward Ops Tests
cargo test --features cuda --test ops_cuda_backward_tests

# Other test files
cargo test --test array_tensor_tests
cargo test --test hooks_tests
cargo test --test init_tests
cargo test --test ops_cpu_backward_tests
cargo test --test ops_cuda_backward_tests --features cuda
cargo test --test ops_edge_cases_tests
cargo test --test ops_overloading_tests
cargo test --test optim_tests
cargo test --test random_ops_tests
cargo test --test sgd_cuda_tests --features cuda
cargo test --test tensor_method_tests
cargo test --features cuda --test tensor_method_cuda_tests_map
cargo test --test test_concat
cargo test --features cuda --test test_cuda_conv2d
cargo test --features cuda --test test_cuda_logsumexp_gradient
cargo test --features cuda --test test_cuda_sum_along_axis
cargo test --test test_expand_dims
cargo test --test test_gradient_checker
cargo test --features cuda --test test_gradient_checker_cuda
```

## Benchmarking

This library includes a suite of benchmarks (see [`benches/tensor_ops_bench.rs`](benches/tensor_ops_bench.rs)) to help you quantify and compare the performance of CPU and CUDA backends for core tensor operations such as matrix multiplication (`matmul`), addition, and activation functions.

The benchmark suite covers a broader range of core tensor operations including matrix multiplication, element-wise operations, transpose, various reductions, and fundamental neural network ops like LogSoftmax.

### Running Benchmarks

You can run the benchmarks for different backends to compare performance. Note that results may vary based on your hardware.

- **CPU benchmarks (default backend):**
  ```bash
  cargo bench --bench tensor_ops_bench
  ```
- **CPU benchmarks (with system OpenBLAS):**
  ```bash
  # Make sure libopenblas-dev (or equivalent) is installed first!
  cargo bench --bench tensor_ops_bench --features cpu_openblas
  ```
- **CUDA benchmarks:**
  ```bash
  cargo bench --bench tensor_ops_bench --no-default-features --features cuda
  ```
- **Compare CPU (OpenBLAS) and CUDA:**
  ```bash
  cargo bench --bench tensor_ops_bench --features cpu_openblas,cuda
  ```

After running, compare the output times for relevant benchmarks (e.g., `cpu_matmul_1024` vs. `gpu_matmul_1024`). This helps you quantify the performance benefits of the CUDA backend over CPU, and the impact of using OpenBLAS for CPU matrix multiplication.

**Tip:** You can add your own benchmarks to `benches/tensor_ops_bench.rs` to measure new operations or custom workloads relevant to your use case.

**Why benchmark?**
- Benchmarking helps you understand the speedup provided by GPU acceleration and optimized CPU libraries.
- It is essential for validating performance improvements and making informed decisions about backend selection for your workloads.

## Profiling

This library includes support for performance profiling using system tools. For detailed performance analysis, benchmarking results, and optimization insights, see [PERFORMANCE.md](docs/PERFORMANCE.md).

### Quick Start - CPU Profiling (Linux)
```bash
# Install perf
sudo apt-get install linux-tools-common linux-tools-generic linux-tools-$(uname -r)

# Profile CPU example
perf record -g --call-graph dwarf ./target/release/examples/train_mnist_cpu
perf report -g 'graph,0.5,caller'
```

### Quick Start - GPU Profiling
```bash
# Ensure CUDA environment is set up
export CUDA_PATH=/usr/local/cuda-11.8  # Adjust version as needed
export CUBLAS_LIB_DIR=$CUDA_PATH/lib64
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}

# Profile with Nsight Systems
nsys profile --stats=true -o mnist_gpu_profile ./target/release/examples/train_mnist_gpu
```

See [PERFORMANCE.md](docs/PERFORMANCE.md) for:
- Detailed profiling instructions
- Performance characteristics
- Benchmark results
- Known optimizations
- Hardware-specific considerations

## Feature Flags

*   `cuda`: Enables CUDA GPU support, including kernels and cuBLAS integration. Requires CUDA Toolkit and environment setup.
*   `serialization`: Enables tensor serialization support using `serde` and `serde_json`. Allows saving and loading tensors to/from JSON files with `save_to_file` and `load_from_file` methods.
*   `mnist`: Enables MNIST dataset loading utilities (`src/data.rs`). Used by examples.
*   `debug_logs`: Enables detailed diagnostic `println!` statements, useful for development and debugging backend operations.
*   `cpu_openblas`: **(Optional)** Enables the use of a system-installed OpenBLAS library for CPU matrix multiplication (`matmul`). When this feature is enabled, the `openblas-src` dependency is activated, and `ndarray` will automatically detect and use OpenBLAS for faster matrix operations. This can significantly improve performance for large matrix multiplications on the CPU compared to the default backend.
    *   **Prerequisite:** You must have the OpenBLAS development library installed on your system *before* building with this feature.
        *   **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install libopenblas-dev`
        *   **Fedora:** `sudo dnf install openblas-devel`
        *   **macOS (Homebrew):** `brew install openblas` (might require setting environment variables like `OPENBLAS_PATH` if not found automatically).
        *   **Windows:** Requires more complex setup, often involving MSYS2 or pre-compiled binaries. Refer to `ndarray` or `openblas-src` documentation.
    *   **Usage:** Compile or run with the feature flag:
        ```bash
        cargo run --features cpu_openblas --example your_example
        cargo build --features cpu_openblas --release
        ```

## Development

### Building Documentation
```bash
# Build docs for all public items, including those behind the 'cuda' feature
cargo doc --all-features --no-deps
# Open the documentation in your browser
cargo doc --open
```

### Code Formatting & Linting
```bash
# Format code according to rustfmt.toml or defaults
cargo fmt
# Run clippy linter with all features, treat warnings as errors
cargo clippy --all-features --all-targets -- -D warnings
```

## Advanced Topics & Future Considerations

### Higher-Order Gradients (H.O.G.)

This library currently implements robust first-order automatic differentiation, which is sufficient for training most common deep learning models. Higher-Order Gradients (e.g., gradients of gradients) are useful for more advanced applications like meta-learning, some reinforcement learning algorithms, and model analysis (e.g., Hessians).

Implementing general H.O.G. with the current dynamic, tape-based autograd system (where backward passes are defined by Rust closures) presents significant architectural challenges. It would likely require:
1.  Making all operations within the backward pass graph-aware themselves, allowing the backward pass to construct its own differentiable computation graph.
2.  Or, a shift towards a more symbolic representation of the computation graph prior to execution.

Given the complexity and the educational focus of this library on core components, **Higher-Order Gradients are currently considered out of scope for immediate implementation.** Future exploration might revisit this if the library's architecture evolves or if specific use cases requiring H.O.G. become a priority. The current focus remains on providing a solid and performant foundation for first-order differentiation and model building.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on pull requests, issues, and the development process.

## Code of Conduct

This project aims to be a welcoming community. Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Copyright (c) 2025 Ramsyana <ramsyana@mac.com>

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Support My Work

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/ramsy)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-%23FFDD00?style=flat-square&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/ramsy)
