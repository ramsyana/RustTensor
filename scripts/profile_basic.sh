#!/bin/bash

# Build in release mode
echo "Building in release mode..."
cargo build --release --all-features

# Run benchmarks
echo "Running benchmarks..."
cargo bench --all-features

# Run the MNIST training example with time profiling
echo "Running MNIST CPU training with time profiling..."
time ./target/release/examples/train_mnist_cpu

# If CUDA is enabled and nsys is available, run GPU profiling
if command -v nsys &> /dev/null; then
    echo "Running GPU profiling with Nsight Systems..."
    nsys profile --stats=true \
        ./target/release/examples/train_mnist_gpu
fi
