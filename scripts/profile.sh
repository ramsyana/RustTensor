#!/bin/bash
set -e  # Exit on error

# Build release target
echo "Building release target..."
cargo build --example train_mnist_gpu --features cuda --release

# Find OUT_DIR
OUT_DIR=$(find target/release/build/rust_tensor_lib-*/out -type d | head -n 1)
if [ -z "$OUT_DIR" ]; then
  echo "Error: Could not find OUT_DIR."
  exit 1
fi
echo "Using OUT_DIR=$OUT_DIR"

# Check if perf is installed
if command -v perf >/dev/null 2>&1; then
    echo "Running CPU profiling with perf..."
    perf record -g --call-graph dwarf \
        ./target/release/examples/train_mnist_cpu
    perf report -g 'graph,0.5,caller'
else
    echo "perf not found, skipping CPU profiling. Install with: sudo apt-get install linux-tools-common linux-tools-generic linux-tools-$(uname -r)"
fi

# If CUDA is enabled, run GPU profiling
if command -v nsys >/dev/null 2>&1; then
    echo "Running GPU profiling with Nsight Systems..."
    # Pass OUT_DIR to nsys
    env OUT_DIR="$OUT_DIR" nsys profile --stats=true \
        ./target/release/examples/train_mnist_gpu
else
    echo "nsys not found, skipping GPU profiling"
fi

# Run benchmarks
echo "Running benchmarks..."
cargo bench
