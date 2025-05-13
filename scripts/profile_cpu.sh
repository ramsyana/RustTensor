#!/bin/bash

# Function to run and time the mnist example
run_mnist_profile() {
    local features=$1
    local output_file="profile_results_${features// /_}.txt"
    
    echo "Running MNIST CPU example with features: $features"
    echo "=== Profile run with features: $features ===" > "$output_file"
    
    # Build with specified features
    cargo build --release --example train_mnist_cpu ${features:+--features "$features"}
    
    # Run multiple times to get average performance
    for i in {1..3}; do
        echo "Run $i:" >> "$output_file"
        RUST_LOG=info time ./target/release/examples/train_mnist_cpu 2>> "$output_file"
        echo "---" >> "$output_file"
    done
    
    echo "Results saved to $output_file"
}

# Run without OpenBLAS
echo "Running baseline CPU profile..."
run_mnist_profile ""

# Run with OpenBLAS if available
if dpkg -l | grep -q libopenblas-dev; then
    echo "Running OpenBLAS CPU profile..."
    run_mnist_profile "cpu_openblas"
else
    echo "OpenBLAS not installed. To install: sudo apt-get install libopenblas-dev"
fi