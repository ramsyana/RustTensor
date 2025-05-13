#!/bin/bash

# Function to check if nsys (Nvidia Nsight Systems) is available
check_nsys() {
    if ! command -v nsys &> /dev/null; then
        echo "Error: nsys command not found. Please ensure NVIDIA Nsight Systems is installed and in your PATH."
        exit 1
    fi
}

# Function to check CUDA environment
check_cuda_env() {
    if [ -z "${CUDA_PATH}" ]; then
        echo "Warning: CUDA_PATH is not set. Using default /usr/local/cuda"
        export CUDA_PATH=/usr/local/cuda
    fi
    
    if [ -z "${LD_LIBRARY_PATH}" ] || [[ ! "${LD_LIBRARY_PATH}" =~ "${CUDA_PATH}/lib64" ]]; then
        echo "Adding CUDA libraries to LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
    fi
}

# Function to run MNIST GPU training with profiling
profile_mnist_gpu() {
    local output_base="mnist_gpu_profile"
    
    echo "Building MNIST GPU example..."
    cargo build --release --features cuda --example train_mnist_gpu
    
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi
    
    echo "Running NVIDIA Nsight Systems profiling..."
    nsys profile \
        --stats=true \
        --output="${output_base}" \
        --trace=cuda,cublas \
        --cuda-memory-usage=true \
        ./target/release/examples/train_mnist_gpu
        
    echo "Analyzing results..."
    nsys stats "${output_base}.nsys-rep"
}

# Function to run tensor operations benchmarks with profiling
profile_benchmarks() {
    local output_base="tensor_ops_profile"
    
    echo "Building benchmarks..."
    cargo build --release --features cuda --bench tensor_ops_bench
    
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi
    
    echo "Running NVIDIA Nsight Systems profiling on benchmarks..."
    nsys profile \
        --stats=true \
        --output="${output_base}" \
        --trace=cuda,cublas \
        --cuda-memory-usage=true \
        ./target/release/deps/tensor_ops_bench
        
    echo "Analyzing results..."
    nsys stats "${output_base}.nsys-rep"
}

# Main execution
main() {
    check_nsys
    check_cuda_env
    
    echo "=== Starting GPU Profiling ==="
    
    # Profile MNIST training
    echo "Profiling MNIST GPU training..."
    profile_mnist_gpu
    
    # Profile benchmarks
    echo -e "\nProfiling tensor operations benchmarks..."
    profile_benchmarks
    
    echo -e "\nProfiling complete. Check the .nsys-rep files for detailed analysis."
    echo "You can open these files with NVIDIA Nsight Systems UI for timeline visualization."
}

main