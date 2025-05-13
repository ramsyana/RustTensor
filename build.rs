use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Handle OpenBLAS linking when cpu_openblas feature is enabled
    if env::var("CARGO_FEATURE_CPU_OPENBLAS").is_ok() {
        println!("cargo:warning=CPU OpenBLAS feature enabled, configuring linking...");
        
        // Link against OpenBLAS
        println!("cargo:rustc-link-lib=openblas");
        
        // Ensure rebuild if relevant env vars change
        println!("cargo:rerun-if-env-changed=OPENBLAS_PATH");
    }
    
    // Handle CUDA feature
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        // Configure cuBLAS linking
        if let Ok(lib_dir) = env::var("CUBLAS_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        }

        // Link against cuBLAS
        if env::var("CUBLAS_STATIC").is_ok() {
            println!("cargo:rustc-link-lib=static=cublas");
        } else {
            println!("cargo:rustc-link-lib=dylib=cublas");
        }

        // Ensure rebuild if relevant env vars change
        println!("cargo:rerun-if-env-changed=CUBLAS_LIB_DIR");
        println!("cargo:rerun-if-env-changed=CUBLAS_STATIC");
        println!("cargo:rerun-if-env-changed=CUBLAS_LIBS");
        println!("cargo:warning=CUDA feature enabled, compiling kernels...");

        // Find nvcc - Use `which` crate first for better cross-platform compatibility
        let nvcc_path = match which::which("nvcc") {
            Ok(path) => path,
            Err(_) => {
                // Fallback to checking CUDA_PATH or common locations if `which` fails
                if let Ok(cuda_path) = env::var("CUDA_PATH") {
                    PathBuf::from(cuda_path).join("bin").join("nvcc")
                } else {
                    // Try default locations (adjust if needed for your system)
                    ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"]
                         .iter()
                         .map(PathBuf::from)
                         .find(|p| p.exists())
                         .expect("nvcc not found. Ensure CUDA Toolkit is installed and nvcc is in PATH, or set CUDA_PATH.")
                }
            }
        };
        println!("cargo:warning=Using nvcc found at: {:?}", nvcc_path);

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        // --- Define Kernels to Compile ---
        let kernels_to_compile = [
            ("src/backend/cuda/kernels/elementwise.cu", "elementwise.ptx"),
            ("src/backend/cuda/kernels/reduction.cu", "reduction.ptx"),
            ("src/backend/cuda/kernels/optimizer.cu", "optimizer.ptx"),
            ("src/backend/cuda/kernels/transpose.cu", "transpose.ptx"),
            (
                "src/backend/cuda/kernels/log_softmax_fused.cu",
                "log_softmax_fused.ptx",
            ),
            ("src/backend/cuda/kernels/conv.cu", "conv.ptx"),
            ("src/backend/cuda/kernels/pooling.cu", "pooling.ptx"),
            ("src/backend/cuda/kernels/array_ops.cu", "array_ops.ptx"),
        ];

        for (src_path, ptx_filename) in kernels_to_compile {
            let ptx_path = out_dir.join(ptx_filename);
            println!(
                "cargo:warning=Compiling {} to {}",
                src_path,
                ptx_path.display()
            );

            // --- Register kernels (for verification or future use) ---
            let _expected_kernels: &[&str] = match src_path {
                "src/backend/cuda/kernels/elementwise.cu" => &[
                    "add_kernel",
                    "mul_kernel",
                    "sub_kernel",
                    "div_kernel",
                    "div_scalar_kernel",
                    "relu_kernel",
                    "exp_kernel",
                    "ln_kernel",
                    "broadcast_kernel",
                    "relu_backward_kernel",
                    "sgd_step_kernel",
                    "sum_along_axis_kernel",
                    "max_along_axis_kernel",
                    "logsumexp_along_axis_kernel",
                    "fill_scalar_kernel",
                    "abs_kernel",
                    "abs_backward_kernel",
                    "sigmoid_kernel",
                    "sigmoid_backward_kernel",
                    "tanh_kernel",
                    "tanh_backward_kernel",
                    "max_backward_kernel",
                    "min_backward_kernel",
                    "prod_backward_kernel",
                    "logsumexp_backward_kernel",
                    "softplus_kernel",
                    "softplus_backward_kernel",
                    "powf_kernel",
                    "powf_backward_kernel",
                    "square_kernel",
                    "square_backward_kernel",
                    "maximum_kernel",
                    "maximum_backward_kernel",
                    "minimum_kernel",
                    "minimum_backward_kernel",
                    "equal_kernel",
                    "greater_kernel",
                    "greater_equal_kernel",
                    "less_kernel",
                    "less_equal_kernel",
                    "not_equal_kernel",
                    "sin_kernel",
                    "sin_backward_kernel",
                    "cos_kernel",
                    "cos_backward_kernel",
                    "tan_kernel",
                    "tan_backward_kernel",
                    "elu_kernel",
                    "elu_backward_kernel",
                    "add_bias_4d_kernel",
                    "clip_kernel",
                    "clip_backward_kernel",
                ],
                "src/backend/cuda/kernels/reduction.cu" => &[
                    "sum_reduction_kernel",
                    "max_reduction_kernel",
                    "min_reduction_kernel",
                    "prod_reduction_kernel",
                    "logsumexp_reduction_kernel",
                    "argmax_along_axis_kernel",
                    "argmin_along_axis_kernel",
                ],
                "src/backend/cuda/kernels/optimizer.cu" => &[
                    "adam_step_kernel",
                    "momentum_sgd_step_kernel",
                    "adagrad_step_kernel",
                    "sgd_step_kernel",
                ],
                "src/backend/cuda/kernels/transpose.cu" => &["transpose_2d_kernel"],
                "src/backend/cuda/kernels/log_softmax_fused.cu" => &["log_softmax_fused_kernel"],
                "src/backend/cuda/kernels/conv.cu" => &["im2col_kernel", "col2im_kernel"],
                "src/backend/cuda/kernels/pooling.cu" => &["max_pool2d_forward_kernel", "max_pool2d_backward_kernel"],
                "src/backend/cuda/kernels/array_ops.cu" => &["slice_kernel", "slice_backward_kernel", "concat_kernel", "concat_backward_kernel"],
                _ => &[],
            };

            println!(
                "[DEBUG build.rs] Registering CUDA kernels for {}: {:?}",
                src_path, _expected_kernels
            );

            // --- End kernel registration ---

            let status = Command::new(&nvcc_path)
                .arg("--ptx") // Output PTX assembly
                .arg("-O3") // Optimization level
                .arg("--use_fast_math") // Can improve performance, use with caution
                // Optional: Add architecture flag if needed, e.g., for specific GPU features
                // .arg("-gencode").arg("arch=compute_75,code=sm_75") // Example for Turing
                // .arg("-gencode").arg("arch=compute_86,code=sm_86") // Example for Ampere
                .arg("-o")
                .arg(&ptx_path)
                .arg(src_path)
                .status()
                .unwrap_or_else(|e| panic!("Failed to execute nvcc for {}: {}", src_path, e));

            if !status.success() {
                panic!("nvcc failed to compile {}", src_path);
            }
            println!("cargo:rerun-if-changed={}", src_path);
        }
    } else {
        println!("cargo:warning=CUDA feature not enabled, skipping kernel compilation.");
    }
}
