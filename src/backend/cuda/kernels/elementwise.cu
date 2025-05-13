#include <stdio.h>  // for printf
#include <math.h>    // for mathematical functions

// Define a small tolerance for floating-point equality checks
#define EQUAL_TOLERANCE 1e-9f

// CUDA kernel for elementwise operations
// Will be populated with add, mul, and other elementwise operations

extern "C" {
    extern "C" __global__ void abs_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = fabsf(x[idx]);
        }
    }
    extern "C" __global__ void abs_backward_kernel(const float* x, const float* grad_out, float* grad_in, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float sign = (x[idx] > 0.0f) ? 1.0f : ((x[idx] < 0.0f) ? -1.0f : 0.0f);
            grad_in[idx] = grad_out[idx] * sign;
        }
    }
    extern "C" __global__ void add_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = a[idx] + b[idx];
        }
    }

    extern "C" __global__ void mul_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = a[idx] * b[idx];
        }
    }

    extern "C" __global__ void relu_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = fmaxf(0.0f, x[idx]);
        }
    }

    extern "C" __global__ void div_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            if (b[idx] == 0.0f) {
                out[idx] = INFINITY;  // Or handle division by zero differently if needed
            } else {
                out[idx] = a[idx] / b[idx];
            }
        }
    }

    extern "C" __global__ void div_scalar_kernel(const float* x, float scalar, float* out, int n) { // Pass scalar by value
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // Handle division by zero if necessary, although host checks it
            if (scalar == 0.0f) {
                out[idx] = INFINITY; // Or NAN
            } else {
                out[idx] = x[idx] / scalar;
            }
        }
    }

    extern "C" __global__ void relu_backward_kernel(const float* input, const float* grad_out, float* grad_in, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            grad_in[idx] = input[idx] > 0.0f ? grad_out[idx] : 0.0f;
        }
    }

    extern "C" __global__ void elu_kernel(const float* x, float* out, float alpha, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float x_val = x[idx];
            if (x_val < 0.0f) {
                out[idx] = alpha * (expf(x_val) - 1.0f);
            } else {
                out[idx] = x_val;
            }
        }
    }

    extern "C" __global__ void elu_backward_kernel(const float* x, const float* grad_out, float* grad_in, float alpha, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float x_val = x[idx];
            float derivative;
            if (x_val < 0.0f) {
                derivative = alpha * expf(x_val);
            } else {
                derivative = 1.0f;
            }
            grad_in[idx] = grad_out[idx] * derivative;
        }
    }

    #ifndef NAN
    #define NAN __int_as_float(0x7fffffff)
    #endif

    extern "C" __global__ void broadcast_kernel(const float* input, float* output,
                                                const int* input_shape, const int* output_shape,
                                                const int* input_strides, const int* output_strides,
                                                int input_ndim, int output_ndim,
                                                int n_output, int n_input) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_output) {
            const int MAX_DIMS = 8;
            if (input_ndim > MAX_DIMS || output_ndim > MAX_DIMS) return;
            int coords[MAX_DIMS];
            int temp = idx;

            // Calculate output coordinates from linear index
            for (int i = output_ndim - 1; i >= 0; --i) {
                if (output_shape[i] > 0) {
                    coords[i] = temp % output_shape[i];
                    temp /= output_shape[i];
                } else {
                    coords[i] = 0;
                }
            }

            // Calculate input_idx using output coordinates and padded input strides
            int input_idx = 0;
            // Special case: if input is scalar (n_input == 1), always use index 0
            if (n_input == 1) {
                input_idx = 0;
            } else {
                for (int i = 0; i < output_ndim; ++i) {
                    input_idx += coords[i] * input_strides[i]; // input_strides is padded and has 0 for broadcast dims
                }
                // Safety check: Ensure calculated input_idx is within bounds
                if (input_idx < 0 || input_idx >= n_input) {
                    output[idx] = NAN;
                    return;
                }
            }
            output[idx] = input[input_idx];
        }
    }

    extern "C" __global__ void exp_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = expf(x[idx]);
        }
    }

    extern "C" __global__ void ln_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = logf(x[idx]);  // Returns -inf for x <= 0
        }
    }

    extern "C" __global__ void sub_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = a[idx] - b[idx];
        }
    }

    extern "C" __global__ void sgd_step_kernel(float* w, const float* dw, float lr, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            w[idx] -= lr * dw[idx];
        }
    }

    extern "C" __global__ void fill_scalar_kernel(float* out, float value, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = value;
        }
    }

    // New custom 2D transpose kernel
    extern "C" __global__ void transpose_2d_kernel(
        const float* input,
        float* output,
        int rows, // original rows
        int cols  // original cols
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // output column index
        int y = blockIdx.y * blockDim.y + threadIdx.y; // output row index
        if (x < rows && y < cols) {
            int input_idx = x * cols + y;           // input[x][y]
            int output_idx = y * rows + x;          // output[y][x]
            output[output_idx] = input[input_idx];
        }
    }
    
    // Efficient kernel for adding bias to 4D tensors (NCHW format)
    // This avoids creating large intermediate tensors for broadcasting
    extern "C" __global__ void add_bias_4d_kernel(
        float* output,       // [N, C, H, W] tensor to add bias to (in-place)
        const float* bias,   // [C] bias tensor
        int N, int C, int H, int W,
        int total_elements
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements) {
            // Calculate NCHW indices from flat index
            int w_idx = idx % W;
            int h_idx = (idx / W) % H;
            int c_idx = (idx / (W * H)) % C;
            int n_idx = idx / (W * H * C);
            
            // Add bias to the output at the current position
            // Only need to index into bias with the channel index
            output[idx] += bias[c_idx];
        }
    }

    // Helper functions for multi-dimensional index calculations
    __device__ inline int multi_dim_to_flat_index(const int* coords, const int* strides, int ndim) {
        int index = 0;
        for (int i = 0; i < ndim; ++i) {
            index += coords[i] * strides[i];
        }
        return index;
    }

    __device__ inline void flat_index_to_multi_dim(int flat_index, const int* shape, int ndim, int* coords) {
        int current_index = flat_index;
        for (int i = ndim - 1; i >= 0; --i) {
            if (shape[i] > 0) { // Avoid division by zero for empty dimensions
                coords[i] = current_index % shape[i];
                current_index /= shape[i];
            } else {
                coords[i] = 0;
            }
        }
    }

    extern "C" __global__ void sum_along_axis_kernel(
        const float* input, float* output,
        const int* input_shape, const int* input_strides,
        const int* output_shape, const int* output_strides,
        int input_ndim, int output_ndim, int axis, int n_output,
        int n_input
    ) {
        int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (output_idx >= n_output) {
            return;
        }

        const int MAX_DIMS = 8;
        if (input_ndim > MAX_DIMS || (output_ndim > 0 && output_ndim > MAX_DIMS)) {
            return;
        }

        // Special case for scalar output
        if (output_ndim == 0) {
            if (output_idx == 0) {  // Only first thread computes the scalar result
                float sum = 0.0f;
                int reduction_dim_size = input_shape[axis];
                int stride_for_axis = input_strides[axis];
                int input_idx_base = 0;  // Start from beginning of input since result is scalar
                
                for (int k = 0; k < reduction_dim_size; ++k) {
                    int current_input_idx = input_idx_base + k * stride_for_axis;
                    if (current_input_idx >= 0 && current_input_idx < n_input) {
                        sum += input[current_input_idx];
                    }
                }
                output[0] = sum;
            }
            return;
        }

        // Regular case for non-scalar output
        int output_coords[MAX_DIMS] = {0};  // Initialize to zero
        int input_coords[MAX_DIMS] = {0};   // Initialize to zero

        // Convert output index to coordinates
        flat_index_to_multi_dim(output_idx, output_shape, output_ndim, output_coords);

        // Map output coordinates to input coordinates, filling in the reduction axis
        int current_out_dim = 0;
        for (int i = 0; i < input_ndim; ++i) {
            if (i == axis) {
                input_coords[i] = 0;  // Will iterate over this dimension
            } else {
                if (current_out_dim < output_ndim) {
                    input_coords[i] = output_coords[current_out_dim];
                    current_out_dim++;
                } else {
                    input_coords[i] = 0;
                }
            }
        }

        // Compute base input index for this output element
        int input_idx_base = multi_dim_to_flat_index(input_coords, input_strides, input_ndim);
        int stride_for_axis = input_strides[axis];
        int reduction_size = input_shape[axis];

        // Sum over the reduction dimension
        float sum = 0.0f;
        for (int k = 0; k < reduction_size; ++k) {
            int input_idx = input_idx_base + k * stride_for_axis;
            if (input_idx >= 0 && input_idx < n_input) {
                sum += input[input_idx];
            }
        }

        output[output_idx] = sum;
    }

    extern "C" __global__ void max_along_axis_kernel(
        const float* input, float* output,
        const int* input_shape, const int* input_strides,
        const int* output_shape, const int* output_strides,
        int input_ndim, int output_ndim, int axis, int n_output,
        int n_input
    ) {
        int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (output_idx >= n_output) {
            return;
        }

        const int MAX_DIMS = 8;
        if (input_ndim > MAX_DIMS || (output_ndim > 0 && output_ndim > MAX_DIMS)) {
            return;
        }

        // Special case for scalar output
        if (output_ndim == 0) {
            if (output_idx == 0) {  // Only first thread computes the scalar result
                float max_val = -INFINITY;
                int reduction_dim_size = input_shape[axis];
                int stride_for_axis = input_strides[axis];
                int input_idx_base = 0;  // Start from beginning of input since result is scalar
                
                for (int k = 0; k < reduction_dim_size; ++k) {
                    int current_input_idx = input_idx_base + k * stride_for_axis;
                    if (current_input_idx >= 0 && current_input_idx < n_input) {
                        max_val = fmaxf(max_val, input[current_input_idx]);
                    }
                }
                output[0] = max_val;
            }
            return;
        }

        // Regular case for non-scalar output
        int output_coords[MAX_DIMS] = {0};  // Initialize to zero
        int input_coords[MAX_DIMS] = {0};   // Initialize to zero

        // Convert output index to coordinates
        flat_index_to_multi_dim(output_idx, output_shape, output_ndim, output_coords);

        // Map output coordinates to input coordinates, filling in the reduction axis
        int current_out_dim = 0;
        for (int i = 0; i < input_ndim; ++i) {
            if (i == axis) {
                input_coords[i] = 0;  // Will iterate over this dimension
            } else {
                if (current_out_dim < output_ndim) {
                    input_coords[i] = output_coords[current_out_dim];
                    current_out_dim++;
                } else {
                    input_coords[i] = 0;
                }
            }
        }

        // Compute base input index for this output element
        int input_idx_base = multi_dim_to_flat_index(input_coords, input_strides, input_ndim);
        int stride_for_axis = input_strides[axis];
        int reduction_size = input_shape[axis];

        // Find max over the reduction dimension
        float max_val = -INFINITY;
        for (int k = 0; k < reduction_size; ++k) {
            int input_idx = input_idx_base + k * stride_for_axis;
            if (input_idx >= 0 && input_idx < n_input) {
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }

        output[output_idx] = max_val;
    }

    /*
     * Computes the logsumexp reduction along a specified axis.
     * logsumexp(x) = log(sum(exp(x - max(x)))) + max(x)
     * This is numerically stable as it avoids overflow in exp().
     */
    extern "C" __global__ void logsumexp_along_axis_kernel(
        const float* input, float* output,
        const int* input_shape, const int* input_strides,
        const int* output_shape, const int* output_strides,
        int input_ndim, int output_ndim, int axis, int n_output,
        int n_input
    ) {
        // Process one output element per thread
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_output) {
            return;
        }

        const int MAX_DIMS = 8;
        if (input_ndim > MAX_DIMS || output_ndim > MAX_DIMS) {
            // Too many dimensions to handle in this kernel
            if (idx == 0) {
                printf("[ERROR] logsumexp_along_axis_kernel: Too many dimensions (input_ndim=%d, output_ndim=%d, MAX_DIMS=%d)\n", 
                       input_ndim, output_ndim, MAX_DIMS);
            }
            return;
        }

        // Convert output index to output coordinates
        int output_coords[MAX_DIMS] = {0};
        int temp = idx;
        for (int i = output_ndim - 1; i >= 0; --i) {
            if (output_shape[i] > 0) {
                output_coords[i] = temp % output_shape[i];
                temp /= output_shape[i];
            }
        }

        // Calculate output index in flattened output tensor
        int output_idx = multi_dim_to_flat_index(output_coords, output_strides, output_ndim);

        // Map output coordinates to input coordinates
        int input_coords[MAX_DIMS] = {0};
        int current_out_dim = 0;
        for (int i = 0; i < input_ndim; ++i) {
            if (i == axis) {
                input_coords[i] = 0;  // Will iterate over this dimension
            } else {
                if (current_out_dim < output_ndim) {
                    input_coords[i] = output_coords[current_out_dim];
                    current_out_dim++;
                } else {
                    input_coords[i] = 0;
                }
            }
        }

        // Compute base input index for this output element
        int input_idx_base = multi_dim_to_flat_index(input_coords, input_strides, input_ndim);
        int stride_for_axis = input_strides[axis];
        int reduction_size = input_shape[axis];

        // Step 1: Find max value along the reduction dimension
        float max_val = -INFINITY;
        for (int k = 0; k < reduction_size; ++k) {
            int input_idx = input_idx_base + k * stride_for_axis;
            if (input_idx >= 0 && input_idx < n_input) {
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }

        // Step 2: Compute sum of exp(x - max_val)
        float sum_exp = 0.0f;
        for (int k = 0; k < reduction_size; ++k) {
            int input_idx = input_idx_base + k * stride_for_axis;
            if (input_idx >= 0 && input_idx < n_input) {
                float shifted_val = input[input_idx] - max_val;
                sum_exp += expf(shifted_val);
            }
        }

        // Step 3: Compute final logsumexp result: log(sum_exp) + max_val
        float result = logf(sum_exp) + max_val;
        output[output_idx] = result;
    }

    extern "C" __global__ void log_softmax_kernel(
        const float* input,
        float* output,
        const float* max_vals,
        const float* sums,
        const int* input_shape,
        const int* reduced_shape,
        const int* reduced_strides,
        int axis,
        int reduced_ndim,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) {
            return;
        }

        // For scalar output (reduced_ndim == 0), directly compute without extra indexing
        if (reduced_ndim == 0) {
            float x = input[idx];
            float max_val = max_vals[0];
            float sum_val = sums[0];
            float shifted_val = x - max_val;
            output[idx] = shifted_val - log1pf(sum_val - expf(shifted_val)); // More numerically stable
            return;
        }

        // For non-scalar case, compute output index by mapping through reduction
        const int MAX_DIMS = 8;
        if (input_shape[0] > MAX_DIMS) return;

        // Convert linear index to input coordinates
        int input_coords[MAX_DIMS] = {0};
        int temp = idx;
        for (int i = input_shape[0] - 1; i >= 0; --i) {
            if (input_shape[i] > 0) {
                input_coords[i] = temp % input_shape[i];
                temp /= input_shape[i];
            }
        }

        // Calculate reduced index by skipping the reduction axis
        int reduced_idx = 0;
        int out_dim = 0;
        for (int i = 0; i < input_shape[0]; ++i) {
            if (i != axis) {
                if (out_dim < reduced_ndim) {
                    reduced_idx += input_coords[i] * reduced_strides[out_dim];
                    out_dim++;
                }
            }
        }

        // Compute log_softmax more carefully
        float x = input[idx];
        float max_val = max_vals[reduced_idx];
        float sum_val = sums[reduced_idx];
        float shifted_val = x - max_val;
        output[idx] = shifted_val - log1pf(sum_val - expf(shifted_val)); // More numerically stable
    }

    /*
     * Computes the backward pass for the max reduction operation.
     * Handles broadcasting of y and grad_output, and ties in max values.
     */
    extern "C" __global__ void max_backward_kernel(
        const float* x,              // Input tensor data
        const float* y,              // Forward output tensor data (max values)
        const float* grad_output,    // Gradient flowing back from the next layer
        float* grad_input,         // Output buffer for the calculated input gradient
        const int* x_shape,          // Shape of x
        const int* y_shape,          // Shape of y
        const int* grad_output_shape,// Shape of grad_output (should match y_shape)
        const int* x_strides,        // Strides of x
        const int* y_strides,        // Strides of y
        const int* grad_output_strides, // Strides of grad_output
        int x_ndim,                  // Number of dimensions of x
        int y_ndim,                  // Number of dimensions of y (can be 0 for global)
        int grad_output_ndim,        // Number of dimensions of grad_output (can be 0 for global)
        int axis,                    // Reduction axis (-1 for global)
        int n_input                  // Total elements in x
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_input) return;

        const int MAX_DIMS = 8; // Or define based on max expected rank
        if (x_ndim > MAX_DIMS || y_ndim > MAX_DIMS || grad_output_ndim > MAX_DIMS) return;

        int x_coords[MAX_DIMS];
        flat_index_to_multi_dim(idx, x_shape, x_ndim, x_coords);

        // Calculate corresponding index in y and grad_output based on reduction type
        int y_idx = 0; // Default for global reduction
        int grad_out_idx = 0; // Default for global reduction

        if (axis != -1) { // Axis reduction
            // Calculate coordinates for y/grad_output by skipping the reduction axis
            int current_y_dim = 0;
            int y_coords[MAX_DIMS];
            int current_grad_out_dim = 0;
            int grad_out_coords[MAX_DIMS];

            for(int i = 0; i < x_ndim; ++i) {
                if (i != axis) {
                     if (current_y_dim < y_ndim) {
                        y_coords[current_y_dim] = x_coords[i];
                        current_y_dim++;
                     }
                     if (current_grad_out_dim < grad_output_ndim) {
                        grad_out_coords[current_grad_out_dim] = x_coords[i];
                        current_grad_out_dim++;
                     }
                }
            }
            // Check if dimensions match expected (sanity check)
             if (current_y_dim == y_ndim && current_grad_out_dim == grad_output_ndim) {
                y_idx = multi_dim_to_flat_index(y_coords, y_strides, y_ndim);
                grad_out_idx = multi_dim_to_flat_index(grad_out_coords, grad_output_strides, grad_output_ndim);
             } else {
                 // This indicates a logic error in coordinate mapping
                 grad_input[idx] = NAN; // Indicate error
                 return;
             }
        }
        // Else: For global reduction (axis == -1), y_idx and grad_out_idx remain 0

        float x_val = x[idx];
        float y_val = y[y_idx];
        float grad_out_val = grad_output[grad_out_idx];

        // Check if this element was the maximum value in its reduction slice
        bool is_max = fabsf(x_val - y_val) < 1e-9f; // Use tolerance for float comparison

        if (!is_max) {
            grad_input[idx] = 0.0f;
            return;
        }

        // This element contributed to the max. Now, count ties within its reduction slice.
        int count = 0;
        if (axis == -1) { // Global reduction: count across the whole input
            for (int k = 0; k < n_input; ++k) {
                if (fabsf(x[k] - y_val) < 1e-9f) {
                    count++;
                }
            }
        } else { // Axis reduction: count only along the specified axis
            int reduction_dim_size = x_shape[axis];
            int stride_for_axis = x_strides[axis];
            // Calculate the starting index of the slice this element belongs to
            int slice_coords[MAX_DIMS];
             for (int d = 0; d < x_ndim; ++d) {
                 slice_coords[d] = x_coords[d];
             }
            slice_coords[axis] = 0; // Start at the beginning of the reduction axis
            int base_idx = multi_dim_to_flat_index(slice_coords, x_strides, x_ndim);

            // Iterate along the reduction axis
            for (int k = 0; k < reduction_dim_size; ++k) {
                int current_check_idx = base_idx + k * stride_for_axis;
                // Ensure the index is within bounds (important for non-contiguous memory?)
                 if (current_check_idx >= 0 && current_check_idx < n_input) {
                    if (fabsf(x[current_check_idx] - y_val) < 1e-9f) {
                        count++;
                    }
                 }
            }
        }

        // Distribute gradient among ties
        if (count > 0) {
            grad_input[idx] = grad_out_val / (float)count;
        } else {
            // Should not happen if is_max is true, but as a fallback
            grad_input[idx] = 0.0f;
        }
    }

    /*
     * Computes the backward pass for the min reduction operation.
     * Logic is identical to max_backward_kernel but checks for minimum value.
     */
    extern "C" __global__ void min_backward_kernel(
        const float* x, const float* y, const float* grad_output,
        float* grad_input,
        const int* x_shape, const int* y_shape, const int* grad_output_shape,
        const int* x_strides, const int* y_strides, const int* grad_output_strides,
        int x_ndim, int y_ndim, int grad_output_ndim,
        int axis, int n_input
    ) {
        // Print kernel configuration information
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("[KERNEL DEBUG] min_backward_kernel launched with:\n");
            printf("  - x_ndim: %d, y_ndim: %d, grad_output_ndim: %d\n", x_ndim, y_ndim, grad_output_ndim);
            printf("  - axis: %d, n_input: %d\n", axis, n_input);
            printf("  - x_shape: [");
            for (int i = 0; i < x_ndim; i++) {
                printf("%d%s", x_shape[i], (i < x_ndim - 1) ? ", " : "");
            }
            printf("]\n");
            printf("  - y_shape: [");
            for (int i = 0; i < y_ndim; i++) {
                printf("%d%s", y_shape[i], (i < y_ndim - 1) ? ", " : "");
            }
            printf("]\n");
            printf("  - grad_output_shape: [");
            for (int i = 0; i < grad_output_ndim; i++) {
                printf("%d%s", grad_output_shape[i], (i < grad_output_ndim - 1) ? ", " : "");
            }
            printf("]\n");
            printf("  - First few input values: ");
            for (int i = 0; i < min(5, n_input); i++) {
                printf("%.2f ", x[i]);
            }
            printf("\n");
            printf("  - First y value: %.6f\n", y[0]);
            printf("  - First grad_output value: %.6f\n", grad_output[0]);
        }

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_input) return;

        const int MAX_DIMS = 8;
        if (x_ndim > MAX_DIMS || y_ndim > MAX_DIMS || grad_output_ndim > MAX_DIMS) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("[KERNEL ERROR] Dimension exceeds MAX_DIMS=%d\n", MAX_DIMS);
            }
            return;
        }

        int x_coords[MAX_DIMS];
        flat_index_to_multi_dim(idx, x_shape, x_ndim, x_coords);

        int y_idx = 0;
        int grad_out_idx = 0;

        if (axis != -1) {
            int current_y_dim = 0;
            int y_coords[MAX_DIMS];
            int current_grad_out_dim = 0;
            int grad_out_coords[MAX_DIMS];
            for(int i = 0; i < x_ndim; ++i) {
                if (i != axis) {
                     if (current_y_dim < y_ndim) { y_coords[current_y_dim++] = x_coords[i]; }
                     if (current_grad_out_dim < grad_output_ndim) { grad_out_coords[current_grad_out_dim++] = x_coords[i]; }
                }
            }
             if (current_y_dim == y_ndim && current_grad_out_dim == grad_output_ndim) {
                y_idx = multi_dim_to_flat_index(y_coords, y_strides, y_ndim);
                grad_out_idx = multi_dim_to_flat_index(grad_out_coords, grad_output_strides, grad_output_ndim);
             } else { 
                 if (idx == 0) {
                     printf("[KERNEL ERROR] Coordinate mapping error: current_y_dim=%d, y_ndim=%d, current_grad_out_dim=%d, grad_output_ndim=%d\n", 
                            current_y_dim, y_ndim, current_grad_out_dim, grad_output_ndim);
                 }
                 grad_input[idx] = NAN; 
                 return; 
             }
        }

        float x_val = x[idx];
        float y_val = y[y_idx];
        float grad_out_val = grad_output[grad_out_idx];

        bool is_min = fabsf(x_val - y_val) < 1e-9f;

        if (!is_min) {
            grad_input[idx] = 0.0f;
            return;
        }

        int count = 0;
        if (axis == -1) {
            for (int k = 0; k < n_input; ++k) {
                if (fabsf(x[k] - y_val) < 1e-9f) { count++; }
            }
        } else {
            int reduction_dim_size = x_shape[axis];
            int stride_for_axis = x_strides[axis];
            int slice_coords[MAX_DIMS];
            for (int d = 0; d < x_ndim; ++d) { slice_coords[d] = x_coords[d]; }
            slice_coords[axis] = 0;
            int base_idx = multi_dim_to_flat_index(slice_coords, x_strides, x_ndim);
            for (int k = 0; k < reduction_dim_size; ++k) {
                int current_check_idx = base_idx + k * stride_for_axis;
                if (current_check_idx >= 0 && current_check_idx < n_input) {
                     if (fabsf(x[current_check_idx] - y_val) < 1e-9f) { count++; }
                }
            }
        }

        if (count > 0) {
            grad_input[idx] = grad_out_val / (float)count;
            
            // Debug output for a few representative threads
            if (idx < 3 || (is_min && idx < 10)) {
                printf("Thread %d: x=%f, min_val=%f, is_min=%d, count=%d, grad_out=%f, grad_in=%f\n", 
                       idx, x_val, y_val, is_min ? 1 : 0, count, grad_out_val, grad_input[idx]);
            }
        } else {
            grad_input[idx] = 0.0f;
            
            // This shouldn't happen if is_min is true
            if (is_min) {
                printf("[WARNING] Thread %d: is_min is true but count is 0. x=%f, min_val=%f\n", 
                       idx, x_val, y_val);
            }
        }
    }

    /*
     * Computes the backward pass for the product reduction operation.
     * grad_input = grad_output * (y / x) = grad_output * prod(others)
     */
    extern "C" __global__ void prod_backward_kernel(
        const float* x, const float* y, const float* grad_output,
        float* grad_input,
        const int* x_shape, const int* y_shape, const int* grad_output_shape,
        const int* x_strides, const int* y_strides, const int* grad_output_strides,
        int x_ndim, int y_ndim, int grad_output_ndim,
        int axis, int n_input
    ) {
        // Print kernel configuration information
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("[KERNEL DEBUG] prod_backward_kernel launched with:\n");
            printf("  - x_ndim: %d, y_ndim: %d, grad_output_ndim: %d\n", x_ndim, y_ndim, grad_output_ndim);
            printf("  - axis: %d, n_input: %d\n", axis, n_input);
            printf("  - x_shape: [");
            for (int i = 0; i < x_ndim; i++) {
                printf("%d%s", x_shape[i], (i < x_ndim - 1) ? ", " : "");
            }
            printf("]\n");
            printf("  - y_shape: [");
            for (int i = 0; i < y_ndim; i++) {
                printf("%d%s", y_shape[i], (i < y_ndim - 1) ? ", " : "");
            }
            printf("]\n");
            printf("  - grad_output_shape: [");
            for (int i = 0; i < grad_output_ndim; i++) {
                printf("%d%s", grad_output_shape[i], (i < grad_output_ndim - 1) ? ", " : "");
            }
            printf("]\n");
            printf("  - First few input values: ");
            for (int i = 0; i < min(5, n_input); i++) {
                printf("%.2f ", x[i]);
            }
            printf("\n");
            printf("  - First y value: %.6f\n", y[0]);
            printf("  - First grad_output value: %.6f\n", grad_output[0]);
        }
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_input) return;

        const int MAX_DIMS = 8;
        if (x_ndim > MAX_DIMS || y_ndim > MAX_DIMS || grad_output_ndim > MAX_DIMS) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("[KERNEL ERROR] Dimension exceeds MAX_DIMS=%d\n", MAX_DIMS);
            }
            return;
        }

        int x_coords[MAX_DIMS];
        flat_index_to_multi_dim(idx, x_shape, x_ndim, x_coords);

        float x_val = x[idx];
        float grad_out_val;
        float y_val;

        // Handle global reduction (axis == -1) specially
        if (axis == -1) {
            // For global reduction, y is a scalar and grad_output is also a scalar
            // Calculate the product of all elements EXCEPT the current one
            float total_prod = y[0]; // Global product result
            grad_out_val = grad_output[0]; // Scalar gradient

            // Debug output for a few threads
            if (idx < 3) {
                printf("Thread %d (global): x=%f, prod_val=%f, grad_out=%f\n", 
                      idx, x_val, total_prod, grad_out_val);
            }

            // For global product, grad_input[i] = grad_out * (prod / x[i])
            if (fabsf(x_val) < 1e-15f) { // If x[idx] is essentially zero
                if (idx < 3) {
                    printf("Thread %d (global): x is near zero (%.2e), setting grad to 0\n", idx, (double)x_val);
                }
                grad_input[idx] = 0.0f;
            } else {
                // Normal case: grad = grad_out * (y / x)
                // This is the product of all elements divided by this element
                // which gives the product of all OTHER elements
                grad_input[idx] = grad_out_val * (total_prod / x_val);
                
                if (idx < 3) {
                    printf("Thread %d (global): grad_input = %f * (%f / %f) = %f\n", 
                          idx, grad_out_val, total_prod, x_val, grad_input[idx]);
                }
            }
            return;
        }

        // For axis-specific reduction:
        // Get the corresponding indices in the gradient and product tensors
        int y_idx = 0;
        int grad_out_idx = 0;

        // Map the x coordinates to y and grad_output coordinates by removing the reduction axis
        int y_coords[MAX_DIMS];
        int grad_out_coords[MAX_DIMS];
        
        int current_y_dim = 0;
        int current_grad_out_dim = 0;
        
        for(int i = 0; i < x_ndim; ++i) {
            if (i != axis) {
                 if (current_y_dim < y_ndim) { y_coords[current_y_dim++] = x_coords[i]; }
                 if (current_grad_out_dim < grad_output_ndim) { grad_out_coords[current_grad_out_dim++] = x_coords[i]; }
            }
        }
         
        if (current_y_dim == y_ndim && current_grad_out_dim == grad_output_ndim) {
            y_idx = multi_dim_to_flat_index(y_coords, y_strides, y_ndim);
            grad_out_idx = multi_dim_to_flat_index(grad_out_coords, grad_output_strides, grad_output_ndim);
        } else { 
            if (idx == 0) {
                printf("[KERNEL ERROR] Coordinate mapping error: current_y_dim=%d, y_ndim=%d, current_grad_out_dim=%d, grad_output_ndim=%d\n", 
                       current_y_dim, y_ndim, current_grad_out_dim, grad_output_ndim);
            }
            grad_input[idx] = NAN; 
            return; 
        }

        y_val = y[y_idx]; // Product result for the slice
        grad_out_val = grad_output[grad_out_idx];

        // Debug output for a few threads
        if (idx < 3) {
            printf("Thread %d (axis=%d): x=%f, prod_val=%f, grad_out=%f\n", 
                  idx, axis, x_val, y_val, grad_out_val);
        }

        // Handle division by zero carefully
        if (fabsf(x_val) < 1e-15f) { // If x[idx] is essentially zero
            if (idx < 3) {
                printf("Thread %d (axis=%d): x is near zero (%.2e), setting grad to 0\n", idx, axis, (double)x_val);
            }
            grad_input[idx] = 0.0f;
        } else {
            // Normal case: grad = grad_out * (y / x)
            grad_input[idx] = grad_out_val * (y_val / x_val);
            
            if (idx < 3) {
                printf("Thread %d (axis=%d): grad_input = %f * (%f / %f) = %f\n", 
                      idx, axis, grad_out_val, y_val, x_val, grad_input[idx]);
            }
        }
    }

    /*
     * Computes the backward pass for the logsumexp reduction operation.
     * grad_input = grad_output * exp(x - y) = grad_output * softmax(x along axis)
     */
    extern "C" __global__ void logsumexp_backward_kernel(
        const float* x, const float* y, const float* grad_output,
        float* grad_input,
        const int* x_shape, const int* y_shape, const int* grad_output_shape,
        const int* x_strides, const int* y_strides, const int* grad_output_strides,
        int x_ndim, int y_ndim, int grad_output_ndim,
        int axis, int n_input
    ) {
        // Print kernel configuration information
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("[KERNEL DEBUG] logsumexp_backward_kernel launched with:\n");
            printf("  - x_ndim: %d, y_ndim: %d, grad_output_ndim: %d\n", x_ndim, y_ndim, grad_output_ndim);
            printf("  - axis: %d, n_input: %d\n", axis, n_input);
            printf("  - x_shape: [");
            for (int i = 0; i < x_ndim; i++) {
                printf("%d%s", x_shape[i], (i < x_ndim - 1) ? ", " : "");
            }
            printf("]\n");
            printf("  - y_shape: [");
            for (int i = 0; i < y_ndim; i++) {
                printf("%d%s", y_shape[i], (i < y_ndim - 1) ? ", " : "");
            }
            printf("]\n");
            printf("  - grad_output_shape: [");
            for (int i = 0; i < grad_output_ndim; i++) {
                printf("%d%s", grad_output_shape[i], (i < grad_output_ndim - 1) ? ", " : "");
            }
            printf("]\n");
            printf("  - First few input values: ");
            for (int i = 0; i < min(5, n_input); i++) {
                printf("%.2f ", x[i]);
            }
            printf("\n");
            printf("  - First y value: %.6f\n", y[0]);
            printf("  - First grad_output value: %.6f\n", grad_output[0]);
        }
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_input) return;

        const int MAX_DIMS = 8;
        if (x_ndim > MAX_DIMS || y_ndim > MAX_DIMS || grad_output_ndim > MAX_DIMS) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("[KERNEL ERROR] Dimension exceeds MAX_DIMS=%d\n", MAX_DIMS);
            }
            return;
        }

        int x_coords[MAX_DIMS];
        flat_index_to_multi_dim(idx, x_shape, x_ndim, x_coords);

        int y_idx = 0;
        int grad_out_idx = 0;

        if (axis != -1) {
            int current_y_dim = 0;
            int y_coords[MAX_DIMS];
            int current_grad_out_dim = 0;
            int grad_out_coords[MAX_DIMS];
            for(int i = 0; i < x_ndim; ++i) {
                if (i != axis) {
                     if (current_y_dim < y_ndim) { y_coords[current_y_dim++] = x_coords[i]; }
                     if (current_grad_out_dim < grad_output_ndim) { grad_out_coords[current_grad_out_dim++] = x_coords[i]; }
                }
            }
             if (current_y_dim == y_ndim && current_grad_out_dim == grad_output_ndim) {
                y_idx = multi_dim_to_flat_index(y_coords, y_strides, y_ndim);
                grad_out_idx = multi_dim_to_flat_index(grad_out_coords, grad_output_strides, grad_output_ndim);
             } else { 
                 if (idx == 0) {
                     printf("[KERNEL ERROR] Coordinate mapping error: current_y_dim=%d, y_ndim=%d, current_grad_out_dim=%d, grad_output_ndim=%d\n", 
                            current_y_dim, y_ndim, current_grad_out_dim, grad_output_ndim);
                 }
                 grad_input[idx] = NAN; 
                 return; 
             }
        }

        float x_val = x[idx];
        float y_val = y[y_idx]; // LogSumExp result for the slice
        float grad_out_val = grad_output[grad_out_idx];

        // grad = grad_output * exp(x - y) which is grad_output * softmax(x)
        float grad_val = grad_out_val * expf(x_val - y_val);
        grad_input[idx] = grad_val;
        
        // Debug output for a few threads
        if (idx < 3) {
            printf("Thread %d: x=%f, logsumexp_val=%f, exp(x-y)=%f, grad_out=%f, grad_in=%f\n", 
                   idx, x_val, y_val, expf(x_val - y_val), grad_out_val, grad_val);
        }
    }

    extern "C" __global__ void tanh_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = tanhf(x[idx]); // Use single-precision tanh
        }
    }

    extern "C" __global__ void tanh_backward_kernel(const float* x, const float* grad_output, float* grad_input, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float tanh_x = tanhf(x[idx]);
            grad_input[idx] = grad_output[idx] * (1.0f - tanh_x * tanh_x);
        }
    }

    extern "C" __global__ void sqrt_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = sqrtf(x[idx]);
        }
    }
    extern "C" __global__ void sqrt_backward_kernel(const float* x, const float* grad_out, float* grad_in, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float sqrt_x = sqrtf(x[idx]);
            grad_in[idx] = grad_out[idx] * 0.5f / sqrt_x;
        }
    }

    extern "C" __global__ void sigmoid_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = 1.0f / (1.0f + expf(-x[idx]));
        }
    }

    extern "C" __global__ void sigmoid_backward_kernel(const float* y, const float* grad_out, float* grad_in, int n) {
        // Assumes y = sigmoid(x) was computed in the forward pass or recomputed
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float y_val = y[idx];
            grad_in[idx] = grad_out[idx] * y_val * (1.0f - y_val);
        }
    }

    extern "C" __global__ void softplus_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // Use a more numerically stable version:
            // max(x, 0) + log(1 + exp(-abs(x)))
            float x_val = x[idx];
            out[idx] = fmaxf(x_val, 0.0f) + logf(1.0f + expf(-fabsf(x_val)));
            // Simpler version (less stable): logf(1.0f + expf(x[idx]));
        }
    }

    extern "C" __global__ void softplus_backward_kernel(const float* x, const float* grad_output, float* grad_input, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            // Derivative is sigmoid(x)
            float sigmoid_x = 1.0f / (1.0f + expf(-x[idx]));
            grad_input[idx] = grad_output[idx] * sigmoid_x;
        }
    }

    extern "C" __global__ void powf_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = powf(a[idx], b[idx]);
        }
    }

    extern "C" __global__ void powf_backward_kernel(
        const float* a, const float* b,
        const float* a_pow_b, // Precomputed result of a^b
        const float* grad_out,
        float* grad_a, float* grad_b, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float a_val = a[idx];
            float b_val = b[idx];
            float a_pow_b_val = a_pow_b[idx]; // Use precomputed value
            float dout = grad_out[idx];

            // Handle edge cases for gradient calculation
            if (a_val < 1e-9f) { // Avoid issues with log(0) or 0^negative
                 grad_a[idx] = 0.0f; // Gradient is 0 or inf, safest to set 0
                 grad_b[idx] = 0.0f; // Gradient involves log(a), which tends to 0 * -inf = 0
            } else {
                // For da: dL/da = dL/dy * b * a^(b-1)
                grad_a[idx] = dout * b_val * powf(a_val, b_val - 1.0f);
                // For db: dL/db = dL/dy * a^b * ln(a)
                grad_b[idx] = dout * a_pow_b_val * logf(a_val);
            }
        }
    }

    extern "C" __global__ void square_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float val = x[idx];
            out[idx] = val * val;
        }
    }

    extern "C" __global__ void square_backward_kernel(const float* x, const float* grad_out, float* grad_in, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            grad_in[idx] = grad_out[idx] * 2.0f * x[idx];
        }
    }

    // Forward: Element-wise maximum
    extern "C" __global__ void maximum_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = fmaxf(a[idx], b[idx]);
        }
    }

    // Backward: Element-wise maximum gradient calculation
    extern "C" __global__ void maximum_backward_kernel(
        const float* a, const float* b, const float* grad_out,
        float* grad_a, float* grad_b, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float a_val = a[idx];
            float b_val = b[idx];
            float grad_out_val = grad_out[idx];
            // Gradient flows if the element contributed to the max
            // For equality, give gradient to 'a' (first input)
            if (fabsf(a_val - b_val) < EQUAL_TOLERANCE) {
                 grad_a[idx] = grad_out_val; // Give full gradient to 'a'
                 grad_b[idx] = 0.0f;         // No gradient to 'b'
            } else if (a_val > b_val) {
                grad_a[idx] = grad_out_val;
                grad_b[idx] = 0.0f;
            } else { // b_val > a_val
                grad_a[idx] = 0.0f;
                grad_b[idx] = grad_out_val;
            }
        }
    }

    // Forward: Element-wise minimum
    extern "C" __global__ void minimum_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = fminf(a[idx], b[idx]);
        }
    }

    // Backward: Element-wise minimum gradient calculation
    extern "C" __global__ void minimum_backward_kernel(
        const float* a, const float* b, const float* grad_out,
        float* grad_a, float* grad_b, int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float a_val = a[idx];
            float b_val = b[idx];
            float grad_out_val = grad_out[idx];
            // Gradient flows if the element contributed to the min
            // For equality, give gradient to 'a' (first input)
            if (fabsf(a_val - b_val) < EQUAL_TOLERANCE) {
                 grad_a[idx] = grad_out_val; // Give full gradient to 'a'
                 grad_b[idx] = 0.0f;         // No gradient to 'b'
            } else if (a_val < b_val) {
                grad_a[idx] = grad_out_val;
                grad_b[idx] = 0.0f;
            } else { // b_val < a_val
                grad_a[idx] = 0.0f;
                grad_b[idx] = grad_out_val;
            }
        }
    }

    /*
     * Computes element-wise equality (a == b) with a tolerance.
     * Outputs 1.0f if abs(a[idx] - b[idx]) < EQUAL_TOLERANCE, 0.0f otherwise.
     */
    extern "C" __global__ void equal_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = (fabsf(a[idx] - b[idx]) < EQUAL_TOLERANCE) ? 1.0f : 0.0f;
        }
    }

    /*
     * Computes element-wise greater comparison (a > b).
     * Outputs 1.0f if a[idx] > b[idx], 0.0f otherwise.
     */
    extern "C" __global__ void greater_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
        }
    }

    /*
     * Computes element-wise greater or equal comparison (a >= b).
     * Outputs 1.0f if a[idx] >= b[idx], 0.0f otherwise.
     */
    extern "C" __global__ void greater_equal_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
        }
    }

    /*
     * Computes element-wise less comparison (a < b).
     * Outputs 1.0f if a[idx] < b[idx], 0.0f otherwise.
     */
    extern "C" __global__ void less_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f;
        }
    }

    /*
     * Computes element-wise less or equal comparison (a <= b).
     * Outputs 1.0f if a[idx] <= b[idx], 0.0f otherwise.
     */
    extern "C" __global__ void less_equal_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
        }
    }

    /*
     * Computes element-wise not equal comparison (a != b) with a tolerance.
     * Outputs 1.0f if abs(a[idx] - b[idx]) >= EQUAL_TOLERANCE, 0.0f otherwise.
     */
    extern "C" __global__ void not_equal_kernel(const float* a, const float* b, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = (fabsf(a[idx] - b[idx]) >= EQUAL_TOLERANCE) ? 1.0f : 0.0f;
        }
    }

    /*
     * Computes element-wise sine: sin(x).
     */
    extern "C" __global__ void sin_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = sinf(x[idx]);
        }
    }

    /*
     * Computes gradient for sine operation: grad_in = cos(x) * grad_out.
     * The derivative of sin(x) is cos(x).
     */
    extern "C" __global__ void sin_backward_kernel(const float* x, const float* grad_out, float* grad_in, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            grad_in[idx] = cosf(x[idx]) * grad_out[idx];
        }
    }
    
    /*
     * Computes element-wise cosine: cos(x).
     */
    extern "C" __global__ void cos_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = cosf(x[idx]);
        }
    }

    /*
     * Computes gradient for cosine operation: grad_in = -sin(x) * grad_out.
     * The derivative of cos(x) is -sin(x).
     */
    extern "C" __global__ void cos_backward_kernel(const float* x, const float* grad_out, float* grad_in, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            grad_in[idx] = -sinf(x[idx]) * grad_out[idx];
        }
    }
    
    /*
     * Computes element-wise tangent: tan(x).
     */
    extern "C" __global__ void tan_kernel(const float* x, float* out, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = tanf(x[idx]);
        }
    }

    /*
     * Computes gradient for tangent operation: grad_in = (1 + tan²(x)) * grad_out.
     * The derivative of tan(x) is 1/cos²(x) = 1 + tan²(x).
     */
    extern "C" __global__ void tan_backward_kernel(const float* x, const float* grad_out, float* grad_in, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float tan_x = tanf(x[idx]);
            float derivative = 1.0f + tan_x * tan_x; // 1 + tan²(x)
            grad_in[idx] = derivative * grad_out[idx];
        }
    }

    /*
     * Clips the values of a tensor to be within [min_val, max_val].
     */
    extern "C" __global__ void clip_kernel(const float* input, float* output, float min_val, float max_val, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = fmaxf(min_val, fminf(input[idx], max_val));
        }
    }

    /*
     * Computes gradient for clip operation.
     * The derivative is 1.0 where input is within [min_val, max_val], and 0.0 otherwise.
     */
    extern "C" __global__ void clip_backward_kernel(
        const float* input,
        const float* grad_output,
        float* grad_input,
        float min_val,
        float max_val,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float val = input[idx];
            grad_input[idx] = (val >= min_val && val <= max_val) ? grad_output[idx] : 0.0f;
        }
    }
}