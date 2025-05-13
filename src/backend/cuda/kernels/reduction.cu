// CUDA kernel for reduction operations

#include <stdio.h>
#include <math.h>

extern "C" {
    __global__ void sum_reduction_kernel(const float* input, float* output, int* dims, int ndim) {
        extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Load input into shared memory
        sdata[tid] = (i < dims[0]) ? input[i] : 0;
        __syncthreads();

        // Do reduction in shared memory
        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        // Write result for this block to global memory
        if (tid == 0) output[blockIdx.x] = sdata[0];
    }

    __global__ void max_reduction_kernel(const float* input, float* output, int* dims, int ndim) {
        extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Debugging info at the start
        if (tid == 0 && blockIdx.x == 0) {
            printf("KERNEL DEBUG: max_reduction_kernel started\n");
            printf("KERNEL DEBUG: dims[0]=%d, ndim=%d\n", dims[0], ndim);
            printf("KERNEL DEBUG: blockDim.x=%d, gridDim.x=%d\n", blockDim.x, gridDim.x);
            printf("KERNEL DEBUG: Total threads: %d\n", blockDim.x * gridDim.x);
            
            // Print first 5 input values
            printf("KERNEL DEBUG: First 5 input values: ");
            for (int j = 0; j < 5 && j < dims[0]; j++) {
                printf("%.2f ", input[j]);
            }
            printf("\n");
        }
        
        // If ndim == 1, this is a global reduction operation
        if (ndim <= 1) {
            // Simple case: treat the entire array as a flat array and find the maximum
            sdata[tid] = (i < dims[0]) ? input[i] : -INFINITY;
            __syncthreads();
            
            // Debug print for first few threads
            if (blockIdx.x == 0 && tid < 8) {
                printf("Block 0, Thread %d: loaded value %f (index %d)\n", tid, sdata[tid], i);
            }
            
            // Standard reduction in shared memory
            for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
                }
                __syncthreads();
            }
            
            // Write result for this block to output
            if (tid == 0) {
                output[blockIdx.x] = sdata[0];
                printf("Block %d: Final max value = %f\n", blockIdx.x, sdata[0]);
            }
        }
        // Otherwise, this is an axis reduction
        else if (ndim == 2 && i < dims[0]) {
            // We're reducing a 2D tensor along an axis
            // For axis=0 reduction (dims = [rows, cols], output = [cols])
            // For axis=1 reduction (dims = [rows, cols], output = [rows])
            
            const int rows = dims[1];  // Number of rows (first dimension)
            const int cols = dims[2];  // Number of columns (second dimension)
            const int axis = dims[3];  // Axis to reduce along (0 = rows, 1 = columns)
            
            // Debug the shape and axis
            if (tid == 0 && blockIdx.x == 0) {
                printf("KERNEL DEBUG: 2D Tensor Reduction - shape=[%d,%d], axis=%d\n", rows, cols, axis);
            }
            
            // Different handling based on which axis we're reducing along
            if (axis == 0) {
                // We're reducing along rows, output will have shape [cols]
                // Each thread processes one column
                const int col = i;
                
                if (col < cols) {
                    float max_val = -INFINITY;
                    
                    // Find max across all rows for this column
                    for (int row = 0; row < rows; row++) {
                        float val = input[row * cols + col];
                        max_val = fmaxf(max_val, val);
                    }
                    
                    // Store result directly (no need for shared memory reduction)
                    output[col] = max_val;
                    
                    if (col < 5) {
                        printf("Reduced column %d: max = %f\n", col, max_val);
                    }
                }
            }
            else if (axis == 1) {
                // We're reducing along columns, output will have shape [rows]
                // Each thread processes one row
                const int row = i;
                
                if (row < rows) {
                    float max_val = -INFINITY;
                    
                    // Find max across all columns for this row
                    for (int col = 0; col < cols; col++) {
                        float val = input[row * cols + col];
                        max_val = fmaxf(max_val, val);
                    }
                    
                    // Store result directly
                    output[row] = max_val;
                    
                    if (row < 5) {
                        printf("Reduced row %d: max = %f\n", row, max_val);
                    }
                }
            }
        }
        else {
            // More complex cases need more specialized handling
            // For now, just output a message
            if (tid == 0 && blockIdx.x == 0) {
                printf("WARNING: Complex tensor shape or axis not implemented in max_reduction_kernel\n");
            }
        }
    }

    __global__ void min_reduction_kernel(const float* input, float* output, int* dims, int ndim) {
        extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Debugging info at the start
        if (tid == 0 && blockIdx.x == 0) {
            printf("KERNEL DEBUG: min_reduction_kernel started\n");
            printf("KERNEL DEBUG: dims[0]=%d, ndim=%d\n", dims[0], ndim);
            printf("KERNEL DEBUG: blockDim.x=%d, gridDim.x=%d\n", blockDim.x, gridDim.x);
            printf("KERNEL DEBUG: Total threads: %d\n", blockDim.x * gridDim.x);
            
            // Print first 5 input values
            printf("KERNEL DEBUG: First 5 input values: ");
            for (int j = 0; j < 5 && j < dims[0]; j++) {
                printf("%.2f ", input[j]);
            }
            printf("\n");
        }
        
        // If ndim == 1, this is a global reduction operation
        if (ndim <= 1) {
            // Simple case: treat the entire array as a flat array and find the minimum
            sdata[tid] = (i < dims[0]) ? input[i] : INFINITY;
            __syncthreads();
            
            // Debug print for first few threads
            if (blockIdx.x == 0 && tid < 8) {
                printf("Block 0, Thread %d: loaded value %f (index %d)\n", tid, sdata[tid], i);
            }
            
            // Standard reduction in shared memory
            for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
                }
                __syncthreads();
            }
            
            // Write result for this block to output
            if (tid == 0) {
                output[blockIdx.x] = sdata[0];
                printf("Block %d: Final min value = %f\n", blockIdx.x, sdata[0]);
            }
        }
        // Otherwise, this is an axis reduction
        else if (ndim == 2 && i < dims[0]) {
            // We're reducing a 2D tensor along an axis
            // For axis=0 reduction (dims = [rows, cols], output = [cols])
            // For axis=1 reduction (dims = [rows, cols], output = [rows])
            
            const int rows = dims[1];  // Number of rows (first dimension)
            const int cols = dims[2];  // Number of columns (second dimension)
            const int axis = dims[3];  // Axis to reduce along (0 = rows, 1 = columns)
            
            // Debug the shape and axis
            if (tid == 0 && blockIdx.x == 0) {
                printf("KERNEL DEBUG: 2D Tensor Reduction - shape=[%d,%d], axis=%d\n", rows, cols, axis);
            }
            
            // Different handling based on which axis we're reducing along
            if (axis == 0) {
                // We're reducing along rows, output will have shape [cols]
                // Each thread processes one column
                const int col = i;
                
                if (col < cols) {
                    float min_val = INFINITY;
                    
                    // Find min across all rows for this column
                    for (int row = 0; row < rows; row++) {
                        float val = input[row * cols + col];
                        min_val = fminf(min_val, val);
                    }
                    
                    // Store result directly (no need for shared memory reduction)
                    output[col] = min_val;
                    
                    if (col < 5) {
                        printf("Reduced column %d: min = %f\n", col, min_val);
                    }
                }
            }
            else if (axis == 1) {
                // We're reducing along columns, output will have shape [rows]
                // Each thread processes one row
                const int row = i;
                
                if (row < rows) {
                    float min_val = INFINITY;
                    
                    // Find min across all columns for this row
                    for (int col = 0; col < cols; col++) {
                        float val = input[row * cols + col];
                        min_val = fminf(min_val, val);
                    }
                    
                    // Store result directly
                    output[row] = min_val;
                    
                    if (row < 5) {
                        printf("Reduced row %d: min = %f\n", row, min_val);
                    }
                }
            }
        }
        else {
            // More complex cases need more specialized handling
            // For now, just output a message
            if (tid == 0 && blockIdx.x == 0) {
                printf("WARNING: Complex tensor shape or axis not implemented in min_reduction_kernel\n");
            }
        }
    }

    __global__ void prod_reduction_kernel(const float* input, float* output, int* dims, int ndim) {
        extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Debugging info at the start
        if (tid == 0 && blockIdx.x == 0) {
            printf("KERNEL DEBUG: prod_reduction_kernel started\n");
            printf("KERNEL DEBUG: dims[0]=%d, ndim=%d\n", dims[0], ndim);
            printf("KERNEL DEBUG: blockDim.x=%d, gridDim.x=%d\n", blockDim.x, gridDim.x);
            printf("KERNEL DEBUG: Total threads: %d\n", blockDim.x * gridDim.x);
            
            // Print first 5 input values
            printf("KERNEL DEBUG: First 5 input values: ");
            for (int j = 0; j < 5 && j < dims[0]; j++) {
                printf("%.2f ", input[j]);
            }
            printf("\n");
        }
        
        // If ndim == 1, this is a global reduction operation
        if (ndim <= 1) {
            // Simple case: treat the entire array as a flat array and find the product
            sdata[tid] = (i < dims[0]) ? input[i] : 1.0f;
            __syncthreads();
            
            // Debug print for first few threads
            if (blockIdx.x == 0 && tid < 8) {
                printf("Block 0, Thread %d: loaded value %f (index %d)\n", tid, sdata[tid], i);
            }
            
            // Standard reduction in shared memory
            for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] *= sdata[tid + s];
                }
                __syncthreads();
            }
            
            // Write result for this block to output
            if (tid == 0) {
                output[blockIdx.x] = sdata[0];
                printf("Block %d: Final product value = %f\n", blockIdx.x, sdata[0]);
            }
        }
        // Otherwise, this is an axis reduction
        else if (ndim == 2 && i < dims[0]) {
            // We're reducing a 2D tensor along an axis
            // For axis=0 reduction (dims = [rows, cols], output = [cols])
            // For axis=1 reduction (dims = [rows, cols], output = [rows])
            
            const int rows = dims[1];  // Number of rows (first dimension)
            const int cols = dims[2];  // Number of columns (second dimension)
            const int axis = dims[3];  // Axis to reduce along (0 = rows, 1 = columns)
            
            // Debug the shape and axis
            if (tid == 0 && blockIdx.x == 0) {
                printf("KERNEL DEBUG: 2D Tensor Reduction - shape=[%d,%d], axis=%d\n", rows, cols, axis);
            }
            
            // Different handling based on which axis we're reducing along
            if (axis == 0) {
                // We're reducing along rows, output will have shape [cols]
                // Each thread processes one column
                const int col = i;
                
                if (col < cols) {
                    float prod_val = 1.0f;
                    
                    // Find product across all rows for this column
                    for (int row = 0; row < rows; row++) {
                        float val = input[row * cols + col];
                        prod_val *= val;
                    }
                    
                    // Store result directly (no need for shared memory reduction)
                    output[col] = prod_val;
                    
                    if (col < 5) {
                        printf("Reduced column %d: product = %f\n", col, prod_val);
                    }
                }
            }
            else if (axis == 1) {
                // We're reducing along columns, output will have shape [rows]
                // Each thread processes one row
                const int row = i;
                
                if (row < rows) {
                    float prod_val = 1.0f;
                    
                    // Find product across all columns for this row
                    for (int col = 0; col < cols; col++) {
                        float val = input[row * cols + col];
                        prod_val *= val;
                    }
                    
                    // Store result directly
                    output[row] = prod_val;
                    
                    if (row < 5) {
                        printf("Reduced row %d: product = %f\n", row, prod_val);
                    }
                }
            }
        }
        else {
            // More complex cases need more specialized handling
            // For now, just output a message
            if (tid == 0 && blockIdx.x == 0) {
                printf("WARNING: Complex tensor shape or axis not implemented in prod_reduction_kernel\n");
            }
        }
    }

    // General reduction kernel for sum/mean/max/min
    // Assumes grid and block configurations handle the reduction properly
    extern "C" __global__ void reduction_kernel(
        const float* input,
        float* output,
        int n,
        int op_type // 0 = sum, 1 = max, 2 = min, 3 = mean
    ) {
        // ... existing kernel code ...
    }

    // Simplified LogSumExp reduction kernel that works in a single pass
    // for single-block inputs (or produces partial results for multi-block)
    extern "C" __global__ void logsumexp_reduction_kernel(
        const float* input,    // Input data
        float* output,         // Output (scalar result)
        float* block_max,      // Temporary storage for block maximums
        int n                  // Total number of elements
    ) {
        extern __shared__ float sdata[]; // Shared memory for reduction
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Step 1: Find max within this thread's data
        float thread_max = -INFINITY;
        if (i < n) {
            thread_max = input[i];
        }
        
        // Load max into shared memory
        sdata[tid] = thread_max;
        __syncthreads();
        
        // Reduce to find block max
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        
        // Write block max to global memory
        if (tid == 0) {
            block_max[blockIdx.x] = sdata[0];
        }
        __syncthreads(); // Make sure block_max is written before proceeding
        
        // For a single-block reduction, we can continue in the same kernel
        float block_max_val = block_max[blockIdx.x];
        
        // Step 2: Calculate exp(x - max) sum
        float thread_sum = 0.0f;
        if (i < n) {
            thread_sum = expf(input[i] - block_max_val);
        }
        
        // Load sum into shared memory
        sdata[tid] = thread_sum;
        __syncthreads();
        
        // Reduce within block to compute sum
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        // Final step: compute log(sum) + max
        if (tid == 0) {
            output[0] = logf(sdata[0]) + block_max_val;
        }
    }

    // Helper functions for multi-dimensional indexing if not already present
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

    /*
     * Computes the index of the maximum value along a specified axis.
     * Output stores indices as float values.
     */
    extern "C" __global__ void argmax_along_axis_kernel(
        const float* input, float* output_indices,
        const int* input_shape, const int* input_strides,
        const int* output_shape, const int* output_strides,
        int input_ndim, int output_ndim, int axis, int n_output,
        int n_input
    ) {
        int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (output_idx >= n_output) {
            return;
        }

        const int MAX_DIMS = 8; // Assuming max 8 dimensions, adjust if needed
        if (input_ndim > MAX_DIMS || output_ndim > MAX_DIMS) {
             // Handle error or log, maybe write NaN to output
            if (output_idx == 0) printf("Error: argmax_along_axis_kernel dimension limit exceeded.\n");
            return;
        }

        int output_coords[MAX_DIMS] = {0};
        int input_coords[MAX_DIMS] = {0};

        // Convert flat output index to multi-dimensional output coordinates
        flat_index_to_multi_dim(output_idx, output_shape, output_ndim, output_coords);

        // Map output coordinates to the base input coordinates (setting reduction axis coord to 0)
        int current_out_dim = 0;
        for (int i = 0; i < input_ndim; ++i) {
            if (i == axis) {
                input_coords[i] = 0; // Start iteration at index 0 for the reduction axis
            } else {
                 // Ensure we don't read past the end of output_coords if output_ndim < input_ndim - 1
                 if (current_out_dim < output_ndim) {
                    input_coords[i] = output_coords[current_out_dim];
                    current_out_dim++;
                 } else {
                     // This case should ideally not happen if shapes are calculated correctly
                     input_coords[i] = 0;
                 }
            }
        }

        // Calculate the base flat index in the input tensor corresponding to the start of the reduction slice
        int input_idx_base = multi_dim_to_flat_index(input_coords, input_strides, input_ndim);
        int stride_for_axis = input_strides[axis];
        int reduction_size = input_shape[axis];

        // Find the index of the maximum value along the reduction axis
        float max_val = -INFINITY; // Initialize with negative infinity
        int max_idx = 0;          // Index within the reduction dimension

        for (int k = 0; k < reduction_size; ++k) {
            int current_input_idx = input_idx_base + k * stride_for_axis;
            // Bounds check for safety, although theoretically unnecessary if base/strides are correct
            if (current_input_idx >= 0 && current_input_idx < n_input) {
                float current_val = input[current_input_idx];
                if (current_val > max_val) {
                    max_val = current_val;
                    max_idx = k; // Store the index *along the reduction axis*
                }
            } else {
                 // Indicate potential error if index goes out of bounds
                 // This might happen with incorrect stride/shape calculation upstream.
                 // printf("Warning: Index %d out of bounds (n_input=%d)\n", current_input_idx, n_input);
            }
        }

        // Write the index (as a float) to the output tensor
        output_indices[output_idx] = (float)max_idx;
    }

    /*
     * Computes the index of the minimum value along a specified axis.
     * Output stores indices as float values.
     */
    extern "C" __global__ void argmin_along_axis_kernel(
        const float* input, float* output_indices,
        const int* input_shape, const int* input_strides,
        const int* output_shape, const int* output_strides,
        int input_ndim, int output_ndim, int axis, int n_output,
        int n_input
    ) {
        int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (output_idx >= n_output) {
            return;
        }

        const int MAX_DIMS = 8; // Assuming max 8 dimensions, adjust if needed
        if (input_ndim > MAX_DIMS || output_ndim > MAX_DIMS) {
             // Handle error or log, maybe write NaN to output
             if (output_idx == 0) printf("Error: argmin_along_axis_kernel dimension limit exceeded.\n");
             return;
        }

        int output_coords[MAX_DIMS] = {0};
        int input_coords[MAX_DIMS] = {0};

        // Convert flat output index to multi-dimensional output coordinates
        flat_index_to_multi_dim(output_idx, output_shape, output_ndim, output_coords);

        // Map output coordinates to the base input coordinates (setting reduction axis coord to 0)
        int current_out_dim = 0;
        for (int i = 0; i < input_ndim; ++i) {
            if (i == axis) {
                input_coords[i] = 0; // Start iteration at index 0 for the reduction axis
            } else {
                 if (current_out_dim < output_ndim) {
                    input_coords[i] = output_coords[current_out_dim];
                    current_out_dim++;
                 } else {
                     input_coords[i] = 0;
                 }
            }
        }

        // Calculate the base flat index in the input tensor corresponding to the start of the reduction slice
        int input_idx_base = multi_dim_to_flat_index(input_coords, input_strides, input_ndim);
        int stride_for_axis = input_strides[axis];
        int reduction_size = input_shape[axis];

        // Find the index of the minimum value along the reduction axis
        float min_val = INFINITY; // Initialize with positive infinity
        int min_idx = 0;          // Index within the reduction dimension

        for (int k = 0; k < reduction_size; ++k) {
            int current_input_idx = input_idx_base + k * stride_for_axis;
             // Bounds check for safety
            if (current_input_idx >= 0 && current_input_idx < n_input) {
                float current_val = input[current_input_idx];
                if (current_val < min_val) {
                    min_val = current_val;
                    min_idx = k; // Store the index *along the reduction axis*
                }
            } else {
                // printf("Warning: Index %d out of bounds (n_input=%d)\n", current_input_idx, n_input);
            }
        }

        // Write the index (as a float) to the output tensor
        output_indices[output_idx] = (float)min_idx;
    }
}