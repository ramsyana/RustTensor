#include "reduction_helpers.cuh"

extern "C" {
    /*
     * Computes the backward pass for the logsumexp reduction operation.
     * grad_input = grad_output * exp(x - y) = grad_output * softmax(x along axis)
     */
    __global__ void logsumexp_backward_kernel(
        const float* x, const float* y, const float* grad_output,
        float* grad_input,
        const int* x_shape, const int* y_shape, const int* grad_output_shape,
        const int* x_strides, const int* y_strides, const int* grad_output_strides,
        int x_ndim, int y_ndim, int grad_output_ndim,
        int axis, int n_input
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_input) return;

        const int MAX_DIMS = 8;
        if (x_ndim > MAX_DIMS || y_ndim > MAX_DIMS || grad_output_ndim > MAX_DIMS) return;

        // Debug info (only for first thread)
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("[KERNEL DEBUG] logsumexp_backward_kernel launched with:\n");
            printf("  x_ndim=%d, y_ndim=%d, grad_output_ndim=%d, axis=%d\n", 
                   x_ndim, y_ndim, grad_output_ndim, axis);
            printf("  x_shape=[%d", x_shape[0]);
            for (int i = 1; i < x_ndim; i++) printf(",%d", x_shape[i]);
            printf("]\n");
        }

        // Convert flat index to input coordinates
        int x_coords[MAX_DIMS];
        flat_index_to_multi_dim(idx, x_shape, x_ndim, x_coords);

        // Compute output coordinates based on reduction axis
        int y_coords[MAX_DIMS];
        int grad_out_coords[MAX_DIMS];

        if (axis == -1) { // Global reduction
            // For global reduction, y and grad_output are scalars
            y_coords[0] = 0;
            grad_out_coords[0] = 0;
        } else { // Axis reduction
            // Map input coordinates to output coordinates by skipping the reduction axis
            int out_dim = 0;
            for (int i = 0; i < x_ndim; i++) {
                if (i != axis) {
                    y_coords[out_dim] = x_coords[i];
                    grad_out_coords[out_dim] = x_coords[i];
                    out_dim++;
                }
            }
        }

        // Get flat indices for y and grad_output
        const int y_idx = multi_dim_to_flat_index(y_coords, y_strides, y_ndim);
        const int grad_out_idx = multi_dim_to_flat_index(grad_out_coords, grad_output_strides, grad_output_ndim);

        // Compute gradient: grad_input = grad_output * exp(x - y)
        const float x_val = x[idx];
        const float y_val = y[y_idx];
        const float grad_out_val = grad_output[grad_out_idx];
        grad_input[idx] = grad_out_val * expf(x_val - y_val);

        // Debug output for first few elements
        if (idx < 3) {
            printf("Thread %d: x=%f, y=%f, grad_out=%f, grad_in=%f\n",
                   idx, x_val, y_val, grad_out_val, grad_input[idx]);
        }
    }
}