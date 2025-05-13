#include <float.h> // For FLT_MIN

extern "C" {

/*
 * MaxPool2D Forward Kernel (NCHW)
 * - input: [N, C, H_in, W_in]
 * - output_val: [N, C, H_out, W_out] -> Stores max values
 * - output_idx: [N, C, H_out, W_out] -> Stores flat index (0 to K_h*K_w-1) of max val *within the window*
 */
__global__ void max_pool2d_forward_kernel(
    const float* input,
    float* output_val,
    float* output_idx, // Store indices as float for simplicity, convert to int on host if needed
    int N, int C, int H_in, int W_in,
    int K_h, int K_w,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x; // Output width index
    int oh = blockIdx.y * blockDim.y + threadIdx.y; // Output height index
    int nc = blockIdx.z;                           // Batch * Channel index

    if (ow >= W_out || oh >= H_out || nc >= N * C) {
        return;
    }

    int n_idx = nc / C;
    int c_idx = nc % C;

    // Top-left corner of the receptive field in the input tensor (potentially padded)
    int h_start = oh * stride_h - pad_h;
    int w_start = ow * stride_w - pad_w;

    float max_val_in_window = -FLT_MAX; // Smallest possible float
    int max_k_flat_idx = 0;             // Flat index within the K_h * K_w window

    for (int kh = 0; kh < K_h; ++kh) {
        for (int kw = 0; kw < K_w; ++kw) {
            int h_curr = h_start + kh;
            int w_curr = w_start + kw;

            float current_val;
            if (h_curr >= 0 && h_curr < H_in && w_curr >= 0 && w_curr < W_in) {
                // Valid input pixel
                int input_flat_idx = ((n_idx * C + c_idx) * H_in + h_curr) * W_in + w_curr;
                current_val = input[input_flat_idx];
            } else {
                // Padded pixel, effectively -infinity for max pooling
                current_val = -FLT_MAX;
            }

            if (current_val > max_val_in_window) {
                max_val_in_window = current_val;
                max_k_flat_idx = kh * K_w + kw; // Store flat index within this window
            }
        }
    }

    int output_flat_idx = ((n_idx * C + c_idx) * H_out + oh) * W_out + ow;
    output_val[output_flat_idx] = max_val_in_window;
    output_idx[output_flat_idx] = (float)max_k_flat_idx;
}


/*
 * MaxPool2D Backward Kernel (NCHW)
 * - grad_output: [N, C, H_out, W_out] -> Gradient from next layer
 * - indices: [N, C, H_out, W_out] -> Flat indices from forward pass
 * - grad_input: [N, C, H_in, W_in] -> Gradient to be computed for input
 */
__global__ void max_pool2d_backward_kernel(
    const float* grad_output,
    const float* indices, // Indices stored as float
    float* grad_input,    // Output gradient
    int N, int C, int H_in, int W_in,
    int K_h, int K_w,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int total_input_elements // For atomicAdd bounds check if needed, though not strictly necessary if grad_input is zeroed
) {
    // Each thread handles one element of grad_output
    int ow = blockIdx.x * blockDim.x + threadIdx.x; // Output width index
    int oh = blockIdx.y * blockDim.y + threadIdx.y; // Output height index
    int nc = blockIdx.z;                           // Batch * Channel index

    if (ow >= W_out || oh >= H_out || nc >= N * C) {
        return;
    }

    int n_idx = nc / C;
    int c_idx = nc % C;

    int output_flat_idx = ((n_idx * C + c_idx) * H_out + oh) * W_out + ow;
    float grad_val = grad_output[output_flat_idx];
    int max_k_flat_idx = (int)indices[output_flat_idx]; // Convert float index back to int

    int kh_max = max_k_flat_idx / K_w;
    int kw_max = max_k_flat_idx % K_w;

    int h_start = oh * stride_h - pad_h;
    int w_start = ow * stride_w - pad_w;

    int h_in_target = h_start + kh_max;
    int w_in_target = w_start + kw_max;

    // Check if the target input pixel (where the max came from) is within valid input bounds
    if (h_in_target >= 0 && h_in_target < H_in && w_in_target >= 0 && w_in_target < W_in) {
        int input_flat_idx = ((n_idx * C + c_idx) * H_in + h_in_target) * W_in + w_in_target;
        // Atomically add the gradient to the corresponding input element
        // This is necessary because multiple output elements might map to the same input element
        // if kernel windows overlap significantly (though less common with max pooling).
        // For max pooling, typically only one output element's max comes from a specific input,
        // but if stride < kernel_size, overlaps occur. atomicAdd handles this.
        // If grad_input is pre-zeroed, and strides >= kernel_size, direct assignment might work for SOME cases,
        // but atomicAdd is safer.
        atomicAdd(&grad_input[input_flat_idx], grad_val);
    }
}

} // extern "C"
