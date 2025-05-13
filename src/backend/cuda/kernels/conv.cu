// CUDA kernels for convolution operations

extern "C" __global__ void im2col_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n, int c, int h, int w,
    int kh, int kw, int oh, int ow,
    int stride_h, int stride_w, int pad_h, int pad_w
) {
    // Calculate output position
    int ow_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int filter_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;

    // Check bounds
    if (ow_idx >= oh * ow || filter_idx >= c * kh * kw || batch_idx >= n) {
        return;
    }

    // Calculate output coordinates
    int oh_idx = ow_idx / ow;
    int ow_idx_mod = ow_idx % ow;

    // Calculate filter coordinates
    int c_idx = filter_idx / (kh * kw);
    int filter_idx_mod = filter_idx % (kh * kw);
    int kh_idx = filter_idx_mod / kw;
    int kw_idx = filter_idx_mod % kw;

    // Calculate input coordinates with padding
    int ih_idx = oh_idx * stride_h - pad_h + kh_idx;
    int iw_idx = ow_idx_mod * stride_w - pad_w + kw_idx;

    // Calculate output index - this is the column-major layout for im2col
    // For matmul: weights [C_out, K_eff] @ im2col [K_eff, SPATIAL_OUT]
    // We need im2col in the format [K_eff, SPATIAL_OUT] where K_eff = c*kh*kw
    int output_idx = filter_idx * (oh * ow) + ow_idx;

    // Calculate input index and check bounds
    if (ih_idx >= 0 && ih_idx < h && iw_idx >= 0 && iw_idx < w) {
        int input_idx = ((batch_idx * c + c_idx) * h + ih_idx) * w + iw_idx;
        output[output_idx] = input[input_idx];
    } else {
        output[output_idx] = 0.0f;
    }
}

extern "C" __global__ void col2im_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n, int c, int h, int w,
    int kh, int kw, int oh, int ow,
    int stride_h, int stride_w, int pad_h, int pad_w
) {
    // Calculate output position
    int ih_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iw_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_c_idx = blockIdx.z;

    // Check bounds
    if (ih_idx >= h || iw_idx >= w || batch_c_idx >= n * c) {
        return;
    }

    int batch_idx = batch_c_idx / c;
    int c_idx = batch_c_idx % c;

    // Initialize output to zero
    int output_idx = ((batch_idx * c + c_idx) * h + ih_idx) * w + iw_idx;
    output[output_idx] = 0.0f;

    // Iterate over all possible filter positions that could contribute to this output
    for (int kh_idx = 0; kh_idx < kh; kh_idx++) {
        for (int kw_idx = 0; kw_idx < kw; kw_idx++) {
            // Calculate output coordinates
            int oh_idx = (ih_idx + pad_h - kh_idx) / stride_h;
            int ow_idx = (iw_idx + pad_w - kw_idx) / stride_w;

            // Check if the output coordinates are valid
            if (oh_idx >= 0 && oh_idx < oh && ow_idx >= 0 && ow_idx < ow &&
                (ih_idx + pad_h - kh_idx) % stride_h == 0 &&
                (iw_idx + pad_w - kw_idx) % stride_w == 0) {
                
                // Calculate filter index
                int filter_idx = c_idx * kh * kw + kh_idx * kw + kw_idx;
                
                // Calculate input index
                int ow_linear = oh_idx * ow + ow_idx;
                int input_idx = (batch_idx * (c * kh * kw) + filter_idx) * (oh * ow) + ow_linear;
                
                // Accumulate
                atomicAdd(&output[output_idx], input[input_idx]);
            }
        }
    }
}
