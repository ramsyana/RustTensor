// log_softmax_fused.cu
extern "C" __global__ void log_softmax_fused_kernel(
    const float* input, float* output, const int* shape, int ndim, int axis, int n_total
) {
    // Each block computes one slice along the reduction axis
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int slice_size = shape[axis];
    int n_slices = n_total / slice_size;
    int slice_idx = blockIdx.x;
    if (slice_idx >= n_slices) return;

    // Compute the base offset for this slice
    int stride = 1;
    for (int i = ndim - 1; i > axis; --i) stride *= shape[i];
    int slice_offset = (slice_idx / stride) * stride * shape[axis] + (slice_idx % stride);

    // Step 1: Find max in slice
    float max_val = -INFINITY;
    for (int i = tid; i < slice_size; i += blockDim.x) {
        int idx = slice_offset + i * stride;
        float val = input[idx];
        if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();
    // Reduction for max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float sum_exp = 0.0f;
    for (int i = tid; i < slice_size; i += blockDim.x) {
        int idx = slice_offset + i * stride;
        sum_exp += expf(input[idx] - max_val);
    }
    sdata[tid] = sum_exp;
    __syncthreads();
    // Reduction for sum
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sum_exp = sdata[0];
    __syncthreads();

    float log_sum_exp = logf(sum_exp) + max_val;

    // Step 3: Write output
    for (int i = tid; i < slice_size; i += blockDim.x) {
        int idx = slice_offset + i * stride;
        output[idx] = input[idx] - log_sum_exp;
    }
}
