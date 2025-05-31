#pragma once

#include <stdio.h>
#include <math.h>

// Helper functions for reduction operations - 64-bit versions for large tensors
__device__ inline long long multi_dim_to_flat_index(const long long* coords, const long long* strides, int ndim) {
    long long index = 0;
    for (int i = 0; i < ndim; ++i) {
        index += coords[i] * strides[i];
    }
    return index;
}

// Original 32-bit version for backward compatibility
__device__ inline int multi_dim_to_flat_index(const int* coords, const int* strides, int ndim) {
    int index = 0;
    for (int i = 0; i < ndim; ++i) {
        index += coords[i] * strides[i];
    }
    return index;
}

// 64-bit version for large tensors
__device__ inline void flat_index_to_multi_dim(long long flat_index, const long long* shape, int ndim, long long* coords) {
    long long current_index = flat_index;
    for (int i = ndim - 1; i >= 0; --i) {
        const long long dim = shape[i];
        if (dim == 0) {
            coords[i] = 0;
            current_index = 0;           // prevent stale remainder
            continue;
        }
        coords[i] = current_index % dim;
        current_index /= dim;
    }
}

// Original 32-bit version for backward compatibility
__device__ inline void flat_index_to_multi_dim(int flat_index, const int* shape, int ndim, int* coords) {
    int current_index = flat_index;
    for (int i = ndim - 1; i >= 0; --i) {
        const int dim = shape[i];
        if (dim == 0) {
            coords[i] = 0;
            current_index = 0;           // prevent stale remainder
            continue;
        }
        coords[i] = current_index % dim;
        current_index /= dim;
    }
}

// Helper function to compute output coordinates for a reduction operation - 64-bit version
__device__ inline void compute_reduction_coords(long long* out_coords, const long long* in_coords, int ndim, int axis) {
    int out_dim = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != axis) {
            out_coords[out_dim++] = in_coords[i];
        }
    }
}

// Original 32-bit version for backward compatibility
__device__ inline void compute_reduction_coords(int* out_coords, const int* in_coords, int ndim, int axis) {
    int out_dim = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != axis) {
            out_coords[out_dim++] = in_coords[i];
        }
    }
}