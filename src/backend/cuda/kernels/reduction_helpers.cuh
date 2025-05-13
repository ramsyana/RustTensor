#pragma once

#include <stdio.h>
#include <math.h>

// Helper functions for reduction operations
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

// Helper function to compute output coordinates for a reduction operation
__device__ inline void compute_reduction_coords(int* out_coords, const int* in_coords, int ndim, int axis) {
    int out_dim = 0;
    for (int i = 0; i < ndim; i++) {
        if (i != axis) {
            out_coords[out_dim++] = in_coords[i];
        }
    }
}