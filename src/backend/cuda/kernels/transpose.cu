// src/backend/cuda/kernels/transpose.cu

#include <stdio.h> // Optional: for printf debugging inside kernel if needed

// Use tiling for better memory coalescence and performance
#define TILE_DIM 16 // Tile width

extern "C" {

/*
 * Transposes a 2D matrix (rows x cols) stored in row-major order.
 * Output matrix will have shape (cols x rows) and also be row-major.
 * Uses shared memory tiling.
 */
__global__ void transpose_2d_kernel(
    const float* input,  // Input matrix data (row-major, rows * cols elements)
    float* output,       // Output matrix data (row-major, cols * rows elements)
    int rows,            // Number of rows in the input matrix
    int cols             // Number of columns in the input matrix
) {
    // Shared memory tile
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    // Calculate global thread indices for the *input* matrix reading
    int read_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int read_col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Calculate global thread indices for the *output* matrix writing
    int write_row = blockIdx.x * TILE_DIM + threadIdx.y; // Transposed block indices
    int write_col = blockIdx.y * TILE_DIM + threadIdx.x;

    // --- Load tile from global memory to shared memory ---
    // Check bounds for reading from the input matrix
    if (read_row < rows && read_col < cols) {
        tile[threadIdx.y][threadIdx.x] = input[read_row * cols + read_col];
    }

    // Synchronize to ensure the entire tile is loaded before proceeding
    __syncthreads();

    // --- Write transposed tile from shared memory to global memory ---
    // Check bounds for writing to the output matrix
    if (write_row < cols && write_col < rows) {
        output[write_row * rows + write_col] = tile[threadIdx.x][threadIdx.y]; // Read transposed from tile
    }
}

} // extern "C"
