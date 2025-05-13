#include "reduction_helpers.cuh"

// Maximum number of dimensions supported
#ifndef MAX_DIMS
#define MAX_DIMS 8
#endif

/**
 * Kernel for slicing a tensor along specified dimensions.
 * 
 * @param input_ptr Input tensor data
 * @param output_ptr Output tensor data (result of slice)
 * @param input_shape_ptr Shape of the input tensor
 * @param input_strides_ptr Strides of the input tensor
 * @param output_shape_ptr Shape of the output tensor
 * @param output_strides_ptr Strides of the output tensor
 * @param ranges_flat_ptr Flattened ranges [start0, end0, start1, end1, ...]
 * @param ndim Number of dimensions
 * @param num_output_elements Total number of elements in the output tensor
 */
extern "C" __global__ void slice_kernel(
    const float* input_ptr,
    float* output_ptr,
    const int* input_shape_ptr,    // Device pointer
    const int* input_strides_ptr,  // Device pointer
    const int* output_shape_ptr,   // Device pointer
    const int* output_strides_ptr, // Device pointer
    const int* ranges_flat_ptr,    // Device pointer to flattened ranges [start0, end0, start1, end1, ...]
    int ndim,
    int num_output_elements
) {
    int out_flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat_idx >= num_output_elements) {
        return;
    }

    // Convert output flat index to multi-dimensional coordinates
    int output_coords[MAX_DIMS];
    flat_index_to_multi_dim(out_flat_idx, output_shape_ptr, ndim, output_coords);

    // Map output coordinates to input coordinates using the ranges
    int input_coords[MAX_DIMS];
    for (int i = 0; i < ndim; ++i) {
        int range_start = ranges_flat_ptr[i * 2 + 0];
        input_coords[i] = range_start + output_coords[i]; // Assuming step 1
    }

    // Convert input coordinates to flat index
    int in_flat_idx = multi_dim_to_flat_index(input_coords, input_strides_ptr, ndim);
    
    // Copy the value from input to output
    output_ptr[out_flat_idx] = input_ptr[in_flat_idx];
}

/**
 * Kernel for the backward pass of the slice operation.
 * 
 * @param grad_output_ptr Gradient with respect to the output of the slice operation
 * @param grad_input_ptr Gradient with respect to the input of the slice operation (output of this kernel)
 * @param grad_input_shape_ptr Shape of the original input tensor
 * @param grad_input_strides_ptr Strides of the original input tensor
 * @param grad_output_shape_ptr Shape of the gradient output tensor
 * @param grad_output_strides_ptr Strides of the gradient output tensor
 * @param ranges_flat_ptr Flattened ranges [start0, end0, start1, end1, ...]
 * @param ndim Number of dimensions
 * @param num_grad_output_elements Total number of elements in the gradient output tensor
 */
extern "C" __global__ void slice_backward_kernel(
    const float* grad_output_ptr,
    float* grad_input_ptr,         // Output of this kernel
    const int* grad_input_shape_ptr, // Shape of the original full input
    const int* grad_input_strides_ptr,
    const int* grad_output_shape_ptr, // Shape of grad_output (same as slice output shape)
    const int* grad_output_strides_ptr,
    const int* ranges_flat_ptr,     // [start0, end0, start1, end1, ...]
    int ndim,
    int num_grad_output_elements // Number of elements in grad_output
) {
    // This kernel copies grad_output elements to the correct slice in grad_input.
    // grad_input should be pre-zeroed.
    int grad_out_flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (grad_out_flat_idx >= num_grad_output_elements) {
        return;
    }

    // Convert gradient output flat index to multi-dimensional coordinates
    int grad_output_coords[MAX_DIMS];
    flat_index_to_multi_dim(grad_out_flat_idx, grad_output_shape_ptr, ndim, grad_output_coords);

    // Map gradient output coordinates to gradient input coordinates using the ranges
    int grad_input_coords[MAX_DIMS];
    for (int i = 0; i < ndim; ++i) {
        int range_start = ranges_flat_ptr[i * 2 + 0];
        grad_input_coords[i] = range_start + grad_output_coords[i]; // Map output coord to input coord
    }

    // Convert gradient input coordinates to flat index
    int grad_in_flat_idx = multi_dim_to_flat_index(grad_input_coords, grad_input_strides_ptr, ndim);
    
    // Copy the value from gradient output to gradient input
    grad_input_ptr[grad_in_flat_idx] = grad_output_ptr[grad_out_flat_idx];
}

/**
 * Kernel for concatenating multiple tensors along a specified axis.
 * 
 * @param input_ptrs_array Array of device pointers to input tensors
 * @param output_ptr Output tensor data (result of concatenation)
 * @param num_input_tensors Number of input tensors
 * @param input_shapes_flat_ptr Flattened shapes of all input tensors [ndim0, s0_0, s0_1, ..., ndim1, s1_0, s1_1, ...]
 * @param input_strides_flat_ptr Flattened strides of all input tensors
 * @param input_offsets_ptr Offsets to find each tensor's shape/strides in the flattened arrays
 * @param output_shape_ptr Shape of the output tensor
 * @param output_strides_ptr Strides of the output tensor
 * @param concat_axis The axis along which to concatenate
 * @param num_output_elements Total number of elements in the output tensor
 */
extern "C" __global__ void concat_kernel(
    const float** input_ptrs_array,      // Array of device pointers to input tensors
    float* output_ptr,
    const int* num_input_tensors_val,    // Pointer to single int value
    const int* input_shapes_flat,        // Flattened shapes: [ndim0, s0_0,s0_1,..., ndim1, s1_0,s1_1,...]
    const int* input_strides_flat,       // Flattened strides
    const int* input_offsets_in_flat_shapes, // Offsets to find each tensor's shape/strides
    const int* output_shape_ptr,
    const int* output_strides_ptr,
    int output_ndim,
    int concat_axis,
    int num_output_elements
) {
    int out_flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_flat_idx >= num_output_elements) {
        return;
    }

    int num_inputs = *num_input_tensors_val;
    int output_coords[MAX_DIMS];
    flat_index_to_multi_dim(out_flat_idx, output_shape_ptr, output_ndim, output_coords);

    int current_dim_offset = 0;
    for (int i = 0; i < num_inputs; ++i) {
        int input_ndim_offset = input_offsets_in_flat_shapes[i * 2 + 0];
        int input_strides_offset = input_offsets_in_flat_shapes[i * 2 + 1];

        const int* current_input_shape = &input_shapes_flat[input_ndim_offset + 1]; // +1 to skip ndim
        int current_input_ndim = input_shapes_flat[input_ndim_offset];
        const int* current_input_strides = &input_strides_flat[input_strides_offset];

        int size_along_axis = current_input_shape[concat_axis];

        if (output_coords[concat_axis] >= current_dim_offset &&
            output_coords[concat_axis] < current_dim_offset + size_along_axis) {
            
            int input_coords[MAX_DIMS];
            for (int d = 0; d < current_input_ndim; ++d) {
                if (d == concat_axis) {
                    input_coords[d] = output_coords[d] - current_dim_offset;
                } else {
                    input_coords[d] = output_coords[d];
                }
            }
            int in_flat_idx = multi_dim_to_flat_index(input_coords, current_input_strides, current_input_ndim);
            output_ptr[out_flat_idx] = input_ptrs_array[i][in_flat_idx];
            return; // Found the source
        }
        current_dim_offset += size_along_axis;
    }
}

/**
 * Kernel for the backward pass of the concat operation.
 * This kernel extracts a slice from grad_output for a specific input tensor.
 * 
 * @param grad_output_ptr Gradient with respect to the output of the concat operation
 * @param grad_input_ptr Gradient with respect to one of the input tensors
 * @param grad_output_shape_ptr Shape of the gradient output tensor
 * @param grad_output_strides_ptr Strides of the gradient output tensor
 * @param grad_input_shape_ptr Shape of the gradient input tensor
 * @param grad_input_strides_ptr Strides of the gradient input tensor
 * @param concat_axis The axis along which tensors were concatenated
 * @param offset_along_axis The offset along the concat axis for this input tensor
 * @param num_grad_input_elements Total number of elements in the gradient input tensor
 */
extern "C" __global__ void concat_backward_kernel(
    const float* grad_output_ptr,
    float* grad_input_ptr,          // Output of this kernel (gradient for one input tensor)
    const int* grad_output_shape_ptr,
    const int* grad_output_strides_ptr,
    const int* grad_input_shape_ptr,
    const int* grad_input_strides_ptr,
    int ndim,
    int concat_axis,
    int offset_along_axis,
    int num_grad_input_elements
) {
    int grad_in_flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (grad_in_flat_idx >= num_grad_input_elements) {
        return;
    }

    // Convert gradient input flat index to multi-dimensional coordinates
    int grad_input_coords[MAX_DIMS];
    flat_index_to_multi_dim(grad_in_flat_idx, grad_input_shape_ptr, ndim, grad_input_coords);

    // Map gradient input coordinates to gradient output coordinates
    int grad_output_coords[MAX_DIMS];
    for (int i = 0; i < ndim; ++i) {
        if (i == concat_axis) {
            grad_output_coords[i] = grad_input_coords[i] + offset_along_axis;
        } else {
            grad_output_coords[i] = grad_input_coords[i];
        }
    }

    // Convert gradient output coordinates to flat index
    int grad_out_flat_idx = multi_dim_to_flat_index(grad_output_coords, grad_output_strides_ptr, ndim);
    
    // Copy the value from gradient output to gradient input
    grad_input_ptr[grad_in_flat_idx] = grad_output_ptr[grad_out_flat_idx];
}
