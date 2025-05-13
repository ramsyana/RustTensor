use crate::error::Error;
use cust::memory::DeviceBuffer;

/// Helper function to calculate strides for a given shape
/// Strides represent the number of elements to skip to move by 1 in each dimension
pub fn calc_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![]; // Return empty strides for 0D tensors (scalars)
    }

    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Compute the output shape for a reduction operation
pub fn compute_reduction_shape(
    input_shape: &[usize],
    axis: Option<usize>,
) -> Result<Vec<usize>, Error> {
    match axis {
        None => {
            Ok(vec![]) // Global reduction results in scalar []
        }
        Some(axis_val) => {
            if input_shape.is_empty() {
                // Handle 0D input
                if axis_val == 0 {
                    return Ok(vec![]);
                }
                // Reducing axis 0 of 0D gives 0D
                else {
                    return Err(Error::InvalidIndex(vec![axis_val]));
                }
            }
            if axis_val >= input_shape.len() {
                return Err(Error::InvalidIndex(vec![axis_val]));
            }
            let mut output_shape = input_shape.to_vec();
            output_shape.remove(axis_val);
            // If the resulting shape is empty (e.g., reduced a 1D array), return []
            if output_shape.is_empty() && input_shape.len() == 1 {
                Ok(vec![])
            } else {
                Ok(output_shape)
            }
        }
    }
}

/// Convert any DeviceCopy type to a device buffer
pub fn to_device_buffer_generic<T: cust::memory::DeviceCopy>(
    data: &[T],
) -> Result<DeviceBuffer<T>, Error> {
    if data.is_empty() {
        // Handle empty slice case appropriately
        println!(
            "[WARN] to_device_buffer_generic called with empty slice. Allocating buffer of size 1."
        );
        let buffer = unsafe { DeviceBuffer::<T>::uninitialized(1)? };
        Ok(buffer)
    } else {
        DeviceBuffer::from_slice(data).map_err(|e| Error::CudaError(e.to_string()))
    }
}
