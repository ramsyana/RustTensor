use crate::array::Array;
use crate::error::Error;
use cust::memory::{CopyDestination, DeviceBuffer};

#[cfg(feature = "serialization")]
use serde::{Serialize, Deserialize, Serializer, Deserializer};
#[cfg(feature = "serialization")]
use serde::ser::SerializeStruct;
#[cfg(feature = "serialization")]
use serde::de::{self, Visitor, MapAccess};
#[cfg(feature = "serialization")]
use std::fmt;

// Helper for CudaError conversion
fn map_cuda_error(e: cust::error::CudaError) -> Error {
    Error::CudaError(e.to_string())
}

pub struct CudaStorage {
    data: DeviceBuffer<f32>,
    shape: Vec<usize>,
}

#[cfg(feature = "serialization")]
impl Serialize for CudaStorage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Transfer data from GPU to host for serialization
        let host_data = self.to_vec().map_err(|e| {
            serde::ser::Error::custom(format!("Failed to transfer data from GPU to host: {}", e))
        })?;
        
        // Serialize both the shape and the host data
        let mut state = serializer.serialize_struct("CudaStorage", 2)?;
        state.serialize_field("shape", &self.shape)?;
        state.serialize_field("data", &host_data)?;
        state.end()
    }
}

#[cfg(feature = "serialization")]
impl<'de> Deserialize<'de> for CudaStorage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Define the fields we expect in the serialized data
        enum Field { Shape, Data }
        
        // Implement a visitor to parse the field names
        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;
                
                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;
                    
                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`shape` or `data`")
                    }
                    
                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "shape" => Ok(Field::Shape),
                            "data" => Ok(Field::Data),
                            _ => Err(de::Error::unknown_field(value, &["shape", "data"])),
                        }
                    }
                }
                
                deserializer.deserialize_identifier(FieldVisitor)
            }
        }
        
        // Implement the main visitor for CudaStorage
        struct CudaStorageVisitor;
        
        impl<'de> Visitor<'de> for CudaStorageVisitor {
            type Value = CudaStorage;
            
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CudaStorage")
            }
            
            fn visit_map<V>(self, mut map: V) -> Result<CudaStorage, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut shape: Option<Vec<usize>> = None;
                let mut host_data: Option<Vec<f32>> = None;
                
                // Extract shape and data from the deserialized map
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Shape => {
                            if shape.is_some() {
                                return Err(de::Error::duplicate_field("shape"));
                            }
                            shape = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if host_data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            host_data = Some(map.next_value()?);
                        }
                    }
                }
                
                // Ensure we have both shape and data
                let shape = shape.ok_or_else(|| de::Error::missing_field("shape"))?;
                let host_data = host_data.ok_or_else(|| de::Error::missing_field("data"))?;
                
                // Create a new CudaStorage and transfer data from host to GPU
                let mut storage = CudaStorage::new(&shape).map_err(|e| {
                    de::Error::custom(format!("Failed to create CudaStorage: {}", e))
                })?;
                
                storage.copy_from_slice(&host_data).map_err(|e| {
                    de::Error::custom(format!("Failed to transfer data from host to GPU: {}", e))
                })?;
                
                Ok(storage)
            }
        }
        
        // Start deserialization
        deserializer.deserialize_struct("CudaStorage", &["shape", "data"], CudaStorageVisitor)
    }
}

impl From<Vec<f32>> for CudaStorage {
    fn from(value: Vec<f32>) -> Self {
        let shape = vec![value.len()];
        let mut storage = CudaStorage::new(&shape).expect("Failed to create CudaStorage from Vec"); // This is a From impl, panic is acceptable
        storage
            .copy_from_slice(&value)
            .expect("Failed to copy Vec data to CudaStorage"); // This is a From impl, panic is acceptable
        storage
    }
}

impl From<Array> for CudaStorage {
    fn from(value: Array) -> Self {
        let shape = value.shape().to_vec();
        let data = value.into_raw_vec();
        let mut storage =
            CudaStorage::new(&shape).expect("Failed to create CudaStorage from Array"); // This is a From impl, panic is acceptable
        storage
            .copy_from_slice(&data)
            .expect("Failed to copy Array data to CudaStorage"); // This is a From impl, panic is acceptable
        storage
    }
}

impl CudaStorage {
    pub fn new(shape: &[usize]) -> Result<Self, Error> {
        debug_println!(
            "[CudaStorage::new] Creating new CudaStorage with shape {:?}",
            shape
        );
        // Handle 0D scalar case: allocate buffer of size 1
        let size = shape.iter().product::<usize>().max(1);
        debug_println!("[CudaStorage::new] Before DeviceBuffer::uninitialized");
        let data = unsafe { DeviceBuffer::<f32>::uninitialized(size) }.map_err(map_cuda_error)?;
        debug_println!("[CudaStorage::new] After DeviceBuffer::uninitialized");
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    pub fn zeros(shape: &[usize]) -> Result<Self, Error> {
        debug_println!(
            "[CudaStorage::zeros] Creating zeroed CudaStorage with shape {:?}",
            shape
        );
        // Handle 0D scalar case: allocate buffer of size 1
        let size = shape.iter().product::<usize>().max(1);
        debug_println!("[CudaStorage::zeros] Before DeviceBuffer::zeroed");
        let data = DeviceBuffer::<f32>::zeroed(size).map_err(map_cuda_error)?;
        debug_println!("[CudaStorage::zeros] After DeviceBuffer::zeroed");
        Ok(Self {
            data,
            shape: shape.to_vec(),
        })
    }

    pub fn from_device_buffer(data: DeviceBuffer<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn from_cpu(array: &crate::array::Array) -> Result<Self, Error> {
        let shape = array.shape();
        let data = array.clone().into_raw_vec();
        let mut storage = Self::new(shape)?;
        storage.copy_from_slice(&data)?;
        Ok(storage)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        // Buffer length is the source of truth for emptiness.
        // Logical shape [] is handled by len() returning >= 1.
        self.data.len() == 0
    }

    /// Sets the shape metadata.
    /// **Warning:** Does not resize the buffer, only updates metadata.
    /// Will panic if the new shape requires more elements than the buffer can hold.
    pub(crate) fn set_shape(&mut self, new_shape: Vec<usize>) {
        let required_len: usize = if new_shape.is_empty() {
            1 // Scalar shape [] requires buffer len >= 1
        } else {
            new_shape.iter().product()
        };

        // Ensure the buffer is large enough for the required elements
        if self.data.len() < required_len {
            // Use panic here because this indicates a serious internal logic error
            // where an operation produced an incompatible shape/buffer combination.
            panic!(
                "Illegal state: Cannot set shape {:?} for CudaStorage buffer of length {} (requires at least {} elements)",
                new_shape, self.data.len(), required_len
            );
        }

        self.shape = new_shape;
    }

    pub fn as_ptr(&self) -> cust::memory::DevicePointer<f32> {
        self.data.as_device_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> cust::memory::DevicePointer<f32> {
        self.data.as_device_ptr()
    }

    pub fn copy_from_slice(&mut self, data: &[f32]) -> Result<(), Error> {
        debug_println!(
            "[CudaStorage::copy_from_slice] Copying from slice of len {}",
            data.len()
        );
        let logical_size = if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        };
        let buffer_len = self.data.len();

        if data.len() != logical_size {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                actual: vec![data.len()],
            });
        }

        // Ensure we don't try to copy more data than the buffer can hold
        let copy_len = std::cmp::min(logical_size, buffer_len);
        if copy_len > 0 {
            debug_println!("[CudaStorage::copy_from_slice] Before self.data.copy_from");
            println!(
                "[DEBUG CudaStorage::copy_from_slice] Copying data to device on thread {:?}",
                std::thread::current().id()
            );
            println!("[DEBUG CudaStorage::copy_from_slice] (No direct API to print current CUDA context in cust 0.3.2)");
            self.data
                .copy_from(&data[..copy_len])
                .map_err(map_cuda_error)?;
            debug_println!("[CudaStorage::copy_from_slice] After self.data.copy_from");
        }

        Ok(())
    }

    pub fn to_vec(&self) -> Result<Vec<f32>, Error> {
        // Determine logical size based on shape, treat 0D scalar as size 1
        let logical_size = if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product::<usize>()
        };

        // Allocate host vec based on logical size
        let mut host_data = vec![0.0f32; logical_size];

        if logical_size > 0 && self.data.len() > 0 {
            // If logical size is 1 (scalar), only copy 1 element, even if buffer is larger
            let copy_len = std::cmp::min(logical_size, self.data.len());
            self.data
                .copy_to(&mut host_data[..copy_len])
                .map_err(map_cuda_error)?;
        }
        Ok(host_data)
    }

    pub fn copy_device_to_device(&mut self, src: &Self) -> Result<(), Error> {
        if self.len() != src.len() {
            return Err(Error::ShapeMismatch {
                expected: self.shape.clone(),
                actual: src.shape.clone(),
            });
        }
        if !self.is_empty() {
            self.data.copy_from(&src.data).map_err(map_cuda_error)?;
        }
        Ok(())
    }

    /// Copies data from another CudaStorage instance (device-to-device).
    /// Shapes must be compatible (same number of elements).
    pub(crate) fn copy_from_storage(&mut self, src: &Self) -> Result<(), Error> {
        // Use len() for comparison as shape check is done in set_data
        if self.data.len() != src.data.len() {
            return Err(Error::InternalLogicError(format!(
                "Buffer length mismatch in copy_from_storage: dst={}, src={}",
                self.data.len(),
                src.data.len()
            )));
        }
        self.data.copy_from(&src.data).map_err(map_cuda_error)?;
        Ok(())
    }

    /// Gets a slice of data from the storage starting at offset and ending at end_offset
    /// 
    /// # Arguments
    /// * `offset` - The starting offset in elements
    /// * `end_offset` - The ending offset in elements (exclusive)
    /// 
    /// # Returns
    /// A vector containing the data from offset to end_offset
    /// 
    /// # Errors
    /// Returns an error if the offset or end_offset are out of bounds or if the CUDA operation fails
    pub fn get_slice(&self, offset: usize, end_offset: usize) -> Result<Vec<f32>, Error> {
        if offset >= self.len() || end_offset > self.len() || offset > end_offset {
            return Err(Error::IndexOutOfBounds {
                index: offset,
                size: self.len(),
            });
        }

        let slice_len = end_offset - offset;
        if slice_len == 0 {
            return Ok(Vec::new());
        }

        let mut host_data = vec![0.0f32; slice_len];
        
        // Copy data from device to host
        self.data.index(offset..(offset + slice_len)).copy_to(&mut host_data).map_err(map_cuda_error)?;
        
        Ok(host_data)
    }

    /// Sets a slice of data in the storage starting at offset
    /// 
    /// # Arguments
    /// * `offset` - The starting offset in elements
    /// * `data` - The data to copy to the device
    /// 
    /// # Returns
    /// Ok(()) if successful
    /// 
    /// # Errors
    /// Returns an error if the offset + data.len() is out of bounds or if the CUDA operation fails
    pub fn set_slice(&mut self, offset: usize, data: &[f32]) -> Result<(), Error> {
        if offset >= self.len() || offset + data.len() > self.len() {
            return Err(Error::IndexOutOfBounds {
                index: offset + data.len() - 1,
                size: self.len(),
            });
        }

        if data.is_empty() {
            return Ok(());
        }

        // Copy data from host to device
        self.data.index(offset..(offset + data.len())).copy_from(data).map_err(map_cuda_error)?;
        
        Ok(())
    }

    /// Copies a slice of data from a source CudaStorage to this storage at a specified offset.
    ///
    /// # Arguments
    /// * `dst_offset_elements`: The starting element index in `self` (destination) where copying will begin.
    /// * `src_storage`: The source `CudaStorage` to copy from.
    /// * `src_offset_elements`: The starting element index in `src_storage` from where data will be read.
    /// * `num_elements`: The number of f32 elements to copy.
    /// * `stream`: The CUDA stream to use for the operation (currently unused).
    ///
    /// # Errors
    /// Returns an error if:
    /// - Copying would write out of bounds in the destination.
    /// - Copying would read out of bounds from the source.
    /// - The CUDA operation fails.
    pub fn copy_from_storage_slice_at_offset(
        &mut self,
        dst_offset_elements: usize,
        src_storage: &Self,
        src_offset_elements: usize,
        num_elements: usize,
        _stream: &cust::stream::Stream, // Unused for now
    ) -> Result<(), Error> {
        if num_elements == 0 {
            return Ok(()); // Nothing to copy
        }

        // Destination bounds check
        if dst_offset_elements.saturating_add(num_elements) > self.data.len() {
            return Err(Error::IndexOutOfBounds {
                index: dst_offset_elements.saturating_add(num_elements).saturating_sub(1),
                size: self.data.len(),
            });
        }

        // Source bounds check
        if src_offset_elements.saturating_add(num_elements) > src_storage.data.len() {
            return Err(Error::IndexOutOfBounds {
                index: src_offset_elements.saturating_add(num_elements).saturating_sub(1),
                size: src_storage.data.len(),
            });
        }

        // Get a slice of data from the source
        let src_slice = src_storage.get_slice(src_offset_elements, src_offset_elements + num_elements)?;
        
        // Copy to the destination at the specified offset
        self.set_slice(dst_offset_elements, &src_slice)?;
        
        Ok(())
    }
}

impl Clone for CudaStorage {
    fn clone(&self) -> Self {
        let mut new_data = unsafe { DeviceBuffer::<f32>::uninitialized(self.len()) }
            .map_err(|e| {
                Error::CudaError(format!("Failed to allocate device buffer for clone: {}", e))
            })
            .expect("Failed to allocate device buffer for clone"); // Clone impl can panic
        if !self.is_empty() {
            new_data
                .copy_from(&self.data)
                .map_err(|e| {
                    Error::CudaError(format!("Failed to copy device buffer for clone: {}", e))
                })
                .expect("Failed to copy device buffer for clone"); // Clone impl can panic
        }
        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }
}

impl std::fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CudaStorage(shape={:?}, len={}, ptr={:?})",
            self.shape,
            self.len(),
            self.data.as_device_ptr()
        )
    }
}

impl std::fmt::Display for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaStorage(shape={:?})", self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::{get_global_context, init_context, CudaContextGuard};
    use serial_test::serial;

    #[cfg(feature = "cuda")]
    #[serial]
    #[test]
    fn test_cuda_storage_copy_slice_at_offset() -> Result<(), Error> {
        // Initialize CUDA context
        init_context(0)?;
        let _guard = CudaContextGuard::new()?;
        let context = get_global_context()?;
        let stream = context.get_stream(); // Get the stream reference

        // Create source storage
        let src_data_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut src_storage = CudaStorage::new(&[src_data_vec.len()])?;
        src_storage.copy_from_slice(&src_data_vec)?;

        // Create destination storage (larger and initialized to zeros)
        let dst_size = 10;
        let mut dst_storage = CudaStorage::zeros(&[dst_size])?;

        // Test 1: Copy a slice to the beginning of dst
        // Copy elements 3.0, 4.0 from src (offset 2, 2 elements) to dst at offset 0
        dst_storage.copy_from_storage_slice_at_offset(0, &src_storage, 2, 2, &stream)?;
        stream.synchronize()?; // Wait for copy to complete

        let dst_vec1 = dst_storage.to_vec()?;
        let expected_dst_vec1 = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(dst_vec1, expected_dst_vec1, "Slice copy to beginning failed");

        // Test 2: Copy a slice to a middle offset in dst
        // Re-initialize dst_storage to zeros
        dst_storage = CudaStorage::zeros(&[dst_size])?;
        // Copy elements 1.0, 2.0, 3.0 from src (offset 0, 3 elements) to dst at offset 3
        dst_storage.copy_from_storage_slice_at_offset(3, &src_storage, 0, 3, &stream)?;
        stream.synchronize()?;

        let dst_vec2 = dst_storage.to_vec()?;
        let expected_dst_vec2 = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(dst_vec2, expected_dst_vec2, "Slice copy to middle failed");

        // Test 3: Bounds checks (out of bounds destination)
        let result_dst_oob =
            dst_storage.copy_from_storage_slice_at_offset(8, &src_storage, 0, 3, &stream);
        assert!(matches!(result_dst_oob, Err(Error::IndexOutOfBounds { .. })));

        // Test 4: Bounds checks (out of bounds source)
        let result_src_oob =
            dst_storage.copy_from_storage_slice_at_offset(0, &src_storage, 4, 3, &stream);
        assert!(matches!(result_src_oob, Err(Error::IndexOutOfBounds { .. })));

        // Test 5: Copy zero elements
        let mut dst_storage_zero_copy = CudaStorage::zeros(&[dst_size])?;
        dst_storage_zero_copy.copy_from_storage_slice_at_offset(0, &src_storage, 0, 0, &stream)?;
        stream.synchronize()?;
        let dst_vec_zero = dst_storage_zero_copy.to_vec()?;
        assert_eq!(dst_vec_zero, vec![0.0; dst_size], "Zero element copy modified destination");

        Ok(())
    }
}
