use crate::error::Error;

#[cfg(feature = "serialization")]
use serde::{Serialize, Deserialize};

#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaBackend;
#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaStorage;
#[cfg(feature = "cuda")]
use crate::backend::Backend;
use ndarray::{ArrayD, Axis, IxDyn, ShapeError};

#[derive(Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Array {
    pub(crate) data: ArrayD<f32>,
}

impl Array {
    pub fn new(data: ArrayD<f32>) -> Self {
        Self { data }
    }

    pub fn into_ndarray(self) -> ArrayD<f32> {
        self.data
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self, Error> {
        let actual_len = data.len();
        let map_err = |_e: ShapeError| Error::ShapeMismatch {
            expected: shape.to_vec(),
            actual: vec![actual_len],
        };
        let array = ArrayD::from_shape_vec(IxDyn(shape), data).map_err(map_err)?;
        Ok(Self { data: array })
    }

    #[cfg(feature = "cuda")]
    pub fn from_cuda(storage: &CudaStorage) -> Result<Self, Error> {
        // Use the CudaBackend::copy_to_host method to get a Vec<f32>
        let data = CudaBackend::copy_to_host(storage)?;
        let shape = CudaBackend::shape(storage).to_vec();
        Self::from_vec(data, &shape)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::ones(IxDyn(shape)),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the array contains no elements
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get_data(&self) -> &ArrayD<f32> {
        &self.data
    }

    pub fn get_data_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.data
    }

    pub fn into_raw_vec(self) -> Vec<f32> {
        self.data.into_raw_vec_and_offset().0
    }

    pub(crate) fn broadcast_to(&self, shape: &[usize]) -> Result<Array, Error> {
        match self.data.broadcast(IxDyn(shape)) {
            Some(broadcasted_view) => Ok(Array::new(broadcasted_view.to_owned())),
            None => Err(Error::IncompatibleShapes {
                op: "broadcast".to_string(),
                shape_a: self.shape().to_vec(),
                shape_b: shape.to_vec(),
            }),
        }
    }

    pub(crate) fn sum_along_axis(&self, axis: usize) -> Result<Array, Error> {
        if axis >= self.data.ndim() {
            return Err(Error::InvalidIndex(vec![axis]));
        }
        Ok(Array::new(self.data.sum_axis(Axis(axis))))
    }

    pub(crate) fn max_along_axis(&self, axis: usize) -> Result<Array, Error> {
        if axis >= self.data.ndim() {
            return Err(Error::InvalidIndex(vec![axis]));
        }
        Ok(Array::new(self.data.map_axis(Axis(axis), |view| {
            view.iter().fold(f32::MIN, |acc, &x| acc.max(x))
        })))
    }

    /// Sets the shape of the array without changing its data.
    /// The new shape must have the same total number of elements.
    pub fn set_shape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let old_size = self.data.len();
        let new_size = shape.iter().product::<usize>();
        if old_size != new_size {
            return Err(Error::IncompatibleShapes {
                op: "set_shape".to_string(),
                shape_a: self.data.shape().to_vec(),
                shape_b: shape.to_vec(),
            });
        }
        // Create a new ndarray with the same data but different shape
        let new_data = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(shape),
            self.data
                .as_slice()
                .expect("Data should be sliceable for set_shape/reshape")
                .to_vec(),
        )
        .map_err(|e| Error::ShapeError(e.to_string()))?;
        self.data = new_data;
        Ok(())
    }

    /// Reshapes the array without changing its data.
    /// The new shape must have the same total number of elements.
    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let old_size = self.data.len();
        let new_size = shape.iter().product::<usize>();
        if old_size != new_size {
            return Err(Error::ShapeMismatch {
                expected: shape.to_vec(),
                actual: self.data.shape().to_vec(),
            });
        }
        // Create a new ndarray with the same data but different shape
        let new_data = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(shape),
            self.data
                .as_slice()
                .expect("Data should be sliceable for set_shape/reshape")
                .to_vec(),
        )
        .map_err(|e| Error::ShapeError(e.to_string()))?;
        self.data = new_data;
        Ok(())
    }
}
