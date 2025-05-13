#[cfg(feature = "cuda")]
use cust;
use std::ffi;
use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Shape error: {0}")]
    ShapeError(String),

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Incompatible shapes for operation {op}: {shape_a:?} and {shape_b:?}")]
    IncompatibleShapes {
        op: String,
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    },

    #[error("Invalid index: {0:?}")]
    InvalidIndex(Vec<usize>),

    #[error("Index out of bounds: index {index}, size {size}")]
    IndexOutOfBounds {
        index: usize,
        size: usize,
    },

    #[error("Dimension mismatch: expected {0}, got {1}")]
    DimensionMismatch(usize, usize),

    #[cfg(feature = "mnist")]
    #[error("CSV parsing error: {0}")]
    CsvError(#[from] csv::Error),

    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    CudaError(String),

    #[cfg(feature = "cuda")]
    #[error("CUDA cuBLAS error: {0}")]
    CublasError(String),

    #[error("CString conversion error: {0}")]
    NulError(#[from] ffi::NulError),

    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),
    
    #[error("I/O error: {0}")]
    IoErrorString(String),
    
    #[cfg(feature = "serialization")]
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[cfg(feature = "serialization")]
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Tensor does not have a gradient")]
    NoGradientError,

    #[error("Operation requires tensor to require grad")]
    RequiresGradError,

    #[error("Operation cannot be performed on empty tensor")]
    EmptyTensor,

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Error during tensor initialization")]
    InitializationError,

    #[error("Internal logic error: {0}")]
    InternalLogicError(String),

    #[error("Gradient check error: analytical={analytical:?}, numerical={numerical:?}, max_rel_error={max_rel_error}, max_abs_error={max_abs_error}, at_index={at_index}")]
    GradientCheckError {
        analytical: Vec<f32>,
        numerical: Vec<f32>,
        max_rel_error: f32,
        max_abs_error: f32,
        at_index: usize,
    },

    #[error("Operation not yet implemented: {0}")]
    Unimplemented(String),

    #[error("Out of memory: {0}")]
    OutOfMemory(String),
}

#[cfg(feature = "cuda")]
impl From<cust::error::CudaError> for Error {
    fn from(err: cust::error::CudaError) -> Self {
        Error::CudaError(err.to_string())
    }
}

/// Specifies the reduction to apply to the output of a loss function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction applied (output has the same shape as input).
    None,
    /// The output is summed over all elements.
    Sum,
    /// The output is averaged over all elements.
    Mean,
}
