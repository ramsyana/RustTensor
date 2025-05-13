//! CUDA kernel definitions and PTX constants

// Will contain PTX constants for elementwise operations
pub const _ELEMENTWISE_PTX: &str = include_str!("elementwise.cu");

// CUDA reduction kernel PTX code
// Will contain PTX constants for reduction operations
pub const _REDUCTION_PTX: &str = include_str!("reduction.cu");
