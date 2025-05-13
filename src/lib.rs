//! A Rust tensor library with automatic differentiation support
//!
//! This library provides a high-performance tensor computation framework with:
//! - CPU and GPU (CUDA) backend support
//! - Automatic differentiation (autograd)
//! - Neural network operations
//! - Optimizers for machine learning
//!
//! # Features
//! - `cuda` - Enables CUDA GPU support (requires CUDA toolkit)
//! - `mnist` - Enables MNIST dataset loading utilities
//!
//! # Example
//! ```rust
//! use rust_tensor_lib::{Tensor, CpuBackend, ops::*};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create tensors
//!     let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
//!     let b = Tensor::<CpuBackend>::from_vec(vec![4.0, 5.0, 6.0], &[3], true)?;
//!     
//!     // Perform operations with automatic gradient tracking
//!     let c = add(&a, &b)?;  // Element-wise addition
//!     
//!     // Sum to scalar for backward pass
//!     let loss = mean(&c, None)?;  // Update mean call to include None as the axis parameter
//!     
//!     // Backward pass (no gradient seed needed for scalar)
//!     loss.backward()?;
//!     
//!     // Access gradients
//!     println!("Gradient of a: {:?}", a.grad());
//!     Ok(())
//! }
//! ```

// --- Central debug_println macro definition ---
/// Conditional logging macro. Prints if 'debug_logs' feature is enabled.
#[cfg(feature = "debug_logs")]
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        ::std::println!("[DEBUG {}] {}", module_path!(), ::std::format_args!($($arg)*))
    };
}

/// Conditional logging macro (disabled version). Does nothing.
#[cfg(not(feature = "debug_logs"))]
#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {};
}

// Declare the modules within the crate
pub mod array;
pub mod backend;
pub mod data;
pub mod error;
pub mod graph;
pub mod hooks;
pub mod init;
pub mod ops;
pub use hooks::{FnHook, Hook};
pub mod optim;

pub mod tensor; // Declare backend module
mod tensor_debug_impl;
pub mod util;

// Only compile and expose this module during testing
pub mod test_utils;

/// Represents the device where a tensor's data resides
#[cfg(feature = "serialization")]
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum Device {
    /// CPU device
    Cpu,
    /// CUDA GPU device with a specific device ID
    #[cfg(feature = "cuda")]
    Cuda(u32),
}

// Re-export the public types for easier use by consumers of the library
pub use array::Array; // Keep Array export for CpuBackend::Storage type visibility
pub use backend::cpu::CpuBackend; // Re-export CpuBackend struct
#[cfg(feature = "cuda")]
pub use backend::cuda::CudaBackend; // Re-export CudaBackend struct
pub use backend::Backend; // Re-export Backend trait
pub use backend::CpuTensor; // Re-export CpuTensor type alias
#[cfg(feature = "cuda")]
pub use backend::CudaTensor; // Re-export CudaTensor type alias
pub use data::*;
pub use error::Error;
pub use error::Reduction;
pub use graph::{Op, OpType};
pub use init::kaiming_uniform; // This function returns Array, may need update if Backend trait changes init
pub use optim::*;

pub use tensor::Tensor; // Re-export Tensor type

// Commented out as conv2d is not directly exported from ops module
// pub use crate::ops::conv2d; // Re-export Tensor type
