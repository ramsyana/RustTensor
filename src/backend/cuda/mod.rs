mod context;
pub(crate) mod kernels;
mod ops;
mod storage;
mod utils;

// Make CudaContextGuard public
pub use context::{get_global_context, init_context, CudaContextGuard};
pub use ops::CudaBackend;
pub use storage::CudaStorage; // Use the one from ops module that implements Backend

// Define CudaTensor type alias here for convenience
pub type CudaTensor = crate::tensor::Tensor<CudaBackend>;
