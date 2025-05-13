#![cfg(feature = "cuda")]

use rust_tensor_lib::{
    backend::cuda::{init_context, CudaBackend, CudaContextGuard},
    ops,
    test_utils::check_gradient,
    Error, Tensor,
};
use serial_test::serial;

#[serial]
#[test]
fn test_cuda_logsumexp_gradient() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create test input tensor with values that will exercise the logsumexp operation
    let x = Tensor::<CudaBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true)?;

    // Test global logsumexp
    let logsumexp_global_fn =
        |inputs: &[Tensor<CudaBackend>]| -> Result<Tensor<CudaBackend>, Error> {
            let lse = ops::logsumexp(&inputs[0], None)?;
            ops::mean(&lse, None) // Convert to scalar for gradient checking
        };

    // Use a higher tolerance for numerical stability
    const LOGSUMEXP_TOLERANCE: f32 = 5e-2;
    check_gradient(
        logsumexp_global_fn,
        &[x.clone()],
        0,
        1e-3,
        LOGSUMEXP_TOLERANCE,
    )?;

    // Test axis logsumexp
    let logsumexp_axis_fn = |inputs: &[Tensor<CudaBackend>]| -> Result<Tensor<CudaBackend>, Error> {
        let lse = ops::logsumexp(&inputs[0], Some(0))?;
        ops::mean(&lse, None) // Convert to scalar for gradient checking
    };

    check_gradient(
        logsumexp_axis_fn,
        &[x.clone()],
        0,
        1e-3,
        LOGSUMEXP_TOLERANCE,
    )?;

    Ok(())
}
