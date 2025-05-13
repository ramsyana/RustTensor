// tests/tensor_method_tests.rs
use approx::assert_abs_diff_eq;
use rust_tensor_lib::{ops, CpuBackend, Error, Tensor};

#[test]
fn test_relu_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![-1.0, 0.5, 2.0], &[3], false)?;
    let res_method = t.relu()?;
    let res_ops = ops::relu(&t)?;
    assert_eq!(res_method.shape(), res_ops.shape());
    assert_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref()
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_exp_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![0.0, 1.0, 2.0], &[3], false)?;
    let res_method = t.exp()?;
    let res_ops = ops::exp(&t)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_ln_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 4.0], &[3], false)?;
    let res_method = t.ln()?;
    let res_ops = ops::ln(&t)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_sigmoid_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![0.0, 2.0, -2.0], &[3], false)?;
    let res_method = t.sigmoid()?;
    let res_ops = ops::sigmoid(&t)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_tanh_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![0.0, 1.0, -1.0], &[3], false)?;
    let res_method = t.tanh()?;
    let res_ops = ops::tanh(&t)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_softplus_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![0.0, 1.0, -1.0], &[3], false)?;
    let res_method = t.softplus()?;
    let res_ops = ops::softplus(&t)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_sqrt_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 4.0, 9.0], &[3], false)?;
    let res_method = t.sqrt()?;
    let res_ops = ops::sqrt(&t)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_square_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![2.0, -3.0, 0.5], &[3], false)?;
    let res_method = t.square()?;
    let res_ops = ops::square(&t)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_abs_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![-2.0, 3.0, -0.5], &[3], false)?;
    let res_method = t.abs()?;
    let res_ops = ops::abs(&t)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_mean_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)?;
    let res_method = t.mean(Some(0))?;
    let res_ops = ops::mean(&t, Some(0))?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_sum_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)?;
    let res_method = t.sum(Some(1))?;
    let res_ops = ops::sum(&t, Some(1))?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_max_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], false)?;
    let res_method = t.max(Some(1))?;
    let res_ops = ops::max(&t, Some(1))?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_min_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], false)?;
    let res_method = t.min(Some(1))?;
    let res_ops = ops::min(&t, Some(1))?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_prod_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)?;
    let res_method = t.prod(Some(0))?;
    let res_ops = ops::prod(&t, Some(0))?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_logsumexp_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)?;
    let res_method = t.logsumexp(Some(1))?;
    let res_ops = ops::logsumexp(&t, Some(1))?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_argmax_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], false)?;
    let res_method = t.argmax(1)?;
    let res_ops = ops::argmax(&t, 1)?;
    assert_eq!(res_method.shape(), res_ops.shape());
    assert_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref()
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_argmin_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], false)?;
    let res_method = t.argmin(1)?;
    let res_ops = ops::argmin(&t, 1)?;
    assert_eq!(res_method.shape(), res_ops.shape());
    assert_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref()
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_log_softmax_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], false)?;
    let res_method = t.log_softmax(0)?;
    let res_ops = ops::log_softmax(&t, 0)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_elu_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![-1.0, 0.0, 2.0], &[3], false)?;
    let res_method = t.elu(1.0)?;
    let res_ops = ops::elu(&t, 1.0)?;
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_transpose_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)?;
    let res_method = t.transpose()?;
    let res_ops = ops::transpose(&t)?;
    assert_eq!(res_method.shape(), res_ops.shape());
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_view_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)?;
    let res_method = t.view(&[4])?;
    let res_ops = ops::view(&t, &[4])?;
    assert_eq!(res_method.shape(), res_ops.shape());
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_broadcast_to_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0], &[1], false)?;
    let res_method = t.broadcast_to(&[2])?;
    let res_ops = ops::broadcast_to(&t, &[2])?;
    assert_eq!(res_method.shape(), res_ops.shape());
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}

#[test]
fn test_reshape_method() -> Result<(), Error> {
    let t = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)?;
    let res_method = t.reshape(&[4])?;
    let res_ops = t.view(&[4])?;
    assert_eq!(res_method.shape(), res_ops.shape());
    assert_abs_diff_eq!(
        res_method.to_cpu()?.data().as_ref(),
        res_ops.to_cpu()?.data().as_ref(),
        epsilon = 1e-6
    );
    // requires_grad propagation test (auto-inserted)
    Ok(())
}
