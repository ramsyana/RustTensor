// Operator overloading tests for &Tensor<B>
use rust_tensor_lib::{CpuBackend, Error, Tensor};

#[test]
fn test_add_overload() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], false)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![3.0, 4.0], &[2], false)?;
    let c = (&a + &b)?;
    let expected_data = vec![4.0, 6.0];
    assert_eq!(c.shape(), vec![2]);
    assert_eq!(c.to_cpu()?.data().as_ref(), expected_data.as_slice());
    Ok(())
}

#[test]
fn test_mul_overload() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![2.0, 3.0], &[2], false)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![4.0, 5.0], &[2], false)?;
    let c = (&a * &b)?;
    let expected_data = vec![8.0, 15.0];
    assert_eq!(c.to_cpu()?.data().as_ref(), expected_data.as_slice());
    Ok(())
}

#[test]
fn test_sub_overload() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![5.0, 7.0], &[2], false)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![2.0, 3.0], &[2], false)?;
    let c = (&a - &b)?;
    let expected_data = vec![3.0, 4.0];
    assert_eq!(c.to_cpu()?.data().as_ref(), expected_data.as_slice());
    Ok(())
}

#[test]
fn test_div_overload() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![6.0, 8.0], &[2], false)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![2.0, 4.0], &[2], false)?;
    let c = (&a / &b)?;
    let expected_data = vec![3.0, 2.0];
    assert_eq!(c.to_cpu()?.data().as_ref(), expected_data.as_slice());
    Ok(())
}

#[test]
fn test_add_overload_requires_grad() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], true)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![3.0, 4.0], &[2], true)?;
    let c = (&a + &b)?;
    assert!(c.requires_grad());
    Ok(())
}

// --- Enhanced Operator Overloading Tests ---

#[test]
fn test_broadcast_add_scalar() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], false)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![10.0], &[1], false)?; // Scalar shape
    let c = (&a + &b)?;
    let expected_data = vec![11.0, 12.0, 13.0];
    assert_eq!(c.shape(), vec![3]);
    assert_eq!(c.to_cpu()?.data().as_ref(), expected_data.as_slice());
    Ok(())
}

#[test]
fn test_broadcast_mul_vector() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![10.0, 100.0], &[2], false)?; // Broadcast along axis 1
    let c = (&a * &b)?;
    let expected_data = vec![10.0, 200.0, 30.0, 400.0];
    assert_eq!(c.shape(), vec![2, 2]);
    assert_eq!(c.to_cpu()?.data().as_ref(), expected_data.as_slice());
    Ok(())
}

#[test]
fn test_operator_chaining() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], false)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![3.0, 4.0], &[2], false)?;
    let c = Tensor::<CpuBackend>::from_vec(vec![5.0, 6.0], &[2], false)?;
    let d = (&(&a + &b)? * &c)?;
    let expected_data = vec![(1.0 + 3.0) * 5.0, (2.0 + 4.0) * 6.0];
    assert_eq!(d.to_cpu()?.data().as_ref(), expected_data.as_slice());
    Ok(())
}

#[test]
fn test_requires_grad_propagation_mixed() -> Result<(), Error> {
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], true)?;
    let b = Tensor::<CpuBackend>::from_vec(vec![3.0, 4.0], &[2], false)?;
    let c = (&a + &b)?;
    assert!(
        c.requires_grad(),
        "requires_grad should propagate if any operand requires grad"
    );
    let d = (&c * &b)?;
    assert!(
        d.requires_grad(),
        "requires_grad should propagate through chained ops"
    );
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use rust_tensor_lib::{
        backend::cuda::{init_context, CudaBackend, CudaContextGuard},
        Tensor,
    };

    fn cuda_tensor(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Tensor<CudaBackend> {
        Tensor::<CudaBackend>::from_vec(data, shape, requires_grad).unwrap()
    }

    #[test]
    fn test_cuda_broadcast_add_scalar() -> Result<(), Error> {
        init_context(0)?;
        let _guard = CudaContextGuard::new()?;
        let a = cuda_tensor(vec![1.0, 2.0, 3.0], &[3], false);
        let b = cuda_tensor(vec![10.0], &[1], false);
        let c = (&a + &b)?;
        let expected_data = vec![11.0, 12.0, 13.0];
        assert_eq!(c.shape(), vec![3]);
        assert_eq!(c.to_cpu()?.data().as_ref(), expected_data.as_slice());
        Ok(())
    }

    #[test]
    fn test_cuda_operator_chaining() -> Result<(), Error> {
        init_context(0)?;
        let _guard = CudaContextGuard::new()?;
        let a = cuda_tensor(vec![1.0, 2.0], &[2], false);
        let b = cuda_tensor(vec![3.0, 4.0], &[2], false);
        let c = cuda_tensor(vec![5.0, 6.0], &[2], false);
        let d = (&(&a + &b)? * &c)?;
        let expected_data = vec![(1.0 + 3.0) * 5.0, (2.0 + 4.0) * 6.0];
        assert_eq!(d.to_cpu()?.data().as_ref(), expected_data.as_slice());
        Ok(())
    }

    #[test]
    fn test_cuda_requires_grad_propagation() -> Result<(), Error> {
        init_context(0)?;
        let _guard = CudaContextGuard::new()?;
        let a = cuda_tensor(vec![1.0, 2.0], &[2], true);
        let b = cuda_tensor(vec![3.0, 4.0], &[2], false);
        let c = (&a + &b)?;
        assert!(c.requires_grad());
        Ok(())
    }
}
