// tests/test_concat.rs
use rust_tensor_lib::{backend::cpu::CpuBackend, error::Error, ops, Backend, CpuTensor, Tensor};

// Helper to create CpuTensor quickly
fn cpu_tensor(data: Vec<f32>, shape: &[usize]) -> CpuTensor {
    Tensor::<CpuBackend>::from_vec(data, shape, false).unwrap()
}

fn cpu_tensor_req_grad(data: Vec<f32>, shape: &[usize]) -> CpuTensor {
    Tensor::<CpuBackend>::from_vec(data, shape, true).unwrap()
}

#[test]
fn test_cpu_concat_forward() -> Result<(), Error> {
    // Test case 1: Concat along axis 0
    let a = cpu_tensor(vec![1.0, 2.0, 3.0], &[1, 3]);
    let b = cpu_tensor(vec![4.0, 5.0, 6.0], &[1, 3]);
    
    let result = ops::concat(&[&a, &b], 0)?;
    assert_eq!(result.shape(), &[2, 3]);
    let result_data = CpuBackend::copy_to_host(&*result.data())?;
    assert_eq!(result_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    
    // Test case 2: Concat along axis 1
    let c = cpu_tensor(vec![1.0, 2.0], &[2, 1]);
    let d = cpu_tensor(vec![3.0, 4.0], &[2, 1]);
    
    let result2 = ops::concat(&[&c, &d], 1)?;
    assert_eq!(result2.shape(), &[2, 2]);
    let result2_data = CpuBackend::copy_to_host(&*result2.data())?;
    assert_eq!(result2_data, vec![1.0, 3.0, 2.0, 4.0]);
    
    // Test case 3: Concat three tensors
    let e = cpu_tensor(vec![1.0, 2.0], &[1, 2]);
    let f = cpu_tensor(vec![3.0, 4.0], &[1, 2]);
    let g = cpu_tensor(vec![5.0, 6.0], &[1, 2]);
    
    let result3 = ops::concat(&[&e, &f, &g], 0)?;
    assert_eq!(result3.shape(), &[3, 2]);
    let result3_data = CpuBackend::copy_to_host(&*result3.data())?;
    assert_eq!(result3_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    
    // Test case 4: Empty tensor list should return error
    let empty_result = ops::concat::<CpuBackend>(&[], 0);
    assert!(empty_result.is_err());
    
    // Test case 5: Incompatible shapes should return error
    let h = cpu_tensor(vec![1.0, 2.0], &[1, 2]);
    let i = cpu_tensor(vec![3.0, 4.0, 5.0], &[1, 3]);
    let incompatible_result = ops::concat(&[&h, &i], 0);
    assert!(incompatible_result.is_err());
    
    // Test case 6: Invalid axis should return error
    let j = cpu_tensor(vec![1.0, 2.0], &[1, 2]);
    let k = cpu_tensor(vec![3.0, 4.0], &[1, 2]);
    let invalid_axis_result = ops::concat(&[&j, &k], 2);
    assert!(invalid_axis_result.is_err());
    
    // Test requires_grad propagation
    let a_grad = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0], &[1, 3]);
    let b_grad = cpu_tensor_req_grad(vec![4.0, 5.0, 6.0], &[1, 3]);
    let result_grad = ops::concat(&[&a_grad, &b_grad], 0)?;
    assert!(result_grad.requires_grad());
    
    // Test with one tensor requiring grad and one not
    let a_grad2 = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0], &[1, 3]);
    let b_no_grad = cpu_tensor(vec![4.0, 5.0, 6.0], &[1, 3]);
    let result_mixed_grad = ops::concat(&[&a_grad2, &b_no_grad], 0)?;
    assert!(result_mixed_grad.requires_grad());
    
    Ok(())
}

#[test]
fn test_cpu_concat_backward() -> Result<(), Error> {
    // Test backward pass for concat operation along axis 0
    let a = cpu_tensor_req_grad(vec![1.0, 2.0], &[1, 2]);
    let b = cpu_tensor_req_grad(vec![3.0, 4.0], &[1, 2]);
    
    let result = ops::concat(&[&a, &b], 0)?;
    assert_eq!(result.shape(), &[2, 2]);
    
    // Create a gradient for the output: [[10, 20], [30, 40]]
    let grad_output = cpu_tensor(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    
    // Set the gradient for the output tensor
    result.set_grad(Some(grad_output.data().clone()));
    
    // Perform backward pass
    result.backward()?;
    
    // Check the input gradients
    // a's gradient should be [10, 20]
    let a_grad = a.grad().unwrap();
    assert_eq!(a_grad.shape(), &[1, 2]);
    let a_grad_data = CpuBackend::copy_to_host(&*a_grad)?;
    assert_eq!(a_grad_data, vec![10.0, 20.0]);
    
    // b's gradient should be [30, 40]
    let b_grad = b.grad().unwrap();
    assert_eq!(b_grad.shape(), &[1, 2]);
    let b_grad_data = CpuBackend::copy_to_host(&*b_grad)?;
    assert_eq!(b_grad_data, vec![30.0, 40.0]);
    
    // Test backward pass for concat operation along axis 1
    let c = cpu_tensor_req_grad(vec![1.0, 2.0], &[2, 1]);
    let d = cpu_tensor_req_grad(vec![3.0, 4.0], &[2, 1]);
    
    let result2 = ops::concat(&[&c, &d], 1)?;
    assert_eq!(result2.shape(), &[2, 2]);
    
    // Create a gradient for the output: [[10, 20], [30, 40]]
    let grad_output2 = cpu_tensor(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
    
    // Set the gradient for the output tensor
    result2.set_grad(Some(grad_output2.data().clone()));
    
    // Perform backward pass
    result2.backward()?;
    
    // Check the input gradients
    // c's gradient should be [[10], [30]]
    let c_grad = c.grad().unwrap();
    assert_eq!(c_grad.shape(), &[2, 1]);
    let c_grad_data = CpuBackend::copy_to_host(&*c_grad)?;
    assert_eq!(c_grad_data, vec![10.0, 30.0]);
    
    // d's gradient should be [[20], [40]]
    let d_grad = d.grad().unwrap();
    assert_eq!(d_grad.shape(), &[2, 1]);
    let d_grad_data = CpuBackend::copy_to_host(&*d_grad)?;
    assert_eq!(d_grad_data, vec![20.0, 40.0]);
    
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    use rust_tensor_lib::{backend::cuda::{CudaBackend, CudaContextGuard, init_context}, CudaTensor};

    // Helper to create CudaTensor quickly
    fn cuda_tensor(data: Vec<f32>, shape: &[usize]) -> CudaTensor {
        Tensor::<CudaBackend>::from_vec(data, shape, false).unwrap()
    }

    fn cuda_tensor_req_grad(data: Vec<f32>, shape: &[usize]) -> CudaTensor {
        Tensor::<CudaBackend>::from_vec(data, shape, true).unwrap()
    }

    #[test]
    fn test_cuda_concat_forward() -> Result<(), Error> {
        // Initialize CUDA context
        init_context(0)?;
        let _guard = CudaContextGuard::new()?;
        // Test case 1: Concat along axis 0
        let a = cuda_tensor(vec![1.0, 2.0, 3.0], &[1, 3]);
        let b = cuda_tensor(vec![4.0, 5.0, 6.0], &[1, 3]);
        
        let result = ops::concat(&[&a, &b], 0)?;
        assert_eq!(result.shape(), &[2, 3]);
        let result_data = result.data().to_vec()?;
        assert_eq!(result_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        // Test case 2: Concat along axis 1
        let c = cuda_tensor(vec![1.0, 2.0], &[2, 1]);
        let d = cuda_tensor(vec![3.0, 4.0], &[2, 1]);
        
        let result2 = ops::concat(&[&c, &d], 1)?;
        assert_eq!(result2.shape(), &[2, 2]);
        let result2_data = result2.data().to_vec()?;
        assert_eq!(result2_data, vec![1.0, 3.0, 2.0, 4.0]);
        
        // Test requires_grad propagation
        let a_grad = cuda_tensor_req_grad(vec![1.0, 2.0, 3.0], &[1, 3]);
        let b_grad = cuda_tensor_req_grad(vec![4.0, 5.0, 6.0], &[1, 3]);
        let result_grad = ops::concat(&[&a_grad, &b_grad], 0)?;
        assert!(result_grad.requires_grad());
        
        Ok(())
    }

    #[test]
    fn test_cuda_concat_backward() -> Result<(), Error> {
        // Initialize CUDA context
        init_context(0)?;
        let _guard = CudaContextGuard::new()?;
        // Test backward pass for concat operation along axis 0
        let a = cuda_tensor_req_grad(vec![1.0, 2.0], &[1, 2]);
        let b = cuda_tensor_req_grad(vec![3.0, 4.0], &[1, 2]);
        
        let result = ops::concat(&[&a, &b], 0)?;
        assert_eq!(result.shape(), &[2, 2]);
        
        // Create a gradient for the output: [[10, 20], [30, 40]]
        let grad_output = cuda_tensor(vec![10.0, 20.0, 30.0, 40.0], &[2, 2]);
        
        // Set the gradient for the output tensor
        result.set_grad(Some(grad_output.data().clone()));
        
        // Perform backward pass
        result.backward()?;
        
        // Check the input gradients
        // a's gradient should be [10, 20]
        let a_grad = a.grad().unwrap();
        assert_eq!(a_grad.shape(), &[1, 2]);
        let a_grad_data = a_grad.to_vec()?;
        assert_eq!(a_grad_data, vec![10.0, 20.0]);
        
        // b's gradient should be [30, 40]
        let b_grad = b.grad().unwrap();
        assert_eq!(b_grad.shape(), &[1, 2]);
        let b_grad_data = b_grad.to_vec()?;
        assert_eq!(b_grad_data, vec![30.0, 40.0]);
        
        Ok(())
    }
}
