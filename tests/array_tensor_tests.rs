// tests/array_tensor_tests.rs
use rust_tensor_lib::{
    backend::cpu::CpuBackend, // Import backend
    Backend,
    CpuTensor, // Import traits/types
    Error,
    Tensor,
};

#[test]
fn test_backend_array_creation() {
    // Test CpuBackend methods via Backend trait
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = &[2, 3];
    let storage = CpuBackend::from_vec(data.clone(), shape).unwrap();

    // Check shape via Backend trait
    assert_eq!(CpuBackend::shape(&storage), shape);
    // Check size via Backend trait
    assert_eq!(CpuBackend::size(&storage), 6);
    // Check data via Backend trait
    let raw_vec = CpuBackend::into_raw_vec(storage).unwrap();
    assert_eq!(raw_vec, data);
}

#[test]
fn test_backend_zeros_ones() {
    let shape = &[2, 3];
    let zeros = CpuBackend::zeros(shape).unwrap();
    let ones = CpuBackend::ones(shape).unwrap();

    // Check shapes
    assert_eq!(CpuBackend::shape(&zeros), shape);
    assert_eq!(CpuBackend::shape(&ones), shape);

    // Check values
    let zeros_data = CpuBackend::into_raw_vec(zeros).unwrap();
    let ones_data = CpuBackend::into_raw_vec(ones).unwrap();

    assert!(zeros_data.iter().all(|&x| x == 0.0));
    assert!(ones_data.iter().all(|&x| x == 1.0));
}

#[test]
fn test_tensor_creation_cpu() {
    // Test Tensor with CpuBackend
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = &[2, 3];
    let storage = CpuBackend::from_vec(data, shape).unwrap();

    // Create tensor using CpuBackend storage
    let tensor: CpuTensor = Tensor::new(storage, true); // Use type alias

    assert!(tensor.requires_grad());
    assert_eq!(tensor.shape(), shape); // Use tensor.shape()

    // Test factory methods on Tensor<CpuBackend>
    let zeros_tensor = Tensor::<CpuBackend>::zeros(shape, false).unwrap();
    assert_eq!(zeros_tensor.shape(), shape);
    assert!(!zeros_tensor.requires_grad());

    let zeros_data = CpuBackend::copy_to_host(&*zeros_tensor.data()).unwrap();
    assert!(zeros_data.iter().all(|&x| x == 0.0));

    let ones_tensor = Tensor::<CpuBackend>::ones(shape, true).unwrap();
    assert_eq!(ones_tensor.shape(), shape);
    assert!(ones_tensor.requires_grad());

    let ones_data = CpuBackend::copy_to_host(&*ones_tensor.data()).unwrap();
    assert!(ones_data.iter().all(|&x| x == 1.0));
}

#[test]
fn test_backend_shape_mismatch() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let shape = &[2, 3]; // Requires 6 elements

    // Test from_vec error using Backend trait
    let result = CpuBackend::from_vec(data, shape);

    assert!(result.is_err());
    match result {
        Err(Error::ShapeMismatch {
            expected,
            actual: _,
        }) => {
            assert_eq!(expected, shape);
            // actual might vary based on ndarray version, checking expected is sufficient
        }
        _ => panic!("Expected ShapeMismatch error"),
    }
}

#[test]
fn test_tensor_data_access() {
    let tensor: CpuTensor = Tensor::new(CpuBackend::from_vec(vec![1.0, 2.0], &[2]).unwrap(), false);
    {
        // Borrow scope
        let data_ref = tensor.data(); // Ref<Array>
        assert_eq!(data_ref.shape(), &[2]);
        let data_vec = CpuBackend::copy_to_host(&*data_ref).unwrap();
        assert_eq!(&data_vec, &[1.0, 2.0]);
    } // Borrow released

    // Test setting data
    let new_storage = CpuBackend::from_vec(vec![3.0, 4.0], &[2]).unwrap();
    let result = tensor.set_data(new_storage);
    assert!(result.is_ok());

    let data_ref_after = tensor.data();
    let data_vec_after = CpuBackend::copy_to_host(&*data_ref_after).unwrap();
    assert_eq!(&data_vec_after, &[3.0, 4.0]);
}

#[test]
fn test_tensor_grad_access() {
    let tensor: CpuTensor = Tensor::zeros(&[2], true).unwrap();
    assert!(tensor.grad().is_none());

    let grad_storage = CpuBackend::ones(&[2]).unwrap();
    tensor.set_grad(Some(grad_storage));

    assert!(tensor.grad().is_some());
    {
        let grad_ref = tensor.grad().unwrap(); // Option<Ref<Array>> -> Ref<Array>
        assert_eq!(grad_ref.shape(), &[2]); // Use shape() on Array
        let grad_data = CpuBackend::copy_to_host(&*grad_ref).unwrap();
        assert_eq!(&grad_data, &[1.0, 1.0]);
    }

    tensor.zero_grad();
    assert!(tensor.grad().is_none());
}
