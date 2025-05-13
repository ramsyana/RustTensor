// tests/init_tests.rs
use rust_tensor_lib::{
    backend::cpu::CpuBackend,
    init::kaiming_uniform, // kaiming_uniform returns Array for CPU
    CpuTensor,             // Import backend types if creating Tensors
    Error,                 // Import Error for matching
    Tensor,
};

#[test]
fn test_kaiming_uniform_basic() {
    // Test the init function directly
    let fan_in = 784;
    let shape = &[128, fan_in];
    let result = kaiming_uniform(fan_in, shape); // Returns Result<Array, Error>
    assert!(result.is_ok());
    let array = result.unwrap();

    assert_eq!(array.shape(), shape);
    let bound = (6.0 / fan_in as f32).sqrt();
    let data_vec = array.into_raw_vec(); // Consumes array
    assert_eq!(data_vec.len(), 128 * 784);
    for &value in &data_vec {
        assert!(
            value >= -bound && value <= bound,
            "Value {} out of bound {}",
            value,
            bound
        );
    }
}

#[test]
fn test_kaiming_uniform_invalid_shape_empty() {
    assert!(matches!(
        kaiming_uniform(10, &[]),
        Err(Error::InvalidOperation(_))
    ));
}

#[test]
fn test_kaiming_uniform_zero_dim() {
    // kaiming_uniform now returns Ok(empty_array) for zero dim shapes
    let result = kaiming_uniform(10, &[10, 0, 5]);
    assert!(result.is_ok());
    let array = result.unwrap();
    assert_eq!(array.shape(), &[10, 0, 5]);
    assert_eq!(array.size(), 0);
}

#[test]
fn test_kaiming_uniform_zero_fan_in() {
    assert!(matches!(
        kaiming_uniform(0, &[10, 10]),
        Err(Error::InvalidOperation(_))
    ));
}

#[test]
fn test_tensor_kaiming_uniform() {
    // Test the Tensor constructor
    let fan_in = 50;
    let shape = &[100, fan_in];

    // Use Tensor::<CpuBackend>::kaiming_uniform
    let tensor_result = Tensor::<CpuBackend>::kaiming_uniform(fan_in, shape, true);
    assert!(tensor_result.is_ok());
    let tensor: CpuTensor = tensor_result.unwrap();

    assert_eq!(tensor.shape(), shape);
    assert!(tensor.requires_grad());

    let bound = (6.0 / fan_in as f32).sqrt();
    let data_ref = tensor.data(); // Ref<Array>
    for &value in data_ref.as_ref() {
        // Use as_ref()
        assert!(value >= -bound && value <= bound);
    }
}
