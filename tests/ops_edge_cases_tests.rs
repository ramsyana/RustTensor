use rust_tensor_lib::ops::cpu_ops::{add, log_softmax, matmul, mean, mul, relu};
use rust_tensor_lib::Array;

#[test]
fn test_matmul_shape_mismatch() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let b = Array::from_vec(vec![4.0, 5.0, 6.0], &[3, 1]).unwrap();

    // Matrices with incompatible dimensions
    let result = matmul(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_mul_broadcasting() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]).unwrap();
    let b = Array::from_vec(vec![2.0], &[1]).unwrap();

    let result = mul(&a, &b).unwrap();
    let expected = vec![2.0, 4.0, 6.0];
    assert_eq!(result.into_raw_vec(), expected);
}

#[test]
fn test_add_mismatched_shapes() {
    let a = Array::from_vec(vec![1.0, 2.0], &[2]).unwrap();
    let b = Array::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();

    let result = add(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_mean_invalid_axis() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    // Test with axis larger than dimensions
    let result = mean(&a, Some(2));
    assert!(result.is_err());
}

#[test]
fn test_relu_empty_tensor() {
    let a = Array::from_vec(vec![], &[0]).unwrap();

    let expected_shape = &[0];

    let result = relu(&a);
    // Assert that the result is Ok
    assert!(result.is_ok());
    let output_array = result.unwrap();
    // Assert that the output array is also empty and has the correct shape
    assert_eq!(output_array.shape(), expected_shape);
    assert_eq!(output_array.size(), 0);
}

#[test]
fn test_log_softmax_invalid_axis() {
    let a = Array::from_vec(vec![1.0, 2.0], &[2]).unwrap();

    // Test with invalid axis
    let result = log_softmax(&a, 1);
    assert!(result.is_err());
}

#[test]
fn test_log_softmax_large_values() {
    let a = Array::from_vec(vec![1e5, 1e5], &[2]).unwrap();

    let result = log_softmax(&a, 0).unwrap();
    let result_vec = result.into_raw_vec();

    // Both values should be close to ln(0.5) ≈ -0.693
    for value in result_vec {
        assert!((value + 0.693).abs() < 1e-3);
    }
}
