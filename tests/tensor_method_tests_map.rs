// tests/tensor_method_tests_map.rs
use rust_tensor_lib::{CpuBackend, Error, Tensor};

#[test]
fn test_tensor_map_cpu() -> Result<(), Error> {
    println!("--- Running test_tensor_map_cpu ---");

    // Case 1: Simple mapping function
    let t1_cpu = Tensor::<CpuBackend>::from_vec(vec![1.0, -2.0, 3.0], &[3], true)?; // Input requires_grad
    let map_fn1 = |x: f32| x * 2.0 + 1.0;
    let mapped_t1_cpu = t1_cpu.map(map_fn1)?;

    assert_eq!(
        mapped_t1_cpu.shape(),
        &[3],
        "Shape mismatch for mapped_t1_cpu"
    );
    let expected_data1 = vec![3.0, -3.0, 7.0]; // (1*2+1), (-2*2+1), (3*2+1)
    assert_eq!(
        mapped_t1_cpu.to_cpu()?.data().as_ref(),
        expected_data1.as_slice(),
        "Data mismatch for mapped_t1_cpu"
    );
    assert!(
        !mapped_t1_cpu.requires_grad(),
        "Output of map (mapped_t1_cpu) should not require_grad"
    );
    println!("CPU Map Test Case 1 Passed: Simple map");

    // Case 2: Mapping an empty tensor
    let t2_cpu_empty = Tensor::<CpuBackend>::from_vec(vec![], &[0, 2], false)?;
    let map_fn2 = |x: f32| x.powi(2); // Square
    let mapped_t2_cpu_empty = t2_cpu_empty.map(map_fn2)?;

    assert_eq!(
        mapped_t2_cpu_empty.shape(),
        &[0, 2],
        "Shape mismatch for mapped_t2_cpu_empty"
    );
    assert!(
        mapped_t2_cpu_empty.to_cpu()?.data().as_ref().is_empty(),
        "Data mismatch for mapped_t2_cpu_empty (should be empty)"
    );
    assert!(
        !mapped_t2_cpu_empty.requires_grad(),
        "Output of map (mapped_t2_cpu_empty) should not require_grad"
    );
    println!("CPU Map Test Case 2 Passed: Empty tensor");

    // Case 3: Mapping a tensor with different shape
    let t3_cpu_matrix = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)?;
    let map_fn3 = |x: f32| x - 0.5;
    let mapped_t3_cpu_matrix = t3_cpu_matrix.map(map_fn3)?;

    assert_eq!(
        mapped_t3_cpu_matrix.shape(),
        &[2, 2],
        "Shape mismatch for mapped_t3_cpu_matrix"
    );
    let expected_data3 = vec![0.5, 1.5, 2.5, 3.5];
    assert_eq!(
        mapped_t3_cpu_matrix.to_cpu()?.data().as_ref(),
        expected_data3.as_slice(),
        "Data mismatch for mapped_t3_cpu_matrix"
    );
    assert!(
        !mapped_t3_cpu_matrix.requires_grad(),
        "Output of map (mapped_t3_cpu_matrix) should not require_grad"
    );
    println!("CPU Map Test Case 3 Passed: Matrix map");

    println!("--- test_tensor_map_cpu finished ---");
    Ok(())
}
