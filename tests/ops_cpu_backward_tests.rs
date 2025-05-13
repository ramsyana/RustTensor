use approx::assert_abs_diff_eq;
use rust_tensor_lib::backend::cpu::CpuBackend;
use rust_tensor_lib::ops;
use rust_tensor_lib::{Backend, Error, Op, OpType, Tensor};
use std::rc::Rc;

// test_cpu_sum_backward remains the same...
#[test]
fn test_cpu_sum_backward() {
    // Test case 1: Global sum
    let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();
    let y = ops::sum(&x, None).unwrap();
    y.backward().unwrap();

    let grad = x.grad().expect("Gradient missing for x (case 1)");
    let grad_data = CpuBackend::copy_to_host(&grad).unwrap();
    assert_eq!(grad_data, vec![1.0, 1.0, 1.0, 1.0]);

    // Test case 2: Sum along axis 0
    let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();
    let y = ops::sum(&x, Some(0)).unwrap();
    y.set_grad(Some(CpuBackend::from_vec(vec![1.0, 1.0], &[2]).unwrap()));
    y.backward().unwrap();

    let grad = x.grad().expect("Gradient missing for x (case 2)");
    let grad_data = CpuBackend::copy_to_host(&grad).unwrap();
    assert_eq!(grad_data, vec![1.0, 1.0, 1.0, 1.0]);

    // Test case 3: Sum along axis 1
    let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true).unwrap();
    let y = ops::sum(&x, Some(1)).unwrap();
    y.set_grad(Some(CpuBackend::from_vec(vec![1.0, 1.0], &[2]).unwrap()));
    y.backward().unwrap();

    let grad = x.grad().expect("Gradient missing for x (case 3)");
    let grad_data = CpuBackend::copy_to_host(&grad).unwrap();
    assert_eq!(grad_data, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_cpu_max_backward_global_with_ties() -> Result<(), Error> {
    let input_data = vec![1.0, 5.0, 2.0, 5.0, 3.0, 0.0];
    let input_shape = &[2, 3];

    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;
    let forward_output_cpu = ops::max(&input_cpu, None)?; // Global max
    let loss_cpu = ops::mean(&forward_output_cpu, None)?; // Scalar loss

    loss_cpu.backward()?; // Calculate analytical gradient

    let grad_cpu_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let actual_grad_vec = grad_cpu_ref.as_ref().to_vec();

    // Expected gradient (dL/dy = 1, 2 ties)
    let expected_grad_vec = vec![0.0, 0.5, 0.0, 0.5, 0.0, 0.0];

    println!("Actual CPU Grad: {:?}", actual_grad_vec);
    println!("Expected CPU Grad: {:?}", expected_grad_vec);

    assert_eq!(
        actual_grad_vec.len(),
        expected_grad_vec.len(),
        "Gradient vector lengths differ"
    );
    for (actual, expected) in actual_grad_vec.iter().zip(expected_grad_vec.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
    }

    Ok(())
}

#[test]
fn test_cpu_min_backward_global_with_ties() -> Result<(), Error> {
    // Input data matching the CUDA test case where min=1.0 occurs twice
    let input_data = vec![1.0, 5.0, 2.0, 3.0, 1.0, 6.0];
    let input_shape = &[2, 3];

    // Create tensor requiring grad
    let input_cpu = Tensor::<CpuBackend>::from_vec(input_data.clone(), input_shape, true)?;

    // Perform forward ops: min (global) -> mean (scalar loss)
    let forward_output_cpu = ops::min(&input_cpu, None)?; // Global min
    let loss_cpu = ops::mean(&forward_output_cpu, None)?; // Scalar loss

    // Perform backward pass
    loss_cpu.backward()?; // Calculate analytical gradient

    // Get the calculated gradient
    let grad_cpu_ref = input_cpu.grad().ok_or(Error::NoGradientError)?;
    let actual_grad_vec = grad_cpu_ref.as_ref().to_vec();

    // Calculate the expected gradient:
    // The minimum value is 1.0, occurring at indices 0 and 4.
    // The gradient from the mean loss is 1.0 (since it's mean of a scalar).
    // This 1.0 gradient should be distributed equally among the minimum value locations.
    // So, indices 0 and 4 should get 1.0 / 2 = 0.5 gradient. Others get 0.
    let expected_grad_vec = vec![0.5, 0.0, 0.0, 0.0, 0.5, 0.0];

    println!("Actual CPU Grad: {:?}", actual_grad_vec);
    println!("Expected CPU Grad: {:?}", expected_grad_vec);

    // Assert the gradients match
    assert_eq!(
        actual_grad_vec.len(),
        expected_grad_vec.len(),
        "Gradient vector lengths differ"
    );
    for (actual, expected) in actual_grad_vec.iter().zip(expected_grad_vec.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
    }

    Ok(())
}

#[test]
fn test_min_backward() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = CpuBackend::from_vec(data, &[2, 3]).unwrap();
    let grad_output = CpuBackend::ones(&[1]).unwrap(); // For global min

    // Create Op
    let op = Op {
        op_type: OpType::Min(None),
        inputs: vec![Tensor::new(x.clone(), false)],
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };

    let grad_input = CpuBackend::min_backward(&op, &grad_output).unwrap();
    let grad_data = CpuBackend::copy_to_host(&grad_input).unwrap();

    // Only the minimum element (1.0) should receive gradient
    assert_eq!(grad_data, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    // Test with axis reduction
    let grad_output = CpuBackend::ones(&[3]).unwrap();
    let op = Op {
        op_type: OpType::Min(Some(0)),
        inputs: vec![Tensor::new(x.clone(), false)],
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };

    let grad_input = CpuBackend::min_backward(&op, &grad_output).unwrap();
    let grad_data = CpuBackend::copy_to_host(&grad_input).unwrap();
    assert_eq!(grad_data, vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_prod_backward() {
    let data = vec![2.0, 3.0, 4.0];
    let x = CpuBackend::from_vec(data, &[3]).unwrap();
    let grad_output = CpuBackend::ones(&[1]).unwrap();

    let op = Op {
        op_type: OpType::Prod(None),
        inputs: vec![Tensor::new(x.clone(), false)],
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };

    let grad_input = CpuBackend::prod_backward(&op, &grad_output).unwrap();
    let grad_data = CpuBackend::copy_to_host(&grad_input).unwrap();

    // Each element should receive gradient = product / element
    assert!((grad_data[0] - 12.0).abs() < 1e-5); // 3 * 4
    assert!((grad_data[1] - 8.0).abs() < 1e-5); // 2 * 4
    assert!((grad_data[2] - 6.0).abs() < 1e-5); // 2 * 3
}

#[test]
fn test_logsumexp_backward() {
    let data = vec![1.0, 2.0, 3.0];
    let x = CpuBackend::from_vec(data, &[3]).unwrap();
    let grad_output = CpuBackend::ones(&[1]).unwrap();

    let op = Op {
        op_type: OpType::LogSumExp(None),
        inputs: vec![Tensor::new(x.clone(), false)],
        backward_fn: Rc::new(|_, _| panic!("Should not be called")),
        cached_outputs: None,
    };

    let grad_input = CpuBackend::logsumexp_backward(&op, &grad_output).unwrap();
    let grad_data = CpuBackend::copy_to_host(&grad_input).unwrap();

    // Gradient should be softmax of input
    let sum_exp = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
    let expected = [
        1.0f32.exp() / sum_exp,
        2.0f32.exp() / sum_exp,
        3.0f32.exp() / sum_exp,
    ];

    for (g, e) in grad_data.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5);
    }
}
