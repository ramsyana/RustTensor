// tests/ops_cpu_tests.rs
use rust_tensor_lib::{backend::cpu::CpuBackend, error::Error, ops, Backend, CpuTensor, Tensor};

// Helper to create CpuTensor quickly
fn cpu_tensor(data: Vec<f32>, shape: &[usize]) -> CpuTensor {
    Tensor::<CpuBackend>::from_vec(data, shape, false).unwrap()
}
fn cpu_tensor_req_grad(data: Vec<f32>, shape: &[usize]) -> CpuTensor {
    Tensor::<CpuBackend>::from_vec(data, shape, true).unwrap()
}

// --- Forward Pass Tests ---
// Added sigmoid forward and backward tests

#[test]
fn test_mse_loss_op() {
    use rust_tensor_lib::Reduction;
    let preds = cpu_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let targets = cpu_tensor(vec![1.5, 2.5, 3.5], &[3]);

    // Elementwise squared differences: [0.25, 0.25, 0.25]
    let expected = [0.25, 0.25, 0.25];

    // None reduction
    let loss_none = ops::mse_loss(&preds, &targets, Reduction::None).unwrap();
    assert_eq!(loss_none.shape(), &[3]);
    for (a, b) in loss_none.data().as_ref().iter().zip(expected.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "mse_loss None reduction: {} vs {}",
            a,
            b
        );
    }

    // Sum reduction
    let loss_sum = ops::mse_loss(&preds, &targets, Reduction::Sum).unwrap();
    assert_eq!(loss_sum.shape(), &[] as &[usize]); // Scalar
    assert!(
        (loss_sum.data().as_ref()[0] - 0.75).abs() < 1e-6,
        "mse_loss Sum reduction"
    );

    // Mean reduction
    let loss_mean = ops::mse_loss(&preds, &targets, Reduction::Mean).unwrap();
    assert_eq!(loss_mean.shape(), &[] as &[usize]); // Scalar
    assert!(
        (loss_mean.data().as_ref()[0] - 0.25).abs() < 1e-6,
        "mse_loss Mean reduction"
    );
}

#[test]
fn test_matmul_op() {
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = cpu_tensor(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let result = ops::matmul(&a, &b).unwrap();
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_eq!(result.data().as_ref(), expected.as_slice());
    assert_eq!(result.shape(), vec![2, 2]);
    assert!(!result.requires_grad());
}

#[test]
fn test_mul_op() {
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = cpu_tensor(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let result = ops::mul(&a, &b).unwrap();
    let expected = vec![5.0, 12.0, 21.0, 32.0];
    assert_eq!(result.data().as_ref(), expected.as_slice());
    assert_eq!(result.shape(), vec![2, 2]);
}

#[test]
fn test_add_op() {
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = cpu_tensor(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);

    let result = ops::add(&a, &b).unwrap();
    let expected = vec![6.0, 8.0, 10.0, 12.0];
    assert_eq!(result.data().as_ref(), expected.as_slice());
    assert_eq!(result.shape(), vec![2, 2]);
}

#[test]
fn test_mean_op() {
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // Test global mean (ops::mean calls B::mean with axis=None)
    let result = ops::mean(&a, None).unwrap();
    let expected = [2.5]; // Mean returns scalar tensor
    assert_eq!(result.shape(), Vec::<usize>::new()); // Scalar shape []
    assert!((result.data().as_ref()[0] - expected[0]).abs() < 1e-6);

    // Test requires_grad propagation
    let a_grad = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mean_grad = ops::mean(&a_grad, None).unwrap();
    assert!(mean_grad.requires_grad());
}

#[test]
fn test_relu_op() {
    let a = cpu_tensor(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2]);
    let result = ops::relu(&a).unwrap();
    let expected = vec![0.0, 0.0, 1.0, 2.0];
    assert_eq!(result.data().as_ref(), expected.as_slice());
    assert_eq!(result.shape(), vec![2, 2]);
}

#[test]
fn test_sigmoid_op() {
    let a = cpu_tensor(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2]);
    let result = ops::sigmoid(&a).unwrap();

    let result_data = result.data();
    let expected = [0.26894143f32, 0.5f32, 0.7310586f32, 0.8807971f32];
    for (actual, expected_val) in result_data.as_ref().iter().zip(expected.iter()) {
        assert!(
            (actual - expected_val).abs() < 1e-6,
            "sigmoid output mismatch: {} vs {}",
            actual,
            expected_val
        );
    }
    assert_eq!(result.shape(), vec![2, 2]);
}

#[test]
fn test_sqrt_op() {
    let a = cpu_tensor(vec![1.0, 4.0, 9.0, 16.0], &[2, 2]);
    let result = ops::sqrt(&a).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(result.data().as_ref(), expected.as_slice());
    assert_eq!(result.shape(), vec![2, 2]);

    // Check that negative input returns an error
    let neg = cpu_tensor(vec![1.0, -1.0], &[2]);
    let err = ops::sqrt(&neg);
    assert!(err.is_err());
}

#[test]
fn test_sigmoid_backward() {
    // Test backward pass for sigmoid using analytical gradients
    let a = cpu_tensor_req_grad(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2]);
    let result = ops::sigmoid(&a).unwrap();
    let loss = ops::mean(&result, None).unwrap();
    loss.backward().unwrap();
    let grad = a.grad().expect("Gradient missing for input");
    let result_data = result.data();
    let n = result.size() as f32;
    // Analytic gradient: sigmoid(x) * (1 - sigmoid(x)) / n
    let expected = result_data
        .as_ref()
        .iter()
        .map(|&s| s * (1.0 - s) / n)
        .collect::<Vec<_>>();
    for (actual, expected_val) in grad.as_ref().iter().zip(expected.iter()) {
        assert!(
            (actual - expected_val).abs() < 1e-6,
            "sigmoid backward mismatch: {} vs {}",
            actual,
            expected_val
        );
    }
    assert_eq!(grad.shape(), &[2, 2]);
}

#[test]
fn test_log_softmax_op() {
    let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let result = ops::log_softmax(&a, 1).unwrap();
    let result_data = result.data();
    let result_slice = result_data.as_ref();

    let expected = [-1.313_262, -0.313_262, -1.313_262, -0.313_262];

    assert_eq!(result.shape(), vec![2, 2]);
    for (actual, expected_val) in result_slice.iter().zip(expected.iter()) {
        assert!((actual - expected_val).abs() < 1e-6);
    }
}

#[test]
fn test_tanh_op() {
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let shape = [5];
    let input = cpu_tensor(data.clone(), &shape);
    let output = ops::tanh(&input).unwrap();
    let expected: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
    for (o, e) in output.data().as_ref().iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-6, "tanh -> {} (expected {})", o, e);
    }
}

#[test]
fn test_softplus_op() {
    let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let shape = [5];
    let input = cpu_tensor(data.clone(), &shape);
    let output = ops::softplus(&input).unwrap();
    // Calculate expected values: log(1 + exp(x))
    let expected: Vec<f32> = data.iter().map(|&x| (1.0 + x.exp()).ln()).collect();
    for (o, e) in output.data().as_ref().iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-6, "softplus -> {} (expected {})", o, e);
    }
}

#[test]
fn test_square_op() {
    let a = cpu_tensor(vec![-2.0, 0.0, 3.0], &[3]);
    let result = ops::square(&a).unwrap();
    let expected = vec![4.0, 0.0, 9.0];
    assert_eq!(result.data().as_ref(), expected.as_slice());
    assert_eq!(result.shape(), vec![3]);
    assert!(!result.requires_grad()); // Check grad requirement propagation

    let a_grad = cpu_tensor_req_grad(vec![-2.0, 3.0], &[2]);
    let result_grad = ops::square(&a_grad).unwrap();
    assert!(result_grad.requires_grad());
}

#[test]
fn test_elu_op() {
    // Test ELU with alpha = 1.0
    let alpha = 1.0;
    let a = cpu_tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let result = ops::elu(&a, alpha).unwrap();

    // Calculate expected values
    let expected: Vec<f32> = vec![
        alpha * ((-2.0f32).exp() - 1.0), // -0.8646647
        alpha * ((-1.0f32).exp() - 1.0), // -0.63212055
        0.0,                             // 0.0
        1.0,                             // 1.0
        2.0,                             // 2.0
    ];

    // Check result values
    for (i, expected_val) in expected.iter().enumerate() {
        assert!(
            (result.data().as_ref()[i] - expected_val).abs() < 1e-6,
            "elu({}) = {} but expected {}",
            a.data().as_ref()[i],
            result.data().as_ref()[i],
            expected_val
        );
    }

    assert_eq!(result.shape(), vec![5]);
    assert!(!result.requires_grad()); // Check grad requirement propagation

    // Test with requires_grad
    let a_grad = cpu_tensor_req_grad(vec![-2.0, 0.0, 2.0], &[3]);
    let result_grad = ops::elu(&a_grad, alpha).unwrap();
    assert!(result_grad.requires_grad());
}

#[test]
fn test_elu_backward() {
    // Test backward pass for ELU using analytical gradients
    let alpha = 1.0;
    let a = cpu_tensor_req_grad(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let result = ops::elu(&a, alpha).unwrap();
    let loss = ops::mean(&result, None).unwrap();
    loss.backward().unwrap();

    let grad = a.grad().expect("Gradient missing for input");
    let n = result.size() as f32;

    // Analytic gradient: (x < 0 ? alpha * exp(x) : 1.0) / n
    let expected = a
        .data()
        .as_ref()
        .iter()
        .map(|&x| {
            let derivative = if x < 0.0 { alpha * x.exp() } else { 1.0 };
            derivative / n
        })
        .collect::<Vec<_>>();

    for (i, (actual, expected_val)) in grad.as_ref().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected_val).abs() < 1e-6,
            "elu backward at index {}: {} vs {}",
            i,
            actual,
            expected_val
        );
    }
    assert_eq!(grad.shape(), &[5]);
}

// --- Autograd Backward Pass Tests ---

#[test]
fn test_add_backward() {
    let a = cpu_tensor_req_grad(vec![1.0, 2.0], &[2]);
    let b = cpu_tensor_req_grad(vec![3.0, 4.0], &[2]);
    let c = ops::add(&a, &b).unwrap();
    let loss = ops::mean(&c, None).unwrap(); // Create scalar loss

    loss.backward().unwrap(); // Backpropagate from loss

    let grad_a_ref = a.grad().expect("Grad A missing");
    let grad_b_ref = b.grad().expect("Grad B missing");
    let n = c.size() as f32; // Number of elements in c (which was averaged)

    // Expected grad = dLoss/dInput = dLoss/dMean * dMean/dC * dC/dInput
    // dLoss/dMean = 1.0 (scalar loss)
    // dMean/dC = 1/n for each element of c
    // dC/dA = 1, dC/dB = 1
    // So, dLoss/dA = 1 * (1/n) * 1 = 1/n
    let expected_grad = [1.0 / n, 1.0 / n];

    assert_eq!(grad_a_ref.shape(), &[2]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad.iter())
        .for_each(|(act, exp)| assert!((act - exp).abs() < 1e-6));

    assert_eq!(grad_b_ref.shape(), &[2]);
    grad_b_ref
        .as_ref()
        .iter()
        .zip(expected_grad.iter())
        .for_each(|(act, exp)| assert!((act - exp).abs() < 1e-6));
}

#[test]
fn test_mul_backward() {
    let a_val = vec![1.0, 2.0];
    let b_val = vec![3.0, 4.0];
    let a = cpu_tensor_req_grad(a_val.clone(), &[2]);
    let b = cpu_tensor_req_grad(b_val.clone(), &[2]);
    let c = ops::mul(&a, &b).unwrap();
    let loss = ops::mean(&c, None).unwrap(); // Scalar loss

    loss.backward().unwrap(); // Backpropagate

    let grad_a_ref = a.grad().expect("Grad A missing");
    let grad_b_ref = b.grad().expect("Grad B missing");
    let n = c.size() as f32;

    // Expected: dLoss/dA = (1/n) * B
    let expected_grad_a = b_val.iter().map(|&bi| bi / n).collect::<Vec<_>>();
    assert_eq!(grad_a_ref.shape(), &[2]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad_a.iter())
        .for_each(|(act, exp)| assert!((act - exp).abs() < 1e-6));

    // Expected: dLoss/dB = (1/n) * A
    let expected_grad_b = a_val.iter().map(|&ai| ai / n).collect::<Vec<_>>();
    assert_eq!(grad_b_ref.shape(), &[2]);
    grad_b_ref
        .as_ref()
        .iter()
        .zip(expected_grad_b.iter())
        .for_each(|(act, exp)| assert!((act - exp).abs() < 1e-6));
}

#[test]
fn test_matmul_backward() {
    let a = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = cpu_tensor_req_grad(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = ops::matmul(&a, &b).unwrap();
    let loss = ops::mean(&c, None).unwrap(); // Scalar loss

    loss.backward().unwrap(); // Backpropagate

    let grad_a_ref = a.grad().expect("Grad A missing");
    let grad_b_ref = b.grad().expect("Grad B missing");

    // Expected gradients dA and dB (as calculated before)
    let expected_grad_a = [2.75, 3.75, 2.75, 3.75];
    let expected_grad_b = [1.0, 1.0, 1.5, 1.5];

    assert_eq!(grad_a_ref.shape(), &[2, 2]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad_a.iter())
        .for_each(|(act, exp)| assert!((act - exp).abs() < 1e-6));

    assert_eq!(grad_b_ref.shape(), &[2, 2]);
    grad_b_ref
        .as_ref()
        .iter()
        .zip(expected_grad_b.iter())
        .for_each(|(act, exp)| assert!((act - exp).abs() < 1e-6));
}

#[test]
fn test_mean_backward() {
    let a = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let loss = ops::mean(&a, None).unwrap(); // Result is already scalar loss

    loss.backward().unwrap(); // Backpropagate

    let grad_a_ref = a.grad().expect("Grad A missing");
    let n = a.size() as f32;

    // Expected grad = 1.0 / n for each element
    let expected_grad = [1.0 / n, 1.0 / n, 1.0 / n, 1.0 / n];

    assert_eq!(grad_a_ref.shape(), &[2, 2]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad.iter())
        .for_each(|(act, exp)| assert!((act - exp).abs() < 1e-6));
}

#[test]
fn test_relu_backward() {
    let a = cpu_tensor_req_grad(vec![-1.0, 0.0, 1.0, 2.0], &[2, 2]);
    let result = ops::relu(&a).unwrap();
    let loss = ops::mean(&result, None).unwrap(); // Scalar loss

    loss.backward().unwrap(); // Backpropagate

    let grad_a_ref = a.grad().expect("Grad A missing");
    let n = a.size() as f32;

    // Expected: dLoss/dA = (1/n) * (1 if A > 0 else 0)
    let expected_grad = [0.0, 0.0, 1.0 / n, 1.0 / n];

    assert_eq!(grad_a_ref.shape(), &[2, 2]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad.iter())
        .for_each(|(act, exp)| assert!((act - exp).abs() < 1e-6));
}

#[test]
fn test_log_softmax_backward() {
    let a = cpu_tensor_req_grad(vec![1.0, 2.0], &[2]);
    let b = ops::log_softmax(&a, 0).unwrap();
    let loss = ops::mean(&b, None).unwrap(); // Scalar loss

    loss.backward().unwrap(); // Backpropagate

    // Expected: dx = dLoss/dB * dB/dA = [0.5, 0.5] * [dB/dA_0, dB/dA_1]
    // From previous calculation: dx = [0.231049, -0.231049]
    let expected_grad_a = [0.231049, -0.231049];

    let grad_a_ref = a.grad().expect("Grad A missing");
    assert_eq!(grad_a_ref.shape(), &[2]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad_a.iter())
        .for_each(|(act, exp)| assert!((act - exp).abs() < 1e-5));
}

#[test]
fn test_grad_accumulation() {
    let a = cpu_tensor_req_grad(vec![2.0], &[]); // Scalar tensor
    let b = ops::mul(&a, &a).unwrap(); // b = a*a = 4.0
    let c = ops::mul(&a, &a).unwrap(); // c = a*a = 4.0, separate node
    let loss = ops::add(&b, &c).unwrap(); // loss = b + c = 8.0

    loss.backward().unwrap(); // dLoss/da = dLoss/db * db/da + dLoss/dc * dc/da = 1*(2a) + 1*(2a) = 4a

    let grad_a_ref = a.grad().expect("Grad A missing");
    assert_eq!(grad_a_ref.shape(), &[] as &[usize]);
    assert!((grad_a_ref.as_ref()[0] - (4.0 * 2.0)).abs() < 1e-6); // 4a = 8.0

    // Check intermediate grads
    let grad_b_ref = b.grad().expect("Grad B missing");
    assert_eq!(grad_b_ref.shape(), &[] as &[usize]);
    assert!((grad_b_ref.as_ref()[0] - 1.0).abs() < 1e-6); // dLoss/db = 1

    let grad_c_ref = c.grad().expect("Grad C missing");
    assert_eq!(grad_c_ref.shape(), &[] as &[usize]);
    assert!((grad_c_ref.as_ref()[0] - 1.0).abs() < 1e-6); // dLoss/dc = 1
}

#[test]
fn test_broadcast_add_backward() {
    // Test scalar-vector broadcasting
    let a = cpu_tensor_req_grad(vec![2.0], &[]); // scalar
    let b = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0], &[3]); // vector
    let c = ops::add(&a, &b).unwrap();
    let loss = ops::mean(&c, None).unwrap();

    loss.backward().unwrap();

    let grad_a_ref = a.grad().expect("Grad A missing");
    let grad_b_ref = b.grad().expect("Grad B missing");
    let n = c.size() as f32;

    println!(
        "\n[DEBUG] grad_a_ref: {:?}, shape: {:?}",
        grad_a_ref.as_ref(),
        grad_a_ref.shape()
    );
    println!(
        "[DEBUG] grad_b_ref: {:?}, shape: {:?}",
        grad_b_ref.as_ref(),
        grad_b_ref.shape()
    );
    println!("[DEBUG] expected_grad_a: {:?}", [1.0]); // correct: sum of 1/3 + 1/3 + 1/3 = 1.0
    println!("[DEBUG] expected_grad_b: {:?}", [1.0 / n, 1.0 / n, 1.0 / n]);

    // For broadcasting add:
    // dA = sum(dout) since scalar is broadcasted
    // dB = dout
    let expected_grad_a = [1.0]; // correct: sum of 1/3 + 1/3 + 1/3 = 1.0
    let expected_grad_b = [1.0 / n, 1.0 / n, 1.0 / n];

    assert_eq!(grad_a_ref.shape(), &[] as &[usize]);
    assert_eq!(grad_b_ref.shape(), &[3]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad_a.iter())
        .for_each(|(act, exp)| {
            println!("[DEBUG] grad_a: actual = {}, expected = {}", act, exp);
            assert!((act - exp).abs() < 1e-6)
        });
    grad_b_ref
        .as_ref()
        .iter()
        .zip(expected_grad_b.iter())
        .for_each(|(act, exp)| {
            println!("[DEBUG] grad_b: actual = {}, expected = {}", act, exp);
            assert!((act - exp).abs() < 1e-6)
        });
}

#[test]
fn test_broadcast_mul_backward() {
    // Test scalar-vector broadcasting
    let a = cpu_tensor_req_grad(vec![2.0], &[]); // scalar
    let b = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0], &[3]); // vector
    let c = ops::mul(&a, &b).unwrap();
    let loss = ops::mean(&c, None).unwrap();

    loss.backward().unwrap();

    let grad_a_ref = a.grad().expect("Grad A missing");
    let grad_b_ref = b.grad().expect("Grad B missing");
    let n = c.size() as f32;

    println!(
        "\n[DEBUG] grad_a_ref: {:?}, shape: {:?}",
        grad_a_ref.as_ref(),
        grad_a_ref.shape()
    );
    println!(
        "[DEBUG] grad_b_ref: {:?}, shape: {:?}",
        grad_b_ref.as_ref(),
        grad_b_ref.shape()
    );
    println!("[DEBUG] expected_grad_a: {:?}", [(1.0 + 2.0 + 3.0) / n]); // sum(B) / n
    println!("[DEBUG] expected_grad_b: {:?}", [2.0 / n, 2.0 / n, 2.0 / n]); // A/n

    // For broadcasting multiply:
    // dA = sum(dout * B)
    // dB = dout * A
    let expected_grad_a = [(1.0 + 2.0 + 3.0) / n]; // sum(B) / n
    let expected_grad_b = [2.0 / n, 2.0 / n, 2.0 / n]; // A/n

    assert_eq!(grad_a_ref.shape(), &[] as &[usize]);
    assert_eq!(grad_b_ref.shape(), &[3]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad_a.iter())
        .for_each(|(act, exp)| {
            println!("[DEBUG] grad_a: actual = {}, expected = {}", act, exp);
            assert!((act - exp).abs() < 1e-6)
        });
    grad_b_ref
        .as_ref()
        .iter()
        .zip(expected_grad_b.iter())
        .for_each(|(act, exp)| {
            println!("[DEBUG] grad_b: actual = {}, expected = {}", act, exp);
            assert!((act - exp).abs() < 1e-6)
        });
}

#[test]
fn test_reduction_backward() {
    // Test sum reduction with axis
    let a = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let sum_col = ops::sum(&a, Some(0)).unwrap(); // sum along rows
    let loss = ops::mean(&sum_col, None).unwrap();

    loss.backward().unwrap();

    let grad_a_ref = a.grad().expect("Grad A missing");
    let n = sum_col.size() as f32;

    println!(
        "\n[DEBUG] grad_a_ref: {:?}, shape: {:?}",
        grad_a_ref.as_ref(),
        grad_a_ref.shape()
    );
    println!(
        "[DEBUG] expected_grad: {:?}",
        [1.0 / n, 1.0 / n, 1.0 / n, 1.0 / n]
    );

    // For sum reduction:
    // dA = replicate gradient uniformly
    let expected_grad = [1.0 / n, 1.0 / n, 1.0 / n, 1.0 / n];

    assert_eq!(grad_a_ref.shape(), &[2, 2]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad.iter())
        .for_each(|(act, exp)| {
            println!("[DEBUG] grad_a: actual = {}, expected = {}", act, exp);
            assert!((act - exp).abs() < 1e-6)
        });

    // Test mean reduction with axis
    let a = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let mean_row = ops::mean(&a, Some(1)).unwrap(); // mean along columns
    let loss = ops::mean(&mean_row, None).unwrap();

    loss.backward().unwrap();

    let grad_a_ref = a.grad().expect("Grad A missing");
    let n = mean_row.size() as f32;
    let m = 3.0; // size of reduction dimension

    println!(
        "\n[DEBUG] grad_a_ref: {:?}, shape: {:?}",
        grad_a_ref.as_ref(),
        grad_a_ref.shape()
    );
    println!(
        "[DEBUG] expected_grad: {:?}",
        [
            1.0 / (n * m),
            1.0 / (n * m),
            1.0 / (n * m),
            1.0 / (n * m),
            1.0 / (n * m),
            1.0 / (n * m),
        ]
    );

    // For mean reduction:
    // dA = gradient / reduction_size
    let expected_grad = [
        1.0 / (n * m),
        1.0 / (n * m),
        1.0 / (n * m),
        1.0 / (n * m),
        1.0 / (n * m),
        1.0 / (n * m),
    ];

    assert_eq!(grad_a_ref.shape(), &[2, 3]);
    grad_a_ref
        .as_ref()
        .iter()
        .zip(expected_grad.iter())
        .for_each(|(act, exp)| {
            println!("[DEBUG] grad_a: actual = {}, expected = {}", act, exp);
            assert!((act - exp).abs() < 1e-6)
        });
}

#[test]
fn test_numerical_stability() {
    // Test log_softmax with large numbers
    let a = cpu_tensor_req_grad(vec![1000.0, 1000.1, 1000.2], &[3]);
    let b = ops::log_softmax(&a, 0).unwrap();
    let loss = ops::mean(&b, None).unwrap();

    loss.backward().unwrap();

    let grad_a_ref = a.grad().expect("Grad A missing");
    assert_eq!(grad_a_ref.shape(), &[3]);

    // Check that gradients are finite and sum to ~0
    let grad_sum: f32 = grad_a_ref.as_ref().iter().sum();
    assert!(
        grad_sum.abs() < 1e-6,
        "log_softmax gradients should sum to 0"
    );
    grad_a_ref.as_ref().iter().for_each(|&x| {
        assert!(!x.is_nan() && !x.is_infinite(), "gradient should be finite");
    });

    // Test exp/ln stability with small/large numbers
    let small = cpu_tensor_req_grad(vec![1e-30], &[]);
    let large = cpu_tensor_req_grad(vec![1e30], &[]);

    // Test ln with small number
    let ln_small = ops::ln(&small).unwrap();
    let loss = ops::mean(&ln_small, None).unwrap();
    loss.backward().unwrap();
    let grad_small = small.grad().expect("Grad small missing");
    assert!(
        !grad_small.as_ref()[0].is_nan() && !grad_small.as_ref()[0].is_infinite(),
        "ln gradient should be finite for small input"
    );

    // Test exp with large number
    let exp_large = ops::exp(&large).unwrap();
    let loss = ops::mean(&exp_large, None).unwrap();
    loss.backward().unwrap();
    let grad_large_opt = large.grad();
    assert!(
        grad_large_opt.is_some(),
        "Gradient for large exp input should exist"
    );
    let grad_large_ref = grad_large_opt.unwrap();
    let grad_val = grad_large_ref.as_ref()[0];

    // Check for positive infinity, as exp(large) * 1/n is still infinite
    assert!(!grad_val.is_nan(), "exp gradient should not be NaN");
    assert!(
        grad_val.is_infinite() && grad_val.is_sign_positive(),
        "exp gradient should be positive infinity for large input, got {}",
        grad_val
    );
}

#[test]
fn test_cpu_binary_cross_entropy_with_logits() {
    use approx::assert_abs_diff_eq;
    use rust_tensor_lib::{ops, CpuBackend, Reduction, Tensor};

    let logits_data = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
    let targets_data = vec![0.0, 1.0, 0.0, 1.0, 1.0]; // Mix of 0s and 1s
    let shape = &[logits_data.len()];

    let logits = Tensor::<CpuBackend>::from_vec(logits_data.clone(), shape, false).unwrap();
    let targets = Tensor::<CpuBackend>::from_vec(targets_data.clone(), shape, false).unwrap();

    // Calculate expected values manually using stable formula:
    // loss = max(x, 0) - x*z + log(1 + exp(-abs(x)))
    let expected_elementwise: Vec<f32> = logits_data
        .iter()
        .zip(targets_data.iter())
        .map(|(&x, &z)| x.max(0.0) - x * z + (1.0 + (-x.abs()).exp()).ln())
        .collect();
    let expected_mean: f32 =
        expected_elementwise.iter().sum::<f32>() / expected_elementwise.len() as f32;
    let expected_sum: f32 = expected_elementwise.iter().sum();

    // Test Reduction::None
    let loss_none =
        ops::binary_cross_entropy_with_logits(&logits, &targets, Reduction::None).unwrap();
    assert_eq!(loss_none.shape(), shape);
    let loss_none_data = loss_none.data();
    for (actual, expected) in loss_none_data
        .as_ref()
        .iter()
        .zip(expected_elementwise.iter())
    {
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-6);
    }

    // Test Reduction::Mean
    let loss_mean =
        ops::binary_cross_entropy_with_logits(&logits, &targets, Reduction::Mean).unwrap();
    assert_eq!(loss_mean.shape(), &[] as &[usize]); // Scalar shape
    assert_abs_diff_eq!(loss_mean.data().as_ref()[0], expected_mean, epsilon = 1e-6);

    // Test Reduction::Sum
    let loss_sum =
        ops::binary_cross_entropy_with_logits(&logits, &targets, Reduction::Sum).unwrap();
    assert_eq!(loss_sum.shape(), &[] as &[usize]); // Scalar shape
    assert_abs_diff_eq!(loss_sum.data().as_ref()[0], expected_sum, epsilon = 1e-6);
}

#[test]
fn test_min_reduction() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = CpuBackend::from_vec(data, &[2, 3]).unwrap();

    // Test global min
    let result = CpuBackend::min(&x, None).unwrap();
    assert_eq!(
        CpuBackend::shape(&result),
        &[] as &[usize],
        "Global min shape should be []"
    );
    assert_eq!(
        CpuBackend::copy_to_host(&result).unwrap()[0],
        1.0,
        "Global min value mismatch"
    );

    // Test min along axis 0
    let result = CpuBackend::min(&x, Some(0)).unwrap();
    assert_eq!(CpuBackend::shape(&result), &[3]);
    assert_eq!(
        CpuBackend::copy_to_host(&result).unwrap(),
        vec![1.0, 2.0, 3.0]
    );

    // Test min along axis 1
    let result = CpuBackend::min(&x, Some(1)).unwrap();
    assert_eq!(CpuBackend::shape(&result), &[2]);
    assert_eq!(CpuBackend::copy_to_host(&result).unwrap(), vec![1.0, 4.0]);
}

#[test]
fn test_prod_reduction() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = CpuBackend::from_vec(data, &[2, 3]).unwrap();

    // Test global product
    let result = CpuBackend::prod(&x, None).unwrap();
    assert_eq!(
        CpuBackend::shape(&result),
        &[] as &[usize],
        "Global prod shape should be []"
    );
    assert_eq!(
        CpuBackend::copy_to_host(&result).unwrap()[0],
        720.0,
        "Global prod value mismatch"
    ); // 1*2*3*4*5*6

    // Test product along axis 0
    let result = CpuBackend::prod(&x, Some(0)).unwrap();
    assert_eq!(CpuBackend::shape(&result), &[3]);
    assert_eq!(
        CpuBackend::copy_to_host(&result).unwrap(),
        vec![4.0, 10.0, 18.0]
    );

    // Test product along axis 1
    let result = CpuBackend::prod(&x, Some(1)).unwrap();
    assert_eq!(CpuBackend::shape(&result), &[2]);
    assert_eq!(CpuBackend::copy_to_host(&result).unwrap(), vec![6.0, 120.0]);
}

#[test]
fn test_logsumexp_reduction() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = CpuBackend::from_vec(data, &[2, 3]).unwrap();

    // Test global logsumexp
    let result = CpuBackend::logsumexp(&x, None).unwrap();
    assert_eq!(
        CpuBackend::shape(&result),
        &[] as &[usize],
        "Global logsumexp shape should be []"
    );
    let expected =
        (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp() + 4.0f32.exp() + 5.0f32.exp() + 6.0f32.exp())
            .ln();
    assert!(
        (CpuBackend::copy_to_host(&result).unwrap()[0] - expected).abs() < 1e-5,
        "Global logsumexp value mismatch"
    );

    // Test logsumexp along axis 0
    let result = CpuBackend::logsumexp(&x, Some(0)).unwrap();
    assert_eq!(CpuBackend::shape(&result), &[3]);
    let expected = [
        (1.0f32.exp() + 4.0f32.exp()).ln(),
        (2.0f32.exp() + 5.0f32.exp()).ln(),
        (3.0f32.exp() + 6.0f32.exp()).ln(),
    ];
    let actual = CpuBackend::copy_to_host(&result).unwrap();
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-5);
    }

    // Test logsumexp along axis 1
    let result = CpuBackend::logsumexp(&x, Some(1)).unwrap();
    assert_eq!(CpuBackend::shape(&result), &[2]);
    let expected = [
        (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln(),
        (4.0f32.exp() + 5.0f32.exp() + 6.0f32.exp()).ln(),
    ];
    let actual = CpuBackend::copy_to_host(&result).unwrap();
    for (a, e) in actual.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-5);
    }
}

#[test]
fn test_powf() {
    use rust_tensor_lib::backend::CpuTensor;
    use rust_tensor_lib::ops;

    // Test case 1: Base > 0, exponent > 0
    let base1 = CpuTensor::from_vec(vec![2.0, 3.0, 4.0], &[3], false).unwrap();
    let exp1 = CpuTensor::from_vec(vec![2.0, 3.0, 0.5], &[3], false).unwrap();
    let result1 = ops::powf(&base1, &exp1).unwrap();

    let expected1 = [
        2.0f32.powf(2.0), // 4.0
        3.0f32.powf(3.0), // 27.0
        4.0f32.powf(0.5), // 2.0
    ];

    assert_eq!(result1.shape(), &[3]);
    for (idx, &expected) in expected1.iter().enumerate() {
        let actual = result1.data().as_ref()[idx];
        assert!(
            (actual - expected).abs() < 1e-5,
            "Case 1 at {}: got {} expected {}",
            idx,
            actual,
            expected
        );
    }

    // Test case 2: scalar exponent using powf_scalar
    let base2 = CpuTensor::from_vec(vec![2.0, 3.0, 4.0], &[3], false).unwrap();
    let exp2 = 2.0;
    let result2 = ops::powf_scalar(&base2, exp2).unwrap();

    let expected2 = [
        2.0f32.powf(2.0), // 4.0
        3.0f32.powf(2.0), // 9.0
        4.0f32.powf(2.0), // 16.0
    ];

    assert_eq!(result2.shape(), &[3]);
    for (idx, &expected) in expected2.iter().enumerate() {
        let actual = result2.data().as_ref()[idx];
        assert!(
            (actual - expected).abs() < 1e-5,
            "Case 2 at {}: got {} expected {}",
            idx,
            actual,
            expected
        );
    }

    // Test case 3: Broadcasting - scalar base, vector exponent
    let base3 = CpuTensor::from_vec(vec![2.0], &[1], false).unwrap();
    let exp3 = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], false).unwrap();
    let result3 = ops::powf(&base3, &exp3).unwrap();

    let expected3 = [
        2.0f32.powf(1.0), // 2.0
        2.0f32.powf(2.0), // 4.0
        2.0f32.powf(3.0), // 8.0
    ];

    assert_eq!(result3.shape(), &[3]);
    for (idx, &expected) in expected3.iter().enumerate() {
        let actual = result3.data().as_ref()[idx];
        assert!(
            (actual - expected).abs() < 1e-5,
            "Case 3 at {}: got {} expected {}",
            idx,
            actual,
            expected
        );
    }

    // Test case 4: Broadcasting - vector base, scalar exponent
    let base4 = CpuTensor::from_vec(vec![1.0, 2.0, 4.0], &[3], false).unwrap();
    let exp4 = CpuTensor::from_vec(vec![0.5], &[1], false).unwrap();
    let result4 = ops::powf(&base4, &exp4).unwrap();

    let expected4 = [
        1.0f32.powf(0.5), // 1.0
        2.0f32.powf(0.5), // ~1.414
        4.0f32.powf(0.5), // 2.0
    ];

    assert_eq!(result4.shape(), &[3]);
    for (idx, &expected) in expected4.iter().enumerate() {
        let actual = result4.data().as_ref()[idx];
        assert!(
            (actual - expected).abs() < 1e-5,
            "Case 4 at {}: got {} expected {}",
            idx,
            actual,
            expected
        );
    }
}

#[test]
fn test_powf_gradient() {

    use rust_tensor_lib::backend::CpuTensor;
    use rust_tensor_lib::ops;

    // Test gradient computation for powf

    // Test case 1: both inputs require grad
    let base = CpuTensor::from_vec(vec![2.0, 3.0, 4.0], &[3], true).unwrap();
    let exp = CpuTensor::from_vec(vec![3.0, 2.0, 0.5], &[3], true).unwrap();

    println!(
        "[DEBUG test] Created base tensor: {:?} with shape {:?}",
        base.data().as_ref(),
        base.shape()
    );
    println!(
        "[DEBUG test] Created exp tensor: {:?} with shape {:?}",
        exp.data().as_ref(),
        exp.shape()
    );

    let result = ops::powf(&base, &exp).unwrap();
    println!(
        "[DEBUG test] powf result: {:?} with shape {:?}",
        result.data().as_ref(),
        result.shape()
    );

    let loss = ops::mean(&result, None).unwrap(); // Scalar loss
    println!(
        "[DEBUG test] mean loss: {:?} with shape {:?}",
        loss.data().as_ref(),
        loss.shape()
    );

    println!("[DEBUG test] Starting backward pass");
    loss.backward().unwrap();
    println!("[DEBUG test] Backward pass completed");

    // Get the actual gradients
    let a_grad = base.grad().unwrap().as_ref().to_vec();
    let b_grad = exp.grad().unwrap().as_ref().to_vec();

    // IMPORTANT: When using ops::mean(&result, None), the gradient is scaled by 1/N where N is the number of elements.
    // In this case, N=3, so all gradients are scaled by 1/3 compared to the mathematical formulation.
    // Therefore, we adjust our expected values accordingly:

    // Check gradients
    // dL/da = dL/dc * dc/da = (1/3) * 3 * 2^2 = 4.0
    // dL/da = (1/3) * 2 * 3^1 = 2.0
    println!("[DEBUG test] a_grad = {:?}", a_grad);
    println!(
        "[DEBUG test] Expected a_grad[0] = 4.0 (1/3 * 3 * 2^2), got {}",
        a_grad[0]
    );
    println!(
        "[DEBUG test] Expected a_grad[1] = 2.0 (1/3 * 2 * 3^1), got {}",
        a_grad[1]
    );

    // Adjusted expected values accounting for the 1/3 factor from mean reduction
    assert!(
        (a_grad[0] - 4.0).abs() < 1e-5,
        "a_grad[0] = {} should be 4.0",
        a_grad[0]
    );
    assert!(
        (a_grad[1] - 2.0).abs() < 1e-5,
        "a_grad[1] = {} should be 2.0",
        a_grad[1]
    );

    // dL/db = dL/dc * dc/db = (1/3) * a^b * ln(a)
    // For a=2, b=3: dL/db = (1/3) * 8 * ln(2) ≈ 1.85
    // For a=3, b=2: dL/db = (1/3) * 9 * ln(3) ≈ 3.30
    let expected_b_grad_0 = (8.0 * 2.0f32.ln()) / 3.0;
    let expected_b_grad_1 = (9.0 * 3.0f32.ln()) / 3.0;

    println!("[DEBUG test] b_grad = {:?}", b_grad);
    println!(
        "[DEBUG test] Expected b_grad[0] = {} (1/3 * 8 * ln(2)), got {}",
        expected_b_grad_0, b_grad[0]
    );
    println!(
        "[DEBUG test] Expected b_grad[1] = {} (1/3 * 9 * ln(3)), got {}",
        expected_b_grad_1, b_grad[1]
    );

    assert!(
        (b_grad[0] - expected_b_grad_0).abs() < 1e-5,
        "b_grad[0] = {} should be {}",
        b_grad[0],
        expected_b_grad_0
    );
    assert!(
        (b_grad[1] - expected_b_grad_1).abs() < 1e-5,
        "b_grad[1] = {} should be {}",
        b_grad[1],
        expected_b_grad_1
    );
}

#[test]
fn test_powf_scalar() {
    use rust_tensor_lib::ops;
    use rust_tensor_lib::backend::CpuTensor;

    // Forward: simple vector ^ scalar
    let base = CpuTensor::from_vec(vec![1.0, 2.0, 4.0, 9.0], &[4], false).unwrap();
    let exp = 0.5f32;
    let result = ops::powf_scalar(&base, exp).unwrap();
    let expected = vec![1.0f32.powf(0.5), 2.0f32.powf(0.5), 4.0f32.powf(0.5), 9.0f32.powf(0.5)];
    assert_eq!(result.shape(), &[4]);
    for (a, b) in result.data().as_ref().iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "powf_scalar forward: {} vs {}", a, b);
    }

    // Forward: matrix ^ scalar
    let base2 = CpuTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false).unwrap();
    let exp2 = 2.0f32;
    let result2 = ops::powf_scalar(&base2, exp2).unwrap();
    let expected2 = vec![1.0f32.powf(2.0), 2.0f32.powf(2.0), 3.0f32.powf(2.0), 4.0f32.powf(2.0)];
    assert_eq!(result2.shape(), &[2, 2]);
    for (a, b) in result2.data().as_ref().iter().zip(expected2.iter()) {
        assert!((a - b).abs() < 1e-6, "powf_scalar forward mat: {} vs {}", a, b);
    }

    // Edge case: exponent 0 (should yield 1)
    let base3 = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], false).unwrap();
    let result3 = ops::powf_scalar(&base3, 0.0).unwrap();
    assert_eq!(result3.data().as_ref(), &[1.0, 1.0, 1.0]);

    // Edge case: exponent 1 (should yield base)
    let result4 = ops::powf_scalar(&base3, 1.0).unwrap();
    assert_eq!(result4.data().as_ref(), base3.data().as_ref());

    // Edge case: negative base and non-integer exponent (should error or NaN)
    let base4 = CpuTensor::from_vec(vec![-4.0, -1.0, 0.0], &[3], false).unwrap();
    let result4 = ops::powf_scalar(&base4, 0.5);
    assert!(result4.is_ok(), "CPU powf_scalar with negative base returns Ok but should produce NaN");
    let tmp = result4.unwrap();
    let binding = tmp.data();
    let vals = binding.as_ref();
    assert!(vals[0].is_nan() && vals[1].is_nan(), "Negative base ^ 0.5 should be NaN");
    assert_eq!(vals[2], 0.0);

    // Gradient check: tensor ^ scalar
    let base_grad = CpuTensor::from_vec(vec![2.0, 3.0, 4.0], &[3], true).unwrap();
    let exp_grad = 3.0f32;
    let out = ops::powf_scalar(&base_grad, exp_grad).unwrap();
    let loss = ops::mean(&out, None).unwrap();
    loss.backward().unwrap();
    let grad = base_grad.grad().unwrap();
    // d/dx (x^c) = c*x^(c-1)
    let expected_grad = vec![exp_grad * 2.0f32.powf(exp_grad - 1.0) / 3.0,
                             exp_grad * 3.0f32.powf(exp_grad - 1.0) / 3.0,
                             exp_grad * 4.0f32.powf(exp_grad - 1.0) / 3.0];
    for (g, eg) in grad.as_ref().iter().zip(expected_grad.iter()) {
        assert!((g - eg).abs() < 1e-5, "powf_scalar grad: {} vs {}", g, eg);
    }
}


#[test]
fn test_maximum_forward_cpu() {
    // Case 1: Tensors of the same shape
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], false).unwrap();
    let b = Tensor::<CpuBackend>::from_vec(vec![4.0, 2.0, 1.0, 5.0], &[2, 2], false).unwrap();
    let c = ops::maximum(&a, &b).unwrap();
    assert_eq!(c.shape(), [2, 2]);
    let c_data = CpuBackend::copy_to_host(&c.data()).unwrap();
    assert_eq!(c_data, vec![4.0, 5.0, 3.0, 5.0]);

    // Case 2: Scalar and vector broadcasting
    let scalar = Tensor::<CpuBackend>::from_vec(vec![3.0], &[], false).unwrap();
    let vector = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 2.0], &[3], false).unwrap();
    let result = ops::maximum(&scalar, &vector).unwrap();
    assert_eq!(result.shape(), [3]);
    let result_data = CpuBackend::copy_to_host(&result.data()).unwrap();
    assert_eq!(result_data, vec![3.0, 5.0, 3.0]);

    // Case 3: Matrix and row vector broadcasting
    let matrix =
        Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let row = Tensor::<CpuBackend>::from_vec(vec![3.0, 1.0, 4.0], &[1, 3], false).unwrap();
    let result = ops::maximum(&matrix, &row).unwrap();
    assert_eq!(result.shape(), [2, 3]);
    let result_data = CpuBackend::copy_to_host(&result.data()).unwrap();
    assert_eq!(result_data, vec![3.0, 2.0, 4.0, 4.0, 5.0, 6.0]);

    // Case 4: Test with negative numbers
    let neg_a = Tensor::<CpuBackend>::from_vec(vec![-1.0, -5.0, -3.0], &[3], false).unwrap();
    let neg_b = Tensor::<CpuBackend>::from_vec(vec![-4.0, -2.0, -7.0], &[3], false).unwrap();
    let neg_result = ops::maximum(&neg_a, &neg_b).unwrap();
    assert_eq!(neg_result.shape(), [3]);
    let neg_result_data = CpuBackend::copy_to_host(&neg_result.data()).unwrap();
    assert_eq!(neg_result_data, vec![-1.0, -2.0, -3.0]);
}

#[test]
fn test_minimum_forward_cpu() {
    // Case 1: Tensors of the same shape
    let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 3.0, 2.0], &[2, 2], false).unwrap();
    let b = Tensor::<CpuBackend>::from_vec(vec![4.0, 2.0, 1.0, 5.0], &[2, 2], false).unwrap();
    let c = ops::minimum(&a, &b).unwrap();
    assert_eq!(c.shape(), [2, 2]);
    let c_data = CpuBackend::copy_to_host(&c.data()).unwrap();
    assert_eq!(c_data, vec![1.0, 2.0, 1.0, 2.0]);

    // Case 2: Scalar and vector broadcasting
    let scalar = Tensor::<CpuBackend>::from_vec(vec![3.0], &[], false).unwrap();
    let vector = Tensor::<CpuBackend>::from_vec(vec![1.0, 5.0, 2.0], &[3], false).unwrap();
    let result = ops::minimum(&scalar, &vector).unwrap();
    assert_eq!(result.shape(), [3]);
    let result_data = CpuBackend::copy_to_host(&result.data()).unwrap();
    assert_eq!(result_data, vec![1.0, 3.0, 2.0]);

    // Case 3: Matrix and row vector broadcasting
    let matrix =
        Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    let row = Tensor::<CpuBackend>::from_vec(vec![3.0, 1.0, 4.0], &[1, 3], false).unwrap();
    let result = ops::minimum(&matrix, &row).unwrap();
    assert_eq!(result.shape(), [2, 3]);
    let result_data = CpuBackend::copy_to_host(&result.data()).unwrap();
    assert_eq!(result_data, vec![1.0, 1.0, 3.0, 3.0, 1.0, 4.0]);

    // Case 4: Test with negative numbers
    let neg_a = Tensor::<CpuBackend>::from_vec(vec![-1.0, -5.0, -3.0], &[3], false).unwrap();
    let neg_b = Tensor::<CpuBackend>::from_vec(vec![-4.0, -2.0, -7.0], &[3], false).unwrap();
    let neg_result = ops::minimum(&neg_a, &neg_b).unwrap();
    assert_eq!(neg_result.shape(), [3]);
    let neg_result_data = CpuBackend::copy_to_host(&neg_result.data()).unwrap();
    assert_eq!(neg_result_data, vec![-4.0, -5.0, -7.0]);
}

#[test]
fn test_greater_op() {
    // Case 1: Same shape
    let a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let b1 = cpu_tensor(vec![2.0, 3.0, 2.0], &[3]);
    let result1 = ops::greater(&a1, &b1).unwrap();
    assert_eq!(result1.shape(), &[3]);
    assert_eq!(result1.data().as_ref(), vec![0.0, 1.0, 0.0]);

    // Case 2: Broadcasting scalar
    let a2 = cpu_tensor(vec![3.0], &[]);
    let b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let result2 = ops::greater(&a2, &b2).unwrap();
    assert_eq!(result2.shape(), &[3]);
    assert_eq!(result2.data().as_ref(), vec![1.0, 0.0, 0.0]);

    // Case 3: Broadcasting vector vs scalar
    let a3 = cpu_tensor(vec![1.0, 4.0, 2.0], &[3]);
    let b3 = cpu_tensor(vec![2.5], &[]);
    let result3 = ops::greater(&a3, &b3).unwrap();
    assert_eq!(result3.shape(), &[3]);
    assert_eq!(result3.data().as_ref(), vec![0.0, 1.0, 0.0]);

    // Case 4: Broadcasting matrix vs row
    let a4 = cpu_tensor(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
    let b4 = cpu_tensor(vec![3.0, 3.0, 3.0], &[1, 3]); // Broadcast this row
    let result4 = ops::greater(&a4, &b4).unwrap();
    assert_eq!(result4.shape(), &[2, 3]);
    assert_eq!(result4.data().as_ref(), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    // Verify not differentiable
    assert!(!result1.requires_grad());
}

#[test]
fn test_greater_equal_op() {
    // Case 1: Same shape including equals
    let a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let b1 = cpu_tensor(vec![2.0, 3.0, 2.0], &[3]);
    let result1 = ops::greater_equal(&a1, &b1).unwrap();
    assert_eq!(result1.shape(), &[3]);
    assert_eq!(result1.data().as_ref(), vec![0.0, 1.0, 1.0]); // 2.0 >= 2.0 is true

    // Case 2: Broadcasting scalar
    let a2 = cpu_tensor(vec![3.0], &[]);
    let b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let result2 = ops::greater_equal(&a2, &b2).unwrap();
    assert_eq!(result2.shape(), &[3]);
    assert_eq!(result2.data().as_ref(), vec![1.0, 1.0, 0.0]); // 3.0 >= 3.0 is true

    // Verify not differentiable
    assert!(!result1.requires_grad());
}

#[test]
fn test_less_op() {
    // Case 1: Same shape
    let a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let b1 = cpu_tensor(vec![2.0, 3.0, 2.0], &[3]);
    let result1 = ops::less(&a1, &b1).unwrap();
    assert_eq!(result1.shape(), &[3]);
    assert_eq!(result1.data().as_ref(), vec![1.0, 0.0, 0.0]); // 1.0 < 2.0 is true

    // Case 2: Broadcasting scalar
    let a2 = cpu_tensor(vec![3.0], &[]);
    let b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let result2 = ops::less(&a2, &b2).unwrap();
    assert_eq!(result2.shape(), &[3]);
    assert_eq!(result2.data().as_ref(), vec![0.0, 0.0, 1.0]); // 3.0 < 4.0 is true

    // Verify not differentiable
    assert!(!result1.requires_grad());
}

#[test]
fn test_less_equal_op() {
    // Case 1: Same shape including equals
    let a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let b1 = cpu_tensor(vec![2.0, 3.0, 2.0], &[3]);
    let result1 = ops::less_equal(&a1, &b1).unwrap();
    assert_eq!(result1.shape(), &[3]);
    assert_eq!(result1.data().as_ref(), vec![1.0, 0.0, 1.0]); // 2.0 <= 2.0 is true

    // Case 2: Broadcasting
    let a2 = cpu_tensor(vec![3.0], &[]);
    let b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let result2 = ops::less_equal(&a2, &b2).unwrap();
    assert_eq!(result2.shape(), &[3]);
    assert_eq!(result2.data().as_ref(), vec![0.0, 1.0, 1.0]); // 3.0 <= 3.0 is true

    // Verify not differentiable
    assert!(!result1.requires_grad());
}

#[test]
fn test_not_equal_op() {
    // Case 1: Same shape with some equalities
    let a1 = cpu_tensor(vec![1.0, 5.0, 2.0], &[3]);
    let b1 = cpu_tensor(vec![2.0, 5.0, 3.0], &[3]);
    let result1 = ops::not_equal(&a1, &b1).unwrap();
    assert_eq!(result1.shape(), &[3]);
    assert_eq!(result1.data().as_ref(), vec![1.0, 0.0, 1.0]); // 5.0 == 5.0 is false for not_equal

    // Case 2: Broadcasting with exact equality
    let a2 = cpu_tensor(vec![3.0], &[]);
    let b2 = cpu_tensor(vec![1.0, 3.0, 4.0], &[3]);
    let result2 = ops::not_equal(&a2, &b2).unwrap();
    assert_eq!(result2.shape(), &[3]);
    assert_eq!(result2.data().as_ref(), vec![1.0, 0.0, 1.0]); // 3.0 == 3.0 is false for not_equal

    // Case 3: Near equality test (floating point tolerance)
    let a3 = cpu_tensor(vec![1.0], &[1]);
    let b3 = cpu_tensor(vec![1.0 + 1e-10], &[1]); // Should be considered equal
    let result3 = ops::not_equal(&a3, &b3).unwrap();
    assert_eq!(result3.shape(), &[1]);
    assert_eq!(result3.data().as_ref(), vec![0.0]); // Should be considered equal within tolerance

    // Verify not differentiable
    assert!(!result1.requires_grad());
}

#[test]
fn test_cpu_softmax_cross_entropy_forward() -> Result<(), Error> {
    use approx::assert_abs_diff_eq;
    use rust_tensor_lib::{ops, CpuBackend, Reduction, Tensor};

    // --- Test Data ---
    // Batch of 2 samples, 3 classes
    let logits_data = vec![
        1.0, 2.0, 3.0, // Sample 1 logits
        0.5, 0.5, 0.5, // Sample 2 logits
    ];
    let targets_data = vec![
        0.0, 0.0, 1.0, // Sample 1: True class is 2 (index 2)
        0.0, 1.0, 0.0, // Sample 2: True class is 1 (index 1)
    ];
    let shape = &[2, 3];
    let axis = 1; // Class dimension

    let logits = Tensor::<CpuBackend>::from_vec(logits_data, shape, false)?;
    let targets = Tensor::<CpuBackend>::from_vec(targets_data, shape, false)?;

    // --- Expected Calculation (Manual/Numpy Reference) ---
    // Sample 1: logits=[1,2,3] -> log_softmax ≈ [-2.4076, -1.4076, -0.4076]
    //           target=[0,0,1] -> loss = -(0*-2.4076 + 0*-1.4076 + 1*-0.4076) = 0.4076
    // Sample 2: logits=[0.5,0.5,0.5] -> log_softmax ≈ [-1.0986, -1.0986, -1.0986] (log(1/3))
    //           target=[0,1,0] -> loss = -(0*-1.0986 + 1*-1.0986 + 0*-1.0986) = 1.0986
    let expected_loss_none_vec = [0.4076059, 1.0986123];
    let expected_loss_mean_val = (0.4076059 + 1.0986123) / 2.0;
    let expected_loss_sum_val = 0.4076059 + 1.0986123;

    // --- Test Reduction::None ---
    let loss_none = ops::softmax_cross_entropy(&logits, &targets, axis, Reduction::None)?;
    assert_eq!(loss_none.shape(), &[2]); // Shape after reducing axis 1
    let loss_none_data = loss_none.data();
    for (i, expected) in expected_loss_none_vec.iter().enumerate() {
        assert_abs_diff_eq!(loss_none_data.as_ref()[i], expected, epsilon = 1e-5);
    }

    // --- Test Reduction::Mean ---
    let loss_mean = ops::softmax_cross_entropy(&logits, &targets, axis, Reduction::Mean)?;
    assert_eq!(loss_mean.shape(), &[] as &[usize]); // Scalar shape
    assert_abs_diff_eq!(
        loss_mean.data().as_ref()[0],
        expected_loss_mean_val,
        epsilon = 1e-5
    );

    // --- Test Reduction::Sum ---
    let loss_sum = ops::softmax_cross_entropy(&logits, &targets, axis, Reduction::Sum)?;
    assert_eq!(loss_sum.shape(), &[] as &[usize]); // Scalar shape
    assert_abs_diff_eq!(
        loss_sum.data().as_ref()[0],
        expected_loss_sum_val,
        epsilon = 1e-5
    );

    Ok(())
}

#[test]
fn test_cpu_softmax_cross_entropy_gradient() -> Result<(), Error> {
    use rust_tensor_lib::{ops, test_utils::check_gradient, CpuBackend, Reduction, Tensor};

    // --- Test Data ---
    let logits_data = vec![1.0, 2.0, 3.0, 0.5, 0.5, 0.5];
    // Targets MUST NOT require grad for gradient checking w.r.t logits
    let targets_data = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let shape = &[2, 3];
    let axis = 1; // Class dimension

    let logits = Tensor::<CpuBackend>::from_vec(logits_data, shape, true)?; // requires_grad = true
    let targets = Tensor::<CpuBackend>::from_vec(targets_data, shape, false)?; // requires_grad = false

    // Define the function for check_gradient (needs to return scalar loss)
    let loss_fn = |inputs: &[Tensor<CpuBackend>]| -> Result<Tensor<CpuBackend>, Error> {
        ops::softmax_cross_entropy(&inputs[0], &targets, axis, Reduction::Mean)
    };

    // --- Gradient Check ---
    // Check gradient only w.r.t logits (input index 0)
    check_gradient(loss_fn, &[logits], 0, 1e-3, 1e-2)?; // Use slightly larger tolerance/epsilon

    Ok(())
}

#[test]
fn test_cpu_slice_forward() -> Result<(), Error> {
    // Test slicing a 2D tensor
    let tensor = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    
    // Test case 1: Slice [:, 1:2] -> [[2], [5]]
    let slice1 = ops::slice(&tensor, &[0..2, 1..2])?;
    assert_eq!(slice1.shape(), &[2, 1]);
    let slice1_data = CpuBackend::copy_to_host(&*slice1.data())?;
    assert_eq!(slice1_data, vec![2.0, 5.0]);
    
    // Test case 2: Slice [0:1, :] -> [[1, 2, 3]]
    let slice2 = ops::slice(&tensor, &[0..1, 0..3])?;
    assert_eq!(slice2.shape(), &[1, 3]);
    let slice2_data = CpuBackend::copy_to_host(&*slice2.data())?;
    assert_eq!(slice2_data, vec![1.0, 2.0, 3.0]);
    
    // Test case 3: Slice [0:1, 0:1] -> [[1]]
    let slice3 = ops::slice(&tensor, &[0..1, 0..1])?;
    assert_eq!(slice3.shape(), &[1, 1]);
    let slice3_data = CpuBackend::copy_to_host(&*slice3.data())?;
    assert_eq!(slice3_data, vec![1.0]);
    
    // Test slicing a 1D tensor
    let tensor_1d = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let slice_1d = ops::slice(&tensor_1d, &[1..3])?;
    assert_eq!(slice_1d.shape(), &[2]);
    let slice_1d_data = CpuBackend::copy_to_host(&*slice_1d.data())?;
    assert_eq!(slice_1d_data, vec![2.0, 3.0]);
    
    // Test empty slice result
    let empty_slice = ops::slice(&tensor, &[0..0, 0..3])?;
    assert_eq!(empty_slice.shape(), &[0, 3]);
    let empty_slice_data = CpuBackend::copy_to_host(&*empty_slice.data())?;
    assert_eq!(empty_slice_data.len(), 0);
    
    // Test requires_grad propagation
    let tensor_grad = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let slice_grad = ops::slice(&tensor_grad, &[0..1, 0..2])?;
    assert!(slice_grad.requires_grad());
    
    // Test error cases
    // Out of bounds range
    let result = ops::slice(&tensor, &[0..3, 0..3]); // First dimension is out of bounds
    assert!(result.is_err());
    
    // Invalid range (start > end)
    let result = ops::slice(&tensor, &[1..0, 0..3]);
    assert!(result.is_err());
    
    // Mismatched number of ranges
    let result = ops::slice(&tensor, &[0..1]);
    assert!(result.is_err());
    
    Ok(())
}

#[test]
fn test_cpu_slice_backward() -> Result<(), Error> {
    // Test backward pass for slice operation
    // Create a 2x2 input tensor with requires_grad=true
    let input = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    
    // Slice the first row: [0:1, :] -> [[1, 2]]
    let slice = ops::slice(&input, &[0..1, 0..2])?;
    assert_eq!(slice.shape(), &[1, 2]);
    let slice_data = CpuBackend::copy_to_host(&*slice.data())?;
    assert_eq!(slice_data, vec![1.0, 2.0]);
    
    // Create a gradient for the output: [[10, 20]]
    let grad_output = cpu_tensor(vec![10.0, 20.0], &[1, 2]);
    
    // Set the gradient for the output tensor
    slice.set_grad(Some(grad_output.data().clone()));
    
    // Perform backward pass
    slice.backward()?;
    
    // Check the input gradient
    // Expected gradient: [[10, 20], [0, 0]]
    // Only the sliced part should have non-zero gradients
    let input_grad = input.grad().unwrap();
    assert_eq!(input_grad.shape(), &[2, 2]);
    let expected_grad = vec![10.0, 20.0, 0.0, 0.0];
    let input_grad_data = CpuBackend::copy_to_host(&*input_grad)?;
    assert_eq!(input_grad_data, expected_grad);
    
    // Test with a different slice: [:, 0:1] -> [[1], [3]]
    let input2 = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let slice2 = ops::slice(&input2, &[0..2, 0..1])?;
    assert_eq!(slice2.shape(), &[2, 1]);
    let slice2_data = CpuBackend::copy_to_host(&*slice2.data())?;
    assert_eq!(slice2_data, vec![1.0, 3.0]);
    
    // Create a gradient for the output: [[30], [40]]
    let grad_output2 = cpu_tensor(vec![30.0, 40.0], &[2, 1]);
    
    // Set the gradient for the output tensor
    slice2.set_grad(Some(grad_output2.data().clone()));
    
    // Perform backward pass
    slice2.backward()?;
    
    // Check the input gradient
    // Expected gradient: [[30, 0], [40, 0]]
    let input_grad2 = input2.grad().unwrap();
    assert_eq!(input_grad2.shape(), &[2, 2]);
    let expected_grad2 = vec![30.0, 0.0, 40.0, 0.0];
    let input_grad2_data = CpuBackend::copy_to_host(&*input_grad2)?;
    assert_eq!(input_grad2_data, expected_grad2);
    
    Ok(())
}

#[test]
fn test_clip_op() {
    // Test forward pass for clip operation
    let a = cpu_tensor(vec![-1.0, 0.5, 2.0, 3.0], &[2, 2]);
    
    // Clip values to [0.0, 2.0]
    let result = ops::clip(&a, 0.0, 2.0).unwrap();
    let expected = vec![0.0, 0.5, 2.0, 2.0];
    assert_eq!(result.data().as_ref(), expected.as_slice());
    assert_eq!(result.shape(), vec![2, 2]);
    
    // Test with different min/max values
    let result2 = ops::clip(&a, -0.5, 1.0).unwrap();
    let expected2 = vec![-0.5, 0.5, 1.0, 1.0];
    assert_eq!(result2.data().as_ref(), expected2.as_slice());
    
    // Test error case: min_val > max_val
    let result3 = ops::clip(&a, 2.0, 1.0);
    assert!(result3.is_err());
}

#[test]
fn test_clip_backward() -> Result<(), Error> {
    // Test backward pass for clip operation
    let input = cpu_tensor_req_grad(vec![-1.0, 0.5, 2.0, 3.0], &[2, 2]);
    
    // Clip values to [0.0, 2.0]
    let clipped = ops::clip(&input, 0.0, 2.0)?;
    assert_eq!(clipped.shape(), &[2, 2]);
    
    // Create a gradient for the output: [[1.0, 1.0], [1.0, 1.0]]
    let grad_output = cpu_tensor(vec![1.0, 1.0, 1.0, 1.0], &[2, 2]);
    
    // Set the gradient for the output tensor
    clipped.set_grad(Some(grad_output.data().clone()));
    
    // Perform backward pass
    clipped.backward()?;
    
    // Check the input gradient
    // Expected gradient: [[0.0, 1.0], [1.0, 0.0]]
    // Only values within the clip range [0.0, 2.0] should have non-zero gradients
    let input_grad = input.grad().unwrap();
    assert_eq!(input_grad.shape(), &[2, 2]);
    let expected_grad = vec![0.0, 1.0, 1.0, 0.0];
    let input_grad_data = CpuBackend::copy_to_host(&*input_grad)?;
    assert_eq!(input_grad_data, expected_grad);
    
    Ok(())
}
