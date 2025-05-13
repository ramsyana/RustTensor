use approx::assert_abs_diff_eq;
use rust_tensor_lib::{ops, CpuBackend};

#[test]
fn test_random_uniform_cpu() {
    let shape = &[1000];
    let low = -1.0;
    let high = 1.0;
    let result = ops::random_uniform::<CpuBackend>(shape, low, high).unwrap();

    // Check shape and requires_grad
    assert_eq!(result.shape(), shape);
    assert!(!result.requires_grad());

    // Check values are within bounds
    let data = result.data();
    for &val in data.as_ref() {
        assert!(
            val >= low && val < high,
            "Value {} out of bounds [{}, {})",
            val,
            low,
            high
        );
    }

    // Check mean is roughly centered (should be close to 0.0 for [-1, 1))
    let mean: f32 = data.as_ref().iter().sum::<f32>() / shape[0] as f32;
    assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1);
}

#[test]
fn test_random_normal_cpu() {
    let shape = &[10000]; // Larger size for better statistical properties
    let mean = 0.0;
    let std_dev = 1.0;
    let result = ops::random_normal::<CpuBackend>(shape, mean, std_dev).unwrap();

    // Check shape and requires_grad
    assert_eq!(result.shape(), shape);
    assert!(!result.requires_grad());

    // Check statistical properties
    let data = result.data();
    let data_slice = data.as_ref();
    let actual_mean: f32 = data_slice.iter().sum::<f32>() / shape[0] as f32;
    let variance: f32 = data_slice
        .iter()
        .map(|&x| (x - actual_mean).powi(2))
        .sum::<f32>()
        / shape[0] as f32;
    let actual_std_dev = variance.sqrt();

    // Use looser tolerances for statistical tests
    assert_abs_diff_eq!(actual_mean, mean, epsilon = 0.1);
    assert_abs_diff_eq!(actual_std_dev, std_dev, epsilon = 0.1);
}

#[test]
fn test_bernoulli_cpu() {
    let shape = &[10000]; // Larger size for better statistical properties
    let p = 0.3;
    let result = ops::bernoulli::<CpuBackend>(shape, p).unwrap();

    // Check shape and requires_grad
    assert_eq!(result.shape(), shape);
    assert!(!result.requires_grad());

    // Check values are only 0.0 or 1.0
    let data = result.data();
    let mut ones_count = 0;
    for &val in data.as_ref() {
        assert!(val == 0.0 || val == 1.0, "Value {} is not 0.0 or 1.0", val);
        if val == 1.0 {
            ones_count += 1;
        }
    }

    // Check proportion of ones is close to p
    let proportion = ones_count as f32 / shape[0] as f32;
    assert_abs_diff_eq!(proportion, p, epsilon = 0.05);
}

// Error cases
#[test]
fn test_random_uniform_invalid_bounds() {
    let shape = &[10];
    let result = ops::random_uniform::<CpuBackend>(shape, 1.0, 0.0); // high < low
    assert!(result.is_err());
}

#[test]
fn test_random_normal_invalid_std_dev() {
    let shape = &[10];
    let result = ops::random_normal::<CpuBackend>(shape, 0.0, -1.0); // negative std_dev
    assert!(result.is_err());
}

#[test]
fn test_bernoulli_invalid_p() {
    let shape = &[10];
    let result = ops::bernoulli::<CpuBackend>(shape, 1.5); // p > 1.0
    assert!(result.is_err());
    let result = ops::bernoulli::<CpuBackend>(shape, -0.5); // p < 0.0
    assert!(result.is_err());
}

// CUDA tests
#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use rust_tensor_lib::backend::cuda::{init_context, CudaBackend, CudaContextGuard};
    use rust_tensor_lib::{ops, Backend};

    use serial_test::serial;

    #[test]
    #[serial]
    fn test_random_uniform_cuda() {
        init_context(0).unwrap();
        let _guard = CudaContextGuard::new().unwrap();

        let shape = &[1000];
        let low = -1.0;
        let high = 1.0;
        let result = ops::random_uniform::<CudaBackend>(shape, low, high).unwrap();

        // Check shape and requires_grad
        assert_eq!(result.shape(), shape);
        assert!(!result.requires_grad());

        // Copy to host for validation
        let data = CudaBackend::copy_to_host(&*result.data()).unwrap();
        for &val in &data {
            assert!(
                val >= low && val < high,
                "Value {} out of bounds [{}, {})",
                val,
                low,
                high
            );
        }

        let mean: f32 = data.iter().sum::<f32>() / shape[0] as f32;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1);
    }

    #[test]
    #[serial]
    fn test_random_normal_cuda() {
        init_context(0).unwrap();
        let _guard = CudaContextGuard::new().unwrap();

        let shape = &[10000];
        let mean = 0.0;
        let std_dev = 1.0;
        let result = ops::random_normal::<CudaBackend>(shape, mean, std_dev).unwrap();

        assert_eq!(result.shape(), shape);
        assert!(!result.requires_grad());

        let data = CudaBackend::copy_to_host(&*result.data()).unwrap();
        let actual_mean: f32 = data.iter().sum::<f32>() / shape[0] as f32;
        let variance: f32 =
            data.iter().map(|&x| (x - actual_mean).powi(2)).sum::<f32>() / shape[0] as f32;
        let actual_std_dev = variance.sqrt();

        assert_abs_diff_eq!(actual_mean, mean, epsilon = 0.1);
        assert_abs_diff_eq!(actual_std_dev, std_dev, epsilon = 0.1);
    }

    #[test]
    #[serial]
    fn test_bernoulli_cuda() {
        init_context(0).unwrap();
        let _guard = CudaContextGuard::new().unwrap();

        let shape = &[10000];
        let p = 0.3;
        let result = ops::bernoulli::<CudaBackend>(shape, p).unwrap();

        assert_eq!(result.shape(), shape);
        assert!(!result.requires_grad());

        let data = CudaBackend::copy_to_host(&*result.data()).unwrap();
        let mut ones_count = 0;
        for &val in &data {
            assert!(val == 0.0 || val == 1.0, "Value {} is not 0.0 or 1.0", val);
            if val == 1.0 {
                ones_count += 1;
            }
        }

        let proportion = ones_count as f32 / shape[0] as f32;
        assert_abs_diff_eq!(proportion, p, epsilon = 0.05);
    }
}
