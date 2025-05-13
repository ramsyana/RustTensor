//! Tests for the hooks system in the Tensor library.

use rust_tensor_lib::backend::Backend;
use rust_tensor_lib::tensor::Tensor;

#[test]
fn test_register_and_clear_hooks() {
    // Create a dummy backend and tensor
    use rust_tensor_lib::backend::cpu::CpuBackend;
    let t = Tensor::<CpuBackend>::new(
        CpuBackend::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        false,
    );

    // Register a dummy hook (should not panic)
    t.register_hook(Box::new(|tensor: &Tensor<CpuBackend>| {
        assert_eq!(tensor.len(), 3);
    }));
    // Clear hooks (should not panic)
    t.clear_hooks();
}

#[test]
fn test_show_and_show_shape_do_not_panic() {
    use rust_tensor_lib::backend::cpu::CpuBackend;
    let t = Tensor::<CpuBackend>::new(
        CpuBackend::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        false,
    );
    t.show("test_show");
    t.show_shape("test_show_shape");
}

#[test]
fn test_raw_hook_method_exists() {
    use rust_tensor_lib::backend::cpu::CpuBackend;
    let t = Tensor::<CpuBackend>::new(
        CpuBackend::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
        false,
    );
    // Just check that raw_hook can be called and doesn't panic
    t.raw_hook(|tensor: &Tensor<CpuBackend>| {
        assert_eq!(tensor.len(), 3);
    });
}
