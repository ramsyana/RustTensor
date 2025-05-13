//! Hook trait and implementations for tensor post-processing.

use crate::tensor::Tensor;

pub trait Hook<B: crate::backend::Backend>: Send + Sync {
    fn call(&self, tensor: &Tensor<B>);
}

impl<B, F> Hook<B> for F
where
    B: crate::backend::Backend,
    F: for<'a> Fn(&'a Tensor<B>) + Send + Sync + 'static,
{
    fn call(&self, tensor: &Tensor<B>) {
        (self)(tensor)
    }
}

/// An owned hook object for post-processing tensors.
///
/// `FnHook` wraps a closure or function pointer and implements the [`Hook`] trait.
///
/// # Example
/// ```rust
/// use rust_tensor_lib::{Tensor, CpuBackend, FnHook};
/// let tensor = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], false).unwrap();
/// let hook = FnHook::new(|t: &Tensor<CpuBackend>| {
///     println!("Hook: ID={}, shape={:?}", t.id(), t.shape());
/// });
/// tensor.register_hook(Box::new(hook));
/// ```
pub struct FnHook<B: crate::backend::Backend> {
    func: Box<dyn for<'a> Fn(&'a Tensor<B>) + Send + Sync>,
}

impl<B: crate::backend::Backend> FnHook<B> {
    /// Creates a new `FnHook` from a closure or function pointer.
    ///
    /// # Example
    /// ```rust
    /// use rust_tensor_lib::{Tensor, CpuBackend, FnHook};
    /// let hook = FnHook::new(|t: &Tensor<CpuBackend>| {
    ///     println!("Hook: ID={}, shape={:?}", t.id(), t.shape());
    /// });
    /// ```
    pub fn new<F>(func: F) -> Self
    where
        F: for<'a> Fn(&'a Tensor<B>) + Send + Sync + 'static,
    {
        Self {
            func: Box::new(func),
        }
    }
}

impl<B: crate::backend::Backend> Hook<B> for FnHook<B> {
    fn call(&self, tensor: &Tensor<B>) {
        (self.func)(tensor)
    }
}
