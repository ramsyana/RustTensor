use crate::backend::Backend;
use crate::tensor::TensorData;
use std::fmt;

impl<B: Backend> fmt::Debug for TensorData<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorData")
            .field("id", &self.id)
            .field("data", &"<backend storage>")
            .field(
                "grad",
                &self.grad.as_ref().map(|_| "Some(<backend storage>)"),
            )
            .field("requires_grad", &self.requires_grad)
            .field("op", &self.op)
            .field("device", &self.device)
            .field("hooks_count", &self.hooks.len())
            .finish()
    }
}
