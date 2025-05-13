// src/optim.rs
use crate::{Backend, Error, Tensor};
use std::fmt::Debug;

#[cfg(test)]
use crate::backend::cpu::CpuBackend; // For tests

/// Stochastic Gradient Descent optimizer.
#[derive(Debug)] // Added Debug
pub struct Sgd<B: Backend> {
    parameters: Vec<Tensor<B>>, // Store Tensors to optimize

    learning_rate: f32,
    // Add momentum state later if needed
    // velocities: Option<Vec<B::Storage>>,
}

impl<B: Backend> Sgd<B> {
    /// Creates a new SGD optimizer.
    ///
    /// # Arguments
    ///
    /// * `parameters` - A vector of tensors whose gradients will be updated.
    /// * `learning_rate` - The step size for gradient updates.
    pub fn new(parameters: Vec<Tensor<B>>, learning_rate: f32) -> Self {
        // Ensure all parameters require grad (optional check)
        // assert!(parameters.iter().all(|p| p.requires_grad()), "Optimizing tensor does not require grad");
        Sgd {
            parameters,
            learning_rate,
        }
    }

    /// Performs a single optimization step.
    /// Updates parameters in-place: `param = param - lr * grad`.
    pub fn step(&mut self) -> Result<(), Error> {
        // First, collect all gradients and verify shapes
        let mut updates = Vec::new();

        for param in &self.parameters {
            if !param.requires_grad() {
                continue;
            }

            // Get parameter shape first
            let param_shape = {
                let inner = param.inner.borrow();
                B::shape(&inner.data).to_vec()
            };

            // Then get gradient if it exists
            if let Some(grad) = param.grad() {
                let grad_data = (*grad).clone();
                if B::shape(&grad_data) != param_shape.as_slice() {
                    eprintln!(
                        "Warning: Optimizer skipping parameter (ID {}) due to shape mismatch: data {:?} vs grad {:?}",
                        param.id(), param_shape, B::shape(&grad_data)
                    );
                    continue;
                }
                updates.push((param, grad_data));
            }
        }

        // Now apply all updates without any gradient borrows active
        for (param, grad_data) in updates {
            let mut inner = param.inner.borrow_mut();
            B::sgd_step(&mut inner.data, &grad_data, self.learning_rate)?;
        }

        Ok(())
    }

    /// Sets the gradients of all tracked parameters to None.
    pub fn zero_grad(&mut self) -> Result<(), Error> {
        for param in &self.parameters {
            if param.requires_grad() {
                param.zero_grad(); // Use Tensor's method
            }
        }
        Ok(())
    }
}

/// Adam optimizer for adaptive moment estimation.
#[derive(Debug)]
pub struct Adam<B: Backend> {
    /// Tensors to optimize
    pub parameters: Vec<Tensor<B>>,
    /// Learning rate (alpha)
    pub learning_rate: f32,
    /// Exponential decay rate for the first moment estimates
    pub beta1: f32,
    /// Exponential decay rate for the second moment estimates
    pub beta2: f32,
    /// Term added to the denominator to improve numerical stability
    pub epsilon: f32,
    /// First moment vectors (running average of gradients)
    pub m_states: Vec<B::Storage>,
    /// Second moment vectors (running average of squared gradients)
    pub v_states: Vec<B::Storage>,
    /// Timestep counter
    pub t: usize,
}

impl<B: Backend> Adam<B> {
    /// Creates a new Adam optimizer.
    ///
    /// # Arguments
    /// * `parameters` - Tensors to optimize
    /// * `learning_rate` - Learning rate (alpha)
    /// * `beta1` - Exponential decay rate for the first moment estimates (default 0.9)
    /// * `beta2` - Exponential decay rate for the second moment estimates (default 0.999)
    /// * `epsilon` - Small value for numerical stability (default 1e-8)
    pub fn new(
        parameters: Vec<Tensor<B>>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Result<Self, Error> {
        let n_params = parameters.len();
        let mut m_states = Vec::with_capacity(n_params);
        let mut v_states = Vec::with_capacity(n_params);
        for param in &parameters {
            let shape = param.shape();
            m_states.push(B::zeros(&shape)?);
            v_states.push(B::zeros(&shape)?);
        }
        Ok(Self {
            parameters,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m_states,
            v_states,
            t: 0,
        })
    }
    /// Sets the gradients of all tracked parameters to None.
    pub fn zero_grad(&mut self) -> Result<(), Error> {
        for param in &self.parameters {
            if param.requires_grad() {
                param.zero_grad();
            }
        }
        Ok(())
    }
    /// Performs a single Adam optimization step.
    /// Updates parameters in-place using Adam update rule.
    pub fn step(&mut self) -> Result<(), Error> {
        self.t += 1; // Increment timestep
        let mut updates_to_perform: Vec<(usize, B::Storage)> = Vec::new();
        for (i, param) in self.parameters.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }
            let param_shape = param.shape();
            if let Some(grad_ref) = param.grad() {
                let grad_storage = (*grad_ref).clone();
                let grad_shape = B::shape(&grad_storage);
                if param_shape != grad_shape {
                    eprintln!(
                        "Warning: Adam optimizer skipping parameter (ID {}) at index {} due to shape mismatch: data {:?} vs grad {:?}",
                        param.id(), i, param_shape, grad_shape
                    );
                    continue;
                }
                updates_to_perform.push((i, grad_storage));
            }
        }
        for (i, grad_storage) in updates_to_perform {
            let mut param_inner = self.parameters[i].inner.borrow_mut();
            let m_state = &mut self.m_states[i];
            let v_state = &mut self.v_states[i];
            B::adam_step(
                &mut param_inner.data,
                &grad_storage,
                m_state,
                v_state,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
                self.t,
            )?;
        }
        Ok(())
    }
}

/// Momentum SGD optimizer.
///
/// Implements classical SGD with momentum. Maintains a velocity vector for each parameter.
#[derive(Debug)]
pub struct MomentumSGD<B: Backend> {
    /// Tensors to optimize
    pub parameters: Vec<Tensor<B>>,
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum hyperparameter (typical default: 0.9)
    pub momentum: f32,
    /// Velocity state for each parameter
    pub velocities: Vec<B::Storage>,
}

impl<B: Backend> MomentumSGD<B> {
    /// Creates a new MomentumSGD optimizer.
    ///
    /// # Arguments
    /// * `parameters` - Tensors to optimize
    /// * `learning_rate` - Learning rate
    /// * `momentum` - Momentum hyperparameter (default 0.9)
    pub fn new(
        parameters: Vec<Tensor<B>>,
        learning_rate: f32,
        momentum: f32,
    ) -> Result<Self, Error> {
        let n_params = parameters.len();
        let mut velocities = Vec::with_capacity(n_params);
        for param in &parameters {
            let shape = param.shape();
            velocities.push(B::zeros(&shape)?);
        }
        Ok(MomentumSGD {
            parameters,
            learning_rate,
            momentum,
            velocities,
        })
    }

    /// Performs a single MomentumSGD optimization step.
    /// Updates parameters and velocities in-place.
    pub fn step(&mut self) -> Result<(), Error> {
        let mut updates = Vec::new();
        for (i, param) in self.parameters.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }
            let param_shape = {
                let inner = param.inner.borrow();
                B::shape(&inner.data).to_vec()
            };
            if let Some(grad) = param.grad() {
                let grad_storage = (*grad).clone();
                if B::shape(&grad_storage) != param_shape.as_slice() {
                    eprintln!(
                        "Warning: MomentumSGD skipping parameter (ID {}) due to shape mismatch: data {:?} vs grad {:?}",
                        param.id(), param_shape, B::shape(&grad_storage)
                    );
                    continue;
                }
                updates.push((i, grad_storage));
            }
        }
        for (i, grad_storage) in updates {
            let mut param_inner = self.parameters[i].inner.borrow_mut();
            let velocity = &mut self.velocities[i];
            B::momentum_sgd_step(
                &mut param_inner.data,
                &grad_storage,
                velocity,
                self.learning_rate,
                self.momentum,
            )?;
        }
        Ok(())
    }
}

/// AdaGrad optimizer.
///
/// Implements the AdaGrad algorithm which adapts the learning rate per parameter
/// based on the historical sum of squared gradients.
#[derive(Debug)]
pub struct AdaGrad<B: Backend> {
    /// Tensors to optimize
    pub parameters: Vec<Tensor<B>>,
    /// Learning rate (alpha)
    pub learning_rate: f32,
    /// Term added to the denominator to improve numerical stability
    pub epsilon: f32,
    /// Accumulated squared gradients state for each parameter
    pub accumulated_sq_grad: Vec<B::Storage>,
}

impl<B: Backend> AdaGrad<B> {
    /// Creates a new AdaGrad optimizer.
    ///
    /// # Arguments
    /// * `parameters` - Tensors to optimize.
    /// * `learning_rate` - Learning rate (alpha).
    /// * `epsilon` - Small value for numerical stability (e.g., 1e-8).
    pub fn new(
        parameters: Vec<Tensor<B>>,
        learning_rate: f32,
        epsilon: f32,
    ) -> Result<Self, Error> {
        let n_params = parameters.len();
        let mut accumulated_sq_grad = Vec::with_capacity(n_params);

        // Initialize accumulated squared gradients state with zeros for each parameter
        for param in &parameters {
            let shape = param.shape();
            accumulated_sq_grad.push(B::zeros(&shape)?);
        }

        Ok(Self {
            parameters,
            learning_rate,
            epsilon,
            accumulated_sq_grad,
        })
    }

    /// Sets the gradients of all tracked parameters to None.
    pub fn zero_grad(&mut self) -> Result<(), Error> {
        for param in &self.parameters {
            if param.requires_grad() {
                param.zero_grad();
            }
        }
        Ok(())
    }

    /// Performs a single AdaGrad optimization step.
    /// Updates parameters and accumulated squared gradients in-place.
    pub fn step(&mut self) -> Result<(), Error> {
        // Collect parameter indices and their corresponding gradients first to avoid borrow issues.
        let mut updates_to_perform: Vec<(usize, B::Storage)> = Vec::new();
        for (i, param) in self.parameters.iter().enumerate() {
            if !param.requires_grad() {
                continue;
            }

            let param_shape = param.shape();

            if let Some(grad_ref) = param.grad() {
                let grad_storage = (*grad_ref).clone();
                let grad_shape = B::shape(&grad_storage);
                if param_shape != grad_shape {
                    eprintln!(
                        "Warning: AdaGrad optimizer skipping parameter (ID {}) at index {} due to shape mismatch: data {:?} vs grad {:?}",
                        param.id(), i, param_shape, grad_shape
                    );
                    continue;
                }
                updates_to_perform.push((i, grad_storage));
            }
        }

        for (i, grad_storage) in updates_to_perform {
            let mut param_inner = self.parameters[i].inner.borrow_mut();
            let accum_sq_grad_state = &mut self.accumulated_sq_grad[i];

            B::adagrad_step(
                &mut param_inner.data,
                &grad_storage,
                accum_sq_grad_state,
                self.learning_rate,
                self.epsilon,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use crate::backend::cuda::{init_context, CudaBackend, CudaContextGuard};
    use crate::{CpuTensor, Tensor};
    use approx::assert_abs_diff_eq;

    fn cpu_tensor_req_grad(data: Vec<f32>, shape: &[usize]) -> CpuTensor {
        Tensor::<CpuBackend>::new(CpuBackend::from_vec(data, shape).unwrap(), true)
    }

    #[test]
    fn test_momentum_sgd_cpu() {
        let initial_params = vec![1.0, 2.0];
        let gradients = vec![0.1, -0.2];
        let shape = &[2];
        let lr = 0.1;
        let momentum = 0.9;
        let param_tensor =
            Tensor::<CpuBackend>::from_vec(initial_params.clone(), shape, true).unwrap();
        let grad_storage = CpuBackend::from_vec(gradients.clone(), shape).unwrap();
        param_tensor.set_grad(Some(grad_storage));
        let mut optimizer =
            MomentumSGD::<CpuBackend>::new(vec![param_tensor.clone()], lr, momentum).unwrap();
        optimizer.step().unwrap();
        // Calculate expected values
        #[allow(clippy::useless_vec)]
        let mut expected_velocity = vec![0.0; 2];
        let mut expected_param = initial_params.clone();
        for i in 0..2 {
            expected_velocity[i] = momentum * expected_velocity[i] + gradients[i];
            expected_param[i] -= lr * expected_velocity[i];
        }
        let actual_param_data = CpuBackend::copy_to_host(&optimizer.parameters[0].data()).unwrap();
        for i in 0..2 {
            assert_abs_diff_eq!(actual_param_data[i], expected_param[i], epsilon = 1e-6);
        }
        let actual_velocity = CpuBackend::copy_to_host(&optimizer.velocities[0]).unwrap();
        for i in 0..2 {
            assert_abs_diff_eq!(actual_velocity[i], expected_velocity[i], epsilon = 1e-6);
        }
    }

    #[cfg(feature = "cuda")]
    #[serial_test::serial]
    #[test]
    fn test_momentum_sgd_cuda() -> Result<(), Error> {
        init_context(0)?;
        let _guard = CudaContextGuard::new()?;
        let initial_params = vec![1.0, 2.0];
        let gradients = vec![0.1, -0.2];
        let shape = &[2];
        let lr = 0.1;
        let momentum = 0.9;
        let param_tensor =
            Tensor::<CudaBackend>::from_vec(initial_params.clone(), shape, true).unwrap();
        let grad_storage = CudaBackend::from_vec(gradients.clone(), shape).unwrap();
        param_tensor.set_grad(Some(grad_storage));
        let mut optimizer =
            MomentumSGD::<CudaBackend>::new(vec![param_tensor.clone()], lr, momentum).unwrap();
        optimizer.step().unwrap();
        #[allow(clippy::useless_vec)]
        let mut expected_velocity = vec![0.0; 2];
        let mut expected_param = initial_params.clone();
        for i in 0..2 {
            expected_velocity[i] = momentum * expected_velocity[i] + gradients[i];
            expected_param[i] -= lr * expected_velocity[i];
        }
        let actual_param_data = CudaBackend::copy_to_host(&optimizer.parameters[0].data())?;
        for i in 0..2 {
            assert_abs_diff_eq!(actual_param_data[i], expected_param[i], epsilon = 1e-6);
        }
        let actual_velocity = CudaBackend::copy_to_host(&optimizer.velocities[0])?;
        for i in 0..2 {
            assert_abs_diff_eq!(actual_velocity[i], expected_velocity[i], epsilon = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_sgd_step() {
        let tensor = cpu_tensor_req_grad(vec![1.0, 2.0, 3.0], &[3]);
        let grad_storage = CpuBackend::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();
        tensor.set_grad(Some(grad_storage)); // Set gradient

        // Pass the tensor itself to the optimizer
        let mut optimizer = Sgd::<CpuBackend>::new(vec![tensor.clone()], 0.1); // Use generic Sgd
        optimizer.step().unwrap();

        // Check parameter data after step using copy_to_host
        let param_data = CpuBackend::copy_to_host(&optimizer.parameters[0].data()).unwrap();

        assert!(
            (param_data[0] - (1.0 - 0.1 * 0.1)).abs() < 1e-6,
            "Value was {}",
            param_data[0]
        ); // 0.99
        assert!(
            (param_data[1] - (2.0 - 0.1 * 0.2)).abs() < 1e-6,
            "Value was {}",
            param_data[1]
        ); // 1.98
        assert!(
            (param_data[2] - (3.0 - 0.1 * 0.3)).abs() < 1e-6,
            "Value was {}",
            param_data[2]
        ); // 2.97
    }

    #[test]
    fn test_zero_grad() {
        let tensor = cpu_tensor_req_grad(vec![1.0, 2.0], &[2]);
        let grad_storage = CpuBackend::from_vec(vec![0.1, 0.2], &[2]).unwrap();
        tensor.set_grad(Some(grad_storage));

        let mut optimizer = Sgd::<CpuBackend>::new(vec![tensor.clone()], 0.1);
        assert!(optimizer.parameters[0].grad().is_some()); // Grad exists

        optimizer.zero_grad().unwrap(); // Call zero_grad

        assert!(optimizer.parameters[0].grad().is_none()); // Grad should be None
    }

    #[test]
    fn test_sgd_step_no_grad_param() {
        // Parameter that requires grad
        let tensor_grad = cpu_tensor_req_grad(vec![1.0, 2.0], &[2]);
        let grad_storage = CpuBackend::from_vec(vec![0.1, 0.2], &[2]).unwrap();
        tensor_grad.set_grad(Some(grad_storage));

        // Parameter that does NOT require grad
        let tensor_no_grad = Tensor::<CpuBackend>::new(
            CpuBackend::from_vec(vec![10.0, 20.0], &[2]).unwrap(),
            false, // requires_grad = false
        );
        tensor_no_grad.set_grad(Some(CpuBackend::from_vec(vec![1.0, 1.0], &[2]).unwrap())); // Set grad anyway (should be ignored)

        let mut optimizer =
            Sgd::<CpuBackend>::new(vec![tensor_grad.clone(), tensor_no_grad.clone()], 0.1);
        optimizer.step().unwrap();

        // Check tensor_grad was updated
        let param1_data = CpuBackend::copy_to_host(&optimizer.parameters[0].data()).unwrap();
        assert!((param1_data[0] - (1.0 - 0.1 * 0.1)).abs() < 1e-6);
        assert!((param1_data[1] - (2.0 - 0.1 * 0.2)).abs() < 1e-6);

        // Check tensor_no_grad was NOT updated
        let param2_data = CpuBackend::copy_to_host(&optimizer.parameters[1].data()).unwrap();
        assert!((param2_data[0] - 10.0).abs() < 1e-6);
        assert!((param2_data[1] - 20.0).abs() < 1e-6);
    }
}
