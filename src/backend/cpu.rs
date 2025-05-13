//! CPU backend implementation using `ndarray`.

use crate::array::Array; // The storage type for this backend
use crate::backend::Backend;
use crate::error::Error;
use crate::graph::{Op, OpType}; // Needed for backward function signatures and OpType
use crate::init; // For kaiming_uniform implementation delegation
use crate::ops::cpu_backward;
use crate::ops::cpu_ops; // Import the module itself
                                         // Random imports for random_uniform
use rand::Rng;
use rand_distr::{Bernoulli, Normal, Uniform};
// Remove unused ndarray imports if Array handles everything
// use ndarray::{ArrayD, Axis, Dimension, IxDyn}; // Import ArrayD, Axis, Dimension and IxDyn for argmax/argmin implementation
use std::fmt::{self, Debug, Display}; // Import formatting traits
                                      // use ndarray::Zip; // Import Zip for elementwise operations
                                      // use crate::util; // For broadcast_shapes and other utility functions

/// Marker struct for the CPU backend.
/// Implements the `Backend` trait using `ndarray` operations via the `Array` wrapper.
#[derive(Debug, Clone, PartialEq, Eq)] // Ensure traits required by Backend bound are derived
pub struct CpuBackend;

// --- Implement Backend Trait for CpuBackend ---

impl Backend for CpuBackend {
    fn device(_storage: &Self::Storage) -> crate::Device {
        crate::Device::Cpu
    }
    type Storage = Array; // The storage type is our Array struct

    fn adagrad_step(
        param: &mut Self::Storage,         // Array
        grad: &Self::Storage,              // Array
        accum_sq_grad: &mut Self::Storage, // Array
        lr: f32,
        epsilon: f32,
    ) -> Result<(), Error> {
        // --- Shape Checks ---
        let param_shape = param.shape();
        if param_shape != grad.shape() {
            return Err(Error::ShapeMismatch {
                expected: param_shape.to_vec(),
                actual: grad.shape().to_vec(),
            });
        }
        if param_shape != accum_sq_grad.shape() {
            return Err(Error::ShapeMismatch {
                expected: param_shape.to_vec(),
                actual: accum_sq_grad.shape().to_vec(),
            });
        }

        // Handle empty tensor case gracefully
        if param.size() == 0 {
            return Ok(());
        }

        // --- Get Mutable NdArray Views ---
        // Use get_data_mut() which returns &mut ArrayD<f32>
        let param_data = param.get_data_mut();
        let accum_sq_grad_data = accum_sq_grad.get_data_mut();
        // Get immutable view for grad
        let grad_data = grad.get_data(); // Returns &ArrayD<f32>

        // --- Perform Element-wise Update using ndarray::Zip ---
        ndarray::Zip::from(param_data)
            .and(accum_sq_grad_data)
            .and(grad_data)
            .for_each(|p, acc, &g| {
                // Update accumulated squared gradient: acc = acc + g^2
                *acc += g * g;
                // Update parameter: p = p - lr * g / (sqrt(acc) + epsilon)
                *p -= lr * g / (acc.sqrt() + epsilon);
            });

        Ok(())
    }

    // --- Optimizer: Momentum SGD ---
    fn momentum_sgd_step(
        param: &mut Self::Storage,    // Array
        grad: &Self::Storage,         // Array
        velocity: &mut Self::Storage, // Array
        lr: f32,
        momentum: f32,
    ) -> Result<(), Error> {
        // Shape checks
        let param_shape = param.shape();
        if param_shape != velocity.shape() {
            return Err(Error::ShapeMismatch {
                expected: param_shape.to_vec(),
                actual: velocity.shape().to_vec(),
            });
        }
        if param_shape != grad.shape() {
            return Err(Error::ShapeMismatch {
                expected: param_shape.to_vec(),
                actual: grad.shape().to_vec(),
            });
        }

        // Handle empty case
        if param.size() == 0 {
            return Ok(());
        }

        // Get mutable references to ndarray data
        let param_data = param.get_data_mut();
        let velocity_data = velocity.get_data_mut();
        // Get immutable reference for grad data
        let grad_data = grad.get_data();

        // Perform element-wise update using ndarray::Zip
        ndarray::Zip::from(param_data)
            .and(velocity_data)
            .and(grad_data)
            .for_each(|p, v, &g| {
                // Update velocity: v = momentum * v + grad
                *v = momentum * *v + g;
                // Update parameter: p = p - lr * v (using the *updated* velocity)
                *p -= lr * *v;
            });

        Ok(())
    }

    // --- Factory Methods ---
    fn zeros(shape: &[usize]) -> Result<Self::Storage, Error> {
        Ok(Array::zeros(shape)) // Use Array's implementation
    }

    fn random_uniform(shape: &[usize], low: f32, high: f32) -> Result<Self::Storage, Error> {
        let size = shape.iter().product::<usize>();
        if size == 0 {
            // Return empty storage matching the shape
            return Ok(Array::zeros(shape));
        }
        let dist = Uniform::new(low, high).map_err(|_| Error::InitializationError)?;
        let mut rng = rand::rng();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(rng.sample(dist));
        }
        Array::from_vec(data, shape)
    }

    fn random_normal(shape: &[usize], mean: f32, std_dev: f32) -> Result<Self::Storage, Error> {
        let size = shape.iter().product::<usize>();
        if size == 0 {
            return Ok(Array::zeros(shape));
        }

        let dist = Normal::new(mean, std_dev).map_err(|_| Error::InitializationError)?;

        let mut rng = rand::rng();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(rng.sample(dist));
        }

        Array::from_vec(data, shape)
    }

    fn bernoulli(shape: &[usize], p: f32) -> Result<Self::Storage, Error> {
        let size = shape.iter().product::<usize>();
        if size == 0 {
            return Ok(Array::zeros(shape));
        }

        let dist = Bernoulli::new(p as f64).map_err(|_| Error::InitializationError)?;

        let mut rng = rand::rng();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let sample: bool = rng.sample(dist);
            data.push(if sample { 1.0f32 } else { 0.0f32 });
        }

        Array::from_vec(data, shape)
    }

    fn ones(shape: &[usize]) -> Result<Self::Storage, Error> {
        Ok(Array::ones(shape)) // Use Array's implementation
    }

    fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self::Storage, Error> {
        Array::from_vec(data, shape) // Use Array's implementation
    }

    fn kaiming_uniform(fan_in: usize, shape: &[usize]) -> Result<Self::Storage, Error> {
        // Delegate to the init module function which returns an Array
        init::kaiming_uniform(fan_in, shape)
    }

    // --- Shape/Data Access ---
    fn shape(storage: &Self::Storage) -> &[usize] {
        storage.shape() // Delegate to Array
    }

    fn size(storage: &Self::Storage) -> usize {
        storage.size() // Delegate to Array
    }

    fn into_raw_vec(storage: Self::Storage) -> Result<Vec<f32>, Error> {
        // CPU operation doesn't really fail, but match trait signature
        Ok(storage.into_raw_vec())
    }

    fn set_data(storage: &mut Self::Storage, data: Self::Storage) -> Result<(), Error> {
        *storage = data;
        Ok(())
    }

    fn set_shape(storage: &mut Self::Storage, shape: &[usize]) -> Result<(), Error> {
        storage.reshape(shape).map_err(|_| Error::ShapeMismatch {
            expected: shape.to_vec(),
            actual: storage.shape().to_vec(),
        })
    }

    // --- Explicit Data Movement ---
    fn copy_to_host(storage: &Self::Storage) -> Result<Vec<f32>, Error> {
        // For CPU, clone internal data
        // Use to_vec() directly on ArrayD which works for both contiguous and non-contiguous arrays
        Ok(storage.get_data().iter().cloned().collect())
    }

    fn update_from_host(storage: &mut Self::Storage, data: &[f32]) -> Result<(), Error> {
        if storage.size() != data.len() {
            return Err(Error::ShapeMismatch {
                expected: storage.shape().to_vec(),
                actual: vec![data.len()],
            });
        }
        storage
            .get_data_mut()
            .as_slice_mut()
            .unwrap()
            .copy_from_slice(data);
        Ok(())
    }

    fn sgd_step(
        param: &mut Self::Storage,
        grad: &Self::Storage,
        learning_rate: f32,
    ) -> Result<(), Error> {
        if param.shape() != grad.shape() {
            return Err(Error::ShapeMismatch {
                expected: param.shape().to_vec(),
                actual: grad.shape().to_vec(),
            });
        }
        // Perform w -= lr * dw in-place
        let param_data = param.get_data_mut();
        let grad_data = grad.get_data();
        param_data.zip_mut_with(grad_data, |w, &dw| *w -= learning_rate * dw);
        Ok(())
    }

    fn adam_step(
        param: &mut Self::Storage,
        grad: &Self::Storage,
        m: &mut Self::Storage,
        v: &mut Self::Storage,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        t: usize,
    ) -> Result<(), Error> {
        // Shape checks
        let param_shape = param.shape();
        if param_shape != grad.shape() {
            return Err(Error::ShapeMismatch {
                expected: param_shape.to_vec(),
                actual: grad.shape().to_vec(),
            });
        }
        if param_shape != m.shape() {
            return Err(Error::ShapeMismatch {
                expected: param_shape.to_vec(),
                actual: m.shape().to_vec(),
            });
        }
        if param_shape != v.shape() {
            return Err(Error::ShapeMismatch {
                expected: param_shape.to_vec(),
                actual: v.shape().to_vec(),
            });
        }
        let param_data = param.get_data_mut();
        let m_data = m.get_data_mut();
        let v_data = v.get_data_mut();
        let grad_data = grad.get_data();
        if t == 0 {
            return Err(Error::InternalLogicError(
                "Adam step called with t=0".to_string(),
            ));
        }
        let beta1_t = beta1.powi(t as i32);
        let beta2_t = beta2.powi(t as i32);
        let bias_correction1 = 1.0 - beta1_t;
        let bias_correction2 = 1.0 - beta2_t;
        ndarray::Zip::from(param_data)
            .and(m_data)
            .and(v_data)
            .and(grad_data)
            .for_each(|p, m, v, &g| {
                *m = beta1 * *m + (1.0 - beta1) * g;
                *v = beta2 * *v + (1.0 - beta2) * (g * g);
                let m_hat = *m / bias_correction1;
                let v_hat = *v / bias_correction2;
                *p -= lr * m_hat / (v_hat.sqrt() + epsilon);
            });
        Ok(())
    }

    // --- Core Mathematical Operations ---
    // Delegate most operations to the implementations in `cpu_ops` module,

    // --- START: Added Placeholders for Task 2.6 ---
    fn max(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error> {
        cpu_ops::max(x, axis)
    }
    fn min(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error> {
        cpu_ops::min(x, axis)
    }
    fn prod(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error> {
        cpu_ops::prod(x, axis)
    }
    fn logsumexp(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error> {
        cpu_ops::logsumexp(x, axis)
    }
    fn argmax(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error> {
        cpu_ops::argmax(x, axis)
    }
    fn argmin(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error> {
        cpu_ops::argmin(x, axis)
    }
    fn max_backward(op: &Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error> {
        if op.inputs.len() != 1 {
            return Err(Error::InvalidOperation(
                "max_backward expects 1 input".into(),
            ));
        }
        let input_data_ref = op.inputs[0].data();
        let input_storage = &*input_data_ref;
        let input_shape = Self::shape(input_storage).to_vec();

        let axis = match op.op_type {
            OpType::Max(axis) => axis,
            _ => {
                return Err(Error::InternalLogicError(
                    "Incorrect OpType for max_backward".into(),
                ))
            }
        };

        // 1. Recompute forward output y = max(x)
        let max_values = Self::max(input_storage, axis)?;

        // 2. Broadcast y back to input shape
        let max_broadcast = match axis {
            None => Self::broadcast_to(&max_values, &input_shape)?,
            Some(ax) => {
                let mut expanded_shape = Self::shape(&max_values).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_max = max_values;
                Self::set_shape(&mut reshaped_max, &expanded_shape)?;
                Self::broadcast_to(&reshaped_max, &input_shape)?
            }
        };

        // 3. Create mask: 1.0 where x == max_val
        let mask = Self::equal(input_storage, &max_broadcast)?;

        // 4. Count ties (number of elements equal to max)
        let count_sum_storage = match axis {
            None => {
                let sum_val = Self::sum_all(&mask)?;
                Self::from_vec(vec![sum_val], &[])? // Create scalar storage
            }
            Some(ax) => Self::sum_along_axis(&mask, ax)?,
        };

        // 5. Broadcast count back to input shape
        let count_broadcast = match axis {
            None => Self::broadcast_to(&count_sum_storage, &input_shape)?,
            Some(ax) => {
                let mut expanded_shape = Self::shape(&count_sum_storage).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_count = count_sum_storage;
                Self::set_shape(&mut reshaped_count, &expanded_shape)?;
                Self::broadcast_to(&reshaped_count, &input_shape)?
            }
        };

        // 6. Broadcast grad_output back to input shape
        let grad_broadcast = match axis {
            None => Self::broadcast_to(grad_output, &input_shape)?,
            Some(ax) => {
                let mut expanded_shape = Self::shape(grad_output).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_grad = grad_output.clone();
                Self::set_shape(&mut reshaped_grad, &expanded_shape)?;
                Self::broadcast_to(&reshaped_grad, &input_shape)?
            }
        };

        // 7. Compute gradient: grad_input = grad_output_bcast * mask / count_bcast
        let epsilon_storage = Self::from_vec(vec![1e-9], &[])?; // Avoid division by zero
        let safe_count = Self::add(
            &count_broadcast,
            &Self::broadcast_to(&epsilon_storage, &input_shape)?,
        )?;
        let grad_input = Self::div(&Self::mul(&grad_broadcast, &mask)?, &safe_count)?;

        Ok(grad_input)
    }

    fn min_backward(op: &Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error> {
        if op.inputs.len() != 1 {
            return Err(Error::InvalidOperation(
                "min_backward expects 1 input".into(),
            ));
        }
        let input_data_ref = op.inputs[0].data();
        let input_storage = &*input_data_ref;
        let input_shape = Self::shape(input_storage).to_vec();

        let axis = match op.op_type {
            OpType::Min(axis) => axis,
            _ => {
                return Err(Error::InternalLogicError(
                    "Incorrect OpType for min_backward".into(),
                ))
            }
        };

        // 1. Recompute forward output y = min(x)
        let min_values = Self::min(input_storage, axis)?;

        // 2. Broadcast y back to input shape
        let min_broadcast = match axis {
            None => Self::broadcast_to(&min_values, &input_shape)?,
            Some(ax) => {
                let mut expanded_shape = Self::shape(&min_values).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_min = min_values;
                Self::set_shape(&mut reshaped_min, &expanded_shape)?;
                Self::broadcast_to(&reshaped_min, &input_shape)?
            }
        };

        // 3. Create mask: 1.0 where x == min_val
        let mask = Self::equal(input_storage, &min_broadcast)?;

        // 4. Count ties (number of elements equal to min)
        let count_sum_storage = match axis {
            None => {
                let sum_val = Self::sum_all(&mask)?;
                println!("[CPU min_backward global] Tie count sum: {}", sum_val);
                Self::from_vec(vec![sum_val], &[])? // Create scalar storage
            }
            Some(ax) => Self::sum_along_axis(&mask, ax)?,
        };

        // 5. Broadcast count back to input shape
        let count_broadcast = match axis {
            None => {
                println!(
                    "[CPU min_backward global] Broadcasting count {:?} to {:?}",
                    Self::shape(&count_sum_storage),
                    input_shape
                );
                Self::broadcast_to(&count_sum_storage, &input_shape)?
            }
            Some(ax) => {
                let mut expanded_shape = Self::shape(&count_sum_storage).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_count = count_sum_storage;
                Self::set_shape(&mut reshaped_count, &expanded_shape)?;
                Self::broadcast_to(&reshaped_count, &input_shape)?
            }
        };

        // 6. Broadcast grad_output back to input shape
        let grad_broadcast = match axis {
            None => {
                println!(
                    "[CPU min_backward global] Broadcasting grad_output {:?} to {:?}",
                    Self::shape(grad_output),
                    input_shape
                );
                Self::broadcast_to(grad_output, &input_shape)?
            }
            Some(ax) => {
                let mut expanded_shape = Self::shape(grad_output).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_grad = grad_output.clone();
                Self::set_shape(&mut reshaped_grad, &expanded_shape)?;
                Self::broadcast_to(&reshaped_grad, &input_shape)?
            }
        };

        // 7. Compute gradient: grad_input = grad_output_bcast * mask / count_bcast
        let epsilon_storage = Self::from_vec(vec![1e-9], &[])?; // Avoid division by zero
        let safe_count = Self::add(
            &count_broadcast,
            &Self::broadcast_to(&epsilon_storage, &input_shape)?,
        )?;
        println!("[CPU min_backward global] Calculating: (grad_bcast * mask) / safe_count");
        let term1 = Self::mul(&grad_broadcast, &mask)?;
        let grad_input = Self::div(&term1, &safe_count)?;
        if let Ok(gi_vec) = Self::copy_to_host(&grad_input) {
            println!(
                "[CPU min_backward global] Final grad_input sample: {:?}",
                &gi_vec[..gi_vec.len().min(10)]
            );
        }
        Ok(grad_input)
    }

    fn prod_backward(op: &Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error> {
        if op.inputs.len() != 1 {
            return Err(Error::InvalidOperation(
                "prod_backward expects 1 input".into(),
            ));
        }
        let input_data_ref = op.inputs[0].data();
        let input_storage = &*input_data_ref;
        let input_shape = Self::shape(input_storage).to_vec();

        let axis = match op.op_type {
            OpType::Prod(axis) => axis,
            _ => {
                return Err(Error::InternalLogicError(
                    "Incorrect OpType for prod_backward".into(),
                ))
            }
        };

        // 1. Recompute forward output y = prod(x)
        let prod_values = Self::prod(input_storage, axis)?;

        // 2. Broadcast y back to input shape
        let prod_broadcast = match axis {
            None => Self::broadcast_to(&prod_values, &input_shape)?,
            Some(ax) => {
                let mut expanded_shape = Self::shape(&prod_values).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_prod = prod_values;
                Self::set_shape(&mut reshaped_prod, &expanded_shape)?;
                Self::broadcast_to(&reshaped_prod, &input_shape)?
            }
        };

        // 3. Broadcast grad_output back to input shape
        let grad_broadcast = match axis {
            None => Self::broadcast_to(grad_output, &input_shape)?,
            Some(ax) => {
                let mut expanded_shape = Self::shape(grad_output).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_grad = grad_output.clone();
                Self::set_shape(&mut reshaped_grad, &expanded_shape)?;
                Self::broadcast_to(&reshaped_grad, &input_shape)?
            }
        };

        // 4. Compute gradient: grad_input = grad_output_bcast * prod_broadcast / x
        let epsilon_storage = Self::from_vec(vec![1e-10], &[])?; // Tiny value
        let safe_input = Self::add(
            input_storage,
            &Self::broadcast_to(&epsilon_storage, &input_shape)?,
        )?;
        let grad_input = Self::div(&Self::mul(&grad_broadcast, &prod_broadcast)?, &safe_input)?;

        Ok(grad_input)
    }

    fn logsumexp_backward(
        op: &Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        if op.inputs.len() != 1 {
            return Err(Error::InvalidOperation(
                "logsumexp_backward expects 1 input".into(),
            ));
        }
        let input_data_ref = op.inputs[0].data();
        let input_storage = &*input_data_ref;
        let input_shape = Self::shape(input_storage).to_vec();

        let axis = match op.op_type {
            OpType::LogSumExp(axis) => axis,
            _ => {
                return Err(Error::InternalLogicError(
                    "Incorrect OpType for logsumexp_backward".into(),
                ))
            }
        };

        // 1. Recompute forward output y = logsumexp(x)
        let lse_values = Self::logsumexp(input_storage, axis)?;

        // 2. Broadcast y back to input shape
        let lse_broadcast = match axis {
            None => Self::broadcast_to(&lse_values, &input_shape)?,
            Some(ax) => {
                let mut expanded_shape = Self::shape(&lse_values).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_lse = lse_values;
                Self::set_shape(&mut reshaped_lse, &expanded_shape)?;
                Self::broadcast_to(&reshaped_lse, &input_shape)?
            }
        };

        // 3. Broadcast grad_output back to input shape
        let grad_broadcast = match axis {
            None => Self::broadcast_to(grad_output, &input_shape)?,
            Some(ax) => {
                let mut expanded_shape = Self::shape(grad_output).to_vec();
                if ax < input_shape.len() {
                    expanded_shape.insert(ax, 1);
                } else {
                    return Err(Error::InvalidIndex(vec![ax]));
                }
                let mut reshaped_grad = grad_output.clone();
                Self::set_shape(&mut reshaped_grad, &expanded_shape)?;
                Self::broadcast_to(&reshaped_grad, &input_shape)?
            }
        };

        // 4. Compute softmax = exp(x - lse_broadcast)
        let shifted = Self::sub(input_storage, &lse_broadcast)?;
        let softmax = Self::exp(&shifted)?;

        // 5. Compute gradient: grad_input = grad_output_bcast * softmax
        let grad_input = Self::mul(&grad_broadcast, &softmax)?;

        Ok(grad_input)
    }
    // --- END: Added Placeholders for Task 2.6 ---

    // --- Added equal implementation ---
    fn equal(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        let a_data = a.get_data();
        let b_data = b.get_data();
        let ndim = a_data.ndim().max(b_data.ndim());
        let mut a_adjusted = a_data.view();
        for _ in 0..(ndim - a_data.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }
        let mut b_adjusted = b_data.view();
        for _ in 0..(ndim - b_data.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }
        let broadcast_shape = crate::util::broadcast_shapes(a_data.shape(), b_data.shape())?;
        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "equal_broadcast_a".to_string(),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            })?;
        let b_broadcast =
            b_adjusted
                .broadcast(broadcast_shape)
                .ok_or_else(|| Error::IncompatibleShapes {
                    op: "equal_broadcast_b".to_string(),
                    shape_a: a_data.shape().to_vec(),
                    shape_b: b_data.shape().to_vec(),
                })?;
        let result_data = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| {
                if (a_val - b_val).abs() < 1e-9 {
                    1.0
                } else {
                    0.0
                }
            });
        Ok(Array::new(result_data))
    }

    fn greater(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        let a_data = a.get_data();
        let b_data = b.get_data();
        let ndim = a_data.ndim().max(b_data.ndim());
        let mut a_adjusted = a_data.view();
        for _ in 0..(ndim - a_data.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }
        let mut b_adjusted = b_data.view();
        for _ in 0..(ndim - b_data.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }
        let broadcast_shape = crate::util::broadcast_shapes(a_data.shape(), b_data.shape())?;
        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "greater_broadcast_a".to_string(),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            })?;
        let b_broadcast =
            b_adjusted
                .broadcast(broadcast_shape)
                .ok_or_else(|| Error::IncompatibleShapes {
                    op: "greater_broadcast_b".to_string(),
                    shape_a: a_data.shape().to_vec(),
                    shape_b: b_data.shape().to_vec(),
                })?;
        let result_data = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| if a_val > b_val { 1.0 } else { 0.0 });
        Ok(Array::new(result_data))
    }

    fn greater_equal(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        let a_data = a.get_data();
        let b_data = b.get_data();
        let ndim = a_data.ndim().max(b_data.ndim());
        let mut a_adjusted = a_data.view();
        for _ in 0..(ndim - a_data.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }
        let mut b_adjusted = b_data.view();
        for _ in 0..(ndim - b_data.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }
        let broadcast_shape = crate::util::broadcast_shapes(a_data.shape(), b_data.shape())?;
        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "greater_equal_broadcast_a".to_string(),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            })?;
        let b_broadcast =
            b_adjusted
                .broadcast(broadcast_shape)
                .ok_or_else(|| Error::IncompatibleShapes {
                    op: "greater_equal_broadcast_b".to_string(),
                    shape_a: a_data.shape().to_vec(),
                    shape_b: b_data.shape().to_vec(),
                })?;
        let result_data = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| if a_val >= b_val { 1.0 } else { 0.0 });
        Ok(Array::new(result_data))
    }

    fn less(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        let a_data = a.get_data();
        let b_data = b.get_data();
        let ndim = a_data.ndim().max(b_data.ndim());
        let mut a_adjusted = a_data.view();
        for _ in 0..(ndim - a_data.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }
        let mut b_adjusted = b_data.view();
        for _ in 0..(ndim - b_data.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }
        let broadcast_shape = crate::util::broadcast_shapes(a_data.shape(), b_data.shape())?;
        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "less_broadcast_a".to_string(),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            })?;
        let b_broadcast =
            b_adjusted
                .broadcast(broadcast_shape)
                .ok_or_else(|| Error::IncompatibleShapes {
                    op: "less_broadcast_b".to_string(),
                    shape_a: a_data.shape().to_vec(),
                    shape_b: b_data.shape().to_vec(),
                })?;
        let result_data = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| if a_val < b_val { 1.0 } else { 0.0 });
        Ok(Array::new(result_data))
    }

    fn less_equal(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        let a_data = a.get_data();
        let b_data = b.get_data();
        let ndim = a_data.ndim().max(b_data.ndim());
        let mut a_adjusted = a_data.view();
        for _ in 0..(ndim - a_data.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }
        let mut b_adjusted = b_data.view();
        for _ in 0..(ndim - b_data.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }
        let broadcast_shape = crate::util::broadcast_shapes(a_data.shape(), b_data.shape())?;
        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "less_equal_broadcast_a".to_string(),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            })?;
        let b_broadcast =
            b_adjusted
                .broadcast(broadcast_shape)
                .ok_or_else(|| Error::IncompatibleShapes {
                    op: "less_equal_broadcast_b".to_string(),
                    shape_a: a_data.shape().to_vec(),
                    shape_b: b_data.shape().to_vec(),
                })?;
        let result_data = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| if a_val <= b_val { 1.0 } else { 0.0 });
        Ok(Array::new(result_data))
    }

    fn not_equal(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        let a_data = a.get_data();
        let b_data = b.get_data();
        let ndim = a_data.ndim().max(b_data.ndim());
        let mut a_adjusted = a_data.view();
        for _ in 0..(ndim - a_data.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }
        let mut b_adjusted = b_data.view();
        for _ in 0..(ndim - b_data.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }
        let broadcast_shape = crate::util::broadcast_shapes(a_data.shape(), b_data.shape())?;
        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "not_equal_broadcast_a".to_string(),
                shape_a: a_data.shape().to_vec(),
                shape_b: b_data.shape().to_vec(),
            })?;
        let b_broadcast =
            b_adjusted
                .broadcast(broadcast_shape)
                .ok_or_else(|| Error::IncompatibleShapes {
                    op: "not_equal_broadcast_b".to_string(),
                    shape_a: a_data.shape().to_vec(),
                    shape_b: b_data.shape().to_vec(),
                })?;
        let result_data = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| {
                if (a_val - b_val).abs() >= 1e-9 {
                    1.0
                } else {
                    0.0
                }
            });
        Ok(Array::new(result_data))
    }

    // --- END: Added equal implementation ---

    // which operate on `&Array`.

    /// 2D max pooling (NCHW). Returns (output, indices)
    fn max_pool2d(
        input: &Self::Storage,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Self::Storage, Self::Storage), Error> {
        cpu_ops::max_pool2d(input, kernel_size, stride, padding)
    }

    /// 2D max pooling backward (NCHW)
    fn max_pool2d_backward(
        op_ctx: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        // Extract the original input and indices from the op context
        if op_ctx.inputs.len() < 2 {
            return Err(Error::InvalidOperation(
                "max_pool2d_backward requires at least 2 inputs (input and indices)".to_string()
            ));
        }
        
        let input = &op_ctx.inputs[0];
        let indices = &op_ctx.inputs[1]; // This contains the indices tensor
        
        // Extract parameters from the op type
        let (kernel_size, stride, padding) = match &op_ctx.op_type {
            crate::graph::OpType::MaxPool2D { kernel_size, stride, padding } => {
                (*kernel_size, *stride, *padding)
            },
            _ => return Err(Error::InternalLogicError(
                "Expected MaxPool2d op type for max_pool2d_backward".to_string()
            )),
        };
        
        // Call the CPU implementation with extracted parameters
        cpu_backward::max_pool2d_backward(
            grad_output,
            &indices.data(),
            &input.shape(),
            kernel_size,
            stride,
            padding,
        )
    }

    fn matmul(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        cpu_ops::matmul(a, b)
    }

    fn conv2d(
        input: &Self::Storage,
        weights: &Self::Storage,
        bias: Option<&Self::Storage>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self::Storage, Error> {
        cpu_ops::conv2d(input, weights, bias, stride, padding)
    }

    fn conv2d_backward(
        input: &Self::Storage,
        weights: &Self::Storage,
        grad_output: &Self::Storage,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Self::Storage, Self::Storage, Option<Self::Storage>), Error> {
        cpu_ops::conv2d_backward(input, weights, grad_output, stride, padding)
    }
    
    fn conv2d_transpose(
        input: &Self::Storage,      // &Array [N, C_in, H_in, W_in]
        weights: &Self::Storage,    // &Array [C_in, C_out, K_h, K_w] (Note: C_in, C_out swapped vs Conv2D)
        bias: Option<&Self::Storage>, // &Array [C_out]
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self::Storage, Error> {
        // Get the shapes of input and weights
        let input_shape = input.shape();
        let weights_shape = weights.shape();
        
        // Validate input dimensions
        if input_shape.len() != 4 {
            return Err(Error::ShapeError(format!("Expected 4D input tensor for conv2d_transpose, got {}D", input_shape.len())));
        }
        
        // Validate weights dimensions
        if weights_shape.len() != 4 {
            return Err(Error::ShapeError(format!("Expected 4D weights tensor for conv2d_transpose, got {}D", weights_shape.len())));
        }
        
        // Extract dimensions
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        
        let weight_in_channels = weights_shape[0];
        let weight_out_channels = weights_shape[1];
        let kernel_height = weights_shape[2];
        let kernel_width = weights_shape[3];
        
        // Validate channel dimensions
        if in_channels != weight_in_channels {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, weight_in_channels, input_height, input_width],
                actual: input_shape.to_vec(),
            });
        }
        
        // Validate bias if provided
        if let Some(bias_tensor) = bias {
            let bias_shape = bias_tensor.shape();
            if bias_shape.len() != 1 || bias_shape[0] != weight_out_channels {
                return Err(Error::ShapeMismatch {
                    expected: vec![weight_out_channels],
                    actual: bias_shape.to_vec(),
                });
            }
        }
        
        // Calculate output dimensions
        // For transposed convolution: output_size = (input_size - 1) * stride + kernel_size - 2 * padding
        let output_height = (input_height - 1) * stride.0 + kernel_height - 2 * padding.0;
        let output_width = (input_width - 1) * stride.1 + kernel_width - 2 * padding.1;
        
        // Create output tensor
        let output_shape = [batch_size, weight_out_channels, output_height, output_width];
        let mut output = Self::zeros(&output_shape)?;
        
        // Get data views
        let input_data = input.get_data();
        let weights_data = weights.get_data();
        let output_data_mut = output.get_data_mut();
        
        // For each batch and output channel
        for n in 0..batch_size {
            for out_c in 0..weight_out_channels {
                // For each input channel
                for in_c in 0..in_channels {
                    // For each position in the input
                    for h_in in 0..input_height {
                        for w_in in 0..input_width {
                            // Calculate the corresponding region in the output
                            let h_out_start = h_in * stride.0;
                            let w_out_start = w_in * stride.1;
                            
                            // For each element in the kernel
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    // Calculate output position
                                    let h_out = h_out_start + kh - padding.0;
                                    let w_out = w_out_start + kw - padding.1;
                                    
                                    // Check if the output position is valid
                                    if h_out < output_height && w_out < output_width {
                                        // Get the input value
                                        let input_val = input_data[[n, in_c, h_in, w_in]];
                                        
                                        // Get the weight value (note the flipped indices for transposed conv)
                                        let weight_val = weights_data[[in_c, out_c, kh, kw]];
                                        
                                        // Update the output
                                        output_data_mut[[n, out_c, h_out, w_out]] += input_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Add bias if provided
                if let Some(bias_tensor) = bias {
                    let bias_data = bias_tensor.get_data();
                    let bias_val = bias_data[out_c];
                    
                    // Add bias to each element in this output channel
                    for h_out in 0..output_height {
                        for w_out in 0..output_width {
                            output_data_mut[[n, out_c, h_out, w_out]] += bias_val;
                        }
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    fn conv2d_transpose_backward(
        input: &Self::Storage,
        weights: &Self::Storage,
        grad_output: &Self::Storage,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Self::Storage, Self::Storage, Option<Self::Storage>), Error> {
        // Get the shapes
        let input_shape = input.shape();
        let weights_shape = weights.shape();
        let grad_output_shape = grad_output.shape();
        
        // Validate dimensions
        if input_shape.len() != 4 || weights_shape.len() != 4 || grad_output_shape.len() != 4 {
            return Err(Error::ShapeError("Expected 4D tensors for conv2d_transpose_backward".to_string()));
        }
        
        // Extract dimensions
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];
        
        let weight_in_channels = weights_shape[0];
        let weight_out_channels = weights_shape[1];
        let kernel_height = weights_shape[2];
        let kernel_width = weights_shape[3];
        
        let output_height = grad_output_shape[2];
        let output_width = grad_output_shape[3];
        
        // Validate channel dimensions
        if in_channels != weight_in_channels {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, weight_in_channels, input_height, input_width],
                actual: input_shape.to_vec(),
            });
        }
        
        if grad_output_shape[1] != weight_out_channels {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, weight_out_channels, output_height, output_width],
                actual: grad_output_shape.to_vec(),
            });
        }
        
        // Initialize gradients
        let mut grad_input = Self::zeros(input_shape)?;
        let mut grad_weights = Self::zeros(weights_shape)?;
        let mut grad_bias = if weight_out_channels > 0 {
            Some(Self::zeros(&[weight_out_channels])?)
        } else {
            None
        };
        
        // Get data views
        let input_data = input.get_data();
        let weights_data = weights.get_data();
        let grad_output_data = grad_output.get_data();
        let grad_input_data_mut = grad_input.get_data_mut();
        let grad_weights_data_mut = grad_weights.get_data_mut();
        
        // Calculate gradient w.r.t input (dL/dX)
        // This is similar to a forward convolution with flipped weights
        for n in 0..batch_size {
            for in_c in 0..in_channels {
                for h_in in 0..input_height {
                    for w_in in 0..input_width {
                        // For each output channel
                        for out_c in 0..weight_out_channels {
                            // Calculate the corresponding region in grad_output
                            let h_out_start = h_in * stride.0;
                            let w_out_start = w_in * stride.1;
                            
                            // For each element in the kernel
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    // Calculate grad_output position
                                    let h_out = h_out_start + kh - padding.0;
                                    let w_out = w_out_start + kw - padding.1;
                                    
                                    // Check if the grad_output position is valid
                                    if h_out < output_height && w_out < output_width {
                                        // Get the grad_output value
                                        let grad_val = grad_output_data[[n, out_c, h_out, w_out]];
                                        
                                        // Get the weight value
                                        let weight_val = weights_data[[in_c, out_c, kh, kw]];
                                        
                                        // Update grad_input
                                        grad_input_data_mut[[n, in_c, h_in, w_in]] += grad_val * weight_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Calculate gradient w.r.t weights (dL/dW)
        for in_c in 0..in_channels {
            for out_c in 0..weight_out_channels {
                for kh in 0..kernel_height {
                    for kw in 0..kernel_width {
                        let mut grad_w = 0.0;
                        
                        // Sum over batch and spatial dimensions
                        for n in 0..batch_size {
                            for h_in in 0..input_height {
                                for w_in in 0..input_width {
                                    // Calculate the corresponding position in grad_output
                                    let h_out = h_in * stride.0 + kh - padding.0;
                                    let w_out = w_in * stride.1 + kw - padding.1;
                                    
                                    // Check if the grad_output position is valid
                                    if h_out < output_height && w_out < output_width {
                                        let input_val = input_data[[n, in_c, h_in, w_in]];
                                        let grad_val = grad_output_data[[n, out_c, h_out, w_out]];
                                        
                                        grad_w += input_val * grad_val;
                                    }
                                }
                            }
                        }
                        
                        grad_weights_data_mut[[in_c, out_c, kh, kw]] = grad_w;
                    }
                }
            }
        }
        
        // Calculate gradient w.r.t bias (dL/dB) if needed
        if let Some(ref mut grad_bias_tensor) = grad_bias {
            let grad_bias_data_mut = grad_bias_tensor.get_data_mut();
            
            // Sum gradients over batch and spatial dimensions
            for out_c in 0..weight_out_channels {
                let mut grad_b = 0.0;
                
                for n in 0..batch_size {
                    for h_out in 0..output_height {
                        for w_out in 0..output_width {
                            grad_b += grad_output_data[[n, out_c, h_out, w_out]];
                        }
                    }
                }
                
                grad_bias_data_mut[out_c] = grad_b;
            }
        }
        
        Ok((grad_input, grad_weights, grad_bias))
    }


    fn mul(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        cpu_ops::mul(a, b)
    }

    fn add(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        cpu_ops::add(a, b)
    }

    // Some ops might be simple enough to implement directly using Array methods
    // We assume Array::sub/div/etc. handle potential errors or rely on ndarray's behavior.
    fn sub(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        // Check if Array::sub exists and returns Result, otherwise use underlying ndarray op
        Ok(Array::new(a.get_data() - b.get_data())) // Direct ndarray op
                                                    // a.sub(b) // If Array::sub exists
    }

    fn div(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        // Add check for division by zero? ndarray might return inf/nan.
        Ok(Array::new(a.get_data() / b.get_data())) // Direct ndarray op
                                                    // a.div(b) // If Array::div exists
    }

    fn div_scalar(x: &Self::Storage, scalar: f32) -> Result<Self::Storage, Error> {
        if scalar == 0.0 {
            return Err(Error::InvalidOperation(
                "Division by zero scalar".to_string(),
            ));
        }
        Ok(Array::new(x.get_data() / scalar)) // Direct ndarray op
                                              // x.div_scalar(scalar) // If Array::div_scalar exists
    }

    fn mul_scalar(x: &Self::Storage, scalar: f32) -> Result<Self::Storage, Error> {
        Ok(Array::new(x.get_data() * scalar)) // Direct ndarray op
    }

    fn transpose(x: &Self::Storage) -> Result<Self::Storage, Error> {
        // Create a contiguous transposed array
        let transposed_view = x.get_data().t();
        let new_shape = transposed_view.shape();
        let mut new_array = ndarray::ArrayD::<f32>::zeros(new_shape);
        new_array.assign(&transposed_view);
        Ok(Array::new(new_array))
    }

    fn broadcast_to(x: &Self::Storage, shape: &[usize]) -> Result<Self::Storage, Error> {
        // Use the dedicated helper method in Array
        x.broadcast_to(shape)
    }

    fn exp(x: &Self::Storage) -> Result<Self::Storage, Error> {
        Ok(Array::new(x.get_data().mapv(|v| v.exp()))) // Direct ndarray op
                                                       // x.exp() // If Array::exp exists
    }

    fn ln(x: &Self::Storage) -> Result<Self::Storage, Error> {
        // Add checks for non-positive inputs?
        Ok(Array::new(x.get_data().mapv(|v| v.ln()))) // Direct ndarray op
                                                      // x.ln() // If Array::ln exists
    }

    fn map<F>(x: &Self::Storage, f: F) -> Result<Self::Storage, Error>
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static,
    {
        Ok(Array::new(x.get_data().mapv(f))) // Direct ndarray op
                                             // x.map(f) // If Array::map exists
    }

    // --- Reduction Operations ---
    fn sum_along_axis(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error> {
        // Use the dedicated helper method in Array
        x.sum_along_axis(axis)
    }

    fn sum_all(x: &Self::Storage) -> Result<f32, Error> {
        // Use ndarray's sum method directly on the data
        Ok(x.get_data().sum())
    }

    fn max_along_axis(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error> {
        // Use the dedicated helper method in Array
        x.max_along_axis(axis)
    }

    fn mean(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error> {
        use crate::ops::cpu_ops;
        cpu_ops::mean(x, axis) // Delegate to cpu_ops function
    }

    // --- Neural Network Specific Operations ---
    fn relu(x: &Self::Storage) -> Result<Self::Storage, Error> {
        use crate::ops::cpu_ops;
        cpu_ops::relu(x) // Delegate to cpu_ops function
    }

    fn elu(x: &Self::Storage, alpha: f32) -> Result<Self::Storage, Error> {
        let x_data = x.get_data(); // Returns &ArrayD<f32>
        if x_data.is_empty() {
            return Ok(Array::zeros(x.shape())); // Handle empty
        }

        // Apply ELU logic element-wise
        let result_data = x_data.mapv(|val| {
            if val < 0.0 {
                alpha * (val.exp() - 1.0)
            } else {
                val
            }
        });

        Ok(Array::new(result_data))
    }

    fn log_softmax(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error> {
        use crate::ops::cpu_ops;
        cpu_ops::log_softmax(x, axis) // Delegate to cpu_ops function
    }

    fn tanh(x: &Self::Storage) -> Result<Self::Storage, Error> {
        Ok(Array::new(x.get_data().mapv(|v| v.tanh())))
    }

    fn tanh_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        if op.inputs.is_empty() {
            return Err(Error::InvalidOperation(
                "Tanh backward requires 1 input".to_string(),
            ));
        }
        let input_data_ref = op.inputs[0].data();
        let input_data = &*input_data_ref; // x

        // Recompute tanh(x) as 'y'
        let y = Self::tanh(input_data)?;

        // Compute derivative: 1 - y^2 = 1 - tanh(x)^2
        let ones = Self::ones(y.shape())?; // Get ones with correct shape
        let y_squared = Self::mul(&y, &y)?;
        let derivative = Self::sub(&ones, &y_squared)?;

        // grad_input = grad_output * derivative
        Self::mul(output_grad, &derivative)
    }

    // --- Backward Operations ---
    // Delegate to the generic backward functions in `cpu_backward` module.
    // These functions are generic over `B: Backend`, so we specify `CpuBackend`.
    fn matmul_backward(
        op: &Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error> {
        let grads = cpu_backward::matmul_backward::<Self>(op, output_grad)?;
        if grads.len() == 2 {
            let mut iter = grads.into_iter();
            Ok((iter.next().unwrap(), iter.next().unwrap()))
        } else {
            Err(Error::InvalidOperation(
                "Matmul backward returned incorrect number of gradients".to_string(),
            ))
        }
    }

    fn mul_backward(
        op: &Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error> {
        let grads = cpu_backward::mul_backward::<Self>(op, output_grad)?;
        if grads.len() == 2 {
            let mut iter = grads.into_iter();
            Ok((iter.next().unwrap(), iter.next().unwrap()))
        } else {
            Err(Error::InvalidOperation(
                "Mul backward returned incorrect number of gradients".to_string(),
            ))
        }
    }

    fn add_backward(
        op: &Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error> {
        let grads = cpu_backward::add_backward::<Self>(op, output_grad)?;
        if grads.len() == 2 {
            let mut iter = grads.into_iter();
            Ok((iter.next().unwrap(), iter.next().unwrap()))
        } else {
            Err(Error::InvalidOperation(
                "Add backward returned incorrect number of gradients".to_string(),
            ))
        }
    }

    fn mean_backward(op: &Op<Self>, output_grad: &Self::Storage) -> Result<Self::Storage, Error> {
        let grads = cpu_backward::mean_backward::<Self>(op, output_grad)?;
        if grads.len() == 1 {
            Ok(grads.into_iter().next().unwrap())
        } else {
            Err(Error::InternalLogicError(
                "Mean backward returned incorrect number of gradients".to_string(),
            ))
        }
    }

    fn sum_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        if op.inputs.len() != 1 {
            return Err(Error::InvalidOperation(
                "Sum backward expects 1 input".into(),
            ));
        }
        let input_shape = op.inputs[0].shape();
        let axis_opt = match op.op_type {
            OpType::Sum(axis) => axis,
            _ => {
                return Err(Error::InternalLogicError(
                    "Incorrect OpType for sum_backward".into(),
                ))
            }
        };

        let output_grad_ndarray = output_grad.get_data();

        let grad_input_ndarray =
            match axis_opt {
                None => {
                    // Global sum - broadcast scalar to input shape
                    if !output_grad_ndarray.shape().is_empty() {
                        return Err(Error::ShapeMismatch {
                            expected: vec![],
                            actual: output_grad_ndarray.shape().to_vec(),
                        });
                    }
                    let scalar_grad = output_grad_ndarray.iter().next().copied().ok_or(
                        Error::InternalLogicError("Global sum grad is not scalar".into()),
                    )?;
                    ndarray::ArrayD::from_elem(ndarray::IxDyn(&input_shape), scalar_grad)
                }
                Some(axis) => {
                    if axis >= input_shape.len() {
                        return Err(Error::InvalidIndex(vec![axis]));
                    }
                    let mut expanded_grad = output_grad_ndarray.clone();
                    expanded_grad.insert_axis_inplace(ndarray::Axis(axis));
                    match expanded_grad.broadcast(ndarray::IxDyn(&input_shape)) {
                        Some(broadcasted_view) => broadcasted_view.to_owned(),
                        None => {
                            return Err(Error::IncompatibleShapes {
                                op: "sum_backward broadcast (CPU)".to_string(),
                                shape_a: output_grad_ndarray.shape().to_vec(),
                                shape_b: input_shape.to_vec(),
                            })
                        }
                    }
                }
            };
        Ok(Array::new(grad_input_ndarray))
    }

    fn relu_backward(op: &Op<Self>, output_grad: &Self::Storage) -> Result<Self::Storage, Error> {
        if op.inputs.len() != 1 {
            return Err(Error::InvalidOperation("Relu requires 1 input".to_string()));
        }
        let grads = cpu_backward::relu_backward::<Self>(op, output_grad)?;
        if grads.len() != 1 {
            return Err(Error::InternalLogicError(
                "Relu backward returned incorrect number of gradients".to_string(),
            ));
        }
        Ok(grads[0].clone())
    }

    fn elu_backward(op: &Op<Self>, output_grad: &Self::Storage) -> Result<Self::Storage, Error> {
        if op.inputs.len() != 1 {
            return Err(Error::InvalidOperation(
                "ELU backward expects 1 input".to_string(),
            ));
        }

        // Extract alpha from the OpType
        let alpha = match op.op_type {
            OpType::Elu(a) => a,
            _ => {
                return Err(Error::InternalLogicError(
                    "Incorrect OpType for ELU backward".to_string(),
                ))
            }
        };

        let x_ref = op.inputs[0].data(); // Ref<Array>
        let x = &*x_ref; // &Array
        let x_data = x.get_data(); // &ArrayD<f32>
        let grad_output_data = output_grad.get_data(); // &ArrayD<f32>

        if x_data.is_empty() {
            return Ok(Array::zeros(x.shape())); // Handle empty
        }

        // Ensure shapes match for element-wise operation
        if x_data.shape() != grad_output_data.shape() {
            return Err(Error::ShapeMismatch {
                expected: x_data.shape().to_vec(),
                actual: grad_output_data.shape().to_vec(),
            });
        }

        // Calculate derivative: alpha * exp(x) if x < 0 else 1.0
        // Multiply by output_grad element-wise
        let mut grad_input_data = ndarray::ArrayD::<f32>::zeros(x_data.raw_dim());
        ndarray::Zip::from(&mut grad_input_data)
            .and(x_data)
            .and(grad_output_data)
            .for_each(|grad_in, &x_val, &grad_out| {
                let derivative = if x_val < 0.0 {
                    alpha * x_val.exp()
                } else {
                    1.0
                };
                *grad_in = grad_out * derivative;
            });

        Ok(Array::new(grad_input_data))
    }

    fn log_softmax_backward(
        op: &Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        // Returns single gradient
        let grads = cpu_backward::log_softmax_backward::<Self>(op, output_grad)?;
        if grads.len() == 1 {
            Ok(grads.into_iter().next().unwrap())
        } else {
            Err(Error::InvalidOperation(
                "LogSoftmax backward returned incorrect number of gradients".to_string(),
            ))
        }
    }

    fn div_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error> {
        if op.inputs.len() != 2 {
            return Err(Error::InvalidOperation("Div requires 2 inputs".to_string()));
        }
        let a = &*op.inputs[0].data();
        let b = &*op.inputs[1].data();

        // For division a/b:
        // da = dout * (1/b)
        // db = dout * (-a/b^2)
        let ones = &Self::ones(b.shape())?;
        let reciprocal_b = Self::div(ones, b)?;
        let b_squared = Self::mul(b, b)?;
        let neg_a_over_b_squared = Self::div_scalar(&Self::div(a, &b_squared)?, -1.0)?;

        // Calculate gradients
        let grad_a = Self::mul(output_grad, &reciprocal_b)?;
        let grad_b = Self::mul(output_grad, &neg_a_over_b_squared)?;

        Ok((grad_a, grad_b))
    }

    fn sub_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error> {
        if op.inputs.len() != 2 {
            return Err(Error::InvalidOperation("Sub requires 2 inputs".to_string()));
        }

        // For subtraction a-b:
        // da = dout
        // db = -dout
        let grad_a = output_grad.clone();
        let grad_b = Self::div_scalar(output_grad, -1.0)?;

        Ok((grad_a, grad_b))
    }

    fn exp_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        if op.inputs.is_empty() {
            return Err(Error::InvalidOperation("Exp requires 1 input".to_string()));
        }

        // For exp(x), the gradient is: grad_in = grad_out * exp(x)
        let x = &*op.inputs[0].data();
        let exp_x = Self::exp(x)?;
        Self::mul(output_grad, &exp_x)
    }

    fn ln_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        if op.inputs.is_empty() {
            return Err(Error::InvalidOperation("Ln requires 1 input".to_string()));
        }

        // For ln(x), the gradient is: grad_in = grad_out * (1/x)
        let x = &*op.inputs[0].data();
        let ones = &Self::ones(x.shape())?;
        let reciprocal_x = Self::div(ones, x)?;
        Self::mul(output_grad, &reciprocal_x)
    }

    fn abs(x: &Self::Storage) -> Result<Self::Storage, Error> {
        Ok(Array::new(x.get_data().mapv(|v| v.abs())))
    }

    fn abs_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        if op.inputs.is_empty() {
            return Err(Error::InvalidOperation("Abs requires 1 input".to_string()));
        }
        let input_data_ref = op.inputs[0].data();
        let input_data = &*input_data_ref;
        let sign_mask = input_data.get_data().mapv(|v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        });
        let sign_mask_storage = Array::new(sign_mask);
        Self::mul(output_grad, &sign_mask_storage)
    }

    fn sigmoid(x: &Self::Storage) -> Result<Self::Storage, Error> {
        Ok(Array::new(x.get_data().mapv(|v| 1.0 / (1.0 + (-v).exp()))))
    }

    fn sigmoid_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        if op.inputs.is_empty() {
            return Err(Error::InvalidOperation(
                "Sigmoid requires 1 input".to_string(),
            ));
        }
        let input_data_ref = op.inputs[0].data();
        let input_data = &*input_data_ref;

        // Recompute sigmoid output y = sigmoid(x)
        let y = Self::sigmoid(input_data)?;

        // Compute derivative y * (1 - y)
        let ones = Self::ones(y.shape())?;
        let one_minus_y = Self::sub(&ones, &y)?;
        let derivative = Self::mul(&y, &one_minus_y)?;

        // grad_input = grad_output * derivative
        Self::mul(output_grad, &derivative)
    }

    fn sqrt(x: &Self::Storage) -> Result<Self::Storage, Error> {
        // Check for negative values
        if x.get_data().iter().any(|&v| v < 0.0) {
            return Err(Error::InvalidOperation("sqrt: input contains negative values".to_string()));
        }
        Ok(Array::new(x.get_data().mapv(|v| v.sqrt())))
    }

    fn sqrt_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        // The gradient for sqrt(x) is grad_output * 0.5 / sqrt(x)
        let input = &op.inputs[0].data();
        let sqrt_input = input.get_data().mapv(|v| v.sqrt());
        let grad = grad_output.get_data();
        let result = grad * &sqrt_input.mapv(|v| 0.5 / v);
        Ok(Array::new(result))
    }

    fn softplus(x: &Self::Storage) -> Result<Self::Storage, Error> {
        // Simple formula: log(1 + exp(x)). May be unstable for large x.
        // A more stable version: x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
        // For now, using the simpler version as requested.
        Ok(Array::new(x.get_data().mapv(|val| (1.0 + val.exp()).ln())))
    }

    fn softplus_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        if op.inputs.is_empty() {
            return Err(Error::InvalidOperation(
                "Softplus backward requires 1 input".to_string(),
            ));
        }
        let input_data_ref = op.inputs[0].data();
        let input_data = &*input_data_ref; // x

        // Derivative is sigmoid(x) = 1 / (1 + exp(-x))
        let sigmoid_x = input_data.get_data().mapv(|val| 1.0 / (1.0 + (-val).exp()));
        let derivative = Array::new(sigmoid_x); // Wrap derivative in Array

        // grad_input = grad_output * derivative
        Self::mul(output_grad, &derivative) // Use Self::mul for potential broadcasting
    }

    fn powf(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        use crate::ops::cpu_ops;
        cpu_ops::powf(a, b)
    }

    fn powf_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error> {
        // Delegate to the specific implementation in cpu_backward module
        cpu_backward::powf_backward(op, output_grad)
    }

    fn square(x: &Self::Storage) -> Result<Self::Storage, Error> {
        // Handle empty tensor case
        if x.size() == 0 {
            return Ok(Array::zeros(x.shape()));
        }

        // Get the data from the input
        let x_data = x.get_data();

        // Efficiently compute element-wise square using ndarray's mapv
        let result_array = x_data.mapv(|val| val * val);

        // Create a new Array from the result
        Ok(Array::new(result_array))
    }

    fn square_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        if op.inputs.len() != 1 {
            return Err(Error::InvalidOperation(format!(
                "square_backward expects 1 input, got {}",
                op.inputs.len()
            )));
        }

        let x_ref = op.inputs[0].data();
        let x = &*x_ref;

        // Gradient of square is 2 * x * dL/dy
        // First, multiply x by 2
        let x_data = x.get_data();
        let two_x_data = x_data.mapv(|val| 2.0 * val);
        let two_x = Array::new(two_x_data);

        // Then multiply by output gradient
        Self::mul(output_grad, &two_x)
    }

    fn maximum(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        let a_data = a.get_data();
        let b_data = b.get_data();

        // --- Broadcasting Logic (similar to cpu_ops::add/mul) ---
        let a_shape = a_data.shape();
        let b_shape = b_data.shape();
        let ndim = a_shape.len().max(b_shape.len());

        let a_view = a_data.view();
        let mut a_adjusted = a_view.clone();
        for _ in 0..(ndim - a_view.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }

        let b_view = b_data.view();
        let mut b_adjusted = b_view.clone();
        for _ in 0..(ndim - b_view.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }

        let broadcast_shape = crate::util::broadcast_shapes(a_shape, b_shape)?;

        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "maximum broadcast A".to_string(),
                shape_a: a_shape.to_vec(),
                shape_b: b_shape.to_vec(),
            })?;
        let b_broadcast =
            b_adjusted
                .broadcast(broadcast_shape)
                .ok_or_else(|| Error::IncompatibleShapes {
                    op: "maximum broadcast B".to_string(),
                    shape_a: a_shape.to_vec(),
                    shape_b: b_shape.to_vec(),
                })?;
        // --- End Broadcasting Logic ---

        // --- Element-wise Max ---
        let result_data = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| a_val.max(b_val)); // Use f32::max

        Ok(crate::array::Array::new(result_data))
    }

    fn maximum_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error> {
        if op.inputs.len() != 2 {
            return Err(Error::InvalidOperation(
                "Maximum requires 2 inputs".to_string(),
            ));
        }
        let a_ref = op.inputs[0].data();
        let b_ref = op.inputs[1].data();
        let a = &*a_ref; // &Array
        let b = &*b_ref; // &Array
        let a_shape = Self::shape(a);
        let b_shape = Self::shape(b);
        let a_data = a.get_data(); // &ArrayD
        let b_data = b.get_data(); // &ArrayD
        let output_grad_data = output_grad.get_data(); // &ArrayD

        // --- Broadcasting Logic (similar to forward) ---
        let ndim = a_shape.len().max(b_shape.len());
        let a_view = a_data.view();
        let mut a_adjusted = a_view.clone();
        for _ in 0..(ndim - a_view.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }
        let b_view = b_data.view();
        let mut b_adjusted = b_view.clone();
        for _ in 0..(ndim - b_view.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }
        let broadcast_shape = crate::util::broadcast_shapes(a_shape, b_shape)?;
        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "maximum_backward broadcast A".to_string(),
                shape_a: a_shape.to_vec(),
                shape_b: b_shape.to_vec(),
            })?;
        let b_broadcast = b_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "maximum_backward broadcast B".to_string(),
                shape_a: a_shape.to_vec(),
                shape_b: b_shape.to_vec(),
            })?;
        let output_grad_broadcast = output_grad_data
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "maximum_backward broadcast Grad".to_string(),
                shape_a: Self::shape(output_grad).to_vec(),
                shape_b: broadcast_shape.to_vec(),
            })?;
        // --- End Broadcasting Logic ---

        // --- Compute Masks ---
        let mask_a = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| if a_val >= b_val { 1.0 } else { 0.0 });

        let mask_b = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| if b_val > a_val { 1.0 } else { 0.0 });

        // --- Compute Potentially Broadcasted Gradients ---
        let grad_a_bcast_data = &output_grad_broadcast * &mask_a;
        let grad_b_bcast_data = &output_grad_broadcast * &mask_b;

        let grad_a_bcast = crate::array::Array::new(grad_a_bcast_data);
        let grad_b_bcast = crate::array::Array::new(grad_b_bcast_data);

        // --- Unbroadcast Gradients ---
        // Use the existing helper from cpu_backward
        let grad_a = crate::ops::cpu_backward::unbroadcast::<Self>(grad_a_bcast, a_shape)?;
        let grad_b = crate::ops::cpu_backward::unbroadcast::<Self>(grad_b_bcast, b_shape)?;

        Ok((grad_a, grad_b))
    }

    fn minimum(a: &Self::Storage, b: &Self::Storage) -> Result<Self::Storage, Error> {
        let a_data = a.get_data();
        let b_data = b.get_data();

        // --- Broadcasting Logic (similar to maximum) ---
        let a_shape = a_data.shape();
        let b_shape = b_data.shape();
        let ndim = a_shape.len().max(b_shape.len());

        let a_view = a_data.view();
        let mut a_adjusted = a_view.clone();
        for _ in 0..(ndim - a_view.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }

        let b_view = b_data.view();
        let mut b_adjusted = b_view.clone();
        for _ in 0..(ndim - b_view.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }

        let broadcast_shape = crate::util::broadcast_shapes(a_shape, b_shape)?;

        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "minimum broadcast A".to_string(),
                shape_a: a_shape.to_vec(),
                shape_b: b_shape.to_vec(),
            })?;
        let b_broadcast =
            b_adjusted
                .broadcast(broadcast_shape)
                .ok_or_else(|| Error::IncompatibleShapes {
                    op: "minimum broadcast B".to_string(),
                    shape_a: a_shape.to_vec(),
                    shape_b: b_shape.to_vec(),
                })?;
        // --- End Broadcasting Logic ---

        // --- Element-wise Min ---
        let result_data = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| a_val.min(b_val)); // Use f32::min

        Ok(crate::array::Array::new(result_data))
    }

    fn minimum_backward(
        op: &crate::graph::Op<Self>,
        output_grad: &Self::Storage,
    ) -> Result<(Self::Storage, Self::Storage), Error> {
        if op.inputs.len() != 2 {
            return Err(Error::InvalidOperation(
                "Minimum backward requires 2 inputs".to_string(),
            ));
        }
        let a_ref = op.inputs[0].data();
        let b_ref = op.inputs[1].data();
        let a = &*a_ref; // &Array
        let b = &*b_ref; // &Array
        let a_shape = Self::shape(a);
        let b_shape = Self::shape(b);
        let a_data = a.get_data(); // &ArrayD
        let b_data = b.get_data(); // &ArrayD
        let output_grad_data = output_grad.get_data(); // &ArrayD

        // --- Broadcasting Logic (similar to maximum_backward) ---
        let ndim = a_shape.len().max(b_shape.len());
        let a_view = a_data.view();
        let mut a_adjusted = a_view.clone();
        for _ in 0..(ndim - a_view.ndim()) {
            a_adjusted = a_adjusted.insert_axis(ndarray::Axis(0));
        }
        let b_view = b_data.view();
        let mut b_adjusted = b_view.clone();
        for _ in 0..(ndim - b_view.ndim()) {
            b_adjusted = b_adjusted.insert_axis(ndarray::Axis(0));
        }
        let broadcast_shape = crate::util::broadcast_shapes(a_shape, b_shape)?;
        let a_broadcast = a_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "minimum_backward broadcast A".to_string(),
                shape_a: a_shape.to_vec(),
                shape_b: b_shape.to_vec(),
            })?;
        let b_broadcast = b_adjusted
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "minimum_backward broadcast B".to_string(),
                shape_a: a_shape.to_vec(),
                shape_b: b_shape.to_vec(),
            })?;
        let output_grad_broadcast = output_grad_data
            .broadcast(broadcast_shape.clone())
            .ok_or_else(|| Error::IncompatibleShapes {
                op: "minimum_backward broadcast Grad".to_string(),
                shape_a: Self::shape(output_grad).to_vec(),
                shape_b: broadcast_shape.to_vec(),
            })?;
        // --- End Broadcasting Logic ---

        // --- Compute Masks ---
        // Gradient flows to 'a' if a <= b
        let mask_a = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| if a_val <= b_val { 1.0 } else { 0.0 });

        // Gradient flows to 'b' if b < a
        let mask_b = ndarray::Zip::from(&a_broadcast)
            .and(&b_broadcast)
            .map_collect(|&a_val, &b_val| if b_val < a_val { 1.0 } else { 0.0 });

        // --- Compute Potentially Broadcasted Gradients ---
        let grad_a_bcast_data = &output_grad_broadcast * &mask_a;
        let grad_b_bcast_data = &output_grad_broadcast * &mask_b;

        let grad_a_bcast = crate::array::Array::new(grad_a_bcast_data);
        let grad_b_bcast = crate::array::Array::new(grad_b_bcast_data);

        // --- Unbroadcast Gradients ---
        // Use the existing helper from cpu_backward
        let grad_a = crate::ops::cpu_backward::unbroadcast::<Self>(grad_a_bcast, a_shape)?;
        let grad_b = crate::ops::cpu_backward::unbroadcast::<Self>(grad_b_bcast, b_shape)?;

        Ok((grad_a, grad_b))
    }

    /// Applies the sine function element-wise: sin(x).
    fn sin(x: &Self::Storage) -> Result<Self::Storage, Error> {
        // Get a reference to the ndarray data
        let x_data = x.get_data();
        
        // Apply sin element-wise using ndarray's map
        let result_data = x_data.mapv(|v| v.sin());
        
        // Wrap the result in our Array type
        Ok(crate::array::Array::new(result_data))
    }

    /// Backward pass for sine activation.
    ///
    /// The derivative of the sine function is given by d(sin(x))/dx = cos(x).
    /// This function computes the gradient of the sine activation with respect to its input.
    fn sin_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        // Validate inputs
        if op.inputs.len() != 1 {
            return Err(Error::InternalLogicError(format!(
                "sin_backward expects 1 input, got {}",
                op.inputs.len()
            )));
        }

        // Get the input tensor
        let x = &*op.inputs[0].data();
        let x_data = x.get_data();
        
        // Get the gradient output
        let grad_output_data = grad_output.get_data();
        
        // Compute cos(x) * grad_output
        let result_data = ndarray::Zip::from(x_data)
            .and(grad_output_data)
            .map_collect(|&x_val, &grad_val| x_val.cos() * grad_val);
        
        // Wrap the result in our Array type
        Ok(crate::array::Array::new(result_data))
    }
    
    /// Applies the cosine function element-wise: cos(x).
    fn cos(x: &Self::Storage) -> Result<Self::Storage, Error> {
        // Get a reference to the ndarray data
        let x_data = x.get_data();
        
        // Apply cos element-wise using ndarray's map
        let result_data = x_data.mapv(|v| v.cos());
        
        // Wrap the result in our Array type
        Ok(crate::array::Array::new(result_data))
    }

    /// Backward pass for cosine activation.
    ///
    /// The derivative of the cosine function is given by d(cos(x))/dx = -sin(x).
    /// This function computes the gradient of the cosine activation with respect to its input.
    fn cos_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        // Validate inputs
        if op.inputs.len() != 1 {
            return Err(Error::InternalLogicError(format!(
                "cos_backward expects 1 input, got {}",
                op.inputs.len()
            )));
        }

        // Get the input tensor
        let x = &*op.inputs[0].data();
        let x_data = x.get_data();
        
        // Get the gradient output
        let grad_output_data = grad_output.get_data();
        
        // Compute -sin(x) * grad_output
        let result_data = ndarray::Zip::from(x_data)
            .and(grad_output_data)
            .map_collect(|&x_val, &grad_val| -x_val.sin() * grad_val);
        
        // Wrap the result in our Array type
        Ok(crate::array::Array::new(result_data))
    }
    
    /// Applies the tangent function element-wise: tan(x).
    fn tan(x: &Self::Storage) -> Result<Self::Storage, Error> {
        // Get a reference to the ndarray data
        let x_data = x.get_data();
        
        // Apply tan element-wise using ndarray's map
        let result_data = x_data.mapv(|v| v.tan());
        
        // Wrap the result in our Array type
        Ok(crate::array::Array::new(result_data))
    }

    /// Backward pass for tangent activation.
    ///
    /// The derivative of the tangent function is given by d(tan(x))/dx = 1 + tan²(x) = 1/cos²(x).
    /// This function computes the gradient of the tangent activation with respect to its input.
    fn tan_backward(
        op: &crate::graph::Op<Self>,
        grad_output: &Self::Storage,
    ) -> Result<Self::Storage, Error> {
        // Validate inputs
        if op.inputs.len() != 1 {
            return Err(Error::InternalLogicError(format!(
                "tan_backward expects 1 input, got {}",
                op.inputs.len()
            )));
        }

        // Get the input tensor
        let x = &*op.inputs[0].data();
        
        // Compute cos(x)
        let cos_x = Self::cos(x)?;
        
        // Compute 1/cos²(x)
        let cos_x_squared = Self::mul(&cos_x, &cos_x)?;
        let ones = Self::ones(cos_x_squared.shape())?;
        let derivative = Self::div(&ones, &cos_x_squared)?;
        
        // Compute grad_output * derivative
        Self::mul(grad_output, &derivative)
    }

    // --- Array Operations ---

    /// Extracts a slice from a tensor along specified dimensions.
    fn slice(x: &Self::Storage, ranges: &[std::ops::Range<usize>]) -> Result<Self::Storage, Error> {
        let x_data = x.get_data(); // &ArrayD<f32>
        if ranges.len() != x_data.ndim() {
            return Err(Error::InvalidOperation("Slice ranges do not match input dimensionality".to_string()));
        }

        // Create a slice for each dimension
        let mut slice_info = Vec::with_capacity(ranges.len());
        for r in ranges {
            slice_info.push(ndarray::SliceInfoElem::from(r.clone()));
        }

        // Perform the slice operation
        let sliced_array = x_data.slice(slice_info.as_slice()).to_owned();
        Ok(Array::new(sliced_array))
    }

    /// Computes the gradient for the slice operation.
    fn slice_backward(op_ctx: &Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error> {
        let (input_shape, ranges) = match &op_ctx.op_type {
            OpType::Slice { input_shape, ranges } => (input_shape, ranges),
            _ => return Err(Error::InternalLogicError("Incorrect OpType for slice_backward".into())),
        };

        // 1. Create a zero tensor with the shape of the original input
        let mut grad_input_data = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(input_shape));

        // 2. Create a slice for each dimension
        let mut slice_info = Vec::with_capacity(ranges.len());
        for r in ranges {
            slice_info.push(ndarray::SliceInfoElem::from(r.clone()));
        }

        // Get a mutable slice view and assign grad_output to it
        let mut slice_view = grad_input_data.slice_mut(slice_info.as_slice());
        slice_view.assign(grad_output.get_data());
        
        Ok(Array::new(grad_input_data))
    }
    
    /// Concatenates multiple tensors along a specified axis.
    fn concat(tensors_data: &[&Self::Storage], axis: usize) -> Result<Self::Storage, Error> {
        if tensors_data.is_empty() {
            return Err(Error::InvalidOperation("Cannot concat empty list of storages".to_string()));
        }
        
        // Get views of all tensors
        let views: Vec<_> = tensors_data.iter().map(|s| s.get_data().view()).collect();
        
        // Use ndarray's concatenate function
        match ndarray::concatenate(ndarray::Axis(axis), &views) {
            Ok(concatenated_array) => Ok(Array::new(concatenated_array)),
            Err(e) => Err(Error::ShapeError(format!("ndarray::concatenate failed: {}", e))),
        }
    }
    
    /// Computes the gradient for the concat operation.
    fn concat_backward(op_ctx: &Op<Self>, grad_output: &Self::Storage) -> Result<Vec<Self::Storage>, Error> {
        let (axis, input_shapes) = match &op_ctx.op_type {
            OpType::Concat { axis, input_shapes } => (*axis, input_shapes),
            _ => return Err(Error::InternalLogicError("Incorrect OpType for concat_backward".into())),
        };

        let grad_output_data = grad_output.get_data(); // &ArrayD<f32>
        let mut grad_inputs = Vec::with_capacity(input_shapes.len());
        let mut current_offset = 0;

        for original_shape in input_shapes {
            let dim_size_along_axis = original_shape[axis];
            
            // Create a slice for each dimension
            let mut slice_info = Vec::with_capacity(grad_output_data.ndim());
            for (d, &len) in grad_output_data.shape().iter().enumerate() {
                if d == axis {
                    slice_info.push(ndarray::SliceInfoElem::from(current_offset..(current_offset + dim_size_along_axis)));
                } else {
                    slice_info.push(ndarray::SliceInfoElem::from(0..len));
                }
            }
            
            // Extract the slice from grad_output
            let grad_slice = grad_output_data.slice(slice_info.as_slice()).to_owned();
            
            // Add the gradient slice to our result
            grad_inputs.push(Array::new(grad_slice));
            current_offset += dim_size_along_axis;
        }
        
        Ok(grad_inputs)
    }

    /// Inserts a new dimension of size 1 at the specified axis.
    fn expand_dims(x: &Self::Storage, axis: usize) -> Result<Self::Storage, Error> {
        let mut new_shape = x.shape().to_vec();
        if axis > new_shape.len() { 
            return Err(Error::InvalidIndex(vec![axis])); 
        }
        new_shape.insert(axis, 1);
        
        // Clone the data
        let mut new_storage = x.clone();
        
        // Reshape the cloned data (metadata-only change for ndarray if data is contiguous)
        new_storage.reshape(&new_shape)?;
        
        Ok(new_storage)
    }

    /// Computes the gradient for the expand_dims operation.
    fn expand_dims_backward(op_ctx: &Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error> {
        let axis = match op_ctx.op_type {
            OpType::ExpandDims { axis } => axis,
            _ => return Err(Error::InternalLogicError("Incorrect OpType for expand_dims_backward".into())),
        };
        
        let original_input_shape = op_ctx.inputs[0].shape(); // This is the target shape for grad_input
        let grad_output_shape = grad_output.shape();
        let mut grad_input_storage = grad_output.clone();

        if grad_output_shape.get(axis).copied() == Some(1) {
            // Simple squeeze: just reshape to original_input_shape
            grad_input_storage.reshape(&original_input_shape)?;
        } else if grad_output_shape.get(axis).is_some() { // Dim exists and is > 1
            // Sum along the expanded axis, then reshape
            let summed_grad = Self::sum_along_axis(grad_output, axis)?;
            grad_input_storage = summed_grad;
            grad_input_storage.reshape(&original_input_shape)?; // Reshape to original
        } else {
            // Axis is out of bounds for grad_output, should not happen with correct forward/backward
            return Err(Error::InternalLogicError("expand_dims_backward: axis out of bounds for grad_output".into()));
        }
        
        Ok(grad_input_storage)
    }

    /// Removes dimensions of size 1 from the tensor.
    fn squeeze(x: &Self::Storage, axis: Option<usize>) -> Result<Self::Storage, Error> {
        let current_shape = x.shape().to_vec();
        
        let new_shape = if let Some(ax) = axis {
            if ax >= current_shape.len() {
                return Err(Error::InvalidIndex(vec![ax]));
            }
            if current_shape[ax] != 1 {
                return Err(Error::InvalidOperation(format!(
                    "Cannot squeeze axis {} of shape {:?} (not size 1)", 
                    ax, current_shape
                )));
            }
            let mut new_shape = current_shape.clone();
            new_shape.remove(ax);
            new_shape
        } else {
            // Remove all dimensions of size 1
            current_shape.into_iter().filter(|&d| d != 1).collect()
        };
        
        // Create a new array with the new shape
        let mut result = x.clone();
        result.reshape(&new_shape)?;
        Ok(result)
    }

    /// Computes the gradient for the squeeze operation.
    fn squeeze_backward(op_ctx: &Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error> {
        let original_input_shape = match &op_ctx.op_type {
            OpType::Squeeze { original_input_shape, .. } => original_input_shape,
            _ => return Err(Error::InternalLogicError("Incorrect OpType".into())),
        };
        
        // Gradient of squeeze is to reshape (expand_dims) grad_output to original input shape.
        let mut new_grad_storage = grad_output.clone();
        new_grad_storage.reshape(original_input_shape)?; // This is set_shape
        
        Ok(new_grad_storage)
    }

    /// Clips the values of a tensor to be within [min_val, max_val].
    fn clip(x: &Self::Storage, min_val: f32, max_val: f32) -> Result<Self::Storage, Error> {
        // Create a new array with the same shape
        let mut result = x.clone();
        
        // Apply the clip operation using mapv
        let _clipped_data = result.get_data_mut().mapv_inplace(|v| v.max(min_val).min(max_val));
        
        Ok(result)
    }

    /// Computes the gradient for the clip operation.
    fn clip_backward(op_ctx: &Op<Self>, grad_output: &Self::Storage) -> Result<Self::Storage, Error> {
        let (min_val, max_val) = match op_ctx.op_type {
            OpType::Clip { min_val, max_val } => (min_val, max_val),
            _ => return Err(Error::InternalLogicError("Incorrect OpType".into())),
        };
        
        let input_data = &op_ctx.inputs[0].data(); // Get the original input tensor
        
        // Create a mask where 1.0 if value is within range, 0.0 otherwise
        let mask = input_data.get_data().mapv(|x_i| {
            if x_i >= min_val && x_i <= max_val { 1.0 } else { 0.0 }
        });
        
        // Multiply grad_output by the mask to get grad_input
        let grad_input_data = grad_output.get_data() * mask;
        
        Ok(Array::new(grad_input_data))
    }
}

// --- Implement Debug, Display, AsRef, AsMut for Array ---
// These are needed to satisfy the Backend::Storage trait bounds for Array.

impl Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Delegate to ndarray's Display implementation for ArrayD
        write!(f, "{}", self.get_data())
    }
}

impl Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Provide a more informative Debug representation
        write!(
            f,
            "Array(shape={:?}, data={:?})",
            self.shape(),
            self.get_data()
        )
    }
}

impl AsRef<[f32]> for Array {
    fn as_ref(&self) -> &[f32] {
        // Use ndarray's as_slice() which panics if not contiguous C-order.
        // Ensure data layout is suitable or handle non-contiguous cases.
        self.get_data()
            .as_slice()
            .expect("CPU Array data is expected to be sliceable")
    }
}

impl AsMut<[f32]> for Array {
    fn as_mut(&mut self) -> &mut [f32] {
        // Use ndarray's as_slice_mut(). Panics if not contiguous C-order.
        self.get_data_mut()
            .as_slice_mut()
            .expect("CPU Array data is expected to be mutably sliceable")
    }
}
