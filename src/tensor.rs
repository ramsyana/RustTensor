#[cfg(feature = "cuda")]
use crate::backend::cuda::CudaBackend;
use crate::{
    backend::{cpu::CpuBackend, Backend},
    error::Error,
    graph::{Op, OpType},
    ops, // Ensure ops is in scope
};
use std::ops::{Add, Div, Mul, Sub};
use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashSet,
    hash::{Hash, Hasher},
    marker::PhantomData,
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn next_id() -> usize {
    TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

use crate::hooks::Hook;

#[cfg(feature = "serialization")]
use serde::{Serialize, Deserialize, Serializer, Deserializer};
#[cfg(feature = "serialization")]
use serde::de;
#[cfg(feature = "serialization")]
use std::fs::File;
#[cfg(feature = "serialization")]
use std::io::{BufReader, BufWriter};
#[cfg(feature = "serialization")]
use std::path::Path;

// Define a serializable representation of a tensor
#[cfg(feature = "serialization")]
#[derive(Serialize, Deserialize)]
struct SerializableTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    grad: Option<Vec<f32>>,
    requires_grad: bool,
    device: crate::Device,
}

// Intermediate representation for cross-device serialization
#[cfg(feature = "serialization")]
#[derive(Serialize, Deserialize, Clone)]
pub struct SerializableTensorIntermediary {
    pub data_vec: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad_vec: Option<Vec<f32>>,
    pub requires_grad: bool,
    pub device_type: String, // "cpu" or "cuda"
}

pub struct TensorData<B: Backend> {
    pub id: usize,
    pub data: B::Storage,
    pub grad: Option<B::Storage>,
    pub requires_grad: bool,
    pub op: Option<Op<B>>,
    pub device: crate::Device,
    pub hooks: Vec<Box<dyn Hook<B>>>,
}

/// A multi-dimensional array with automatic differentiation support.
///
/// The `Tensor` struct is the core type of this library, representing an n-dimensional array
/// that supports automatic differentiation. It can be used with different backends (CPU/GPU)
/// and tracks gradients for optimization.
///
/// # Example
/// ```rust
/// use rust_tensor_lib::{Tensor, CpuBackend, ops::*};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create a tensor with gradient tracking
///     let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
///     
///     // Perform operations
///     let y = exp(&x)?;
///     
///     // Sum to scalar for backward pass
///     let loss = mean(&y, None)?;
///     
///     // Backward pass (no gradient seed needed for scalar)
///     loss.backward()?;
///     
///     // Access gradients
///     println!("Gradient of x: {:?}", x.grad());
///     Ok(())
/// }
/// ```
///
/// # Type Parameters
/// * `B` - The backend type that implements the `Backend` trait
///
/// # Examples
/// ```rust
/// use rust_tensor_lib::{Tensor, CpuBackend, ops::*};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create tensors with gradient tracking
///     let a = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], true)?;
///     let b = Tensor::<CpuBackend>::from_vec(vec![3.0, 4.0], &[2], true)?;
///
///     // Perform operations
///     let c = add(&a, &b)?;
///     
///     // Sum to scalar for backward pass
///     let loss = mean(&c, None)?;
///     loss.backward()?;
///
///     // Access gradients
///     assert!(a.grad().is_some());
///     Ok(())
/// }
/// ```
#[derive(Debug)]
pub struct Tensor<B: Backend> {
    pub(crate) inner: Rc<RefCell<TensorData<B>>>,
    _backend: PhantomData<B>,
}

#[cfg(feature = "serialization")]
impl<B: Backend> Serialize for Tensor<B> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Access the inner data through RefCell
        let inner = self.inner.borrow();
        
        // Create a serializable representation
        let serializable = SerializableTensor {
            // Convert data to Vec<f32> regardless of backend
            data: B::copy_to_host(&inner.data).map_err(|e| {
                serde::ser::Error::custom(format!("Failed to convert tensor data to vec: {}", e))
            })?,
            
            // Store shape
            shape: B::shape(&inner.data).to_vec(),
            
            // Convert grad to Vec<f32> if it exists
            grad: if let Some(ref grad) = inner.grad {
                Some(B::copy_to_host(grad).map_err(|e| {
                    serde::ser::Error::custom(format!("Failed to convert tensor grad to vec: {}", e))
                })?)
            } else {
                None
            },
            
            // Store metadata
            requires_grad: inner.requires_grad,
            device: inner.device,
        };
        
        // Serialize the representation
        serializable.serialize(serializer)
    }
}

#[cfg(feature = "serialization")]
impl<B: Backend> Tensor<B> {
    /// Converts this tensor to a SerializableTensorIntermediary for cross-device serialization
    pub fn to_intermediary(&self) -> Result<SerializableTensorIntermediary, Error> {
        let inner = self.inner.borrow();
        
        // Determine device type string
        let device_type = match inner.device {
            crate::Device::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            crate::Device::Cuda(_) => "cuda",
        }.to_string();
        
        Ok(SerializableTensorIntermediary {
            data_vec: B::copy_to_host(&inner.data)?,
            shape: B::shape(&inner.data).to_vec(),
            grad_vec: if let Some(ref grad) = inner.grad {
                Some(B::copy_to_host(grad)?)
            } else {
                None
            },
            requires_grad: inner.requires_grad,
            device_type,
        })
    }
    
    /// Creates a tensor from a SerializableTensorIntermediary
    pub fn from_intermediary(intermediary: SerializableTensorIntermediary) -> Result<Self, Error> {
        // Create a temporary storage to determine the backend's device type
        let temp_storage = B::zeros(&[1])?;
        let backend_device = B::device(&temp_storage);
        
        // Convert backend device to string representation for comparison
        let expected_device_type = match backend_device {
            crate::Device::Cpu => "cpu",
            #[cfg(feature = "cuda")]
            crate::Device::Cuda(_) => "cuda",
        };
        
        // Check device type compatibility
        if intermediary.device_type != expected_device_type {
            return Err(Error::InternalLogicError(
                format!("Device mismatch: tensor was serialized from '{}' but trying to deserialize to '{}'", 
                        intermediary.device_type, expected_device_type)
            ));
        }
        
        // Use the actual device from the backend
        let device = backend_device;
        
        // Create storage for the main tensor data
        let data = B::from_host_vec(
            intermediary.data_vec,
            &intermediary.shape,
            device,
        )?;
        
        // Create storage for gradient if it exists
        let grad = if let Some(grad_data) = intermediary.grad_vec {
            Some(B::from_host_vec(
                grad_data,
                &intermediary.shape,
                device,
            )?)
        } else {
            None
        };
        
        // Create the tensor data
        let tensor_data = TensorData {
            id: next_id(), // Generate a new ID
            data,
            grad,
            requires_grad: intermediary.requires_grad,
            op: None, // Don't serialize computation graph
            device,
            hooks: Vec::new(), // Don't serialize hooks
        };
        
        // Create and return the tensor
        Ok(Tensor {
            inner: Rc::new(RefCell::new(tensor_data)),
            _backend: PhantomData,
        })
    }
    
    // File I/O methods are defined elsewhere in this file
}

#[cfg(feature = "serialization")]
impl<'de, B: Backend> Deserialize<'de> for Tensor<B> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize into our intermediate representation
        let serializable = SerializableTensor::deserialize(deserializer)?;
        
        // Check device compatibility by creating a small tensor and checking its device
        let temp_storage = B::zeros(&[1])
            .map_err(|e| de::Error::custom(format!("Failed to create temporary storage: {}", e)))?;
        let expected_device = B::device(&temp_storage);
        if serializable.device != expected_device {
            return Err(de::Error::custom(format!(
                "Device mismatch: tensor was serialized with device {:?} but trying to deserialize with {:?}",
                serializable.device, expected_device
            )));
        }
        
        // Create storage for the main tensor data
        let data = B::from_host_vec(
            serializable.data,
            &serializable.shape,
            serializable.device,
        ).map_err(|e| {
            de::Error::custom(format!("Failed to create tensor storage: {}", e))
        })?;
        
        // Create storage for gradient if it exists
        let grad = if let Some(grad_data) = serializable.grad {
            Some(B::from_host_vec(
                grad_data,
                &serializable.shape,
                serializable.device,
            ).map_err(|e| {
                de::Error::custom(format!("Failed to create gradient storage: {}", e))
            })?)
        } else {
            None
        };
        
        // Create the tensor data
        let tensor_data = TensorData {
            id: next_id(), // Generate a new ID
            data,
            grad,
            requires_grad: serializable.requires_grad,
            op: None, // Don't serialize computation graph
            device: serializable.device,
            hooks: Vec::new(), // Don't serialize hooks
        };
        
        // Create and return the tensor
        Ok(Tensor {
            inner: Rc::new(RefCell::new(tensor_data)),
            _backend: PhantomData,
        })
    }
}

impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> PartialEq for Tensor<B> {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl<B: Backend> Eq for Tensor<B> {}

impl<B: Backend> Hash for Tensor<B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

// --- Add Operator Overloading ---
impl<'b, B: Backend> Add<&'b Tensor<B>> for &Tensor<B> {
    type Output = Result<Tensor<B>, Error>;

    /// Performs element-wise addition using the `+` operator.
    /// Calls `rust_tensor_lib::ops::add`.
    ///
    /// # Example
    /// ```rust
    /// use rust_tensor_lib::{Tensor, CpuBackend};
    /// let a = Tensor::<CpuBackend>::from_vec(vec![1.0], &[1], false).unwrap();
    /// let b = Tensor::<CpuBackend>::from_vec(vec![2.0], &[1], false).unwrap();
    /// let c = (&a + &b)?; // c will be a tensor with value [3.0]
    /// # Ok::<(), rust_tensor_lib::Error>(())
    /// ```
    fn add(self, rhs: &'b Tensor<B>) -> Self::Output {
        ops::add(self, rhs)
    }
}

impl<'b, B: Backend> Mul<&'b Tensor<B>> for &Tensor<B> {
    type Output = Result<Tensor<B>, Error>;

    /// Performs element-wise multiplication using the `*` operator.
    /// Calls `rust_tensor_lib::ops::mul`.
    /// Use `ops::matmul` for matrix multiplication.
    ///
    /// # Example
    /// ```rust
    /// use rust_tensor_lib::{Tensor, CpuBackend};
    /// let a = Tensor::<CpuBackend>::from_vec(vec![2.0], &[1], false).unwrap();
    /// let b = Tensor::<CpuBackend>::from_vec(vec![3.0], &[1], false).unwrap();
    /// let c = (&a * &b)?; // c will be a tensor with value [6.0]
    /// # Ok::<(), rust_tensor_lib::Error>(())
    /// ```
    fn mul(self, rhs: &'b Tensor<B>) -> Self::Output {
        ops::mul(self, rhs)
    }
}

impl<'b, B: Backend> Sub<&'b Tensor<B>> for &Tensor<B> {
    type Output = Result<Tensor<B>, Error>;

    /// Performs element-wise subtraction using the `-` operator.
    /// Calls `rust_tensor_lib::ops::sub`.
    ///
    /// # Example
    /// ```rust
    /// use rust_tensor_lib::{Tensor, CpuBackend};
    /// let a = Tensor::<CpuBackend>::from_vec(vec![5.0], &[1], false).unwrap();
    /// let b = Tensor::<CpuBackend>::from_vec(vec![2.0], &[1], false).unwrap();
    /// let c = (&a - &b)?; // c will be a tensor with value [3.0]
    /// # Ok::<(), rust_tensor_lib::Error>(())
    /// ```
    fn sub(self, rhs: &'b Tensor<B>) -> Self::Output {
        ops::sub(self, rhs)
    }
}

impl<'b, B: Backend> Div<&'b Tensor<B>> for &Tensor<B> {
    type Output = Result<Tensor<B>, Error>;

    /// Performs element-wise division using the `/` operator.
    /// Calls `rust_tensor_lib::ops::div`.
    ///
    /// # Example
    /// ```rust
    /// use rust_tensor_lib::{Tensor, CpuBackend};
    /// let a = Tensor::<CpuBackend>::from_vec(vec![6.0], &[1], false).unwrap();
    /// let b = Tensor::<CpuBackend>::from_vec(vec![2.0], &[1], false).unwrap();
    /// let c = (&a / &b)?; // c will be a tensor with value [3.0]
    /// # Ok::<(), rust_tensor_lib::Error>(())
    /// ```
    fn div(self, rhs: &'b Tensor<B>) -> Self::Output {
        ops::div(self, rhs)
    }
}

impl<B: Backend> Tensor<B> {
    #[cfg(feature = "serialization")]
    /// Saves the tensor to a file in a portable format.
    ///
    /// This method serializes the tensor data, shape, gradient (if present),
    /// and other metadata to a file. The computation graph and hooks are not saved.
    ///
    /// # Arguments
    /// * `path` - The path where the tensor should be saved
    ///
    /// # Returns
    /// * `Result<(), Error>` - Ok if successful, Error otherwise
    ///
    /// # Errors
    /// * Returns an error if the file cannot be created or written to
    /// * Returns an error if serialization fails
    ///
    /// # Example
    /// ```rust,no_run
    /// # use rust_tensor_lib::{Tensor, CpuBackend, ops::*};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
    /// x.save_to_file("tensor.bin")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), crate::error::Error> {
        let file = File::create(path).map_err(|e| {
            crate::error::Error::IoErrorString(format!("Failed to create file: {}", e))
        })?;
        let writer = BufWriter::new(file);
        
        #[cfg(feature = "serialization")]
        serde_json::to_writer(writer, self).map_err(|e| {
            crate::error::Error::SerializationError(format!("Failed to serialize tensor: {}", e))
        })?;
        
        Ok(())
    }
    
    #[cfg(feature = "serialization")]
    /// Loads a tensor from a file.
    ///
    /// This method deserializes a tensor that was previously saved with `save_to_file`.
    /// A new tensor ID is generated during loading.
    ///
    /// # Arguments
    /// * `path` - The path to the saved tensor file
    ///
    /// # Returns
    /// * `Result<Tensor<B>, Error>` - The loaded tensor if successful, Error otherwise
    ///
    /// # Errors
    /// * Returns an error if the file cannot be opened or read
    /// * Returns an error if deserialization fails
    ///
    /// # Example
    /// ```rust,no_run
    /// # use rust_tensor_lib::{Tensor, CpuBackend};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let loaded = Tensor::<CpuBackend>::load_from_file("tensor.bin")?;
    /// println!("Loaded tensor shape: {:?}", loaded.shape());
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, crate::error::Error> {
        let file = File::open(path).map_err(|e| {
            crate::error::Error::IoErrorString(format!("Failed to open file: {}", e))
        })?;
        let reader = BufReader::new(file);
        
        #[cfg(feature = "serialization")]
        let tensor = serde_json::from_reader(reader).map_err(|e| {
            crate::error::Error::DeserializationError(format!("Failed to deserialize tensor: {}", e))
        })?;
        
        #[cfg(not(feature = "serialization"))]
        let tensor = {
            return Err(crate::error::Error::InvalidOperation(
                "Serialization feature is not enabled".to_string(),
            ));
        };
        
        Ok(tensor)
    }
    /// Registers a new hook to be run after this tensor's value is computed.
    pub fn register_hook(&self, hook: Box<dyn crate::hooks::Hook<B>>) {
        self.inner.borrow_mut().hooks.push(hook);
    }

    /// Clears all registered hooks from this tensor.
    pub fn clear_hooks(&self) {
        self.inner.borrow_mut().hooks.clear();
    }

    /// Registers a raw closure as a hook.
    /// Accepts any closure or function pointer that implements `for<'a> Fn(&'a Tensor<B>)`.
    pub fn raw_hook<F>(&self, func: F)
    where
        F: for<'a> Fn(&'a Tensor<B>) + Send + Sync + 'static,
    {
        self.register_hook(Box::new(func));
    }

    /// Executes all registered hooks for this tensor.
    /// This should be called by `ops::*` functions after the tensor's data is finalized.
    pub(crate) fn run_hooks(&self) {
        let inner = self.inner.borrow();
        if !inner.hooks.is_empty() {
            for hook in &self.inner.borrow().hooks {
                hook.call(self);
            }
        }
    }
    /// Applies the ReLU activation function element-wise.
    /// Calls `rust_tensor_lib::ops::relu`.
    pub fn relu(&self) -> Result<Tensor<B>, Error> {
        ops::relu(self)
    }

    /// Applies the exponential function element-wise.
    /// Calls `rust_tensor_lib::ops::exp`.
    pub fn exp(&self) -> Result<Tensor<B>, Error> {
        ops::exp(self)
    }

    /// Applies the natural logarithm element-wise.
    /// Calls `rust_tensor_lib::ops::ln`.
    pub fn ln(&self) -> Result<Tensor<B>, Error> {
        ops::ln(self)
    }

    /// Applies the sigmoid activation function element-wise.
    /// Calls `rust_tensor_lib::ops::sigmoid`.
    pub fn sigmoid(&self) -> Result<Tensor<B>, Error> {
        ops::sigmoid(self)
    }

    /// Applies the hyperbolic tangent (tanh) activation function element-wise.
    /// Calls `rust_tensor_lib::ops::tanh`.
    pub fn tanh(&self) -> Result<Tensor<B>, Error> {
        ops::tanh(self)
    }

    /// Applies the softplus activation function element-wise: log(1 + exp(x)).
    /// Calls `rust_tensor_lib::ops::softplus`.
    pub fn softplus(&self) -> Result<Tensor<B>, Error> {
        ops::softplus(self)
    }

    /// Computes the element-wise square root.
    /// Calls `rust_tensor_lib::ops::sqrt`.
    pub fn sqrt(&self) -> Result<Tensor<B>, Error> {
        ops::sqrt(self)
    }

    /// Computes the element-wise square (x^2).
    /// Calls `rust_tensor_lib::ops::square`.
    pub fn square(&self) -> Result<Tensor<B>, Error> {
        ops::square(self)
    }

    /// Computes the element-wise absolute value (|x|).
    /// Calls `rust_tensor_lib::ops::abs`.
    pub fn abs(&self) -> Result<Tensor<B>, Error> {
        ops::abs(self)
    }

    /// Applies a function element-wise using the backend's `map` implementation.
    ///
    /// # ⚠️ IMPORTANT: BREAKS AUTOGRAD GRAPH ⚠️
    /// This operation breaks the automatic differentiation graph. The resulting tensor
    /// will not have gradient tracking, and gradients cannot flow through this operation
    /// during backpropagation. If you need gradients, use built-in operations instead.
    ///
    /// Calls `B::map`.
    ///
    /// # Arguments
    /// * `f`: A closure that takes an `f32` and returns an `f32`. The closure
    ///      must be `Send + Sync + 'static`.
    ///
    /// # Returns
    /// A new `Tensor` containing the results of applying `f` to each element, with
    /// `requires_grad` set to `false` regardless of the input tensor's setting.
    ///
    /// # Example
    /// ```
    /// use rust_tensor_lib::{Tensor, CpuBackend, ops};
    /// 
    /// let x = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true).unwrap();
    /// 
    /// // Apply a custom function (square root) to each element
    /// let y = x.map(|v| v.sqrt()).unwrap();
    /// 
    /// // Note: y will have requires_grad = false
    /// assert_eq!(y.requires_grad(), false);
    /// 
    /// // If we compute backward through another path, x will still get gradients
    /// let z = x.mul_scalar(2.0).unwrap();
    /// 
    /// // Sum to scalar for backward pass
    /// let loss = ops::sum(&z, None).unwrap();
    /// loss.backward().unwrap();
    /// 
    /// // x has gradients from z, but not from y
    /// assert!(x.grad().is_some());
    /// ```
    pub fn map<F>(&self, f: F) -> Result<Tensor<B>, Error>
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static,
    {
        let output_data = B::map(&*self.data(), f)?;
        // map operation typically does not participate in autograd unless specifically designed
        Ok(Tensor::new(output_data, false))
    }

    /// Computes the mean of elements along a given axis or globally.
    /// Calls `rust_tensor_lib::ops::mean`.
    ///
    /// # Arguments
    /// * `axis`: The axis along which to compute the mean. If `None`, computes the global mean.
    pub fn mean(&self, axis: Option<usize>) -> Result<Tensor<B>, Error> {
        ops::mean(self, axis)
    }

    /// Computes the sum of elements along a given axis or globally.
    /// Calls `rust_tensor_lib::ops::sum`.
    ///
    /// # Arguments
    /// * `axis`: The axis along which to compute the sum. If `None`, computes the global sum.
    pub fn sum(&self, axis: Option<usize>) -> Result<Tensor<B>, Error> {
        ops::sum(self, axis)
    }

    /// Computes the maximum of elements along a given axis or globally.
    /// Calls `rust_tensor_lib::ops::max`.
    pub fn max(&self, axis: Option<usize>) -> Result<Tensor<B>, Error> {
        ops::max(self, axis)
    }

    /// Computes the minimum of elements along a given axis or globally.
    /// Calls `rust_tensor_lib::ops::min`.
    pub fn min(&self, axis: Option<usize>) -> Result<Tensor<B>, Error> {
        ops::min(self, axis)
    }

    /// Computes the product of elements along a given axis or globally.
    /// Calls `rust_tensor_lib::ops::prod`.
    pub fn prod(&self, axis: Option<usize>) -> Result<Tensor<B>, Error> {
        ops::prod(self, axis)
    }

    /// Computes the log-sum-exp of elements along a given axis or globally.
    /// Calls `rust_tensor_lib::ops::logsumexp`.
    pub fn logsumexp(&self, axis: Option<usize>) -> Result<Tensor<B>, Error> {
        ops::logsumexp(self, axis)
    }

    /// Returns the indices of the maximum values along an axis. Not differentiable.
    /// Calls `rust_tensor_lib::ops::argmax`.
    pub fn argmax(&self, axis: usize) -> Result<Tensor<B>, Error> {
        ops::argmax(self, axis)
    }

    /// Returns the indices of the minimum values along an axis. Not differentiable.
    /// Calls `rust_tensor_lib::ops::argmin`.
    pub fn argmin(&self, axis: usize) -> Result<Tensor<B>, Error> {
        ops::argmin(self, axis)
    }

    /// Applies the log-softmax function along a specified axis.
    /// Calls `rust_tensor_lib::ops::log_softmax`.
    pub fn log_softmax(&self, axis: usize) -> Result<Tensor<B>, Error> {
        ops::log_softmax(self, axis)
    }

    /// Applies the ELU activation function element-wise.
    /// Calls `rust_tensor_lib::ops::elu`.
    pub fn elu(&self, alpha: f32) -> Result<Tensor<B>, Error> {
        ops::elu(self, alpha)
    }

    /// Transposes the tensor (typically swaps the last two dimensions for 2D).
    /// Calls `rust_tensor_lib::ops::transpose`.
    pub fn transpose(&self) -> Result<Tensor<B>, Error> {
        ops::transpose(self)
    }

    /// Returns a new tensor with the same data but a different compatible shape.
    /// Calls `rust_tensor_lib::ops::view`. The total number of elements must remain the same.
    /// Note: `reshape` might be a better name if it potentially copies data, `view` implies no copy. Let's stick to `view` for now.
    pub fn view(&self, shape: &[usize]) -> Result<Tensor<B>, Error> {
        ops::view(self, shape)
    }

    /// Broadcasts the tensor to a new compatible shape.
    /// Calls `rust_tensor_lib::ops::broadcast_to`.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor<B>, Error> {
        ops::broadcast_to(self, shape)
    }

    /// Alias for `view`. Returns a new tensor with the same data but a different compatible shape.
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor<B>, Error> {
        self.view(shape)
    }
    /// Creates a new tensor from backend storage.
    ///
    /// # Arguments
    /// * `data` - The storage from the backend containing the tensor data
    /// * `requires_grad` - Whether this tensor should track gradients
    ///
    /// # Returns
    /// Creates a new tensor from storage.
    pub fn new(data: B::Storage, requires_grad: bool) -> Self {
        Self {
            inner: Rc::new(RefCell::new(TensorData {
                id: next_id(),
                data: data.clone(),
                grad: None,
                requires_grad,
                op: None,
                device: B::device(&data),
                hooks: Vec::new(),
            })),
            _backend: PhantomData,
        }
    }

    /// Creates a new tensor with an associated operation for autograd
    pub fn new_with_op(
        data: B::Storage,
        requires_grad: bool,
        op_type: Option<OpType>,
        inputs: Vec<Tensor<B>>,
    ) -> Self {
        let tensor = Self::new(data, requires_grad);
        if let Some(op_type) = op_type {
            let op = Op::new(op_type, inputs.clone(), move |op_ctx, grad_output| {
                eprintln!(
                    "[!!!] Inside Op::new backward closure for OpType: {:?} (Tensor ID: {}) [!!!]",
                    op_ctx.op_type,
                    op_ctx.inputs.first().map(|t| t.id()).unwrap_or(usize::MAX)
                );
                match op_ctx.op_type {
                    OpType::Add => {
                        eprintln!("[!!!]   Matching OpType::Add - Attempting to call B::add_backward... [!!!]");
                        B::add_backward(op_ctx, grad_output).map(|(a, b)| vec![a, b])
                    }
                    OpType::Mul => B::mul_backward(op_ctx, grad_output).map(|(a, b)| vec![a, b]),
                    OpType::Matmul => {
                        eprintln!("[!!!]   Matching OpType::Matmul - Attempting to call B::matmul_backward... [!!!]");
                        B::matmul_backward(op_ctx, grad_output).map(|(a, b)| vec![a, b])
                    }
                    OpType::Mean(_) => B::mean_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Relu => B::relu_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::LogSoftmax(_) => B::log_softmax_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Sum(_) => B::sum_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Max(_) => B::max_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Min(_) => B::min_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Prod(_) => B::prod_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::LogSumExp(_) => B::logsumexp_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::ArgMax(_) => Err(Error::Unimplemented("Backward pass for ArgMax is not implemented".to_string())),
                    OpType::ArgMin(_) => Err(Error::Unimplemented("Backward pass for ArgMin is not implemented".to_string())),
                    OpType::Sub => B::sub_backward(op_ctx, grad_output).map(|(a, b)| vec![a, b]),
                    OpType::Div => B::div_backward(op_ctx, grad_output).map(|(a, b)| vec![a, b]),
                    OpType::Exp => B::exp_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Ln => B::ln_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Broadcast => {
                        // For broadcast operation, we need to sum along the broadcasted dimensions
                        // to get back to the original shape
                        let input = &*op_ctx.inputs[0].data();
                        let input_shape = B::shape(input);
                        Ok(vec![crate::ops::cpu_backward::unbroadcast::<B>(
                            grad_output.clone(),
                            input_shape,
                        )?])
                    }
                    OpType::View => {
                        // For view operation, we just need to reshape the gradient back to the input shape
                        let input = &*op_ctx.inputs[0].data();
                        let input_shape = B::shape(input);
                        let mut grad = grad_output.clone();
                        B::set_shape(&mut grad, input_shape)?;
                        Ok(vec![grad])
                    }
                    OpType::Abs => {
                        let grad_x = B::abs_backward(op_ctx, grad_output)?;
                        Ok(vec![grad_x])
                    }
                    OpType::Sigmoid => {
                        let grad_x = B::sigmoid_backward(op_ctx, grad_output)?;
                        Ok(vec![grad_x])
                    }
                    OpType::Tanh => {
                        let grad_x = B::tanh_backward(op_ctx, grad_output)?;
                        Ok(vec![grad_x])
                    }
                    OpType::Sqrt => {
                        let grad_x = B::sqrt_backward(op_ctx, grad_output)?;
                        Ok(vec![grad_x])
                    }
                    OpType::Softplus => B::softplus_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Powf => B::powf_backward(op_ctx, grad_output).map(|(a, b)| vec![a, b]),
                    OpType::Square => B::square_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Maximum => {
                        B::maximum_backward(op_ctx, grad_output).map(|(a, b)| vec![a, b])
                    }
                    OpType::Minimum => {
                        B::minimum_backward(op_ctx, grad_output).map(|(a, b)| vec![a, b])
                    }
                    OpType::Elu(_) => B::elu_backward(op_ctx, grad_output).map(|x| vec![x]),
                    OpType::Transpose => B::transpose(grad_output).map(|x| vec![x]),
                    OpType::Sin => B::sin_backward(op_ctx, grad_output).map(|g| vec![g]),
                    OpType::Cos => Err(Error::Unimplemented("Backward pass for Cos is not yet implemented".to_string())),
                    OpType::Tan => Err(Error::Unimplemented("Backward pass for Tan is not yet implemented".to_string())),
                    OpType::Concat { .. } => B::concat_backward(op_ctx, grad_output),
                    OpType::Conv2d { stride, padding } => {
                        // Ensure op_ctx.inputs has input, weights, and bias (even if bias is dummy)
                        if op_ctx.inputs.len() < 2 {
                            return Err(Error::InternalLogicError("Conv2d backward expects at least 2 inputs (input, weights)".to_string()));
                        }
                        let input_data = &*op_ctx.inputs[0].data();
                        let weights_data = &*op_ctx.inputs[1].data();
                        // Bias is optional, but ops::conv2d should always pass a dummy bias tensor if None
                        // So op_ctx.inputs[2] should always exist
                        let bias_shape = op_ctx.inputs.get(2).map(|b| b.shape());
                        B::conv2d_backward(input_data, weights_data, grad_output, stride, padding)
                            .map(|(gi, gw, gb_opt)| {
                                // Always return 3 grads to match inputs: input, weights, bias
                                if let Some(gb) = gb_opt {
                                    vec![gi, gw, gb]
                                } else if let Some(bias_shape) = bias_shape {
                                    // If backend returns None for bias grad, create a dummy zeros grad
                                    match B::zeros(&bias_shape) {
                                        Ok(dummy_bias_grad) => vec![gi, gw, dummy_bias_grad],
                                        Err(e) => panic!("Failed to create dummy bias grad for conv2d backward: {:?}", e),
                                    }
                                } else {
                                    // Should not happen if ops::conv2d always passes 3 inputs
                                    panic!("Conv2d backward: bias grad missing and bias input missing")
                                }
                            })
                    },
                    OpType::Conv2DTranspose { .. } => {
                        // Currently unimplemented, return error
                        Err(Error::Unimplemented("Backward pass for Conv2DTranspose is not yet implemented".to_string()))
                    },
                    OpType::MaxPool2D { .. } => {
                        // kernel_size, stride, padding are in op_ctx.op_type
                        // op_ctx.inputs[0] is the original input to max_pool2d
                        // op_ctx.inputs[1] must be the indices tensor saved by ops::max_pool2d
                        B::max_pool2d_backward(op_ctx, grad_output)
                            .map(|grad_input| vec![grad_input]) // Expects only grad_input
                    },
                    OpType::Slice { .. } => {
                        // The slice_backward function will use the input_shape and ranges from op_ctx.op_type
                        B::slice_backward(op_ctx, grad_output).map(|grad_input| vec![grad_input])
                    },
                    OpType::ExpandDims { .. } => {
                        // The expand_dims_backward function will use the axis from op_ctx.op_type
                        B::expand_dims_backward(op_ctx, grad_output).map(|grad_input| vec![grad_input])
                    },
                    OpType::Squeeze { .. } => {
                        // The squeeze_backward function will use the original_input_shape from op_ctx.op_type
                        B::squeeze_backward(op_ctx, grad_output).map(|grad_input| vec![grad_input])
                    },
                    OpType::Clip { .. } => {
                        // The clip_backward function will use the min_val and max_val from op_ctx.op_type
                        B::clip_backward(op_ctx, grad_output).map(|grad_input| vec![grad_input])
                    },
                    OpType::DivScalar(_) => {
                        // Currently unimplemented, return error
                        Err(Error::InvalidOperation("Backward pass for DivScalar operation not implemented yet".to_string()))
                    },
                    OpType::MulScalar(scalar) => {
                        // For multiplication by scalar, the gradient is just the scalar itself
                        let grad_x = B::mul_scalar(grad_output, scalar)?;
                        Ok(vec![grad_x])
                    },
                }
            });
            tensor.set_op(op);
        }
        tensor
    }

    /// Creates a new tensor with zeros in all elements.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor to create
    /// * `requires_grad` - Whether this tensor should track gradients
    ///
    /// # Returns
    /// A `Result` containing either the new tensor or an error
    ///
    /// # Errors
    /// Returns an error if the backend fails to create the storage
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Result<Self, Error> {
        Ok(Self::new(B::zeros(shape)?, requires_grad))
    }

    /// Creates a new tensor with ones in all elements.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor to create
    /// * `requires_grad` - Whether this tensor should track gradients
    ///
    /// # Returns
    /// A `Result` containing either the new tensor or an error
    ///
    /// # Errors
    /// Returns an error if the backend fails to create the storage
    pub fn ones(shape: &[usize], requires_grad: bool) -> Result<Self, Error> {
        Ok(Self::new(B::ones(shape)?, requires_grad))
    }

    /// Creates a new tensor from a vector of f32 values.
    ///
    /// # Arguments
    /// * `data` - The vector of values to initialize the tensor with
    /// * `shape` - The desired shape of the tensor
    /// * `requires_grad` - Whether this tensor should track gradients
    ///
    /// # Returns
    /// A `Result` containing either the new tensor or an error
    ///
    /// # Errors
    /// Returns an error if:
    /// * The product of dimensions in `shape` doesn't match `data.len()`
    /// * The backend fails to create the storage
    pub fn from_vec(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Result<Self, Error> {
        Ok(Self::new(B::from_vec(data, shape)?, requires_grad))
    }

    /// Initialize weights using Kaiming uniform initialization via the Backend
    pub fn kaiming_uniform(
        fan_in: usize,
        shape: &[usize],
        requires_grad: bool,
    ) -> Result<Self, Error> {
        let data = B::kaiming_uniform(fan_in, shape)?;
        Ok(Self::new(data, requires_grad))
    }

    /// Gets an immutable reference to the underlying storage.
    ///
    /// # Returns
    /// A `Ref` to the backend's storage type
    pub fn data(&self) -> Ref<'_, B::Storage> {
        Ref::map(self.inner.borrow(), |inner| &inner.data)
    }

    /// Gets a mutable reference to the underlying storage.
    ///
    /// # Warning
    /// Directly mutating data can break the computation graph. Use with caution.
    #[allow(unused)]
    pub(crate) fn data_mut(&self) -> RefMut<'_, B::Storage> {
        RefMut::map(self.inner.borrow_mut(), |inner| &mut inner.data)
    }

    /// Sets the data of this tensor using Backend's set_data. Consumes `data`.
    pub fn set_data(&self, data: B::Storage) -> Result<(), Error> {
        let mut inner = self.inner.borrow_mut();
        B::set_data(&mut inner.data, data)
    }

    /// Gets the shape of the tensor.
    ///
    /// # Returns
    /// A vector containing the dimensions of the tensor
    pub fn shape(&self) -> Vec<usize> {
        let inner = self.inner.borrow();
        B::shape(&inner.data).to_vec()
    }

    /// Gets the total number of elements in the tensor.
    ///
    /// # Returns
    /// The product of all dimensions in the tensor's shape
    pub fn size(&self) -> usize {
        let inner = self.inner.borrow();
        B::size(&inner.data)
    }

    /// Checks if the tensor requires gradient computation.
    ///
    /// # Returns
    /// `true` if the tensor is tracking gradients, `false` otherwise
    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// Gets an immutable reference to the gradient if it exists.
    ///
    /// # Returns
    /// `Some(Ref<B::Storage>)` if the tensor has a gradient, `None` otherwise
    pub fn grad(&self) -> Option<Ref<'_, B::Storage>> {
        Ref::filter_map(self.inner.borrow(), |inner| inner.grad.as_ref()).ok()
    }

    /// Gets a mutable reference (via RefMut) to the gradient storage if it exists.
    #[allow(unused)]
    pub(crate) fn grad_mut(&self) -> Option<RefMut<'_, B::Storage>> {
        RefMut::filter_map(self.inner.borrow_mut(), |inner| inner.grad.as_mut()).ok()
    }

    /// Sets the gradient storage of this tensor. Replaces existing gradient.
    pub fn set_grad(&self, grad: Option<B::Storage>) {
        let mut inner = self.inner.borrow_mut();
        if let Some(ref g) = grad {
            if B::shape(&inner.data) != B::shape(g) {
                panic!(
                    "Gradient shape mismatch for Tensor ID {}: data {:?} vs grad {:?}",
                    inner.id,
                    B::shape(&inner.data),
                    B::shape(g)
                );
            }
        }

        inner.grad = grad;
    }

    /// Clear the gradient of this tensor (sets it to None)
    pub fn zero_grad(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.grad = None;
    }

    /// Add gradient to this tensor's gradient (accumulate if gradient exists).
    pub(crate) fn add_grad(&self, grad_to_add: B::Storage) -> Result<(), Error> {
        // Step 1: Verify shapes and get existing gradient (read-only borrows)
        let (data_shape, existing_grad_opt) = {
            // Scope for borrow
            let inner = self.inner.borrow(); // Immutable borrow
            (B::shape(&inner.data).to_vec(), inner.grad.clone()) // Clone Option<Storage>
        }; // Borrow released

        let grad_shape = B::shape(&grad_to_add).to_vec();
        #[cfg(feature = "cuda")]
        {
            use std::println;
            println!("[add_grad][CUDA] Tensor ID: {}", self.id());
            println!("[add_grad][CUDA] data_shape: {:?}", data_shape);
            println!("[add_grad][CUDA] grad_shape: {:?}", grad_shape);
            if let Some(ref existing_grad) = existing_grad_opt {
                let shape = B::shape(existing_grad);
                println!("[add_grad][CUDA] existing_grad shape: {:?}", shape);
                if let Ok(vec) = B::copy_to_host(existing_grad) {
                    println!(
                        "[add_grad][CUDA] existing_grad sample: {:?} ... {:?}",
                        &vec[..vec.len().min(3)],
                        &vec[vec.len().saturating_sub(3)..]
                    );
                }
            } else {
                println!("[add_grad][CUDA] existing_grad: None");
            }
            if let Ok(vec) = B::copy_to_host(&grad_to_add) {
                println!(
                    "[add_grad][CUDA] grad_to_add sample: {:?} ... {:?}",
                    &vec[..vec.len().min(3)],
                    &vec[vec.len().saturating_sub(3)..]
                );
            }
        }
        if data_shape != grad_shape {
            // Use the more specific Error::IncompatibleShapes
            return Err(Error::IncompatibleShapes {
                op: format!("add_grad on Tensor ID {}", self.id()),
                shape_a: data_shape, // Tensor's data shape
                shape_b: grad_shape, // Incoming gradient shape
            });
        }

        // Step 2: Compute the *new* total gradient value WITHOUT holding borrows on self.inner
        let new_total_grad = match existing_grad_opt {
            Some(existing_grad) => {
                #[cfg(feature = "cuda")]
                println!("[add_grad][CUDA] Calling B::add for accumulation...");
                let result = B::add(&existing_grad, &grad_to_add)?;
                #[cfg(feature = "cuda")]
                {
                    use std::println;
                    if let Ok(vec) = B::copy_to_host(&result) {
                        println!(
                            "[add_grad][CUDA] new_total_grad (accumulated) sample: {:?} ... {:?}",
                            &vec[..vec.len().min(3)],
                            &vec[vec.len().saturating_sub(3)..]
                        );
                    }
                }
                result
            }
            None => {
                let result = grad_to_add.clone();
                #[cfg(feature = "cuda")]
                {
                    use std::println;
                    println!("[add_grad][CUDA] Initializing gradient (no existing grad).");
                    if let Ok(vec) = B::copy_to_host(&result) {
                        println!(
                            "[add_grad][CUDA] new_total_grad (initialized) sample: {:?} ... {:?}",
                            &vec[..vec.len().min(3)],
                            &vec[vec.len().saturating_sub(3)..]
                        );
                    }
                }
                result
            }
        };

        // Step 3: Update the tensor's gradient field (minimal mutable borrow)
        {
            let mut inner = self.inner.borrow_mut(); // Acquire mutable borrow JUST for assignment
            inner.grad = Some(new_total_grad);
        } // Borrow released

        Ok(())
    }

    /// Computes gradients through the computation graph.
    ///
    /// This method performs backpropagation starting from this tensor,
    /// computing gradients for all tensors in the graph that require gradients.
    ///
    /// # Returns
    /// A `Result` indicating success or containing an error
    ///
    /// # Errors
    /// Returns an error if gradient computation fails at any point
    pub fn backward(&self) -> Result<(), Error> {
        if !self.requires_grad() {
            return Ok(()); // No grad required, nothing to do
        }

        // --- Phase 0: Initialize Root Gradient ---
        // Ensure the root node (self) has a gradient to start the process.
        if self.size() == 1 {
            let needs_init = self.inner.borrow().grad.is_none();
            if needs_init {
                // For scalar output, default initial gradient is 1.0
                self.set_grad(Some(B::ones(&[])?));
            }
        } else {
            // For non-scalar output, a gradient must have been provided externally (e.g., seed grad)
            if self.inner.borrow().grad.is_none() {
                return Err(Error::InvalidOperation(
                    "backward() called on non-scalar tensor without pre-existing gradient. Seed gradient required.".to_string()
                ));
            }
        }

        // --- Phase 1: Build Topological Sort ---
        let mut sorted_nodes = self.build_topo_sort(); // Get nodes (leaves first, root last)
        sorted_nodes.reverse(); // Reverse to process from root to leaves

        // Track visited nodes to avoid double accumulation
        let mut visited = HashSet::new();

        // --- Phase 2: Traverse Graph and Propagate Gradients ---
        // Iterate through sorted nodes (now from root towards the leaves)
        for node_tensor in sorted_nodes.iter() {
            let node_id = node_tensor.id();
            if visited.contains(&node_id) {
                continue; // Skip if already processed
            }
            visited.insert(node_id);

            // Get the operation and the *accumulated* output gradient for this node
            let (op_opt, output_grad_opt) = {
                // Scope for borrow
                let inner = node_tensor.inner.borrow(); // Immutable borrow
                (inner.op.clone(), inner.grad.clone()) // Clone Option<Op> and Option<Storage>
            }; // Borrow released

            // Only proceed if there's an operation AND a gradient has flowed to this node
            if let (Some(op), Some(output_grad_data)) = (op_opt, output_grad_opt) {
                // Compute gradients for the inputs of this operation
                let input_grads_vec = (op.backward_fn)(&op, &output_grad_data)?;

                if op.inputs.len() != input_grads_vec.len() {
                    return Err(Error::InvalidOperation(format!(
                        "Backward function for op {} (Node ID: {}) returned {} gradients, expected {}",
                        op.op_type,
                        node_tensor.id(),
                        input_grads_vec.len(),
                        op.inputs.len()
                    )));
                }

                // Accumulate the computed gradients onto the input tensors
                for (input_tensor, grad_storage) in op.inputs.iter().zip(input_grads_vec) {
                    if input_tensor.requires_grad() {
                        input_tensor.add_grad(grad_storage)?; // Accumulate immediately
                    }
                }
            }
        }

        Ok(())
    }

    /// Builds a topologically sorted list of tensors in the graph ending at `self`.
    ///
    /// Uses DFS post-order traversal.
    fn build_topo_sort(&self) -> Vec<Tensor<B>> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        fn visit<B: Backend>(
            node: &Tensor<B>,
            visited: &mut HashSet<usize>,
            sorted: &mut Vec<Tensor<B>>,
            visiting: &mut HashSet<usize>,
        ) {
            let node_id = node.id();
            if visited.contains(&node_id) {
                return;
            }
            if !visiting.insert(node_id) {
                panic!(
                    "Cycle detected in computation graph involving Tensor ID {}",
                    node_id
                );
            }

            if let Some(op) = node.inner.borrow().op.as_ref() {
                for input in &op.inputs {
                    visit(input, visited, sorted, visiting);
                }
            }

            visiting.remove(&node_id);
            visited.insert(node_id);
            sorted.push(node.clone());
        }

        visit(self, &mut visited, &mut sorted, &mut visiting);
        sorted
    }

    /// Gets the unique identifier of this tensor.
    ///
    /// # Returns
    /// The unique ID of this tensor, used for graph operations and equality comparison
    pub fn id(&self) -> usize {
        self.inner.borrow().id
    }

    /// Sets the operation that produced this tensor. Internal use by ops functions.
    pub(crate) fn set_op(&self, op: Op<B>) {
        let mut inner = self.inner.borrow_mut();
        // Only leaf nodes (inputs/parameters) should have op set multiple times (to None usually).
        // If an intermediate node's op is overwritten, it breaks the graph.
        if inner.op.is_some() {
            // This might indicate a logic error where an intermediate tensor is being reused improperly.
            // Consider panicking or logging a warning.
            // panic!("Cannot overwrite existing operation for tensor ID {}", inner.id);
        }
        inner.op = Some(op);
    }

    /// Copies the tensor's data to the CPU, returning a new `Tensor<CpuBackend>`.
    pub fn to_cpu(&self) -> Result<Tensor<CpuBackend>, Error> {
        let inner_borrow = self.inner.borrow();
        let current_data = &inner_borrow.data;
        let requires_grad = inner_borrow.requires_grad;
        let current_shape = B::shape(current_data); // Get shape via Backend trait

        // Use B::copy_to_host, assuming it handles CPU->CPU efficiently (e.g., clone)
        let host_vec = B::copy_to_host(current_data)?;
        let cpu_storage = CpuBackend::from_vec(host_vec, current_shape)?;

        // Drop the borrow before creating the new tensor
        drop(inner_borrow);

        // Create the new CPU tensor
        Ok(Tensor::<CpuBackend>::new(cpu_storage, requires_grad))
    }

    #[cfg(feature = "cuda")]
    /// Copies the tensor's data to the specified CUDA device, returning a new `Tensor<CudaBackend>`.
    /// Note: Currently assumes the global context corresponds to the device_id.
    pub fn to_gpu(&self, _device_id: u32) -> Result<Tensor<CudaBackend>, Error> {
        // NOTE: This method assumes the CUDA context for the target device
        // is already initialized and active *by the caller* (e.g., in main).

        let inner_borrow = self.inner.borrow();
        let current_data = &inner_borrow.data;
        let requires_grad = inner_borrow.requires_grad;
        let current_shape = B::shape(current_data);

        // Assume B::copy_to_host gets data to CPU, then CudaBackend::from_vec moves to GPU
        let host_vec = B::copy_to_host(current_data)?;
        let cuda_storage = CudaBackend::from_vec(host_vec, current_shape)?;

        // Drop borrow
        drop(inner_borrow);

        // Create the new Cuda tensor
        Ok(Tensor::<CudaBackend>::new(cuda_storage, requires_grad))
    }

    /// Returns the size (number of elements) in the tensor
    pub fn len(&self) -> usize {
        self.size()
    }

    /// Checks if the tensor has zero elements.
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Returns the device where the tensor data is located.
    pub fn device(&self) -> crate::Device {
        self.inner.borrow().device
    }

    /// Prints the tensor's ID and shape to standard output.
    /// Returns a clone of the tensor for chaining.
    pub fn show_shape(&self, label: &str) -> Self {
        println!(
            "[Tensor Hook] {}: ID={}, Shape={:?}",
            label,
            self.id(),
            self.shape()
        );
        self.clone()
    }

    /// Prints the tensor's ID, shape, and a sample of its data to standard output.
    /// **Note:** This involves copying data from the device (if applicable), which can incur overhead.
    /// Returns a clone of the tensor for chaining.
    pub fn show(&self, label: &str) -> Self {
        let id = self.id();
        let shape = self.shape(); // Get shape before potential borrow for data
        print!("[Tensor Hook] {}: ID={}, Shape={:?}", label, id, shape);
        match B::copy_to_host(&*self.data()) {
            Ok(data_vec) => {
                let limit = 10; // Limit printed elements
                if data_vec.len() <= limit {
                    println!(", Data={:?}", data_vec);
                } else {
                    println!(
                        ", Data(sample)={:?}...{:?}",
                        &data_vec[..limit / 2],
                        &data_vec[data_vec.len() - limit / 2..]
                    );
                }
            }
            Err(e) => {
                println!(", Data=<Error copying to host: {}>", e);
            }
        }
        self.clone()
    }

    /// Extracts a slice from the tensor along specified dimensions.
    ///
    /// # Arguments
    /// * `ranges` - A slice of ranges, one for each dimension, specifying the slice to extract.
    ///              Each range is in the form `start..end` where `start` is inclusive and `end` is exclusive.
    ///
    /// # Returns
    /// A new tensor containing the sliced data
    ///
    /// # Errors
    /// Returns an error if:
    /// * The number of ranges doesn't match the number of dimensions in the tensor
    /// * Any range is out of bounds for its corresponding dimension
    /// * Any range has start > end
    ///
    /// # Examples
    /// ```
    /// use rust_tensor_lib::{CpuTensor, backend::Backend};
    /// 
    /// let tensor = CpuTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false).unwrap();
    /// // Extract the first row
    /// let slice = tensor.slice(&[0..1, 0..3]).unwrap();
    /// assert_eq!(slice.shape(), &[1, 3]);
    /// let slice_data = rust_tensor_lib::backend::cpu::CpuBackend::copy_to_host(&*slice.data()).unwrap();
    /// assert_eq!(slice_data, vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Tensor<B>, Error> {
        crate::ops::slice(self, ranges)
    }

    /// Inserts a new dimension of size 1 at the specified axis.
    ///
    /// # Arguments
    /// * `axis` - The axis at which to insert the new dimension
    ///
    /// # Returns
    /// A new tensor with an additional dimension of size 1 at the specified axis
    ///
    /// # Errors
    /// Returns an error if the axis is invalid (greater than self.shape().len())
    ///
    /// # Examples
    /// ```
    /// use rust_tensor_lib::{CpuTensor, backend::Backend};
    /// 
    /// let tensor = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], false).unwrap();
    /// // Insert a new dimension at axis 0
    /// let expanded = tensor.expand_dims(0).unwrap();
    /// assert_eq!(expanded.shape(), &[1, 3]);
    /// 
    /// // Insert a new dimension at axis 1
    /// let expanded = tensor.expand_dims(1).unwrap();
    /// assert_eq!(expanded.shape(), &[3, 1]);
    /// ```
    pub fn expand_dims(&self, axis: usize) -> Result<Tensor<B>, Error> {
        crate::ops::expand_dims(self, axis)
    }

    /// Removes dimensions of size 1 from the tensor.
    ///
    /// # Arguments
    /// * `axis` - The axis to squeeze out. If None, all dimensions of size 1 are removed.
    ///
    /// # Returns
    /// A new tensor with the specified dimension(s) of size 1 removed
    ///
    /// # Errors
    /// Returns an error if the axis is invalid or not of size 1
    ///
    /// # Examples
    /// ```
    /// use rust_tensor_lib::{CpuTensor, backend::Backend};
    /// 
    /// let tensor = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3, 1], false).unwrap();
    /// // Squeeze specific axis
    /// let squeezed = tensor.squeeze(Some(0)).unwrap();
    /// assert_eq!(squeezed.shape(), &[3, 1]);
    /// 
    /// // Squeeze all dimensions of size 1
    /// let squeezed = tensor.squeeze(None).unwrap();
    /// assert_eq!(squeezed.shape(), &[3]);
    /// ```
    pub fn squeeze(&self, axis: Option<usize>) -> Result<Tensor<B>, Error> {
        crate::ops::squeeze(self, axis)
    }

    /// Clips the values of a tensor to be within [min_val, max_val].
    ///
    /// # Arguments
    /// * `min_val` - The minimum value to clip to
    /// * `max_val` - The maximum value to clip to
    ///
    /// # Returns
    /// A new tensor with all values clipped to the range [min_val, max_val]
    ///
    /// # Errors
    /// Returns an error if min_val > max_val or if the backend fails to compute the clip
    ///
    /// # Examples
    /// ```
    /// use rust_tensor_lib::{CpuTensor, backend::Backend, backend::cpu::CpuBackend};
    /// 
    /// let tensor = CpuTensor::from_vec(vec![-1.0, 0.5, 2.0], &[3], false).unwrap();
    /// let clipped = tensor.clip(0.0, 1.0).unwrap();
    /// 
    /// // Values are clipped to [0.0, 1.0]
    /// let expected = CpuTensor::from_vec(vec![0.0, 0.5, 1.0], &[3], false).unwrap();
    /// 
    /// // Compare the values in the tensors
    /// let clipped_storage = clipped.data();
    /// let expected_storage = expected.data();
    /// let clipped_data = clipped_storage.get_data();
    /// let expected_data = expected_storage.get_data();
    /// assert_eq!(clipped_data.shape(), expected_data.shape());
    /// 
    /// for (a, b) in clipped_data.iter().zip(expected_data.iter()) {
    ///     assert!((a - b).abs() < 1e-5);
    /// }
    /// ```
    pub fn clip(&self, min_val: f32, max_val: f32) -> Result<Tensor<B>, Error> {
        crate::ops::clip(self, min_val, max_val)
    }
    
    /// Multiplies the tensor by a scalar value element-wise.
    ///
    /// # Arguments
    /// * `scalar` - The scalar value to multiply by
    ///
    /// # Returns
    /// A new tensor with all elements multiplied by the scalar
    ///
    /// # Errors
    /// Returns an error if the backend fails to compute the multiplication
    ///
    /// # Examples
    /// ```
    /// use rust_tensor_lib::{CpuTensor, backend::Backend};
    /// 
    /// let tensor = CpuTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], false).unwrap();
    /// let scaled = tensor.mul_scalar(2.5).unwrap();
    /// 
    /// // Values are multiplied by 2.5
    /// let data_ref = scaled.data();
    /// let scaled_data = data_ref.get_data();
    /// assert!((scaled_data[[0]] - 2.5).abs() < 1e-5);
    /// assert!((scaled_data[[1]] - 5.0).abs() < 1e-5);
    /// assert!((scaled_data[[2]] - 7.5).abs() < 1e-5);
    /// ```
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor<B>, Error> {
        crate::ops::mul_scalar(self, scalar)
    }
}
