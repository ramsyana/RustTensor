use crate::backend::Backend;
use crate::error::Error;
use crate::tensor::Tensor;

use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum OpType {
    Matmul,
    Mul,
    Add,
    Mean(Option<usize>),
    Relu,
    LogSoftmax(usize),
    Sum(Option<usize>),
    Sub,
    Div,
    Exp,
    Ln,
    Broadcast,
    View,
    Abs,
    Sigmoid,
    Sqrt,
    Tanh,
    Softplus,
    // New reduction operations
    Max(Option<usize>),
    Min(Option<usize>),
    Prod(Option<usize>),
    LogSumExp(Option<usize>),
    // Forward-only index operations
    ArgMax(usize),
    ArgMin(usize),
    // New operation
    Powf,
    Square,
    Maximum,
    Minimum,
    Elu(f32),
    Transpose,
    Sin,
    Cos, // Placeholder for cos
    Tan, // Placeholder for tan
    Conv2d { stride: (usize, usize), padding: (usize, usize) },
    MaxPool2D { kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize) },
    Conv2DTranspose { stride: (usize, usize), padding: (usize, usize), output_padding: (usize, usize) },
    // Array operations
    Slice {
        // Store original input shape and ranges for backward pass
        input_shape: Vec<usize>,
        ranges: Vec<std::ops::Range<usize>>,
    },
    Concat {
        axis: usize,
        input_shapes: Vec<Vec<usize>>, // Store shapes of original tensors for backward
    },
    ExpandDims { axis: usize },
    Squeeze { axis: Option<usize>, original_input_shape: Vec<usize> },
    Clip { min_val: f32, max_val: f32 },
    DivScalar(f32),
    MulScalar(f32),
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpType::Matmul => write!(f, "Matmul"),
            OpType::Mul => write!(f, "Mul"),
            OpType::Add => write!(f, "Add"),
            OpType::Mean(None) => write!(f, "Mean(global)"),
            OpType::Mean(Some(axis)) => write!(f, "Mean(axis={})", axis),
            OpType::Relu => write!(f, "ReLU"),
            OpType::LogSoftmax(axis) => write!(f, "LogSoftmax(axis={})", axis),
            OpType::Sum(None) => write!(f, "Sum(global)"),
            OpType::Sum(Some(axis)) => write!(f, "Sum(axis={})", axis),
            OpType::Sub => write!(f, "Sub"),
            OpType::Div => write!(f, "Div"),
            OpType::Exp => write!(f, "Exp"),
            OpType::Ln => write!(f, "Ln"),
            OpType::Broadcast => write!(f, "Broadcast"),
            OpType::View => write!(f, "View"),
            OpType::Abs => write!(f, "Abs"),
            OpType::Sigmoid => write!(f, "Sigmoid"),
            OpType::Sqrt => write!(f, "Sqrt"),
            OpType::Tanh => write!(f, "Tanh"),
            OpType::Softplus => write!(f, "Softplus"),
            OpType::Max(None) => write!(f, "Max(global)"),
            OpType::Max(Some(axis)) => write!(f, "Max(axis={})", axis),
            OpType::Min(None) => write!(f, "Min(global)"),
            OpType::Min(Some(axis)) => write!(f, "Min(axis={})", axis),
            OpType::Prod(None) => write!(f, "Prod(global)"),
            OpType::Prod(Some(axis)) => write!(f, "Prod(axis={})", axis),
            OpType::LogSumExp(None) => write!(f, "LogSumExp(global)"),
            OpType::LogSumExp(Some(axis)) => write!(f, "LogSumExp(axis={})", axis),
            OpType::ArgMax(axis) => write!(f, "ArgMax(axis={})", axis),
            OpType::ArgMin(axis) => write!(f, "ArgMin(axis={})", axis),
            OpType::Powf => write!(f, "Powf"),
            OpType::Square => write!(f, "Square"),
            OpType::Maximum => write!(f, "Maximum"),
            OpType::Minimum => write!(f, "Minimum"),
            OpType::Elu(alpha) => write!(f, "Elu(alpha={})", alpha),
            OpType::Transpose => write!(f, "Transpose"),
            OpType::Sin => write!(f, "Sin"),
            OpType::Cos => write!(f, "Cos"),
            OpType::Tan => write!(f, "Tan"),
            OpType::Conv2d { stride, padding } => write!(f, "Conv2d(stride={:?}, padding={:?})", stride, padding),
            OpType::MaxPool2D { kernel_size, stride, padding } => write!(f, "MaxPool2D(kernel_size={:?}, stride={:?}, padding={:?})", kernel_size, stride, padding),
            OpType::Conv2DTranspose { stride, padding, output_padding } => write!(f, "Conv2DTranspose(stride={:?}, padding={:?}, output_padding={:?})", stride, padding, output_padding),
            OpType::Slice { ranges, .. } => write!(f, "Slice(ranges={:?})", ranges),
            OpType::Concat { axis, .. } => write!(f, "Concat(axis={})", axis),
            OpType::ExpandDims { axis } => write!(f, "ExpandDims(axis={})", axis),
            OpType::Squeeze { axis, .. } => write!(f, "Squeeze(axis={:?})", axis),
            OpType::Clip { min_val, max_val } => write!(f, "Clip(min={}, max={})", min_val, max_val),
            OpType::DivScalar(scalar) => write!(f, "DivScalar(scalar={})", scalar),
            OpType::MulScalar(scalar) => write!(f, "MulScalar(scalar={})", scalar),
        }
    }
}

impl OpType {
    /// Returns the axis for operations that have one (e.g., reduction operations)
    /// Returns None for operations that have no axis or are global reductions
    pub fn get_axis(&self) -> Option<usize> {
        match self {
            OpType::Mean(axis) => *axis,
            OpType::Sum(axis) => *axis,
            OpType::Max(axis) => *axis,
            OpType::Min(axis) => *axis,
            OpType::Prod(axis) => *axis,
            OpType::LogSumExp(axis) => *axis,
            OpType::LogSoftmax(axis) => Some(*axis),
            OpType::ArgMax(axis) => Some(*axis),
            OpType::ArgMin(axis) => Some(*axis),
            OpType::Concat { axis, .. } => Some(*axis),
            OpType::ExpandDims { axis } => Some(*axis),
            OpType::Squeeze { axis, .. } => *axis,
            OpType::Slice { .. } => None,
            OpType::Clip { .. } => None,
            _ => None,
        }
    }
}

#[allow(type_alias_bounds)]
type BackwardFn<B: Backend> = dyn Fn(&Op<B>, &B::Storage) -> Result<Vec<B::Storage>, Error>;

#[derive(Clone)]
pub struct Op<B: Backend> {
    pub op_type: OpType,
    pub inputs: Vec<Tensor<B>>,
    pub backward_fn: Rc<BackwardFn<B>>,
    pub cached_outputs: Option<B::Storage>, // Cache for storing forward outputs to reuse in backward pass
}

impl<B: Backend> Op<B> {
    pub fn new(
        op_type: OpType,
        inputs: Vec<Tensor<B>>,
        backward_fn: impl Fn(&Op<B>, &B::Storage) -> Result<Vec<B::Storage>, Error> + 'static,
    ) -> Self {
        Self {
            op_type,
            inputs,
            backward_fn: Rc::new(backward_fn),
            cached_outputs: None, // Initialize as None, will be set during forward pass if needed
        }
    }
}

impl<B: Backend> fmt::Debug for Op<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Op")
            .field("op_type", &self.op_type)
            .field(
                "inputs",
                &self.inputs.iter().map(|t| t.id()).collect::<Vec<_>>(),
            )
            .field("backward_fn", &"<closure>")
            .finish()
    }
}
