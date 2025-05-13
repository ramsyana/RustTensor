// CUDA context management using rustacuda
// Will handle device initialization, streams, and kernel loading

use crate::error::Error;
use cublas_sys;
use cust::context::{Context, CurrentContext};
use cust::device::Device;
use cust::function::Function;
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, Once};

struct SendSyncRawHandle(cublas_sys::cublasHandle_t);
unsafe impl Send for SendSyncRawHandle {}
unsafe impl Sync for SendSyncRawHandle {}

pub struct CudaContext {
    pub(crate) _context: Context,
    #[allow(dead_code)]
    device: Device,
    stream: Stream,
    cublas_handle: SendSyncRawHandle,
    modules: HashMap<String, Arc<Module>>,
    kernels: HashMap<String, Function<'static>>,
}

lazy_static! {
    static ref GLOBAL_CUDA_CONTEXT: Mutex<Option<Arc<CudaContext>>> = Mutex::new(None);
    static ref CUDA_INIT: Once = Once::new();
}

impl CudaContext {
    fn new(device_id: u32) -> Result<Self, Error> {
        cust::init(cust::CudaFlags::empty())?;
        let device = Device::get_device(device_id).map_err(|e| Error::CudaError(e.to_string()))?;
        let context = Context::new(device).map_err(|e| Error::CudaError(e.to_string()))?;
        let stream =
            Stream::new(StreamFlags::DEFAULT, None).map_err(|e| Error::CudaError(e.to_string()))?;

        // Create cuBLAS handle
        let mut handle = std::ptr::null_mut();
        unsafe {
            let status = cublas_sys::cublasCreate_v2(&mut handle);
            if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(Error::CudaError(
                    "Failed to create cuBLAS handle".to_string(),
                ));
            }
        }

        let mut instance = Self {
            _context: context,
            device,
            stream,
            cublas_handle: SendSyncRawHandle(handle),
            modules: HashMap::new(),
            kernels: HashMap::new(),
        };

        instance.load_kernels()?;
        debug_println!(
            "[DEBUG CudaContext] Loaded kernels: {:?}",
            instance.kernels.keys().collect::<Vec<_>>()
        );
        Ok(instance)
    }

    fn load_kernel_module(&mut self, name: &str, ptx_path: &Path) -> Result<(), Error> {
        debug_println!(
            "Loading kernel module: {} from {}",
            name,
            ptx_path.display()
        );
        let ptx_str = fs::read_to_string(ptx_path).map_err(|e| {
            Error::CudaError(format!(
                "Failed to read PTX file {}: {}",
                ptx_path.display(),
                e
            ))
        })?;

        CurrentContext::set_current(&self._context)?;

        let module = Module::from_ptx(&ptx_str, &[])
            .map_err(|e| Error::CudaError(format!("Failed to load module {}: {}", name, e)))?;

        let arc_module = Arc::new(module);

        debug_println!("[load_kernel_module] Loading kernels for module: {}", name);
        let expected_kernels: &[&str] = match name {
            "elementwise" => &[
                "add_kernel",
                "mul_kernel",
                "sub_kernel",
                "div_kernel",
                "div_scalar_kernel",
                "relu_kernel",
                "exp_kernel",
                "ln_kernel",
                "broadcast_kernel",
                "relu_backward_kernel",
                "sgd_step_kernel",
                "sum_along_axis_kernel",
                "max_along_axis_kernel",
                "logsumexp_along_axis_kernel",
                "fill_scalar_kernel",
                "abs_kernel",
                "abs_backward_kernel",
                "sigmoid_kernel",
                "sigmoid_backward_kernel",
                "tanh_kernel",
                "tanh_backward_kernel",
                "max_backward_kernel",
                "min_backward_kernel",
                "prod_backward_kernel",
                "logsumexp_backward_kernel",
                "softplus_kernel",
                "softplus_backward_kernel",
                "powf_kernel",
                "powf_backward_kernel",
                "sqrt_kernel",
                "sqrt_backward_kernel",
                "square_kernel",
                "square_backward_kernel",
                "maximum_kernel",
                "maximum_backward_kernel",
                "minimum_kernel",
                "minimum_backward_kernel",
                "equal_kernel",
                "greater_kernel",
                "greater_equal_kernel",
                "less_kernel",
                "less_equal_kernel",
                "not_equal_kernel",
                "elu_kernel",
                "elu_backward_kernel",
                "sin_kernel",
                "sin_backward_kernel",
                "cos_kernel",
                "cos_backward_kernel",
                "tan_kernel",
                "tan_backward_kernel",
                "clip_kernel",
                "clip_backward_kernel",
                "add_bias_4d_kernel",
            ],
            "reduction" => &[
                "sum_reduction_kernel",
                "max_reduction_kernel",
                "min_reduction_kernel",
                "prod_reduction_kernel",
                "logsumexp_reduction_kernel",
                "argmax_along_axis_kernel",
                "argmin_along_axis_kernel",
            ],
            "optimizer" => &[
                "adam_step_kernel",
                "momentum_sgd_step_kernel",
                "adagrad_step_kernel",
            ],
            "transpose" => &[
                "transpose_2d_kernel",
            ],
            "log_softmax_fused" => &["log_softmax_fused_kernel"],
            "conv" => &["im2col_kernel", "col2im_kernel"],
            "pooling" => &["max_pool2d_forward_kernel", "max_pool2d_backward_kernel"],
            "array_ops" => &["slice_kernel", "slice_backward_kernel"],
            _ => {
                return Err(Error::CudaError(format!(
                    "Unknown module name: {}. Expected 'elementwise', 'reduction', 'optimizer', 'transpose', 'log_softmax_fused', 'conv', 'pooling', or 'array_ops'",
                    name
                )))
            }
        };

        let mut functions_to_add = HashMap::new();
        for &kernel_name in expected_kernels {
            debug_println!(
                "[load_kernel_module] Attempting to load kernel: {}",
                kernel_name
            );
            match arc_module.get_function(kernel_name) {
                Ok(func) => {
                    debug_println!("Successfully loaded kernel: {}", kernel_name);
                    let static_func =
                        unsafe { std::mem::transmute::<Function<'_>, Function<'static>>(func) };
                    functions_to_add.insert(kernel_name.to_string(), static_func);
                }
                Err(e) => {
                    return Err(Error::CudaError(format!(
                        "Failed to load kernel '{}' from module '{}': {}",
                        kernel_name, name, e
                    )));
                }
            }
        }

        self.modules.insert(name.to_string(), arc_module);
        self.kernels.extend(functions_to_add);
        debug_println!("Successfully loaded all kernels for module: {}", name);
        Ok(())
    }

    fn load_kernels(&mut self) -> Result<(), Error> {
        let out_dir = std::env::var("OUT_DIR")
            .map_err(|_| Error::InternalLogicError("OUT_DIR not set".to_string()))?;
        let elementwise_ptx_path = Path::new(&out_dir).join("elementwise.ptx");
        let reduction_ptx_path = Path::new(&out_dir).join("reduction.ptx");
        let optimizer_ptx_path = Path::new(&out_dir).join("optimizer.ptx");
        let transpose_ptx_path = Path::new(&out_dir).join("transpose.ptx");
        let log_softmax_fused_ptx_path = Path::new(&out_dir).join("log_softmax_fused.ptx");
        let conv_ptx_path = Path::new(&out_dir).join("conv.ptx");
        let pooling_ptx_path = Path::new(&out_dir).join("pooling.ptx");
        let array_ops_ptx_path = Path::new(&out_dir).join("array_ops.ptx");

        self.load_kernel_module("elementwise", &elementwise_ptx_path)?;
        self.load_kernel_module("reduction", &reduction_ptx_path)?;
        self.load_kernel_module("optimizer", &optimizer_ptx_path)?;
        self.load_kernel_module("transpose", &transpose_ptx_path)?;
        self.load_kernel_module("log_softmax_fused", &log_softmax_fused_ptx_path)?;
        self.load_kernel_module("conv", &conv_ptx_path)?;
        self.load_kernel_module("pooling", &pooling_ptx_path)?;
        self.load_kernel_module("array_ops", &array_ops_ptx_path)?;
        Ok(())
    }

    pub fn get_stream(&self) -> &Stream {
        &self.stream
    }

    pub fn get_cublas_handle(&self) -> cublas_sys::cublasHandle_t {
        self.cublas_handle.0
    }

    pub fn get_kernel(&self, name: &str) -> Option<&Function<'static>> {
        self.kernels.get(name)
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        debug_println!("Dropping CudaContext");
    }
}

pub fn init_context(device_id: u32) -> Result<(), Error> {
    CUDA_INIT.call_once(|| {
        let mut global_ctx_guard = match GLOBAL_CUDA_CONTEXT.lock() {
            Ok(guard) => guard,
            Err(_) => {
                eprintln!("FATAL: CUDA context mutex was poisoned during initialization");
                return;
            }
        };

        if global_ctx_guard.is_none() {
            debug_println!("Initializing CUDA context for device {}...", device_id);
            match CudaContext::new(device_id) {
                Ok(context) => {
                    *global_ctx_guard = Some(Arc::new(context));
                    debug_println!("CUDA context initialization successful.");
                }
                Err(e) => {
                    eprintln!("FATAL: Failed to initialize CUDA context: {}", e);
                }
            }
        } else {
            debug_println!("CUDA context already initialized (skipped initialization logic).");
        }
    });

    let final_check_guard = match GLOBAL_CUDA_CONTEXT.lock() {
        Ok(guard) => guard,
        Err(_) => {
            eprintln!("FATAL: CUDA context mutex was poisoned after initialization check");
            return Err(Error::InternalLogicError(
                "CUDA context mutex was poisoned after initialization check".to_string(),
            ));
        }
    };

    if CUDA_INIT.is_completed() && final_check_guard.is_some() {
        Ok(())
    } else {
        Err(Error::CudaError(
            "CUDA context initialization failed or context is not available.".into(),
        ))
    }
}

pub fn get_global_context() -> Result<Arc<CudaContext>, Error> {
    let global_ctx = GLOBAL_CUDA_CONTEXT
        .lock()
        .map_err(|_| Error::InternalLogicError("CUDA context mutex was poisoned".to_string()))?;

    match global_ctx.as_ref() {
        Some(ctx) => Ok(ctx.clone()),
        None => Err(Error::CudaError(
            "CUDA context not initialized. Call init_context first.".into(),
        )),
    }
}

pub struct CudaContextGuard {
    _context_arc: Arc<CudaContext>,
}

impl CudaContextGuard {
    pub fn new() -> Result<Self, Error> {
        debug_println!(
            "[CudaContextGuard::new] Enter (thread {:?})",
            std::thread::current().id()
        );
        let context = get_global_context()?;
        let result = CurrentContext::set_current(&context._context).map_err(|e| {
            Error::InternalLogicError(format!("Failed to set current CUDA context: {}", e))
        });
        debug_println!("[CudaContextGuard::new] set_current result: {:?}", result);
        let guard = match result {
            Ok(_) => Ok(CudaContextGuard {
                _context_arc: context,
            }),
            Err(e) => Err(e),
        };
        debug_println!(
            "[CudaContextGuard::new] Exit (thread {:?})",
            std::thread::current().id()
        );
        guard
    }
}

// No Drop implementation needed - RAII of the Arc<CudaContext> handles cleanup
