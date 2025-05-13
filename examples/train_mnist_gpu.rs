// examples/train_mnist_gpu.rs
use rust_tensor_lib::{
    backend::{cuda::CudaBackend, cpu::CpuBackend},
    debug_println,
    ops,
    optim::Sgd,
    Backend,
    Error,
    Tensor,
};

// Import CudaContextGuard from a public path
use rust_tensor_lib::backend::cuda::CudaContextGuard;
use rust_tensor_lib::backend::cuda::init_context;
use std::time::Instant;

// --- Constants ---
const BATCH_SIZE: usize = 64;
const NUM_EPOCHS: usize = 3;
const LEARNING_RATE: f32 = 0.01;
const HIDDEN_SIZE: usize = 128;
const INPUT_SIZE: usize = 784; // 28x28 pixels
const NUM_CLASSES: usize = 10;
const MNIST_TRAIN_PATH: &str = "data/mnist_train.csv";
const MNIST_TEST_PATH: &str = "data/mnist_test.csv";

// --- Model Definition (Simple MLP) ---
// Model is generic over the Backend
struct MlpModel<B: Backend> {
    w1: Tensor<B>,
    w2: Tensor<B>,
    // Add biases later if needed
    // b1: Tensor<B>,
    // b2: Tensor<B>,
}

impl<B: Backend> MlpModel<B> {
    /// Creates a new MLP model with Kaiming initialized weights.
    fn new(input_size: usize, hidden_size: usize, num_classes: usize) -> Result<Self, Error> {
        // Use generic Tensor factory method
        let w1 = Tensor::<B>::kaiming_uniform(input_size, &[input_size, hidden_size], true)?;
        let w2 = Tensor::<B>::kaiming_uniform(hidden_size, &[hidden_size, num_classes], true)?;
        Ok(Self { w1, w2 })
    }

    /// Performs a forward pass through the model using generic ops.
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>, Error> {
        debug_println!("forward: x shape = {:?}", x.shape());
        let h = ops::matmul(x, &self.w1)?;
        debug_println!("forward: h shape = {:?}", h.shape());
        let h_relu = ops::relu(&h)?;
        debug_println!("forward: h_relu shape = {:?}", h_relu.shape());
        let logits = ops::matmul(&h_relu, &self.w2)?;
        debug_println!("forward: logits shape = {:?}", logits.shape());
        Ok(logits)
    }

    /// Returns a list of trainable parameters (cloned Tensors).
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![self.w1.clone(), self.w2.clone()]
    }
}

// --- Loss Function ---
/// Computes the mean negative log likelihood (cross-entropy loss for one-hot labels).
/// Loss = -mean(sum(y_true * log_softmax(logits), axis=1))
fn nll_loss<B: Backend>(log_probs: &Tensor<B>, y_true: &Tensor<B>) -> Result<Tensor<B>, Error> {
    // log_probs shape: [batch_size, num_classes]
    // y_true shape: [batch_size, num_classes] (one-hot)
    debug_println!(
        "nll_loss: log_probs shape = {:?}, y_true shape = {:?}",
        log_probs.shape(),
        y_true.shape()
    );

    // 1. Element-wise product: y_true * log_probs
    let elementwise_loss = ops::mul(log_probs, y_true)?;
    debug_println!(
        "nll_loss: elementwise_loss shape = {:?}",
        elementwise_loss.shape()
    );

    // 2. Sum along class dimension (axis=1) to get per-example loss
    let per_example_loss = ops::sum(&elementwise_loss, Some(1))?; // Sum over classes
    debug_println!(
        "nll_loss: per_example_loss shape = {:?}",
        per_example_loss.shape()
    );

    // 3. Compute mean over batch (keeping as scalar)
    let mean_loss = ops::mean(&per_example_loss, None)?;
    debug_println!("nll_loss: mean_loss shape = {:?}", mean_loss.shape());

    // 4. Negate the result (maintaining scalar shape)
    let neg_mean_loss = ops::mul(&mean_loss, &Tensor::<B>::from_vec(vec![-1.0], &[], false)?)?;

    Ok(neg_mean_loss) // Keep as scalar tensor with shape []
}

// --- Helper function to transfer tensors to GPU ---
fn to_gpu<B: Backend>(tensor: &Tensor<B>) -> Result<Tensor<CudaBackend>, Error> {
    // Use the built-in to_gpu method with device 0
    tensor.to_gpu(0)
}

// --- Helper function to transfer tensors to CPU ---
fn to_cpu<B: Backend>(tensor: &Tensor<B>) -> Result<Tensor<CpuBackend>, Error> {
    // Use the built-in to_cpu method
    tensor.to_cpu()
}

// --- Helper function to load MNIST data from CSV ---
fn load_mnist_data(path: &str, normalize: bool) -> Result<(Tensor<CpuBackend>, Tensor<CpuBackend>), Error> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    
    println!("Loading MNIST data from {}...", path);
    let file = File::open(path).map_err(|e| Error::InvalidOperation(format!("Failed to open {}: {}", path, e)))?;
    let reader = BufReader::new(file);
    
    let mut labels = Vec::new();
    let mut images = Vec::new();
    let mut num_samples = 0;
    
    for line in reader.lines() {
        let line = line.map_err(|e| Error::InvalidOperation(format!("Failed to read line: {}", e)))?;
        let values: Vec<f32> = line
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|e| Error::InvalidOperation(format!("Failed to parse value: {}", e)))?;
        
        if values.is_empty() {
            continue;
        }
        
        // First value is the label (0-9)
        let label = values[0] as usize;
        if label >= NUM_CLASSES {
            return Err(Error::InvalidOperation(format!("Invalid label: {}", label)));
        }
        
        // One-hot encode the label
        let mut one_hot = vec![0.0; NUM_CLASSES];
        one_hot[label] = 1.0;
        labels.extend_from_slice(&one_hot);
        
        // Rest of values are pixel intensities (0-255)
        let mut pixels = values[1..].to_vec();
        if normalize {
            // Normalize to [0, 1]
            for pixel in &mut pixels {
                *pixel /= 255.0;
            }
        }
        images.extend_from_slice(&pixels);
        
        num_samples += 1;
    }
    
    if num_samples == 0 {
        return Err(Error::InvalidOperation("No samples found in dataset".to_string()));
    }
    
    // Create tensors
    let x = Tensor::<CpuBackend>::from_vec(images, &[num_samples, INPUT_SIZE], false)?;
    let y = Tensor::<CpuBackend>::from_vec(labels, &[num_samples, NUM_CLASSES], false)?;
    
    println!("Loaded {} samples from {}", num_samples, path);
    Ok((x, y))
}

// --- Helper function to get a random batch from a dataset ---
fn get_random_batch<B: Backend>(
    x: &Tensor<B>,
    y: &Tensor<B>,
    batch_size: usize,
) -> Result<(Tensor<B>, Tensor<B>), Error> {
    let num_samples = x.shape()[0];
    if batch_size > num_samples {
        return Err(Error::InvalidOperation(format!(
            "Batch size ({}) exceeds dataset size ({})",
            batch_size, num_samples
        )));
    }
    
    // For simplicity, just take the first batch_size samples
    // In a real implementation, you would randomly select samples
    
    // Create a new tensor with the first batch_size samples
    let x_shape = x.shape();
    let y_shape = y.shape();
    
    // Copy the data for the batch
    let x_data = B::copy_to_host(&*x.data())?;
    let y_data = B::copy_to_host(&*y.data())?;
    
    // Calculate the size of each sample
    let x_sample_size = x_shape.iter().skip(1).product::<usize>();
    let y_sample_size = y_shape.iter().skip(1).product::<usize>();
    
    // Extract the batch data
    let mut batch_x_data = Vec::with_capacity(batch_size * x_sample_size);
    let mut batch_y_data = Vec::with_capacity(batch_size * y_sample_size);
    
    for i in 0..batch_size {
        let x_start = i * x_sample_size;
        let x_end = x_start + x_sample_size;
        batch_x_data.extend_from_slice(&x_data[x_start..x_end]);
        
        let y_start = i * y_sample_size;
        let y_end = y_start + y_sample_size;
        batch_y_data.extend_from_slice(&y_data[y_start..y_end]);
    }
    
    // Create new batch shapes
    let mut batch_x_shape = x_shape.to_vec();
    batch_x_shape[0] = batch_size;
    
    let mut batch_y_shape = y_shape.to_vec();
    batch_y_shape[0] = batch_size;
    
    // Create new tensors
    let batch_x = Tensor::<B>::from_vec(batch_x_data, &batch_x_shape, x.requires_grad())?;
    let batch_y = Tensor::<B>::from_vec(batch_y_data, &batch_y_shape, y.requires_grad())?;
    
    Ok((batch_x, batch_y))
}

// --- Main Training Function ---
fn main() -> Result<(), Error> {
    // --- Initialize CUDA Context ---
    // Explicitly initialize CUDA context before creating the guard
    init_context(0)?; // Initialize with device ID 0
    let _cuda_guard = CudaContextGuard::new()?; // Use underscore to avoid unused variable warning
    // Ensure the guard is not optimized away
    println!("CUDA context initialized successfully");
    
    // --- Setup ---
    println!("Backend: CUDA GPU");
    type ActiveBackend = CudaBackend; // Explicitly choose backend

    // --- Load Data ---
    println!("Loading MNIST dataset...");
    if !std::path::Path::new(MNIST_TRAIN_PATH).exists() || !std::path::Path::new(MNIST_TEST_PATH).exists() {
        println!("MNIST dataset not found at {} or {}", MNIST_TRAIN_PATH, MNIST_TEST_PATH);
        return Err(Error::InvalidOperation("MNIST dataset not found".to_string()));
    }

    // Load MNIST data manually since we don't have the load_mnist_csv function
    let (x_train_cpu, y_train_cpu) = load_mnist_data(MNIST_TRAIN_PATH, true)?;
    let (x_test_cpu, y_test_cpu) = load_mnist_data(MNIST_TEST_PATH, true)?;

    // Transfer data to GPU
    let x_train = to_gpu(&x_train_cpu)?;
    let y_train = to_gpu(&y_train_cpu)?;
    let x_test = to_gpu(&x_test_cpu)?;
    let y_test = to_gpu(&y_test_cpu)?;

    println!(
        "Data loaded: Train: {} samples, Test: {} samples",
        x_train.shape()[0],
        x_test.shape()[0]
    );

    // --- Initialize Model ---
    let model = MlpModel::<ActiveBackend>::new(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)?;

    // --- Setup Optimizer ---
    let mut optimizer = Sgd::new(model.parameters(), LEARNING_RATE);

    // --- Training Loop ---
    println!("\nTraining for {} epochs...", NUM_EPOCHS);
    let start_time = Instant::now();

    for epoch in 0..NUM_EPOCHS {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        // Get number of samples
        let num_samples = x_train.shape()[0];
        if num_samples == 0 {
            println!("Training set empty, skipping training.");
            break;
        }

        // Process training set in batches
        for i in (0..num_samples).step_by(BATCH_SIZE) {
            let current_batch_size = std::cmp::min(BATCH_SIZE, num_samples - i);
            if current_batch_size == 0 {
                continue;
            }

            // Get batch
            let (bx, by) = get_random_batch(&x_train, &y_train, current_batch_size)?;

            // --- Forward pass ---
            let logits = model.forward(&bx)?;
            let log_probs = ops::log_softmax(&logits, 1)?;
            let loss = nll_loss(&log_probs, &by)?;

            // Transfer loss to CPU for printing
            let loss_cpu = to_cpu(&loss)?;
            let loss_data = CpuBackend::copy_to_host(&*loss_cpu.data())?;
            let loss_value = loss_data[0];
            epoch_loss += loss_value;
            num_batches += 1;

            // --- Backward pass ---
            optimizer.zero_grad()?;
            loss.backward()?;
            optimizer.step()?;

            // Print progress
            if i % (BATCH_SIZE * 100) == 0 {
                println!(
                    "Epoch: {}/{}, Batch: {}/{}, Loss: {:.4}",
                    epoch + 1,
                    NUM_EPOCHS,
                    i / BATCH_SIZE,
                    (num_samples + BATCH_SIZE - 1) / BATCH_SIZE,
                    loss_value
                );
            }
        } // End batch loop

        let avg_epoch_loss = if num_batches > 0 {
            epoch_loss / num_batches as f32
        } else {
            0.0
        };
        println!(
            "====> Epoch: {} Average loss: {:.4}",
            epoch + 1,
            avg_epoch_loss
        );
    } // End epoch loop

    let _training_time = start_time.elapsed();
    println!("Training completed in {:.2?}", _training_time);

    // --- Evaluation ---
    println!("\nEvaluating on test set...");
    evaluate(&model, &x_test, &y_test, &x_test_cpu, &y_test_cpu)?;

    Ok(())
}

/// Evaluates the model on the given dataset.
fn evaluate(
    model: &MlpModel<CudaBackend>,
    x_test: &Tensor<CudaBackend>,
    y_test: &Tensor<CudaBackend>,
    _x_test_cpu: &Tensor<CpuBackend>,
    _y_test_cpu: &Tensor<CpuBackend>,
) -> Result<(), Error> {
    let mut correct = 0;
    let num_samples = x_test.shape()[0];
    if num_samples == 0 {
        println!("Test set empty, skipping evaluation.");
        return Ok(());
    }
    let num_classes = y_test.shape()[1];

    // Process test set in batches
    for i in (0..num_samples).step_by(BATCH_SIZE) {
        let current_batch_size = std::cmp::min(BATCH_SIZE, num_samples - i);
        if current_batch_size == 0 {
            continue;
        }

        // Get batch on GPU
        let (bx, by) = get_random_batch(x_test, y_test, current_batch_size)?;

        // --- Forward pass (no grad tracking needed) ---
        let logits = model.forward(&bx)?;
        // Get log probabilities for prediction comparison
        let log_probs = ops::log_softmax(&logits, 1)?;

        // Transfer predictions to CPU for evaluation
        let log_probs_cpu = to_cpu(&log_probs)?;
        let by_cpu = to_cpu(&by)?;

        // --- Get predictions from log_probs ---
        let log_probs_data_ref = log_probs_cpu.data();
        let log_probs_slice = log_probs_data_ref.as_ref(); // &[f32]
        let by_data_ref = by_cpu.data();
        let by_slice = by_data_ref.as_ref(); // &[f32]

        for j in 0..current_batch_size {
            // Find predicted class index (max log_prob)
            let probs_start = j * num_classes;
            let probs_end = probs_start + num_classes;
            if probs_end > log_probs_slice.len() {
                continue;
            } // Bounds check
            let sample_log_probs = &log_probs_slice[probs_start..probs_end];

            let pred_idx = sample_log_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(usize::MAX); // Use MAX if error

            // Find true class index (where value is 1.0 in one-hot vector)
            let labels_start = j * num_classes;
            let labels_end = labels_start + num_classes;
            if labels_end > by_slice.len() {
                continue;
            } // Bounds check
            let sample_labels = &by_slice[labels_start..labels_end];

            let true_idx = sample_labels.iter().position(|&x| (x - 1.0).abs() < 1e-6);

            if let Some(ti) = true_idx {
                if pred_idx == ti {
                    correct += 1;
                }
            } else {
                // This might happen if labels are not strictly one-hot
                eprintln!(
                    "Warning: No true label (1.0) found in test sample {}",
                    i + j
                );
            }
        }
    } // End batch loop

    let accuracy = if num_samples > 0 {
        100.0 * correct as f32 / num_samples as f32
    } else {
        0.0
    };
    println!(
        "Test Accuracy: {}/{} ({:.2}%)",
        correct, num_samples, accuracy
    );

    Ok(())
}

// Helper extension trait for Tensor shape printing (Optional)
#[allow(dead_code)]
trait ShapeExt {
    fn dims(&self) -> String;
}
impl ShapeExt for Vec<usize> {
    fn dims(&self) -> String {
        format!("{:?}", self)
    }
}
