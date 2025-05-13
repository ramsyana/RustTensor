// examples/train_mnist_cpu.rs
use rust_tensor_lib::{
    backend::cpu::CpuBackend, // Import specific backend
    data,                     // Data loading functions
    debug_println,
    ops,        // Tensor operations (relu, matmul, etc.)
    optim::Sgd, // Optimizer
    Backend,    // Type alias Tensor<CpuBackend>
    Error,
    Tensor,
};
use std::time::Instant;
// ndarray::slicing::s might be useful if accessing ndarray data directly
// use ndarray::s;

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

// --- Main Training Function ---
fn main() -> Result<(), Error> {
    // --- Setup ---
    debug_println!("Backend: CPU");
    type ActiveBackend = CpuBackend; // Explicitly choose backend

    // --- Load Data ---
    debug_println!("Loading MNIST dataset...");
    if !std::path::Path::new(MNIST_TRAIN_PATH).exists()
        || !std::path::Path::new(MNIST_TEST_PATH).exists()
    {
        eprintln!(
            "Error: MNIST CSV files not found at '{}' and '{}'",
            MNIST_TRAIN_PATH, MNIST_TEST_PATH
        );
        eprintln!(
            "Please download/generate the MNIST CSV files and place them in a 'data/' directory."
        );
        return Err(Error::InvalidOperation("MNIST data not found".to_string()));
    }
    // Use generic load_csv specifying the backend
    let (x_train, y_train): (Tensor<ActiveBackend>, Tensor<ActiveBackend>) =
        data::load_csv::<ActiveBackend>(MNIST_TRAIN_PATH)?;
    let (x_test, y_test): (Tensor<ActiveBackend>, Tensor<ActiveBackend>) =
        data::load_csv::<ActiveBackend>(MNIST_TEST_PATH)?;
    println!(
        "Loaded Train: x{:?} y{:?}, Test: x{:?} y{:?}",
        x_train.shape(),
        y_train.shape(),
        x_test.shape(),
        y_test.shape()
    );

    // --- Initialize Model and Optimizer ---
    debug_println!("Initializing model parameters...");
    let model = MlpModel::<ActiveBackend>::new(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)?;
    // Use generic Sgd specifying the backend
    let mut optimizer = Sgd::<ActiveBackend>::new(model.parameters(), LEARNING_RATE);

    // --- Training Loop ---
    debug_println!("Starting training ({} epochs)...", NUM_EPOCHS);
    let start_time = Instant::now();

    for epoch in 0..NUM_EPOCHS {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        let num_samples = x_train.shape()[0];

        // Consider shuffling data each epoch
        // let mut indices: Vec<usize> = (0..num_samples).collect();
        // indices.shuffle(&mut thread_rng()); // Need import

        for batch_start in (0..num_samples).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(num_samples);
            let current_batch_size = batch_end - batch_start;
            if current_batch_size == 0 {
                continue;
            }

            // --- Get Batch ---
            // Use generic get_random_batch (requires data shuffling or use fixed slice for simplicity)
            let (bx, by) = data::get_random_batch(&x_train, &y_train, current_batch_size)?;

            // --- Training Step ---
            optimizer.zero_grad()?; // Clear gradients for model parameters
            let logits = model.forward(&bx)?; // Forward pass using generic ops
            debug_println!("Training: logits shape = {:?}", logits.shape());
            let log_probs = ops::log_softmax(&logits, 1)?; // Apply log-softmax (axis 1 for classes)
            debug_println!("Training: log_probs shape = {:?}", log_probs.shape());

            // --- Loss computation ---
            let loss = nll_loss(&log_probs, &by)?; // Compute loss (scalar tensor)
            debug_println!("Training: loss shape = {:?}", loss.shape());

            // --- Backward pass ---
            loss.backward()?; // Backward pass (compute gradients)
            optimizer.step()?; // Update parameters using gradients

            // Accumulate loss (loss tensor is scalar)
            // Access scalar value using AsRef<[f32]>
            let loss_value = {
                let loss_data_ref = loss.data();
                let loss_slice = loss_data_ref.as_ref();
                // Safely get the first element (scalar value)
                *loss_slice.first().ok_or(Error::InternalLogicError(
                    "Loss tensor is not scalar".to_string(),
                ))?
            };
            epoch_loss += loss_value;
            num_batches += 1;

            // Print progress (optional)
            if (num_batches * BATCH_SIZE) % (BATCH_SIZE * 100) < BATCH_SIZE {
                // Print every 100 batches approx
                println!(
                    "Epoch {} [{}/{} ({:.0}%)]\tLoss: {:.6}",
                    epoch + 1,
                    batch_end,
                    num_samples,
                    100.0 * batch_end as f32 / num_samples as f32,
                    epoch_loss / num_batches as f32 // Average loss for epoch so far
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
    debug_println!("Training completed in {:.2?}", _training_time);

    // --- Evaluation ---
    debug_println!("\nEvaluating on test set...");
    // Pass ActiveBackend explicitly if evaluate is not generic yet, otherwise infer
    evaluate::<ActiveBackend>(&model, &x_test, &y_test)?;

    Ok(())
}

/// Evaluates the model on the given dataset.
fn evaluate<B: Backend>(
    model: &MlpModel<B>,
    x_test: &Tensor<B>,
    y_test: &Tensor<B>,
) -> Result<(), Error>
where
    B::Storage: AsRef<[f32]>, // Bound needed for CPU slice access
{
    let mut correct = 0;
    let num_samples = x_test.shape()[0];
    if num_samples == 0 {
        debug_println!("Test set empty, skipping evaluation.");
        return Ok(());
    }
    let num_classes = y_test.shape()[1];

    // Process test set in batches
    for i in (0..num_samples).step_by(BATCH_SIZE) {
        let current_batch_size = std::cmp::min(BATCH_SIZE, num_samples - i);
        if current_batch_size == 0 {
            continue;
        }

        // Use get_random_batch or fixed slicing
        let (bx, by) = data::get_random_batch(x_test, y_test, current_batch_size)?;

        // --- Forward pass (no grad tracking needed ideally) ---
        // TODO: Add no_grad context if available
        let logits = model.forward(&bx)?;
        // Get log probabilities for prediction comparison
        let log_probs = ops::log_softmax(&logits, 1)?;

        // --- Get predictions from log_probs ---
        let log_probs_data_ref = log_probs.data();
        let log_probs_slice = log_probs_data_ref.as_ref(); // &[f32]
        let by_data_ref = by.data();
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
