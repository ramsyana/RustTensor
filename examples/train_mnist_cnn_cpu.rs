// Minimal boilerplate for MNIST CNN CPU training example
// Step 05.04.01.02: Imports, constants, and main structure

use rust_tensor_lib::{
    backend::cpu::CpuBackend,
    data,
    ops,
    optim::Adam,
    Backend,
    Error,
    Tensor,
    Reduction,
};
// We'll use the generic ops functions
use std::time::Instant;

// Hyperparameters and constants
const BATCH_SIZE: usize = 64;
const NUM_EPOCHS: usize = 5;
const LEARNING_RATE: f32 = 0.001;
const _IMAGE_DIM: usize = 28;
const NUM_CLASSES: usize = 10;
const MNIST_TRAIN_PATH: &str = "data/mnist_train.csv";
const MNIST_TEST_PATH: &str = "data/mnist_test.csv";
// MLP-specific
const _HIDDEN_SIZE: usize = 128;

fn evaluate_cnn<B: Backend>(model: &MnistCnnModel<B>, x_test: &Tensor<B>, y_test: &Tensor<B>, _batch_size: usize) -> Result<f32, Error> {
    let mut correct = 0;
    let mut total = 0;
    
    // Process the entire test set at once to simplify
    // This works because MNIST test set is small enough to fit in memory
    let batch_x = x_test;
    let batch_y = y_test;
    
    let logits = model.forward(&batch_x)?;
    let preds = ops::argmax(&logits, 1)?;
    let labels = ops::argmax(&batch_y, 1)?;
    
    // Compare predictions with labels - use not_equal and subtract from 1 to get equality
    let ones = Tensor::<B>::ones(&preds.shape(), false)?;
    let not_equal_result = ops::not_equal(&preds, &labels)?;
    let correct_predictions = ops::sub(&ones, &not_equal_result)?;
    let batch_correct = correct_predictions.sum(None)?;
    correct += batch_correct.to_cpu()?.data().as_ref()[0] as i32;
    total += batch_x.shape()[0] as i32;
    
    Ok(correct as f32 / total as f32)
}

fn main() -> Result<(), Error> {
    // Check if MNIST data files exist
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
    
    // Load data
    let (x_train, y_train) = data::load_csv::<CpuBackend>(MNIST_TRAIN_PATH)?;
    let (x_test, y_test) = data::load_csv::<CpuBackend>(MNIST_TEST_PATH)?;

    println!(
        "Loaded Train: x{:?} y{:?}, Test: x{:?} y{:?}",
        x_train.shape(),
        y_train.shape(),
        x_test.shape(),
        y_test.shape()
    );

    // Reshape x_train/x_test from [N, 784] to [N, 1, 28, 28]
    let x_train = x_train.view(&[x_train.shape()[0], 1, 28, 28])?;
    let x_test = x_test.view(&[x_test.shape()[0], 1, 28, 28])?;

    // Print original shapes for debugging
    println!("Original shapes - x_train: {:?}, y_train: {:?}", x_train.shape(), y_train.shape());
    
    // For simplicity, we'll create a new smaller dataset with just the first 1000 samples
    let subset_size = 1000;
    
    // Create new tensors with the first subset_size samples
    let mut x_train_data = Vec::with_capacity(subset_size * x_train.shape()[1]);
    let mut y_train_data = Vec::with_capacity(subset_size * y_train.shape()[1]);
    
    // Copy the first subset_size samples
    for i in 0..subset_size {
        if i >= x_train.shape()[0] {
            break;
        }
        
        // Get a reference to the data for this iteration
        let x_data_ref = x_train.data();
        let y_data_ref = y_train.data();
        let x_cpu = x_data_ref.as_ref();
        let y_cpu = y_data_ref.as_ref();
        
        // Copy x data
        let start_x = i * x_train.shape()[1];
        let end_x = start_x + x_train.shape()[1];
        x_train_data.extend_from_slice(&x_cpu[start_x..end_x]);
        
        // Copy y data
        let start_y = i * y_train.shape()[1];
        let end_y = start_y + y_train.shape()[1];
        y_train_data.extend_from_slice(&y_cpu[start_y..end_y]);
    }
    
    // Create new tensors
    let x_train_subset = Tensor::<CpuBackend>::from_vec(
        x_train_data, 
        &[subset_size, 1, 28, 28], 
        false
    )?;

    let y_train_subset = Tensor::<CpuBackend>::from_vec(
        y_train_data, 
        &[subset_size, y_train.shape()[1]], 
        false
    )?;

    println!("Subset shapes - x_train: {:?}, y_train: {:?}", x_train_subset.shape(), y_train_subset.shape());
    
    // Initialize model
    let model = MnistCnnModel::<CpuBackend>::new()?;

    // Initialize optimizer
    let mut optimizer = Adam::new(model.parameters(), LEARNING_RATE, 0.9, 0.999, 1e-8)?;
    
    // Training loop
    for epoch in 0..NUM_EPOCHS {
        let start_time = Instant::now();
        let mut train_loss = 0.0;
        let mut num_batches = 0;
        
        // For each epoch, we'll use the entire subset
        let batch_x = x_train_subset.clone();
        let batch_y = y_train_subset.clone();
        
        optimizer.zero_grad()?;
        
        let logits = model.forward(&batch_x)?;
        let loss = ops::softmax_cross_entropy(&logits, &batch_y, 1, Reduction::Mean)?;

        loss.backward()?;
        optimizer.step()?;

        train_loss += loss.to_cpu()?.data().as_ref()[0];
        num_batches += 1;
        
        let avg_loss = train_loss / num_batches as f32;
        let elapsed = start_time.elapsed();
        
        // Evaluate on test set
        let test_acc = evaluate_cnn(&model, &x_test, &y_test, BATCH_SIZE)?;
        
        println!(
            "Epoch {} | Loss: {:.4} | Test Acc: {:.2}% | Time: {:.2}s",
            epoch + 1,
            avg_loss,
            test_acc * 100.0,
            elapsed.as_secs_f32(),
        );
    }
    
    Ok(())
}

// Step 05.04.04.01: Define the MnistCnnModel struct
#[derive(Debug, Clone)]
pub struct MnistCnnModel<B: Backend> {
    pub conv1_w: Tensor<B>,
    pub conv1_b: Tensor<B>,
    pub conv2_w: Tensor<B>,
    pub conv2_b: Tensor<B>,
    pub fc1_w: Tensor<B>,
    pub fc1_b: Tensor<B>,
    pub fc2_w: Tensor<B>,
    pub fc2_b: Tensor<B>,
}

impl<B: Backend> MnistCnnModel<B> {
    pub fn new() -> Result<Self, Error> {
        // Conv1: 1 input channel, 16 output channels, kernel 5x5
        let conv1_w = Tensor::<B>::kaiming_uniform(1 * 5 * 5, &[16, 1, 5, 5], true)?;
        let conv1_b = Tensor::<B>::zeros(&[16], true)?;
        // Conv2: 16 input channels, 32 output channels, kernel 5x5
        let conv2_w = Tensor::<B>::kaiming_uniform(16 * 5 * 5, &[32, 16, 5, 5], true)?;
        let conv2_b = Tensor::<B>::zeros(&[32], true)?;
        // After conv+pool: [N, 32, 7, 7] => flat_size = 32*7*7
        let flat_size = 32 * 7 * 7;
        // FC1: flat_size -> 128
        let fc1_w = Tensor::<B>::kaiming_uniform(flat_size, &[flat_size, 128], true)?;
        let fc1_b = Tensor::<B>::zeros(&[128], true)?;
        // FC2: 128 -> 10
        let fc2_w = Tensor::<B>::kaiming_uniform(128, &[128, NUM_CLASSES], true)?;
        let fc2_b = Tensor::<B>::zeros(&[NUM_CLASSES], true)?;
        Ok(Self {
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        })
    }
    pub fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>, Error> {
        // x: [N, 1, 28, 28]
        let n = x.shape()[0];
        let c1 = ops::conv2d(x, &self.conv1_w, Some(&self.conv1_b), (1, 1), (2, 2))?; // [N, 16, 28, 28]
        let r1 = ops::relu(&c1)?;
        let p1 = ops::max_pool2d(&r1, (2, 2), (2, 2), (0, 0))?; // [N, 16, 14, 14]
        let c2 = ops::conv2d(&p1, &self.conv2_w, Some(&self.conv2_b), (1, 1), (2, 2))?; // [N, 32, 14, 14]
        let r2 = ops::relu(&c2)?;
        let p2 = ops::max_pool2d(&r2, (2, 2), (2, 2), (0, 0))?; // [N, 32, 7, 7]
        let flat_size = 32 * 7 * 7;
        let flat = p2.view(&[n, flat_size])?;
        let fc1 = ops::matmul(&flat, &self.fc1_w)?;
        let fc1 = ops::add(&fc1, &self.fc1_b.broadcast_to(&[n, 128])?)?;
        let fc1_out = ops::relu(&fc1)?;
        let logits = ops::matmul(&fc1_out, &self.fc2_w)?;
        let logits = ops::add(&logits, &self.fc2_b.broadcast_to(&[n, NUM_CLASSES])?)?;
        Ok(logits)
    }
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        vec![
            self.conv1_w.clone(),
            self.conv1_b.clone(),
            self.conv2_w.clone(),
            self.conv2_b.clone(),
            self.fc1_w.clone(),
            self.fc1_b.clone(),
            self.fc2_w.clone(),
            self.fc2_b.clone(),
        ]
    }
}
