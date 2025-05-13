use rust_tensor_lib::{
    backend::cuda::{CudaBackend, CudaContextGuard},
    data,
    ops,
    optim::Adam,
    Backend,
    Error,
    Tensor,
    Reduction,

};
use std::time::Instant;


const BATCH_SIZE: usize = 64;
const NUM_EPOCHS: usize = 5;
const LEARNING_RATE: f32 = 0.001;
const IMAGE_DIM: usize = 28;
const NUM_CLASSES: usize = 10;
const MNIST_TRAIN_PATH: &str = "data/mnist_train.csv";
const MNIST_TEST_PATH: &str = "data/mnist_test.csv";
const _HIDDEN_SIZE: usize = 128;

fn evaluate_cnn<B: Backend>(model: &MnistCnnModel<B>, x_test: &Tensor<B>, y_test: &Tensor<B>, _batch_size: usize) -> Result<f32, Error> {
    let mut correct = 0;
    let mut total = 0;
    let batch_x = x_test;
    let batch_y = y_test;
    let logits = model.forward(&batch_x)?;
    let preds = ops::argmax(&logits, 1)?;
    let labels = ops::argmax(&batch_y, 1)?;
    let ones = Tensor::<B>::ones(&preds.shape(), false)?;
    let not_equal_result = ops::not_equal(&preds, &labels)?;
    let correct_predictions = ops::sub(&ones, &not_equal_result)?;
    let batch_correct = correct_predictions.sum(None)?;
    correct += batch_correct.to_cpu()?.data().as_ref()[0] as i32;
    total += batch_x.shape()[0] as i32;
    Ok(correct as f32 / total as f32)
}

fn main() -> Result<(), Error> {
    // Initialize CUDA context
    rust_tensor_lib::backend::cuda::init_context(0)?;
    let _guard = CudaContextGuard::new()?;
    if !std::path::Path::new(MNIST_TRAIN_PATH).exists() || !std::path::Path::new(MNIST_TEST_PATH).exists() {
        eprintln!("Error: MNIST CSV files not found at '{}' and '{}'", MNIST_TRAIN_PATH, MNIST_TEST_PATH);
        eprintln!("Please download/generate the MNIST CSV files and place them in a 'data/' directory.");
        return Err(Error::InvalidOperation("MNIST data not found".to_string()));
    }
    // Load data directly to GPU
    let (x_train, y_train) = data::load_csv::<CudaBackend>(MNIST_TRAIN_PATH)?;
    let (x_test, y_test) = data::load_csv::<CudaBackend>(MNIST_TEST_PATH)?;

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

    println!("Original shapes - x_train: {:?}, y_train: {:?}", x_train.shape(), y_train.shape());

    // Use a smaller subset of the training data for faster training
    let subset_size = 1000.min(x_train.shape()[0]);
    
    // Create new tensors with the first subset_size samples (like the CPU example)
    let mut x_train_data = Vec::with_capacity(subset_size * 1 * 28 * 28);
    let mut y_train_data = Vec::with_capacity(subset_size * y_train.shape()[1]);
    for i in 0..subset_size {
        if i >= x_train.shape()[0] { break; }
        let x_data_ref = x_train.data();
        let y_data_ref = y_train.data();
        let x_cpu = x_data_ref.to_vec().unwrap();
        let y_cpu = y_data_ref.to_vec().unwrap();
        // x shape: [N, 1, 28, 28] => stride = 1*28*28
        let x_stride = 1*28*28;
        let start_x = i * x_stride;
        let end_x = start_x + x_stride;
        x_train_data.extend_from_slice(&x_cpu[start_x..end_x]);
        // y shape: [N, 10] => stride = 10
        let y_stride = y_train.shape()[1];
        let start_y = i * y_stride;
        let end_y = start_y + y_stride;
        y_train_data.extend_from_slice(&y_cpu[start_y..end_y]);
    }
    let x_train_subset = Tensor::<CudaBackend>::from_vec(
        x_train_data,
        &[subset_size, 1, 28, 28],
        false
    )?;
    let y_train_subset = Tensor::<CudaBackend>::from_vec(
        y_train_data,
        &[subset_size, y_train.shape()[1]],
        false
    )?;
    println!("Subset shapes - x_train: {:?}, y_train: {:?}", x_train_subset.shape(), y_train_subset.shape());
    let x_test = x_test;
    let y_test = y_test;
    let model = MnistCnnModel::<CudaBackend>::new()?;
    let mut optimizer = Adam::new(model.parameters(), LEARNING_RATE, 0.9, 0.999, 1e-8)?;
    for epoch in 0..NUM_EPOCHS {
        let start_time = Instant::now();
        let mut train_loss = 0.0;
        let mut num_batches = 0;
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

#[derive(Debug, Clone)]
pub struct SimpleMLP<B: Backend> {
    pub fc1_w: Tensor<B>,
    pub fc1_b: Tensor<B>,
    pub fc2_w: Tensor<B>,
    pub fc2_b: Tensor<B>,
}

impl<B: Backend> SimpleMLP<B> {
    pub fn new(
        hidden_size: usize,
        num_classes: usize,
    ) -> Result<Self, Error> {
        // Simple 2-layer MLP: input -> hidden -> output
        let input_size = IMAGE_DIM * IMAGE_DIM; // 784 for MNIST
        
        // First layer weights (input -> hidden)
        let fc1_w = Tensor::<B>::kaiming_uniform(input_size, &[input_size, hidden_size], true)?;
        let fc1_b = Tensor::<B>::zeros(&[hidden_size], true)?;
        
        // Output layer weights (hidden -> num_classes)
        let fc2_w = Tensor::<B>::kaiming_uniform(hidden_size, &[hidden_size, num_classes], true)?;
        let fc2_b = Tensor::<B>::zeros(&[num_classes], true)?;
        
        Ok(Self {
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        })
    }
    
    pub fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>, Error> {
        let batch_size = x.shape()[0];
        let input_size = IMAGE_DIM * IMAGE_DIM;
        
        // Reshape input if needed
        let x_reshaped = if x.shape().len() > 2 {
            x.view(&[batch_size, input_size])?
        } else {
            x.clone()
        };
        
        // First layer
        let fc1 = ops::matmul(&x_reshaped, &self.fc1_w)?;
        let fc1 = ops::add(&fc1, &self.fc1_b.broadcast_to(&[batch_size, self.fc1_b.shape()[0]])?)?;
        let fc1_relu = ops::relu(&fc1)?;
        
        // Output layer
        let fc2 = ops::matmul(&fc1_relu, &self.fc2_w)?;
        let logits = ops::add(&fc2, &self.fc2_b.broadcast_to(&[batch_size, self.fc2_b.shape()[0]])?)?;
        
        Ok(logits)
    }
    
    pub fn parameters(&self) -> Vec<Tensor<B>> {
        vec![
            self.fc1_w.clone(),
            self.fc1_b.clone(),
            self.fc2_w.clone(),
            self.fc2_b.clone(),
        ]
    }
}

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
        let conv1_w = Tensor::<B>::kaiming_uniform(1 * 5 * 5, &[16, 1, 5, 5], true)?;
        let conv1_b = Tensor::<B>::zeros(&[16], true)?;
        let conv2_w = Tensor::<B>::kaiming_uniform(16 * 5 * 5, &[32, 16, 5, 5], true)?;
        let conv2_b = Tensor::<B>::zeros(&[32], true)?;
        let flat_size = 32 * 7 * 7;
        let fc1_w = Tensor::<B>::kaiming_uniform(flat_size, &[flat_size, 128], true)?;
        let fc1_b = Tensor::<B>::zeros(&[128], true)?;
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
        let n = x.shape()[0];
        let c1 = ops::conv2d(x, &self.conv1_w, Some(&self.conv1_b), (1, 1), (2, 2))?;
        let r1 = ops::relu(&c1)?;
        let p1 = ops::max_pool2d(&r1, (2, 2), (2, 2), (0, 0))?;
        let c2 = ops::conv2d(&p1, &self.conv2_w, Some(&self.conv2_b), (1, 1), (2, 2))?;
        let r2 = ops::relu(&c2)?;
        let p2 = ops::max_pool2d(&r2, (2, 2), (2, 2), (0, 0))?;
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
