//! Sine wave regression example using a simple MLP on CPU backend.
//!
//! Trains to fit noisy y = sin(x) using MSE loss. Logs training loss per epoch.

use rust_tensor_lib::{
    backend::cpu::CpuBackend,
    ops,
    optim::Adam,
    Backend, Error, Tensor, Reduction,
};
use rand::Rng;
// thread_rng is deprecated, use rand::thread_rng() directly
use rand_distr::StandardNormal;
use std::f32::consts::PI;

const NUM_SAMPLES: usize = 512;
const NUM_EPOCHS: usize = 2000;
const LEARNING_RATE: f32 = 0.01;
const HIDDEN_SIZE: usize = 32;
const NOISE_STD_DEV: f32 = 0.15;

struct SimpleMlp<B: Backend> {
    linear1_w: Tensor<B>,
    linear1_b: Tensor<B>,
    linear2_w: Tensor<B>,
    linear2_b: Tensor<B>,
}

impl<B: Backend> SimpleMlp<B> {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self, Error> {
        // Kaiming uniform init for weights, zeros for biases
        let linear1_w = Tensor::<B>::kaiming_uniform(input_size, &[input_size, hidden_size], true)?;
        let linear1_b = Tensor::<B>::zeros(&[hidden_size], true)?;
        let linear2_w = Tensor::<B>::kaiming_uniform(hidden_size, &[hidden_size, output_size], true)?;
        let linear2_b = Tensor::<B>::zeros(&[output_size], true)?;
        Ok(Self { linear1_w, linear1_b, linear2_w, linear2_b })
    }

    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>, Error> {
        let n = x.shape()[0];
        let h = ops::add(&ops::matmul(x, &self.linear1_w)?, &self.linear1_b.broadcast_to(&[n, HIDDEN_SIZE])?)?;
        let h_act = ops::tanh(&h)?;
        let y_pred = ops::add(&ops::matmul(&h_act, &self.linear2_w)?, &self.linear2_b.broadcast_to(&[n, 1])?)?;
        Ok(y_pred)
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![
            self.linear1_w.clone(),
            self.linear1_b.clone(),
            self.linear2_w.clone(),
            self.linear2_b.clone(),
        ]
    }
}

fn main() -> Result<(), Error> {
    // Generate synthetic data
    let mut rng = rand::rng();
    let mut x_vals = Vec::with_capacity(NUM_SAMPLES);
    let mut y_vals = Vec::with_capacity(NUM_SAMPLES);
    for i in 0..NUM_SAMPLES {
        let x = -PI + 2.0 * PI * (i as f32) / (NUM_SAMPLES as f32);
        let noise: f32 = rng.sample::<f32, _>(StandardNormal) * NOISE_STD_DEV;
        let y = x.sin() + noise;
        x_vals.push(x);
        y_vals.push(y);
    }
    let x_tensor = Tensor::<CpuBackend>::from_vec(x_vals, &[NUM_SAMPLES, 1], false)?;
    let y_noisy_tensor = Tensor::<CpuBackend>::from_vec(y_vals, &[NUM_SAMPLES, 1], false)?;

    // Model and optimizer
    let model = SimpleMlp::<CpuBackend>::new(1, HIDDEN_SIZE, 1)?;
    let mut optimizer = Adam::new(
        model.parameters(),
        LEARNING_RATE,
        0.9,    // beta1
        0.999,  // beta2
        1e-8    // epsilon
    )?;

    // Training loop
    for epoch in 0..NUM_EPOCHS {
        optimizer.zero_grad()?;
        let y_pred = model.forward(&x_tensor)?;
        let loss = ops::mse_loss(&y_pred, &y_noisy_tensor, Reduction::Mean)?;
        loss.backward()?;
        optimizer.step()?;
        if epoch % 100 == 0 || epoch == NUM_EPOCHS - 1 {
            let loss_vec = loss.data().as_ref().to_vec();
            let loss_val = loss_vec[0];
            println!("Epoch {:4}: MSE loss = {:.6}", epoch, loss_val);
        }
    }

    // Optional: Evaluate on dense grid for plotting
    let dense_points = 200;
    let mut dense_x = Vec::with_capacity(dense_points);
    for i in 0..dense_points {
        let x = -PI + 2.0 * PI * (i as f32) / (dense_points as f32);
        dense_x.push(x);
    }
    let dense_x_tensor = Tensor::<CpuBackend>::from_vec(dense_x.clone(), &[dense_points, 1], false)?;
    let y_final_pred = model.forward(&dense_x_tensor)?;
    let y_final_pred_vec = y_final_pred.data().as_ref().to_vec();

    // Print a few predictions as a sample
    println!("\nSample predictions after training:");
    for i in (0..dense_points).step_by(dense_points / 10) {
        println!("x = {:>6.3}, pred = {:>7.4}, true = {:>7.4}", dense_x[i], y_final_pred_vec[i], dense_x[i].sin());
    }

    // Plotting: You can use plotters crate to visualize results.
    // Plot dense_x vs y_final_pred (model), dense_x vs sin(x) (true), x_vals vs y_vals (noisy data)
    // See plotters documentation for details.

    Ok(())
}
