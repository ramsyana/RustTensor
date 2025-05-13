//! Character-level LSTM RNN example using CPU backend.
//!
//! Trains a character-level RNN to predict the next character in a sequence.
//! Implements LSTM cell manually using existing tensor operations.

// Define debug print macros based on feature flag
// When debug_logs feature is enabled, use println!, otherwise use a no-op macro
#[cfg(feature = "debug_logs")]
#[cfg(not(feature = "debug_logs"))]
macro_rules! debug_println {
    ($($arg:tt)*) => {};
}

use rust_tensor_lib::{
    backend::cpu::CpuBackend,
    ops,
    optim::Adam,
    Backend, Error, Tensor, Reduction,
};
use std::collections::HashMap;
use rand::Rng;
use rand::rngs::ThreadRng;
use std::ops::Deref;
use std::time::Instant;

// Constants
const SEQ_LENGTH: usize = 25;
const HIDDEN_SIZE: usize = 64;
const _BATCH_SIZE: usize = 1; // Single batch for simplicity
const NUM_EPOCHS: usize = 100;
const LEARNING_RATE: f32 = 0.01;
const SAMPLE_LENGTH: usize = 200; // Length of text to generate during sampling

/// LSTM parameters struct to hold all weight matrices and biases
struct LstmManualParams<B: Backend> {
    // Input weights: [input_size, hidden_size]
    wx_f: Tensor<B>, // Forget gate input weights
    wx_i: Tensor<B>, // Input gate input weights
    wx_c: Tensor<B>, // Cell gate input weights
    wx_o: Tensor<B>, // Output gate input weights

    // Recurrent weights: [hidden_size, hidden_size]
    uh_f: Tensor<B>, // Forget gate recurrent weights
    uh_i: Tensor<B>, // Input gate recurrent weights
    uh_c: Tensor<B>, // Cell gate recurrent weights
    uh_o: Tensor<B>, // Output gate recurrent weights

    // Biases: [hidden_size]
    bf: Tensor<B>, // Forget gate bias
    bi: Tensor<B>, // Input gate bias
    bc: Tensor<B>, // Cell gate bias
    bo: Tensor<B>, // Output gate bias

    // Output layer
    why: Tensor<B>, // [hidden_size, vocab_size]
    by: Tensor<B>,  // [vocab_size]
}

impl<B: Backend> LstmManualParams<B> {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self, Error> {
        // Input weights: [input_size, hidden_size]
        let wx_f = Tensor::<B>::kaiming_uniform(input_size, &[input_size, hidden_size], true)?;
        let wx_i = Tensor::<B>::kaiming_uniform(input_size, &[input_size, hidden_size], true)?;
        let wx_c = Tensor::<B>::kaiming_uniform(input_size, &[input_size, hidden_size], true)?;
        let wx_o = Tensor::<B>::kaiming_uniform(input_size, &[input_size, hidden_size], true)?;
        // Recurrent weights: [hidden_size, hidden_size]
        let uh_f = Tensor::<B>::kaiming_uniform(hidden_size, &[hidden_size, hidden_size], true)?;
        let uh_i = Tensor::<B>::kaiming_uniform(hidden_size, &[hidden_size, hidden_size], true)?;
        let uh_c = Tensor::<B>::kaiming_uniform(hidden_size, &[hidden_size, hidden_size], true)?;
        let uh_o = Tensor::<B>::kaiming_uniform(hidden_size, &[hidden_size, hidden_size], true)?;
        // Biases
        let forget_bias_data = vec![1.0; hidden_size];
        let bf = Tensor::<B>::from_vec(forget_bias_data, &[hidden_size], true)?;
        let bi = Tensor::<B>::zeros(&[hidden_size], true)?;
        let bc = Tensor::<B>::zeros(&[hidden_size], true)?;
        let bo = Tensor::<B>::zeros(&[hidden_size], true)?;
        // Output layer: why [hidden_size, output_size], by [output_size]
        let why = Tensor::<B>::kaiming_uniform(hidden_size, &[hidden_size, output_size], true)?;
        let by = Tensor::<B>::zeros(&[output_size], true)?;
        Ok(Self {
            wx_f, wx_i, wx_c, wx_o,
            uh_f, uh_i, uh_c, uh_o,
            bf, bi, bc, bo,
            why, by,
        })
    }
    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![
            self.wx_f.clone(), self.wx_i.clone(), self.wx_c.clone(), self.wx_o.clone(),
            self.uh_f.clone(), self.uh_i.clone(), self.uh_c.clone(), self.uh_o.clone(),
            self.bf.clone(), self.bi.clone(), self.bc.clone(), self.bo.clone(),
            self.why.clone(), self.by.clone(),
        ]
    }
}

/// LSTM cell implementation using manual operations
fn lstm_cell<B: Backend>(
    x: &Tensor<B>,
    h_prev: &Tensor<B>,
    c_prev: &Tensor<B>,
    params: &LstmManualParams<B>
) -> Result<(Tensor<B>, Tensor<B>), Error> {
    let batch_size = x.shape()[0];
    let hidden_size = h_prev.shape()[1];
    // Forget gate
    let f_gate_input = ops::add(
        &ops::matmul(x, &params.wx_f)?,
        &ops::matmul(h_prev, &params.uh_f)?
    )?;
    let f_gate = ops::sigmoid(&ops::add(&f_gate_input, &params.bf.broadcast_to(&[batch_size, hidden_size])?)?)?;
    // Input gate
    let i_gate_input = ops::add(
        &ops::matmul(x, &params.wx_i)?,
        &ops::matmul(h_prev, &params.uh_i)?
    )?;
    let i_gate = ops::sigmoid(&ops::add(&i_gate_input, &params.bi.broadcast_to(&[batch_size, hidden_size])?)?)?;
    // Cell candidate
    let c_hat_candidate = ops::add(
        &ops::matmul(x, &params.wx_c)?,
        &ops::matmul(h_prev, &params.uh_c)?
    )?;
    let c_hat = ops::tanh(&ops::add(&c_hat_candidate, &params.bc.broadcast_to(&[batch_size, hidden_size])?)?)?;
    // Output gate
    let o_gate_input = ops::add(
        &ops::matmul(x, &params.wx_o)?,
        &ops::matmul(h_prev, &params.uh_o)?
    )?;
    let o_gate = ops::sigmoid(&ops::add(&o_gate_input, &params.bo.broadcast_to(&[batch_size, hidden_size])?)?)?;
    // Next cell state
    let fc = ops::mul(&f_gate, c_prev)?;
    let ic = ops::mul(&i_gate, &c_hat)?;
    let c_next = ops::add(&fc, &ic)?;
    // Next hidden state
    let tanh_c = ops::tanh(&c_next)?;
    let h_next = ops::mul(&o_gate, &tanh_c)?;
    Ok((h_next, c_next))
}

/// Forward pass through the LSTM network
fn _forward<B: Backend>(
    inputs: &[Tensor<B>],
    h0: &Tensor<B>,
    c0: &Tensor<B>,
    params: &LstmManualParams<B>
) -> Result<(Vec<Tensor<B>>, Vec<Tensor<B>>, Vec<Tensor<B>>), Error> {
    let mut h = h0.clone();
    let mut c = c0.clone();
    let mut hidden_states = Vec::with_capacity(inputs.len() + 1);
    let mut cell_states = Vec::with_capacity(inputs.len() + 1);
    let mut outputs = Vec::with_capacity(inputs.len());
    
    hidden_states.push(h.clone());
    cell_states.push(c.clone());
    
    for input in inputs {
        // Process through LSTM cell
        let (h_next, c_next) = lstm_cell(input, &h, &c, params)?;
        h = h_next;
        c = c_next;
        
        // Store states
        hidden_states.push(h.clone());
        cell_states.push(c.clone());
        
        // Compute output - matrix multiply h [1, hidden_size] with why [hidden_size, vocab_size]
        // The matrix multiplication is h [1, hidden_size] * why [hidden_size, vocab_size] = [1, vocab_size]
        let output = ops::matmul(&h, &params.why)?;
        // Add bias and push to outputs
        let output = ops::add(&output, &params.by.broadcast_to(&[h.shape()[0], params.by.shape()[0]])?)?;
        outputs.push(output);
    }
    
    Ok((hidden_states, cell_states, outputs))
}

/// Generate a sample text using the trained model
// Sample function with generic type parameter
fn sample<B: Backend>(
    seed_char: usize,
    _char_to_ix: &HashMap<char, usize>,
    ix_to_char: &HashMap<usize, char>,
    h: &Tensor<B>,
    c: &Tensor<B>,
    params: &LstmManualParams<B>,
    length: usize,
    vocab_size: usize
) -> Result<String, Error> {
    let mut result = String::new();
    let mut curr_char_ix = seed_char;
    let mut curr_h = h.clone();
    let mut curr_c = c.clone();
    let mut rng = ThreadRng::default();
    for _ in 0..length {
        // Create one-hot encoding for current character
        let mut x_data = vec![0.0; vocab_size];
        x_data[curr_char_ix] = 1.0;
        let x = Tensor::<B>::from_vec(x_data, &[1, vocab_size], false)?;
        // Forward pass through LSTM cell
        let (next_h, next_c) = lstm_cell(&x, &curr_h, &curr_c, params)?;
        curr_h = next_h;
        curr_c = next_c;
        // Get output logits: [1, vocab_size]
        let logits = ops::matmul(&curr_h, &params.why)?;
        let logits = ops::add(&logits, &params.by.broadcast_to(&[1, vocab_size])?)?;
        // Use log_softmax + exp for probabilities
        let log_probs = ops::log_softmax(&logits, 1)?;
        let probs = ops::exp(&log_probs)?;
        // Multinomial sampling
        let mut next_char_ix = 0;
        let r: f32 = rng.random_range(0.0..1.0);
        let mut cumsum = 0.0;
        let shape = probs.shape();
        let size = shape.iter().product::<usize>();
        let mut probs_vec = vec![0.0; size];
        if let Ok(cpu_tensor) = probs.to_cpu() {
            let data_ref = cpu_tensor.data();
            if let Some(slice) = data_ref.deref().get_data().as_slice() {
                probs_vec = slice.to_vec();
            }
        }
        for (i, &p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                next_char_ix = i;
                break;
            }
        }
        if let Some(&c) = ix_to_char.get(&next_char_ix) {
            result.push(c);
        }
        curr_char_ix = next_char_ix;
    }
    Ok(result)
}

/// Generate text using the LSTM model following the steps outlined in the documentation
/// 
/// This function implements text generation with the following steps:
/// 1. Initialize h, c to zeros
/// 2. Create one-hot vector for the seed character
/// 3. Loop for the desired generation length:
///    a. Process through LSTM cell
///    b. Calculate logits
///    c. Apply softmax to get probabilities
///    d. Sample the next character (or use argmax for greedy sampling)
///    e. Convert next index to character and append to result
///    f. Create one-hot vector for next character
fn generate_text<B: Backend>(
    params: &LstmManualParams<B>,
    ix_to_char: &HashMap<usize, char>,
    _char_to_ix: &HashMap<char, usize>,
    seed_char_ix: usize,
    length: usize,
) -> Result<String, Error> {
    let vocab_size = ix_to_char.len();
    let batch_size = 1;
    
    // Initialize h and c to zeros as per the requirements
    let h = Tensor::<B>::zeros(&[batch_size, HIDDEN_SIZE], false)?;
    let c = Tensor::<B>::zeros(&[batch_size, HIDDEN_SIZE], false)?;
    
    // Create result string and initialize with the seed character
    let mut result = String::new();
    if let Some(&ch) = ix_to_char.get(&seed_char_ix) {
        result.push(ch);
    }
    
    // Create one-hot vector for the seed character
    let mut x_data = vec![0.0; vocab_size];
    x_data[seed_char_ix] = 1.0;
    let mut x = Tensor::<B>::from_vec(x_data, &[batch_size, vocab_size], false)?;
    
    // Current hidden and cell states
    let mut curr_h = h;
    let mut curr_c = c;
    
    // Loop for the desired generation length
    for _ in 0..length {
        // 1. Process through LSTM cell
        let (next_h, next_c) = lstm_cell(&x, &curr_h, &curr_c, params)?;
        curr_h = next_h;
        curr_c = next_c;
        
        // 2. Calculate logits: logits = ops::add(&ops::matmul(&h, &params.Why)?, &params.by.broadcast_to(...)?)?
        let logits = ops::matmul(&curr_h, &params.why)?;
        let logits = ops::add(&logits, &params.by.broadcast_to(&[batch_size, vocab_size])?)?;
        
        // 3. Apply log_softmax and then exp to get probabilities
        let log_probs = ops::log_softmax(&logits, 1)?;
        let probs = ops::exp(&log_probs)?;
        
        // 4. Sample the next character index (or use argmax for greedy sampling)
        // Option 1: Greedy sampling (argmax)
        // let next_ix = ops::argmax(&probs, 1, false)?;
        // let next_ix_val = next_ix.to_cpu()?.data().deref().get_data().as_slice().unwrap()[0] as usize;
        
        // Option 2: Probabilistic sampling
        let mut next_ix_val = 0;
        let mut rng = ThreadRng::default();
        let r: f32 = rng.random_range(0.0..1.0);
        let mut cumsum = 0.0;
        
        // Get probability values from tensor
        let probs_cpu = probs.to_cpu()?;
        let probs_data = probs_cpu.data().deref().get_data().as_slice().unwrap().to_vec();
        
        for (i, &p) in probs_data.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                next_ix_val = i;
                break;
            }
        }
        
        // 5. Convert next_ix back to a character and append to the result string
        if let Some(&ch) = ix_to_char.get(&next_ix_val) {
            result.push(ch);
        }
        
        // 6. Create the one-hot vector for next_ix to be the input for the next step
        let mut next_x_data = vec![0.0; vocab_size];
        next_x_data[next_ix_val] = 1.0;
        x = Tensor::<B>::from_vec(next_x_data, &[batch_size, vocab_size], false)?;
    }
    
    Ok(result)
}

fn main() -> Result<(), Error> {
    // Define a simple input text
    let text = "The quick brown fox jumps over the lazy dog. \
                The five boxing wizards jump quickly. \
                How vexingly quick daft zebras jump!";
    
    // Create vocabulary
    let mut chars = std::collections::HashSet::new();
    for c in text.chars() {
        chars.insert(c);
    }
    
    let mut char_to_ix = HashMap::new();
    let mut ix_to_char = HashMap::new();
    
    for (i, c) in chars.iter().enumerate() {
        char_to_ix.insert(*c, i);
        ix_to_char.insert(i, *c);
    }
    
    let vocab_size = chars.len();
    println!("Vocabulary size: {}", vocab_size);
    
    // Convert text to indices
    let data_ix: Vec<usize> = text.chars().map(|c| *char_to_ix.get(&c).unwrap()).collect();
    
    // Prepare training data (input/target pairs)
    // Create sequence pairs for training
    let mut sequence_pairs = Vec::new();
    
    for i in 0..(data_ix.len() - SEQ_LENGTH - 1) {
        let input_seq = data_ix[i..(i + SEQ_LENGTH)].to_vec();
        let target_seq = data_ix[(i + 1)..(i + SEQ_LENGTH + 1)].to_vec();
        sequence_pairs.push((input_seq, target_seq));
    }
    
    println!("Created {} training sequences", sequence_pairs.len());
    
    // Initialize model parameters
    let params = LstmManualParams::<CpuBackend>::new(vocab_size, HIDDEN_SIZE, vocab_size)?;
    
    // Initialize optimizer
    let mut optimizer = Adam::new(
        params.parameters(),
        LEARNING_RATE,
        0.9,    // beta1
        0.999,  // beta2
        1e-8    // epsilon
    )?;
    
    // Training loop
    let start_time = Instant::now();
    
    for epoch in 0..NUM_EPOCHS {
        let mut total_loss = 0.0;
        let mut sequence_count = 0;
        
        // Initialize h_prev and c_prev to zero tensors
        let mut _h_prev = Tensor::<CpuBackend>::zeros(&[1, HIDDEN_SIZE], false)?;
        let mut _c_prev = Tensor::<CpuBackend>::zeros(&[1, HIDDEN_SIZE], false)?;
        
        // Iterate through sequence pairs
        for (seq_idx, (inputs_ix, targets_ix)) in sequence_pairs.iter().enumerate() {
            // Zero gradients
            optimizer.zero_grad()?;
            
            // Detach h_prev and c_prev from previous sequence's graph
            // We do this by creating new tensors with the same data
            let mut h = Tensor::<CpuBackend>::zeros(&[1, HIDDEN_SIZE], true)?;
            let mut c = Tensor::<CpuBackend>::zeros(&[1, HIDDEN_SIZE], true)?;
            
            // Initialize sequence loss
            let mut sequence_loss = Tensor::<CpuBackend>::zeros(&[], true)?;
            
            // Unroll LSTM: Loop through each timestep in the sequence
            for t in 0..SEQ_LENGTH {
                // Create one-hot input tensor for inputs_ix[t]
                let mut x_data = vec![0.0; vocab_size];
                x_data[inputs_ix[t]] = 1.0;
                let x_t = Tensor::<CpuBackend>::from_vec(x_data, &[1, vocab_size], true)?;
                
                // Process through LSTM cell
                let (h_new, c_new) = lstm_cell(&x_t, &h, &c, &params)?;
                
                // Calculate output logits
                let logits = ops::add(
                    &ops::matmul(&h_new, &params.why)?,
                    &params.by.broadcast_to(&[1, vocab_size])?
                )?;
                
                // Create one-hot target tensor for targets_ix[t]
                // Note: target tensors should NOT require gradients
                let mut y_target_data = vec![0.0; vocab_size];
                y_target_data[targets_ix[t]] = 1.0;
                let y_target_t = Tensor::<CpuBackend>::from_vec(y_target_data, &[1, vocab_size], false)?;
                
                // Calculate loss for this step using softmax cross entropy
                let step_loss = ops::softmax_cross_entropy(&logits, &y_target_t, 1, Reduction::Mean)?;
                
                // Accumulate loss
                sequence_loss = ops::add(&sequence_loss, &step_loss)?;
                
                // Update h and c for next timestep
                h = h_new;
                c = c_new;
            }
            
            // Backpropagate through the unrolled steps
            sequence_loss.backward()?;
            
            // Optional: Implement gradient clipping if needed
            // (Not implemented here for simplicity)
            
            // Update parameters
            optimizer.step()?;
            
            // Update h_prev and c_prev for next sequence
            _h_prev = h;
            _c_prev = c;
            
            // Get loss value
            let mut loss_val = 0.0;
            if let Ok(cpu_tensor) = sequence_loss.to_cpu() {
                let data_ref = cpu_tensor.data();
                if let Some(slice) = data_ref.deref().get_data().as_slice() {
                    if !slice.is_empty() {
                        loss_val = slice[0];
                    }
                }
            }
            
            total_loss += loss_val;
            sequence_count += 1;
            
            // Log progress periodically
            if (seq_idx + 1) % 100 == 0 || seq_idx == sequence_pairs.len() - 1 {
                let elapsed = start_time.elapsed().as_secs_f32();
                let avg_loss = total_loss / sequence_count as f32;
                println!("Epoch {}/{}, Sequence {}/{}: Avg Loss = {:.6}, Time: {:.2}s", 
                         epoch + 1, NUM_EPOCHS, seq_idx + 1, sequence_pairs.len(), avg_loss, elapsed);
            }
        }
        
        // Print epoch summary
        let avg_epoch_loss = total_loss / sequence_count as f32;
        println!("Epoch {}/{} completed: Avg Loss = {:.6}", epoch + 1, NUM_EPOCHS, avg_epoch_loss);
        
        // Generate sample text every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 || epoch == NUM_EPOCHS - 1 {
            // Use a random seed character
            let mut rng = ThreadRng::default();
            let seed_idx = data_ix[rng.random_range(0..data_ix.len())];
            
            // Initialize h and c for sampling
            let h0 = Tensor::<CpuBackend>::zeros(&[1, HIDDEN_SIZE], false)?;
            let c0 = Tensor::<CpuBackend>::zeros(&[1, HIDDEN_SIZE], false)?;
            
            // Generate text using the original sample function
            let sample_text = sample(
                seed_idx,
                &char_to_ix,
                &ix_to_char,
                &h0,
                &c0,
                &params,
                SAMPLE_LENGTH,
                vocab_size
            )?;
            
            println!("\nSample text at epoch {} (using sample function):", epoch + 1);
            println!("{}", sample_text);
            
            // Generate text using the new generate_text function
            let generated_text = generate_text(
                &params,
                &ix_to_char,
                &char_to_ix,
                seed_idx,
                SAMPLE_LENGTH
            )?;
            
            println!("\nGenerated text at epoch {} (using generate_text function):", epoch + 1);
            println!("{}", generated_text);
        }
    }
    
    Ok(())
}
