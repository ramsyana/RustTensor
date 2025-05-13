#[cfg(feature = "cuda")]
use rust_tensor_lib::{Tensor, CpuBackend, CudaBackend, ops};
#[cfg(feature = "cuda")]
use rust_tensor_lib::backend::cuda::{init_context, CudaContextGuard};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Only run this example if CUDA is available
    #[cfg(not(feature = "cuda"))]
    {
        println!("This example requires the 'cuda' feature to be enabled.");
        println!("Run with: cargo run --example cuda_serialization --features=\"cuda,serialization\"");
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        // Initialize CUDA context
        init_context(0)?;
        let _guard = CudaContextGuard::new()?;
        
        // Create a tensor on CPU
        let cpu_tensor = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)?;
        println!("Original CPU tensor: {:?}", cpu_tensor.data());
        
        // Move tensor to GPU
        let cuda_tensor = cpu_tensor.to_gpu(0)?;
        println!("Tensor moved to CUDA device");
        
        // Perform some operations on GPU
        // Create scalar tensors for multiplication and addition
        let scalar_two = Tensor::<CudaBackend>::from_vec(vec![2.0], &[], false)?;
        let scalar_one = Tensor::<CudaBackend>::from_vec(vec![1.0], &[], false)?;
        
        // Multiply by 2 and add 1
        let cuda_tensor = ops::mul(&cuda_tensor, &scalar_two)?;
        let cuda_tensor = ops::add(&cuda_tensor, &scalar_one)?;
        
        // Save the CUDA tensor directly to a file
        // This will automatically transfer data from GPU to CPU during serialization
        println!("Saving CUDA tensor to file...");
        cuda_tensor.save_to_file("cuda_tensor.json")?;
        println!("CUDA tensor saved successfully!");
        
        // Load the tensor back, but to CPU this time using the intermediary pattern
        println!("Loading tensor to CPU using intermediary pattern...");
        
        // First, load the JSON file into a generic Value
        let file_content = std::fs::read_to_string("cuda_tensor.json")?;
        let json_value: serde_json::Value = serde_json::from_str(&file_content)?;
        
        // Create an intermediary from the JSON data
        let data = json_value["data"].as_array().unwrap();
        let shape = json_value["shape"].as_array().unwrap();
        let requires_grad = json_value["requires_grad"].as_bool().unwrap();
        
        // Extract the serialized tensor data
        use rust_tensor_lib::tensor::SerializableTensorIntermediary;
        
        // Create the intermediary with CPU device type
        let intermediary = SerializableTensorIntermediary {
            data_vec: data.iter().map(|v| v.as_f64().unwrap() as f32).collect(),
            shape: shape.iter().map(|v| v.as_u64().unwrap() as usize).collect(),
            grad_vec: None, // We'll ignore gradients for this example
            requires_grad,
            device_type: "cpu".to_string(), // Set to CPU
        };
        
        // Create a CPU tensor from the intermediary
        let loaded_cpu_tensor = Tensor::<CpuBackend>::from_intermediary(intermediary)?;
        println!("Loaded CPU tensor: {:?}", loaded_cpu_tensor.data());
        
        // Load the tensor back to GPU
        println!("Loading tensor to GPU...");
        let loaded_cuda_tensor = Tensor::<CudaBackend>::load_from_file("cuda_tensor.json")?;
        
        // Move the loaded CUDA tensor to CPU for verification
        let verification_tensor = loaded_cuda_tensor.to_cpu()?;
        println!("Verification tensor: {:?}", verification_tensor.data());
        
        // Verify the loaded tensors have the same data
        assert_eq!(loaded_cpu_tensor.shape(), verification_tensor.shape());
        
        // Compare the data values (need to compare on CPU)
        let cpu_tensor_data = loaded_cpu_tensor.data();
        let verification_tensor_data = verification_tensor.data();
        
        // Get the ndarray data
        let cpu_data = cpu_tensor_data.get_data();
        let verification_data = verification_tensor_data.get_data();
        
        // Convert to slices and compare
        let cpu_slice = cpu_data.as_slice().unwrap();
        let verification_slice = verification_data.as_slice().unwrap();
        
        assert_eq!(cpu_slice.len(), verification_slice.len(), "Data length mismatch");
        
        for i in 0..cpu_slice.len() {
            let diff = (cpu_slice[i] - verification_slice[i]).abs();
            assert!(diff < 1e-5, "Values at index {} differ: {} vs {}", i, cpu_slice[i], verification_slice[i]);
        }
        
        println!("Verification successful: loaded tensors match across devices!");
        
        Ok(())
    }
}
