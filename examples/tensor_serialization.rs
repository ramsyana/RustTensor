use rust_tensor_lib::{Tensor, CpuBackend, ops};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Create a tensor with some data
    let tensor = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true)?;
    println!("Original tensor: {:?}", tensor.data());
    
    // Apply some operations to create a more complex tensor
    // Create scalar tensors for multiplication and addition
    let scalar_two = Tensor::<CpuBackend>::from_vec(vec![2.0], &[], false)?;
    let scalar_one = Tensor::<CpuBackend>::from_vec(vec![1.0], &[], false)?;
    
    // Multiply by 2 and add 1
    let tensor = ops::mul(&tensor, &scalar_two)?;
    let tensor = ops::add(&tensor, &scalar_one)?;
    println!("Modified tensor: {:?}", tensor.data());
    
    // Save the tensor to a file
    println!("Saving tensor to file...");
    tensor.save_to_file("tensor.json")?;
    println!("Tensor saved successfully!");
    
    // Load the tensor from the file
    println!("Loading tensor from file...");
    let loaded_tensor = Tensor::<CpuBackend>::load_from_file("tensor.json")?;
    println!("Loaded tensor: {:?}", loaded_tensor.data());
    
    // Verify the tensors are the same
    assert_eq!(tensor.shape(), loaded_tensor.shape());
    
    // Compare the data values - properly handling the temporary values
    let tensor_data = tensor.data();
    let loaded_tensor_data = loaded_tensor.data();
    
    // Get the ndarray data
    let original_data = tensor_data.get_data();
    let loaded_data = loaded_tensor_data.get_data();
    
    // Convert to slices and compare
    let original_slice = original_data.as_slice().unwrap();
    let loaded_slice = loaded_data.as_slice().unwrap();
    
    assert_eq!(original_slice.len(), loaded_slice.len(), "Data length mismatch");
    
    for i in 0..original_slice.len() {
        let diff = (original_slice[i] - loaded_slice[i]).abs();
        assert!(diff < 1e-5, "Values at index {} differ: {} vs {}", i, original_slice[i], loaded_slice[i]);
    }
    // Data comparison is done above
    println!("Verification successful: loaded tensor matches original!");
    
    // Demonstrate that requires_grad is preserved
    println!("Original requires_grad: {}", tensor.requires_grad());
    println!("Loaded requires_grad: {}", loaded_tensor.requires_grad());
    
    // Show that we can continue using the loaded tensor in computations
    let result = ops::mean(&loaded_tensor, None)?;
    println!("Mean of loaded tensor: {:?}", result.data());
    
    Ok(())
}
