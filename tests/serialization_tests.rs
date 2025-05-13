#![cfg(feature = "serialization")] // Only compile this test module if "serialization" is enabled

use rust_tensor_lib::{Backend, CpuBackend, Error, Tensor};

// Import CUDA-specific types when the cuda feature is enabled
#[cfg(feature = "cuda")]
use rust_tensor_lib::CudaBackend;

#[cfg(feature = "cuda")]
use rust_tensor_lib::backend::cuda::{init_context, CudaContextGuard};

#[cfg(feature = "cuda")]
use serial_test::serial; // For sequential CUDA tests

// Import the intermediate serialization type
#[cfg(feature = "cuda")]
use rust_tensor_lib::tensor::SerializableTensorIntermediary;

fn check_tensor_equality<B: Backend>(t1: &Tensor<B>, t2: &Tensor<B>) -> Result<(), Error> {
    assert_eq!(t1.shape(), t2.shape(), "Shapes differ");
    assert_eq!(t1.requires_grad(), t2.requires_grad(), "requires_grad differs");
    assert_eq!(t1.device(), t2.device(), "device differs");

    let d1_vec = B::copy_to_host(&*t1.data())?;
    let d2_vec = B::copy_to_host(&*t2.data())?;
    assert_eq!(d1_vec, d2_vec, "Data differs");

    match (t1.grad(), t2.grad()) {
        (Some(g1_ref), Some(g2_ref)) => {
            let g1_vec = B::copy_to_host(&*g1_ref)?;
            let g2_vec = B::copy_to_host(&*g2_ref)?;
            assert_eq!(g1_vec, g2_vec, "Gradients differ");
        }
        (None, None) => { /* Both None, okay */ }
        _ => panic!("Gradient presence differs"),
    }
    Ok(())
}

#[test]
fn test_cpu_tensor_serialization_deserialization() -> Result<(), Error> {
    let original_tensor = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0, 3.0], &[3], true)?;
    original_tensor.set_grad(Some(CpuBackend::from_vec(vec![0.1, 0.2, 0.3], &[3])?));

    let serialized = serde_json::to_string(&original_tensor)
        .map_err(|e| Error::InternalLogicError(format!("Serialization failed: {}", e)))?;
    println!("CPU Serialized: {}", serialized);

    let deserialized_tensor: Tensor<CpuBackend> = serde_json::from_str(&serialized)
        .map_err(|e| Error::InternalLogicError(format!("Deserialization failed: {}", e)))?;

    check_tensor_equality(&original_tensor, &deserialized_tensor)?;
    Ok(())
}

#[test]
fn test_cpu_tensor_serialization_no_grad() -> Result<(), Error> {
    let original_tensor = Tensor::<CpuBackend>::from_vec(vec![1.0, 2.0], &[2], false)?;

    let serialized = serde_json::to_string(&original_tensor).unwrap();
    let deserialized_tensor: Tensor<CpuBackend> = serde_json::from_str(&serialized).unwrap();

    check_tensor_equality(&original_tensor, &deserialized_tensor)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[serial]
#[test]
fn test_cuda_tensor_serialization_deserialization() -> Result<(), Error> {
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let original_tensor = Tensor::<CudaBackend>::from_vec(vec![10.0, 20.0, 30.0], &[3], true)?;
    original_tensor.set_grad(Some(CudaBackend::from_vec(vec![1.1, 2.2, 3.3], &[3])?));

    let serialized = serde_json::to_string(&original_tensor)
        .map_err(|e| Error::InternalLogicError(format!("Serialization failed: {}", e)))?;
    println!("CUDA Serialized: {}", serialized);

    // Deserialize back into a CUDA tensor. This requires an active CUDA context.
    let deserialized_tensor: Tensor<CudaBackend> = serde_json::from_str(&serialized)
        .map_err(|e| Error::InternalLogicError(format!("Deserialization failed: {}", e)))?;

    check_tensor_equality(&original_tensor, &deserialized_tensor)?;
    Ok(())
}

#[cfg(feature = "cuda")]
#[serial]
#[test]
fn test_cuda_to_cpu_serialization() -> Result<(), Error> {
    // Scenario: Serialize a CUDA tensor, deserialize as CPU tensor
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    let cuda_tensor = Tensor::<CudaBackend>::from_vec(vec![7.0, 8.0], &[2], false)?;
    let serialized_cuda = serde_json::to_string(&cuda_tensor).unwrap();
    println!("CUDA Serialized for CPU load: {}", serialized_cuda);

    // Attempt to deserialize as a CPU tensor
    let deserialized_as_cpu: Result<Tensor<CpuBackend>, _> = serde_json::from_str(&serialized_cuda);
    
    // This SHOULD fail with the current Deserialize impl for Tensor<B> if device types mismatch
    match deserialized_as_cpu {
        Ok(_) => panic!("Should have failed to deserialize CUDA tensor as CPU tensor directly with current impl."),
        Err(e) => {
            println!("Correctly failed to deserialize CUDA as CPU: {}", e);
            assert!(e.to_string().contains("Device mismatch"));
        }
    }
    
    // We need to convert the standard serialization format to our intermediary format
    // First, let's parse the JSON to get the raw data
    let json_value: serde_json::Value = serde_json::from_str(&serialized_cuda)
        .map_err(|e| Error::InternalLogicError(format!("Failed to parse JSON: {}", e)))?;
    
    // Extract the fields we need
    let data = json_value["data"].as_array().unwrap();
    let shape = json_value["shape"].as_array().unwrap();
    let requires_grad = json_value["requires_grad"].as_bool().unwrap();
    
    // Convert to our intermediary format
    let intermediary = SerializableTensorIntermediary {
        data_vec: data.iter().map(|v| v.as_f64().unwrap() as f32).collect(),
        shape: shape.iter().map(|v| v.as_u64().unwrap() as usize).collect(),
        grad_vec: None, // No gradient in this test
        requires_grad,
        device_type: "cuda".to_string(),
    };
    
    // Create a CPU-compatible intermediary by changing the device type
    let cpu_intermediary = SerializableTensorIntermediary {
        data_vec: intermediary.data_vec.clone(),
        shape: intermediary.shape.clone(),
        grad_vec: intermediary.grad_vec.clone(),
        requires_grad: intermediary.requires_grad,
        device_type: "cpu".to_string(),
    };
    
    // Use the from_intermediary method to create a CPU tensor
    let manually_converted_cpu_tensor = Tensor::<CpuBackend>::from_intermediary(cpu_intermediary)?;

    // Now check equality with an equivalent CPU tensor
    let original_cpu_equivalent = Tensor::<CpuBackend>::from_vec(vec![7.0, 8.0], &[2], false)?;
    
    // Check only the data values, not the entire tensor (since device will differ)
    let d1_vec = CpuBackend::copy_to_host(&*manually_converted_cpu_tensor.data())?;
    let d2_vec = CpuBackend::copy_to_host(&*original_cpu_equivalent.data())?;
    assert_eq!(d1_vec, d2_vec, "Data differs");
    assert_eq!(manually_converted_cpu_tensor.shape(), original_cpu_equivalent.shape(), "Shapes differ");
    assert_eq!(manually_converted_cpu_tensor.requires_grad(), original_cpu_equivalent.requires_grad(), "requires_grad differs");

    Ok(())
}

// Test using the to_intermediary and from_intermediary methods for cross-device serialization
#[cfg(feature = "cuda")]
#[serial]
#[test]
fn test_cross_device_serialization_with_intermediary() -> Result<(), Error> {
    // Initialize CUDA context
    init_context(0)?;
    let _guard = CudaContextGuard::new()?;

    // Create a CUDA tensor
    let cuda_tensor = Tensor::<CudaBackend>::from_vec(vec![1.5, 2.5, 3.5], &[3], true)?;
    cuda_tensor.set_grad(Some(CudaBackend::from_vec(vec![0.5, 0.6, 0.7], &[3])?));
    
    // Convert to intermediary representation
    let intermediary = cuda_tensor.to_intermediary()?;
    
    // Verify intermediary properties
    assert_eq!(intermediary.device_type, "cuda");
    assert_eq!(intermediary.shape, vec![3]);
    assert_eq!(intermediary.data_vec, vec![1.5, 2.5, 3.5]);
    assert!(intermediary.requires_grad);
    assert!(intermediary.grad_vec.is_some());
    
    // Attempt to create a CPU tensor from the intermediary
    // This should fail because the device types don't match
    let cpu_from_cuda_result = Tensor::<CpuBackend>::from_intermediary(intermediary.clone());
    assert!(cpu_from_cuda_result.is_err());
    let err_msg = cpu_from_cuda_result.unwrap_err().to_string();
    assert!(err_msg.contains("Device mismatch"));
    
    // Create a new intermediary with CPU device type
    let cpu_intermediary = SerializableTensorIntermediary {
        data_vec: intermediary.data_vec.clone(),
        shape: intermediary.shape.clone(),
        grad_vec: intermediary.grad_vec.clone(),
        requires_grad: intermediary.requires_grad,
        device_type: "cpu".to_string(),
    };
    
    // Now create a CPU tensor from the modified intermediary
    let cpu_tensor = Tensor::<CpuBackend>::from_intermediary(cpu_intermediary)?;
    
    // Verify the CPU tensor has the correct data
    assert_eq!(cpu_tensor.shape(), &[3]);
    assert_eq!(CpuBackend::copy_to_host(&*cpu_tensor.data())?, vec![1.5, 2.5, 3.5]);
    assert!(cpu_tensor.requires_grad());
    assert!(cpu_tensor.grad().is_some());
    
    Ok(())
}
