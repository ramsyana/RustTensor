// src/data.rs
use crate::{Backend, Error, Tensor};
use rand::seq::index;

#[cfg(feature = "mnist")]
use csv::ReaderBuilder;

#[cfg(feature = "mnist")]
/// Loads data from a CSV file into Tensors for the specified Backend.
/// Assumes the first column is the label (0-9) and the rest are features.
/// Normalizes features to [0, 1] by dividing by 255.0.
/// Converts labels to one-hot encoding (assuming 10 classes).
pub fn load_csv<B: Backend>(path: &str) -> Result<(Tensor<B>, Tensor<B>), Error> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false) // Assuming no header row
        .from_path(path)
        .map_err(Error::CsvError)?;

    let mut features_flat = Vec::new();
    let mut labels = Vec::new();
    let mut num_features: Option<usize> = None;
    let num_classes = 10; // MNIST specific

    for (row_idx, result) in reader.records().enumerate() {
        let record = result.map_err(Error::CsvError)?;

        if record.is_empty() {
            return Err(Error::InvalidOperation(format!(
                "Empty row found in CSV at index {}",
                row_idx
            )));
        }

        // Parse label (first column)
        let label_str = record.get(0).ok_or_else(|| {
            Error::InvalidOperation(format!("Missing label column at row {}", row_idx))
        })?;
        let label = label_str.parse::<f32>().map_err(|e| {
            Error::InvalidOperation(format!(
                "Invalid label format at row {}: '{}' - {}",
                row_idx, label_str, e
            ))
        })?;

        // Validate label
        if !(0.0..num_classes as f32).contains(&label) || label.fract() != 0.0 {
            return Err(Error::InvalidOperation(format!(
                "Invalid label value at row {}: {}. Expected integer 0-{}.",
                row_idx,
                label,
                num_classes - 1
            )));
        }
        labels.push(label as usize); // Store as usize index

        // Parse and normalize features (rest of columns)
        let current_features: Vec<f32> = record
            .iter()
            .skip(1)
            .enumerate()
            .map(|(col_idx, s)| {
                s.parse::<f32>().map_err(|e| {
                    Error::InvalidOperation(format!(
                        "Invalid feature format at row {}, column {}: '{}' - {}",
                        row_idx,
                        col_idx + 1,
                        s,
                        e
                    ))
                })
            })
            .map(|r| r.map(|v| v / 255.0))
            .collect::<Result<_, _>>()?;

        // Determine/validate number of features
        match num_features {
            None => {
                if current_features.is_empty() {
                    return Err(Error::InvalidOperation(
                        "Row with label but no features found.".to_string(),
                    ));
                }
                num_features = Some(current_features.len());
            }
            Some(n_feat) if current_features.len() != n_feat => {
                return Err(Error::InvalidOperation(format!(
                    "Inconsistent number of features at row {}. Expected {}, found {}",
                    row_idx,
                    n_feat,
                    current_features.len()
                )));
            }
            Some(_) => {} // Consistent number of features
        }
        features_flat.extend(current_features);
    }

    let num_samples = labels.len();
    let final_num_features = num_features.unwrap_or(0); // Default to 0 if file was empty

    if num_samples == 0 {
        // Return empty tensors if CSV was empty or header-only
        let empty_features = B::from_vec(vec![], &[0, final_num_features])?;
        let empty_labels = B::from_vec(vec![], &[0, num_classes])?;
        return Ok((
            Tensor::new(empty_features, false),
            Tensor::new(empty_labels, false),
        ));
    }

    // Create one-hot encoded labels
    let mut one_hot = vec![0.0; num_samples * num_classes];
    for (i, &label_idx) in labels.iter().enumerate() {
        // Bounds check already done during parsing
        one_hot[i * num_classes + label_idx] = 1.0;
    }

    // Create storage using the backend's factory method
    let x_storage = B::from_vec(features_flat, &[num_samples, final_num_features])?;
    let y_storage = B::from_vec(one_hot, &[num_samples, num_classes])?;

    Ok((Tensor::new(x_storage, false), Tensor::new(y_storage, false)))
}

pub fn get_random_batch<B: Backend>(
    x: &Tensor<B>,
    y: &Tensor<B>,
    batch_size: usize,
) -> Result<(Tensor<B>, Tensor<B>), Error>
where
    B::Storage: Clone,
{
    let x_shape = x.shape();
    let y_shape = y.shape();

    if x_shape.is_empty() || x_shape[0] == 0 {
        return Err(Error::EmptyTensor);
    }
    if y_shape.is_empty() || y_shape[0] == 0 {
        return Err(Error::EmptyTensor);
    }

    let num_samples = x_shape[0];
    if y_shape[0] != num_samples {
        return Err(Error::ShapeMismatch {
            expected: vec![num_samples],
            actual: vec![y_shape[0]],
        });
    }
    if batch_size == 0 {
        return Err(Error::InvalidOperation(
            "Batch size cannot be zero".to_string(),
        ));
    }

    if batch_size > num_samples {
        return Err(Error::DimensionMismatch(batch_size, num_samples));
    }

    // Get random indices directly using sampling without replacement
    let mut rng = rand::rng(); // Use rand::rng() as recommended
    let batch_indices = index::sample(&mut rng, num_samples, batch_size).into_vec();

    // Determine feature dimensions
    let num_features_x = x_shape.get(1).copied().unwrap_or(1); // Assume 1 if 1D
    let num_classes_y = y_shape.get(1).copied().unwrap_or(1); // Assume 1 if 1D

    // Pre-allocate vectors for batch data
    let mut batch_data_x = Vec::with_capacity(batch_size * num_features_x);
    let mut batch_data_y = Vec::with_capacity(batch_size * num_classes_y);

    // Access underlying data by copying to host Vec<f32>
    let x_ref = x.data();
    let y_ref = y.data();
    // Get data as host Vec<f32>
    let x_data_vec = B::copy_to_host(&*x_ref)?;
    let y_data_vec = B::copy_to_host(&*y_ref)?;

    // Copy data for selected indices
    for &idx in &batch_indices {
        let start_x = idx * num_features_x;
        let end_x = start_x + num_features_x;
        if end_x <= x_data_vec.len() {
            batch_data_x.extend_from_slice(&x_data_vec[start_x..end_x]);
        } else {
            return Err(Error::InternalLogicError(format!(
                "Index out of bounds accessing x data slice at index {}",
                idx
            )));
        }

        let start_y = idx * num_classes_y;
        let end_y = start_y + num_classes_y;
        if end_y <= y_data_vec.len() {
            batch_data_y.extend_from_slice(&y_data_vec[start_y..end_y]);
        } else {
            return Err(Error::InternalLogicError(format!(
                "Index out of bounds accessing y data slice at index {}",
                idx
            )));
        }
    }

    // Create batch tensors
    let batch_x = B::from_vec(batch_data_x, &[batch_size, num_features_x])?;
    let batch_y = B::from_vec(batch_data_y, &[batch_size, num_classes_y])?;

    Ok((Tensor::new(batch_x, false), Tensor::new(batch_y, false)))
}

#[cfg(all(test, feature = "mnist"))]
mod mnist_tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use std::io::{self, Write};
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_csv_with_small_dataset() -> Result<(), Error> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "5,0,1,2,3")?;
        writeln!(temp_file, "8,255,253,251,249")?;

        let path = temp_file.path().to_str().ok_or_else(|| {
            Error::IoError(io::Error::new(
                io::ErrorKind::Other,
                "Temp file path is not valid UTF-8",
            ))
        })?;

        let (x, y) = load_csv::<CpuBackend>(path)?;

        assert_eq!(x.shape(), vec![2, 4]);
        assert_eq!(y.shape(), vec![2, 10]);

        let x_data_ref = x.data();
        let x_slice = x_data_ref.as_ref();
        assert!((x_slice[0] - 0.0 / 255.0).abs() < 1e-6);
        assert!((x_slice[4] - 255.0 / 255.0).abs() < 1e-6);

        let y_data_ref = y.data();
        let y_slice = y_data_ref.as_ref();
        assert_eq!(y_slice[5], 1.0);
        assert_eq!(y_slice[18], 1.0);
        assert_eq!(y_slice[0], 0.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;

    #[test]
    fn test_get_random_batch() -> Result<(), Error> {
        let x = Tensor::<CpuBackend>::new(
            CpuBackend::from_vec(vec![1., 2., 3., 4., 5., 6.], &[3, 2])?,
            false,
        );
        let y = Tensor::<CpuBackend>::new(
            CpuBackend::from_vec(vec![1., 0., 0., 1., 1., 0.], &[3, 2])?,
            false,
        );

        let batch_size = 2;
        let (batch_x, batch_y) = get_random_batch(&x, &y, batch_size)?;

        assert_eq!(batch_x.shape(), vec![batch_size, 2]);
        assert_eq!(batch_y.shape(), vec![batch_size, 2]);
        assert_eq!(batch_x.size(), batch_size * 2);
        Ok(())
    }

    #[test]
    fn test_get_random_batch_size_too_large() -> Result<(), Error> {
        let x = Tensor::<CpuBackend>::new(CpuBackend::zeros(&[1, 2])?, false);
        let y = Tensor::<CpuBackend>::new(CpuBackend::zeros(&[1, 2])?, false);
        let result = get_random_batch(&x, &y, 2);
        assert!(matches!(result, Err(Error::DimensionMismatch(2, 1))));
        Ok(())
    }

    #[test]
    fn test_get_random_batch_zero_size() -> Result<(), Error> {
        let x = Tensor::<CpuBackend>::new(CpuBackend::zeros(&[1, 2])?, false);
        let y = Tensor::<CpuBackend>::new(CpuBackend::zeros(&[1, 2])?, false);
        let result = get_random_batch(&x, &y, 0);
        assert!(
            matches!(result, Err(Error::InvalidOperation(msg)) if msg == "Batch size cannot be zero")
        );
        Ok(())
    }

    #[test]
    fn test_get_random_batch_empty_input() -> Result<(), Error> {
        let x = Tensor::<CpuBackend>::new(CpuBackend::from_vec(vec![], &[0, 2])?, false);
        let y = Tensor::<CpuBackend>::new(CpuBackend::from_vec(vec![], &[0, 2])?, false);
        let result = get_random_batch(&x, &y, 1);
        assert!(matches!(result, Err(Error::EmptyTensor)));
        Ok(())
    }
}
