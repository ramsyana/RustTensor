use crate::array::Array;
use crate::error::Error;
use rand_distr::{Distribution, Uniform};

/// Initialize weights using Kaiming uniform initialization.
/// Returns an `Array` suitable for `CpuBackend`.
pub fn kaiming_uniform(fan_in: usize, shape: &[usize]) -> Result<Array, Error> {
    if shape.is_empty() {
        return Err(Error::InvalidOperation(
            "Cannot initialize tensor with empty shape".to_string(),
        ));
    }

    // Check for zero dimensions - return empty array if found
    if shape.iter().any(|&dim| dim == 0) {
        return Ok(Array::zeros(shape));
    }

    if fan_in == 0 {
        return Err(Error::InvalidOperation(
            "Fan-in cannot be zero for Kaiming initialization".to_string(),
        ));
    }

    // Calculate bounds for uniform distribution: bound = sqrt(6 / fan_in)
    let bound = (6.0 / fan_in as f32).sqrt();
    if bound.is_nan() || bound.is_infinite() {
        return Err(Error::InitializationError);
    }

    // Uniform::new can fail if low > high, though not possible here with -bound, bound
    let dist = Uniform::new(-bound, bound).map_err(|_| Error::InitializationError)?;

    let mut rng = rand::rng();

    // Use checked_product for potentially large shapes
    let size: usize = shape
        .iter()
        .try_fold(1usize, |acc, &x| acc.checked_mul(x))
        .ok_or_else(|| {
            Error::InvalidOperation("Shape dimensions multiply to overflow usize".to_string())
        })?;

    // Generate data directly into Vec::with_capacity for potential efficiency
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        data.push(dist.sample(&mut rng));
    }

    // Create Array from the random data
    Array::from_vec(data, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaiming_uniform_basic() {
        let fan_in = 784;
        let shape = &[128, fan_in];
        let result = kaiming_uniform(fan_in, shape);
        assert!(result.is_ok());
        let array = result.unwrap();

        assert_eq!(array.shape(), shape);
        let bound = (6.0 / fan_in as f32).sqrt();
        let data_vec = array.into_raw_vec();
        assert_eq!(data_vec.len(), 128 * 784);
        for &value in &data_vec {
            assert!(
                value >= -bound && value <= bound,
                "Value {} out of bound {}",
                value,
                bound
            );
        }
    }

    #[test]
    fn test_kaiming_uniform_invalid_shape_empty() {
        assert!(matches!(
            kaiming_uniform(10, &[]),
            Err(Error::InvalidOperation(_))
        ));
    }

    #[test]
    fn test_kaiming_uniform_zero_dim() {
        let result = kaiming_uniform(10, &[10, 0, 5]);
        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!(array.shape(), &[10, 0, 5]);
        assert_eq!(array.size(), 0);
    }

    #[test]
    fn test_kaiming_uniform_zero_fan_in() {
        assert!(matches!(
            kaiming_uniform(0, &[10, 10]),
            Err(Error::InvalidOperation(_))
        ));
    }
}
