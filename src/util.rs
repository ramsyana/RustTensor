/// Compute the broadcasted shape for two shapes, following numpy broadcasting rules.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, crate::error::Error> {
    let mut result = Vec::new();
    let ndim = std::cmp::max(a.len(), b.len());
    for i in 0..ndim {
        let a_dim = if i >= ndim - a.len() {
            a[i - (ndim - a.len())]
        } else {
            1
        };
        let b_dim = if i >= ndim - b.len() {
            b[i - (ndim - b.len())]
        } else {
            1
        };
        if a_dim == b_dim || a_dim == 1 || b_dim == 1 {
            result.push(std::cmp::max(a_dim, b_dim));
        } else {
            return Err(crate::error::Error::InvalidOperation(format!(
                "Cannot broadcast dimension {} from {} to {}",
                i, a_dim, b_dim
            )));
        }
    }
    Ok(result)
}
