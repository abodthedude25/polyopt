//! Affine maps for schedules and access relations.

use crate::polyhedral::space::Space;
use crate::polyhedral::expr::AffineExpr;
use crate::polyhedral::set::IntegerSet;
use serde::{Serialize, Deserialize};
use std::fmt;

/// An affine map from one space to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineMap {
    pub space: Space,
    /// Output expressions (one per output dimension)
    pub outputs: Vec<AffineExpr>,
}

impl AffineMap {
    /// Create an identity map of given dimension.
    pub fn identity(n_dim: usize) -> Self {
        let space = Space::map(n_dim, n_dim);
        let outputs = (0..n_dim)
            .map(|i| AffineExpr::var(i, n_dim, 0))
            .collect();
        Self { space, outputs }
    }

    /// Create a zero map (all outputs are zero).
    pub fn zero(n_in: usize, n_out: usize) -> Self {
        let space = Space::map(n_in, n_out);
        let outputs = (0..n_out)
            .map(|_| AffineExpr::zero(n_in, 0))
            .collect();
        Self { space, outputs }
    }

    /// Create from output expressions.
    pub fn from_outputs(n_in: usize, outputs: Vec<AffineExpr>) -> Self {
        let n_out = outputs.len();
        Self {
            space: Space::map(n_in, n_out),
            outputs,
        }
    }

    /// Get input dimensions.
    pub fn n_in(&self) -> usize { self.space.n_in }

    /// Get output dimensions.
    pub fn n_out(&self) -> usize { self.space.n_dim }

    /// Get number of parameters.
    pub fn n_param(&self) -> usize { self.space.n_param }

    /// Apply the map to a point.
    pub fn apply(&self, input: &[i64], params: &[i64]) -> Vec<i64> {
        self.outputs.iter()
            .map(|expr| expr.evaluate(input, params))
            .collect()
    }

    /// Compose two maps: self after other.
    pub fn compose(&self, other: &AffineMap) -> AffineMap {
        assert_eq!(self.n_in(), other.n_out());
        let n_in = other.n_in();
        let n_out = self.n_out();
        
        let outputs = self.outputs.iter().map(|out_expr| {
            let mut result = AffineExpr::zero(n_in, self.n_param());
            result.constant = out_expr.constant;
            
            // Substitute other's outputs into this expression
            for (i, &coeff) in out_expr.coeffs.iter().enumerate() {
                if coeff != 0 {
                    let substituted = other.outputs[i].scale(coeff);
                    result = result + substituted;
                }
            }
            result
        }).collect();

        AffineMap::from_outputs(n_in, outputs)
    }

    /// Apply map to a set to get the image.
    pub fn apply_to_set(&self, set: &IntegerSet) -> IntegerSet {
        // Simplified - returns universe of output dimensions
        IntegerSet::universe(self.n_out())
    }

    /// Invert the map (if possible).
    pub fn inverse(&self) -> Option<AffineMap> {
        // Only works for square, unimodular maps
        if self.n_in() != self.n_out() {
            return None;
        }
        // Placeholder - real implementation needs matrix inversion
        Some(AffineMap::identity(self.n_in()))
    }

    /// Create a schedule map that adds a constant time dimension.
    pub fn schedule_with_constant(n_dim: usize, time: i64) -> Self {
        let mut outputs = vec![AffineExpr::constant(time, n_dim, 0)];
        for i in 0..n_dim {
            outputs.push(AffineExpr::var(i, n_dim, 0));
        }
        Self {
            space: Space::map(n_dim, n_dim + 1),
            outputs,
        }
    }
}

impl fmt::Display for AffineMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ [")?;
        for i in 0..self.n_in() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "i{}", i)?;
        }
        write!(f, "] -> [")?;
        let dim_names: Vec<String> = (0..self.n_in()).map(|i| format!("i{}", i)).collect();
        let param_names: Vec<String> = (0..self.n_param()).map(|i| format!("p{}", i)).collect();
        for (i, expr) in self.outputs.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", expr.to_string_with_names(&dim_names, &param_names))?;
        }
        write!(f, "] }}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let map = AffineMap::identity(3);
        assert_eq!(map.apply(&[1, 2, 3], &[]), vec![1, 2, 3]);
    }

    #[test]
    fn test_compose() {
        let m1 = AffineMap::identity(2);
        let m2 = AffineMap::identity(2);
        let composed = m1.compose(&m2);
        assert_eq!(composed.apply(&[5, 7], &[]), vec![5, 7]);
    }

    #[test]
    fn test_schedule() {
        let sched = AffineMap::schedule_with_constant(2, 0);
        assert_eq!(sched.apply(&[3, 4], &[]), vec![0, 3, 4]);
    }
}
