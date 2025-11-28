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

    /// Swap two dimensions in the schedule.
    pub fn interchange(&self, dim1: usize, dim2: usize) -> Self {
        assert!(dim1 < self.n_out() && dim2 < self.n_out());
        let mut outputs = self.outputs.clone();
        outputs.swap(dim1, dim2);
        Self {
            space: self.space.clone(),
            outputs,
        }
    }

    /// Skew dimension i by factor * dimension j: i' = i + factor * j
    pub fn skew(&self, target_dim: usize, source_dim: usize, factor: i64) -> Self {
        assert!(target_dim < self.n_out() && source_dim < self.n_out());
        let mut outputs = self.outputs.clone();
        let source_expr = outputs[source_dim].scale(factor);
        outputs[target_dim] = outputs[target_dim].clone() + source_expr;
        Self {
            space: self.space.clone(),
            outputs,
        }
    }

    /// Scale a dimension by a factor.
    pub fn scale_dim(&self, dim: usize, factor: i64) -> Self {
        assert!(dim < self.n_out());
        let mut outputs = self.outputs.clone();
        outputs[dim] = outputs[dim].scale(factor);
        Self {
            space: self.space.clone(),
            outputs,
        }
    }

    /// Shift a dimension by a constant offset.
    pub fn shift_dim(&self, dim: usize, offset: i64) -> Self {
        assert!(dim < self.n_out());
        let mut outputs = self.outputs.clone();
        outputs[dim].constant += offset;
        Self {
            space: self.space.clone(),
            outputs,
        }
    }

    /// Add a new output dimension at the given position.
    pub fn add_dim(&self, pos: usize, expr: AffineExpr) -> Self {
        let mut outputs = self.outputs.clone();
        outputs.insert(pos, expr);
        Self {
            space: Space::map(self.n_in(), outputs.len()),
            outputs,
        }
    }

    /// Remove an output dimension.
    pub fn remove_dim(&self, pos: usize) -> Self {
        assert!(pos < self.n_out());
        let mut outputs = self.outputs.clone();
        outputs.remove(pos);
        Self {
            space: Space::map(self.n_in(), outputs.len()),
            outputs,
        }
    }

    /// Create a tiling transformation for a single dimension.
    /// Transforms: [i0, i1, ..., in] -> [i_dim/tile, i0, i1, ..., in]
    /// where the original dimension i_dim is replaced with i_dim mod tile
    pub fn tile_dim(n_dim: usize, dim: usize, tile_size: i64) -> Self {
        assert!(dim < n_dim);
        let n_param = 0;
        
        // Output has n_dim + 1 dimensions: [tile_iter, orig_dims with tiled one modified]
        let mut outputs = Vec::with_capacity(n_dim + 1);
        
        // First output: floor(i_dim / tile_size)
        let tile_iter = AffineExpr::var(dim, n_dim, n_param).floordiv(tile_size);
        outputs.push(tile_iter);
        
        // Rest of the outputs: original dimensions
        for i in 0..n_dim {
            outputs.push(AffineExpr::var(i, n_dim, n_param));
        }
        
        Self {
            space: Space::map(n_dim, n_dim + 1),
            outputs,
        }
    }

    /// Create a full tiling transformation for multiple dimensions.
    pub fn tile(n_dim: usize, tile_sizes: &[i64]) -> Self {
        let n_tiled = tile_sizes.len().min(n_dim);
        let n_out = n_dim + n_tiled;
        let n_param = 0;
        
        let mut outputs = Vec::with_capacity(n_out);
        
        // Add tile iterators first (interleaved with point iterators)
        for d in 0..n_tiled {
            // Tile iterator: floor(i_d / tile_size)
            outputs.push(AffineExpr::var(d, n_dim, n_param).floordiv(tile_sizes[d]));
            // Point iterator: i_d (mod is handled by domain constraints)
            outputs.push(AffineExpr::var(d, n_dim, n_param));
        }
        
        // Add remaining untiled dimensions
        for d in n_tiled..n_dim {
            outputs.push(AffineExpr::var(d, n_dim, n_param));
        }
        
        Self {
            space: Space::map(n_dim, n_out),
            outputs,
        }
    }

    /// Create a permutation map.
    pub fn permutation(perm: &[usize]) -> Self {
        let n = perm.len();
        let outputs = perm.iter()
            .map(|&i| AffineExpr::var(i, n, 0))
            .collect();
        Self {
            space: Space::map(n, n),
            outputs,
        }
    }

    /// Get the coefficient matrix of this map.
    pub fn coefficient_matrix(&self) -> Vec<Vec<i64>> {
        self.outputs.iter()
            .map(|expr| expr.coeffs.clone())
            .collect()
    }

    /// Check if this is an identity map.
    pub fn is_identity(&self) -> bool {
        if self.n_in() != self.n_out() {
            return false;
        }
        for (i, expr) in self.outputs.iter().enumerate() {
            if expr.constant != 0 {
                return false;
            }
            for (j, &coeff) in expr.coeffs.iter().enumerate() {
                let expected = if i == j { 1 } else { 0 };
                if coeff != expected {
                    return false;
                }
            }
        }
        true
    }

    /// Check if this is a permutation (bijective linear map).
    pub fn is_permutation(&self) -> bool {
        if self.n_in() != self.n_out() {
            return false;
        }
        // Each output should be exactly one input variable
        let n = self.n_in();
        let mut used = vec![false; n];
        
        for expr in &self.outputs {
            if expr.constant != 0 {
                return false;
            }
            let mut found = None;
            for (i, &coeff) in expr.coeffs.iter().enumerate() {
                if coeff == 1 {
                    if found.is_some() {
                        return false; // Multiple non-zero coefficients
                    }
                    found = Some(i);
                } else if coeff != 0 {
                    return false; // Non-unit coefficient
                }
            }
            match found {
                Some(i) if !used[i] => used[i] = true,
                _ => return false,
            }
        }
        true
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
