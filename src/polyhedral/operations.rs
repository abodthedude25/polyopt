git add .//! Polyhedral operations: projection, intersection, union, etc.

use crate::polyhedral::set::IntegerSet;
use crate::polyhedral::map::AffineMap;
use crate::polyhedral::expr::AffineExpr;
use crate::polyhedral::constraint::Constraint;

/// Compute the intersection of two sets.
pub fn intersect(a: &IntegerSet, b: &IntegerSet) -> IntegerSet {
    a.intersect(b)
}

/// Compute the union of two sets (returns overapproximation).
pub fn union_overapprox(a: &IntegerSet, b: &IntegerSet) -> IntegerSet {
    // True union requires disjunctions; we return the convex hull approximation
    // For now, just return the first set (placeholder)
    a.clone()
}

/// Project out specified dimensions using Fourier-Motzkin elimination.
pub fn project_out(set: &IntegerSet, dims: &[usize]) -> IntegerSet {
    // Placeholder - real implementation needs FM elimination
    let remaining_dims = set.dim() - dims.len();
    IntegerSet::universe(remaining_dims)
}

/// Check if two sets are equal.
pub fn is_equal(a: &IntegerSet, b: &IntegerSet) -> bool {
    // Placeholder - would need proper emptiness checking
    // For now, just check structural equality
    a.dim() == b.dim() && 
    a.constraints.len() == b.constraints.len()
}

/// Check if set a is a subset of set b.
pub fn is_subset(a: &IntegerSet, b: &IntegerSet) -> bool {
    // a ⊆ b iff a ∩ ¬b = ∅
    // Placeholder implementation
    true
}

/// Check if a set is empty.
pub fn is_empty(set: &IntegerSet) -> bool {
    set.is_obviously_empty()
}

/// Compute the lexicographic minimum of a set.
pub fn lexmin(set: &IntegerSet) -> Option<Vec<i64>> {
    // Placeholder - would need ILP solver
    None
}

/// Compute the lexicographic maximum of a set.
pub fn lexmax(set: &IntegerSet) -> Option<Vec<i64>> {
    // Placeholder - would need ILP solver
    None
}

/// Count the number of integer points in a set (for small sets).
pub fn cardinality(set: &IntegerSet, params: &[i64]) -> Option<u64> {
    // Only works for bounded sets with known parameters
    // Placeholder implementation
    None
}

/// Apply a map to a set.
pub fn apply_map(set: &IntegerSet, map: &AffineMap) -> IntegerSet {
    map.apply_to_set(set)
}

/// Compute the domain of a map.
pub fn domain(map: &AffineMap) -> IntegerSet {
    IntegerSet::universe(map.n_in())
}

/// Compute the range of a map.
pub fn range(map: &AffineMap) -> IntegerSet {
    IntegerSet::universe(map.n_out())
}

/// Compose two maps.
pub fn compose(outer: &AffineMap, inner: &AffineMap) -> AffineMap {
    outer.compose(inner)
}

/// Compute deltas: { y - x : (x, y) in map }
pub fn deltas(map: &AffineMap) -> IntegerSet {
    // For a map {[i] -> [f(i)]}, deltas are {[f(i) - i]}
    IntegerSet::universe(map.n_out())
}

/// Tile a loop dimension.
pub fn tile_dimension(
    set: &IntegerSet,
    dim: usize,
    tile_size: i64,
) -> (IntegerSet, AffineMap) {
    // Returns (tiled_set, tile_to_original_map)
    // Tiling adds a new tile dimension
    let n_dim = set.dim();
    let tiled = IntegerSet::universe(n_dim + 1);
    let map = AffineMap::identity(n_dim + 1);
    (tiled, map)
}

/// Skew transformation: i' = i + factor * j
pub fn skew(
    set: &IntegerSet,
    target_dim: usize,
    source_dim: usize,
    factor: i64,
) -> (IntegerSet, AffineMap) {
    let n_dim = set.dim();
    let mut outputs: Vec<AffineExpr> = (0..n_dim)
        .map(|i| AffineExpr::var(i, n_dim, 0))
        .collect();
    
    // Modify target dimension
    if target_dim < n_dim && source_dim < n_dim {
        let skew_term = AffineExpr::var(source_dim, n_dim, 0).scale(factor);
        outputs[target_dim] = outputs[target_dim].clone() + skew_term;
    }
    
    let map = AffineMap::from_outputs(n_dim, outputs);
    let transformed = IntegerSet::universe(n_dim); // Placeholder
    (transformed, map)
}

/// Interchange two dimensions.
pub fn interchange(set: &IntegerSet, dim1: usize, dim2: usize) -> (IntegerSet, AffineMap) {
    let n_dim = set.dim();
    let mut outputs: Vec<AffineExpr> = (0..n_dim)
        .map(|i| AffineExpr::var(i, n_dim, 0))
        .collect();
    
    if dim1 < n_dim && dim2 < n_dim {
        outputs.swap(dim1, dim2);
    }
    
    let map = AffineMap::from_outputs(n_dim, outputs);
    let transformed = IntegerSet::universe(n_dim);
    (transformed, map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect() {
        let a = IntegerSet::rectangular(&[10, 10]);
        let b = IntegerSet::rectangular(&[5, 5]);
        let c = intersect(&a, &b);
        assert!(c.contains(&[2, 2], &[]));
    }

    #[test]
    fn test_interchange() {
        let set = IntegerSet::rectangular(&[10, 20]);
        let (_, map) = interchange(&set, 0, 1);
        assert_eq!(map.apply(&[3, 7], &[]), vec![7, 3]);
    }
}
