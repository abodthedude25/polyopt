//! Loop tiling transformation.
//!
//! Tiling (also called blocking) transforms a loop nest to improve cache locality
//! by processing data in smaller blocks that fit in cache.
//!
//! Example:
//! ```text
//! for i = 0 to N:
//!   for j = 0 to M:
//!     A[i][j] = ...
//! ```
//! becomes:
//! ```text
//! for ii = 0 to N step Ti:
//!   for jj = 0 to M step Tj:
//!     for i = ii to min(ii + Ti, N):
//!       for j = jj to min(jj + Tj, M):
//!         A[i][j] = ...
//! ```

use crate::ir::pir::{PolyProgram, PolyStmt};
use crate::analysis::{Dependence, Direction};
use crate::polyhedral::map::AffineMap;
use crate::polyhedral::expr::AffineExpr;
use crate::transform::Transform;
use anyhow::Result;
use std::collections::HashSet;

/// Loop tiling transformation.
#[derive(Debug, Clone)]
pub struct Tiling {
    /// Tile sizes for each dimension (starting from outermost)
    pub tile_sizes: Vec<i64>,
    /// Which dimensions to tile (None = all dimensions up to tile_sizes.len())
    pub dimensions: Option<Vec<usize>>,
}

impl Tiling {
    /// Create a new tiling transformation with specified tile sizes.
    pub fn new(tile_sizes: Vec<i64>) -> Self {
        Self { 
            tile_sizes,
            dimensions: None,
        }
    }

    /// Create a tiling transformation with uniform tile size for n dimensions.
    pub fn with_default_size(n_dim: usize, size: i64) -> Self {
        Self {
            tile_sizes: vec![size; n_dim],
            dimensions: None,
        }
    }

    /// Specify which dimensions to tile.
    pub fn with_dimensions(mut self, dims: Vec<usize>) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Apply tiling to a single statement's schedule.
    /// Apply tiling to a statement's schedule.
    pub fn tile_schedule(&self, stmt: &PolyStmt) -> AffineMap {
        let n_dim = stmt.depth();
        let n_param = stmt.domain.n_param();
        
        // Determine which dimensions to tile
        let dims_to_tile: Vec<usize> = match &self.dimensions {
            Some(dims) => dims.iter()
                .filter(|&&d| d < n_dim && d < self.tile_sizes.len())
                .copied()
                .collect(),
            None => (0..n_dim.min(self.tile_sizes.len())).collect(),
        };
        
        if dims_to_tile.is_empty() {
            return stmt.schedule.clone();
        }
        
        let n_tiled = dims_to_tile.len();
        let n_out = n_dim + n_tiled;
        
        let mut outputs = Vec::with_capacity(n_out);
        
        // Build the tiled schedule: [t0, i0, t1, i1, ..., remaining]
        // where ti = floor(original_i / tile_size)
        
        let tiled_set: HashSet<usize> = dims_to_tile.iter().copied().collect();
        
        // First pass: add tile and point iterators for tiled dimensions
        for (idx, &d) in dims_to_tile.iter().enumerate() {
            let tile_size = self.tile_sizes[idx];
            
            // Get the original schedule expression for this dimension
            let orig_expr = if d < stmt.schedule.n_out() {
                stmt.schedule.outputs[d].clone()
            } else {
                AffineExpr::var(d, n_dim, n_param)
            };
            
            // Tile iterator: floor(expr / tile_size)
            let tile_expr = orig_expr.floordiv(tile_size);
            outputs.push(tile_expr);
            
            // Point iterator: original expression (actual mod is done by bounds)
            outputs.push(orig_expr);
        }
        
        // Second pass: add non-tiled dimensions
        for d in 0..n_dim {
            if !tiled_set.contains(&d) {
                let expr = if d < stmt.schedule.n_out() {
                    stmt.schedule.outputs[d].clone()
                } else {
                    AffineExpr::var(d, n_dim, n_param)
                };
                outputs.push(expr);
            }
        }
        
        AffineMap::from_outputs(n_dim, outputs)
    }

    /// Apply tiling directly to an AffineMap schedule.
    pub fn apply_to_schedule(&self, schedule: &AffineMap) -> AffineMap {
        let n_dim = schedule.n_in();
        let n_param = 0; // Assume no parameters for now
        
        // Determine which dimensions to tile
        let dims_to_tile: Vec<usize> = match &self.dimensions {
            Some(dims) => dims.iter()
                .filter(|&&d| d < n_dim && d < self.tile_sizes.len())
                .copied()
                .collect(),
            None => (0..n_dim.min(self.tile_sizes.len())).collect(),
        };
        
        if dims_to_tile.is_empty() {
            return schedule.clone();
        }
        
        let tiled_set: HashSet<usize> = dims_to_tile.iter().copied().collect();
        let mut outputs = Vec::new();
        
        // First pass: add tile and point iterators for tiled dimensions
        for (idx, &d) in dims_to_tile.iter().enumerate() {
            let tile_size = self.tile_sizes[idx];
            
            let orig_expr = if d < schedule.n_out() {
                schedule.outputs[d].clone()
            } else {
                AffineExpr::var(d, n_dim, n_param)
            };
            
            // Tile iterator
            outputs.push(orig_expr.floordiv(tile_size));
            // Point iterator
            outputs.push(orig_expr);
        }
        
        // Second pass: add non-tiled dimensions
        for d in 0..n_dim {
            if !tiled_set.contains(&d) {
                let expr = if d < schedule.n_out() {
                    schedule.outputs[d].clone()
                } else {
                    AffineExpr::var(d, n_dim, n_param)
                };
                outputs.push(expr);
            }
        }
        
        AffineMap::from_outputs(n_dim, outputs)
    }

    /// Check if tiling at the specified dimensions is legal.
    pub fn is_tiling_legal_at(&self, deps: &[Dependence], dims: &[usize]) -> bool {
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            // For tiling to be legal, we need all dependence directions
            // at the tiled dimensions to be non-negative (forward or equal)
            for &d in dims {
                if d < dep.direction.len() {
                    match dep.direction[d] {
                        Direction::Gt => return false, // Backward dependence
                        Direction::Star => {
                            // Unknown direction - check distance if available
                            if let Some(ref dist) = dep.distance {
                                if let Some(&d_val) = dist.get(d) {
                                    if d_val < 0 {
                                        return false;
                                    }
                                }
                            }
                        }
                        _ => {} // Lt, Eq, Le, Ge are all fine
                    }
                }
            }
        }
        true
    }
}

impl Transform for Tiling {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        let mut changed = false;
        
        for stmt in &mut program.statements {
            let new_schedule = self.tile_schedule(stmt);
            if new_schedule.n_out() != stmt.schedule.n_out() {
                stmt.schedule = new_schedule;
                changed = true;
            }
        }
        
        Ok(changed)
    }

    fn is_legal(&self, program: &PolyProgram, deps: &[Dependence]) -> bool {
        // Determine dimensions to tile
        let max_depth = program.statements.iter()
            .map(|s| s.depth())
            .max()
            .unwrap_or(0);
        
        let dims_to_tile: Vec<usize> = match &self.dimensions {
            Some(dims) => dims.clone(),
            None => (0..max_depth.min(self.tile_sizes.len())).collect(),
        };
        
        self.is_tiling_legal_at(deps, &dims_to_tile)
    }

    fn name(&self) -> &str {
        "tiling"
    }
}

/// Rectangular tiling with the same tile size in all dimensions.
pub fn tile_rectangular(program: &mut PolyProgram, tile_size: i64) -> Result<bool> {
    let max_depth = program.statements.iter()
        .map(|s| s.depth())
        .max()
        .unwrap_or(0);
    
    let tiling = Tiling::with_default_size(max_depth, tile_size);
    tiling.apply(program)
}

/// Apply tiling only if legal according to dependence analysis.
pub fn tile_if_legal(
    program: &mut PolyProgram,
    tile_sizes: Vec<i64>,
    deps: &[Dependence],
) -> Result<bool> {
    let tiling = Tiling::new(tile_sizes);
    
    if tiling.is_legal(program, deps) {
        tiling.apply(program)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedral::set::IntegerSet;
    use crate::ir::pir::StmtId;
    use crate::analysis::DependenceKind;
    use crate::polyhedral::dependence::DependenceRelation;

    fn make_test_stmt(id: u64, depth: usize) -> PolyStmt {
        PolyStmt {
            id: StmtId::new(id),
            name: format!("S{}", id),
            domain: IntegerSet::universe(depth),
            schedule: AffineMap::identity(depth),
            reads: vec![],
            writes: vec![],
            body: crate::ir::pir::StmtBody::Assignment { target: crate::ir::pir::AccessExpr { array: "A".to_string(), indices: vec![] }, expr: crate::ir::pir::ComputeExpr::Int(0) },
            span: crate::utils::location::Span::default(),
        }
    }

    #[test]
    fn test_tiling_schedule() {
        let stmt = make_test_stmt(0, 2);
        let tiling = Tiling::new(vec![32, 32]);
        
        let new_sched = tiling.tile_schedule(&stmt);
        
        // Should have 4 output dimensions: t0, i0, t1, i1
        assert_eq!(new_sched.n_out(), 4);
    }

    #[test]
    fn test_tiling_legal() {
        let deps = vec![];
        let tiling = Tiling::new(vec![32]);
        
        // No dependences means tiling is always legal
        assert!(tiling.is_tiling_legal_at(&deps, &[0]));
    }

    #[test]
    fn test_tiling_illegal_backward_dep() {
        let deps = vec![Dependence {
            source: StmtId::new(0),
            target: StmtId::new(0),
            kind: DependenceKind::Flow,
            relation: DependenceRelation::universe(1, 1, 0),
            distance: Some(vec![-1]), // Backward dependence
            direction: vec![Direction::Gt],
            array: "A".to_string(),
            level: Some(0),
            is_loop_independent: false,
        }];
        
        let tiling = Tiling::new(vec![32]);
        
        // Backward dependence means tiling is illegal
        assert!(!tiling.is_tiling_legal_at(&deps, &[0]));
    }
    
    #[test]
    fn test_tiling_legal_forward_dep() {
        let deps = vec![Dependence {
            source: StmtId::new(0),
            target: StmtId::new(0),
            kind: DependenceKind::Flow,
            relation: DependenceRelation::universe(1, 1, 0),
            distance: Some(vec![1]), // Forward dependence
            direction: vec![Direction::Lt],
            array: "A".to_string(),
            level: Some(0),
            is_loop_independent: false,
        }];
        
        let tiling = Tiling::new(vec![32]);
        
        // Forward dependence means tiling is legal
        assert!(tiling.is_tiling_legal_at(&deps, &[0]));
    }
}
