//! Loop unrolling transformation.
//!
//! Unrolling replicates the loop body multiple times to reduce loop overhead
//! and enable instruction-level parallelism.
//!
//! Example:
//! ```text
//! for i = 0 to N:
//!   A[i] = B[i] + 1
//! ```
//! After unrolling by factor 4:
//! ```text
//! for i = 0 to N step 4:
//!   A[i] = B[i] + 1
//!   A[i+1] = B[i+1] + 1
//!   A[i+2] = B[i+2] + 1
//!   A[i+3] = B[i+3] + 1
//! // Handle remainder
//! ```

use crate::ir::pir::{PolyProgram, PolyStmt};
use crate::analysis::Dependence;
use crate::polyhedral::map::AffineMap;
use crate::polyhedral::expr::AffineExpr;
use crate::transform::Transform;
use anyhow::Result;

/// Loop unrolling transformation.
#[derive(Debug, Clone)]
pub struct Unrolling {
    /// Dimension to unroll
    pub dimension: usize,
    /// Unrolling factor
    pub factor: usize,
    /// Whether to generate cleanup code for remainder
    pub generate_cleanup: bool,
}

impl Unrolling {
    /// Create a new unrolling transformation.
    pub fn new(dimension: usize, factor: usize) -> Self {
        Self {
            dimension,
            factor,
            generate_cleanup: true,
        }
    }

    /// Create unrolling without cleanup code.
    pub fn without_cleanup(dimension: usize, factor: usize) -> Self {
        Self {
            dimension,
            factor,
            generate_cleanup: false,
        }
    }

    /// Set whether to generate cleanup code.
    pub fn with_cleanup(mut self, generate: bool) -> Self {
        self.generate_cleanup = generate;
        self
    }

    /// Apply unrolling to a statement's schedule.
    /// In the polyhedral model, unrolling is represented by modifying
    /// the schedule to separate iterations.
    fn unroll_schedule(&self, stmt: &PolyStmt) -> AffineMap {
        let n_dim = stmt.depth();
        let n_param = stmt.domain.n_param();
        
        if self.dimension >= n_dim || self.factor <= 1 {
            return stmt.schedule.clone();
        }
        
        // Unrolling in polyhedral model: add an unroll dimension
        // Original: [i0, ..., id, ..., in] where id is the dimension to unroll
        // After: [i0, ..., floor(id/factor), id mod factor, ..., in]
        
        let mut outputs = Vec::new();
        
        for d in 0..n_dim {
            if d == self.dimension {
                // Add floor division for the main loop
                let floor_expr = AffineExpr::var(d, n_dim, n_param)
                    .floordiv(self.factor as i64);
                outputs.push(floor_expr);
                
                // The original dimension represents the offset within unrolled block
                outputs.push(AffineExpr::var(d, n_dim, n_param));
            } else {
                outputs.push(AffineExpr::var(d, n_dim, n_param));
            }
        }
        
        AffineMap::from_outputs(n_dim, outputs)
    }
}

impl Transform for Unrolling {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        if self.factor <= 1 {
            return Ok(false);
        }
        
        let mut changed = false;
        
        for stmt in &mut program.statements {
            let new_schedule = self.unroll_schedule(stmt);
            if new_schedule.n_out() != stmt.schedule.n_out() {
                stmt.schedule = new_schedule;
                changed = true;
            }
        }
        
        Ok(changed)
    }

    fn is_legal(&self, _program: &PolyProgram, _deps: &[Dependence]) -> bool {
        // Unrolling is always legal - it doesn't change the order of iterations
        self.factor > 1
    }

    fn name(&self) -> &str {
        "unrolling"
    }
}

/// Strip-mining transformation (related to tiling but for a single dimension).
#[derive(Debug, Clone)]
pub struct StripMining {
    /// Dimension to strip-mine
    pub dimension: usize,
    /// Strip size
    pub strip_size: i64,
}

impl StripMining {
    /// Create a new strip-mining transformation.
    pub fn new(dimension: usize, strip_size: i64) -> Self {
        Self {
            dimension,
            strip_size,
        }
    }

    /// Apply strip-mining to a statement's schedule.
    fn strip_mine_schedule(&self, stmt: &PolyStmt) -> AffineMap {
        let n_dim = stmt.depth();
        let n_param = stmt.domain.n_param();
        
        if self.dimension >= n_dim || self.strip_size <= 1 {
            return stmt.schedule.clone();
        }
        
        // Strip-mining: [i0, ..., id, ..., in] -> [i0, ..., floor(id/strip), id, ..., in]
        let mut outputs = Vec::new();
        
        for d in 0..n_dim {
            if d == self.dimension {
                // Add strip iterator
                let strip_expr = AffineExpr::var(d, n_dim, n_param)
                    .floordiv(self.strip_size);
                outputs.push(strip_expr);
            }
            outputs.push(AffineExpr::var(d, n_dim, n_param));
        }
        
        AffineMap::from_outputs(n_dim, outputs)
    }
}

impl Transform for StripMining {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        if self.strip_size <= 1 {
            return Ok(false);
        }
        
        let mut changed = false;
        
        for stmt in &mut program.statements {
            let new_schedule = self.strip_mine_schedule(stmt);
            if new_schedule.n_out() != stmt.schedule.n_out() {
                stmt.schedule = new_schedule;
                changed = true;
            }
        }
        
        Ok(changed)
    }

    fn is_legal(&self, _program: &PolyProgram, _deps: &[Dependence]) -> bool {
        // Strip-mining is always legal
        self.strip_size > 1
    }

    fn name(&self) -> &str {
        "strip_mining"
    }
}

/// Unroll and jam transformation (unroll outer, fuse inner).
#[derive(Debug, Clone)]
pub struct UnrollAndJam {
    /// Outer dimension to unroll
    pub outer_dim: usize,
    /// Inner dimension to jam
    pub inner_dim: usize,
    /// Unrolling factor
    pub factor: usize,
}

impl UnrollAndJam {
    /// Create a new unroll-and-jam transformation.
    pub fn new(outer_dim: usize, inner_dim: usize, factor: usize) -> Self {
        Self {
            outer_dim,
            inner_dim,
            factor,
        }
    }

    /// Apply unroll-and-jam to a statement's schedule.
    fn unroll_and_jam_schedule(&self, stmt: &PolyStmt) -> AffineMap {
        let n_dim = stmt.depth();
        let n_param = stmt.domain.n_param();
        
        if self.outer_dim >= n_dim || self.inner_dim >= n_dim || self.factor <= 1 {
            return stmt.schedule.clone();
        }
        
        // Unroll-and-jam combines strip-mining of outer with reordering
        let mut outputs = Vec::new();
        
        for d in 0..n_dim {
            if d == self.outer_dim {
                // Strip outer dimension
                outputs.push(AffineExpr::var(d, n_dim, n_param)
                    .floordiv(self.factor as i64));
            }
            outputs.push(AffineExpr::var(d, n_dim, n_param));
        }
        
        AffineMap::from_outputs(n_dim, outputs)
    }
}

impl Transform for UnrollAndJam {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        if self.factor <= 1 {
            return Ok(false);
        }
        
        let mut changed = false;
        
        for stmt in &mut program.statements {
            let new_schedule = self.unroll_and_jam_schedule(stmt);
            if new_schedule.n_out() != stmt.schedule.n_out() {
                stmt.schedule = new_schedule;
                changed = true;
            }
        }
        
        Ok(changed)
    }

    fn is_legal(&self, _program: &PolyProgram, deps: &[Dependence]) -> bool {
        // Similar legality to interchange - need to check dependences
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            // Check if unroll-and-jam would violate dependences
            if let Some(ref dist) = dep.distance {
                let d_outer = dist.get(self.outer_dim).copied().unwrap_or(0);
                let d_inner = dist.get(self.inner_dim).copied().unwrap_or(0);
                
                // If there's a backward dependence on inner and forward on outer,
                // unroll-and-jam may be illegal
                if d_outer > 0 && d_inner < 0 {
                    return false;
                }
            }
        }
        true
    }

    fn name(&self) -> &str {
        "unroll_and_jam"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedral::set::IntegerSet;
    use crate::ir::pir::StmtId;

    fn make_test_stmt(id: u64, depth: usize) -> PolyStmt {
        PolyStmt {
            id: StmtId::new(id),
            name: format!("S{}", id),
            domain: IntegerSet::universe(depth),
            schedule: AffineMap::identity(depth),
            reads: vec![],
            writes: vec![],
            body: crate::ir::pir::StmtBody::Assignment {
                target: crate::ir::pir::AccessExpr { array: "A".to_string(), indices: vec![] },
                expr: crate::ir::pir::ComputeExpr::Int(0),
            },
            span: crate::utils::location::Span::default(),
        }
    }

    #[test]
    fn test_unrolling() {
        let stmt = make_test_stmt(0, 1);
        let unroll = Unrolling::new(0, 4);
        
        let new_sched = unroll.unroll_schedule(&stmt);
        
        // Should have 2 dimensions now: floor(i/4) and i
        assert_eq!(new_sched.n_out(), 2);
    }

    #[test]
    fn test_strip_mining() {
        let stmt = make_test_stmt(0, 2);
        let strip = StripMining::new(0, 32);
        
        let new_sched = strip.strip_mine_schedule(&stmt);
        
        // Should have 3 dimensions: floor(i/32), i, j
        assert_eq!(new_sched.n_out(), 3);
    }

    #[test]
    fn test_unroll_factor_1() {
        let stmt = make_test_stmt(0, 1);
        let unroll = Unrolling::new(0, 1);
        
        // Factor of 1 should not change schedule
        let new_sched = unroll.unroll_schedule(&stmt);
        assert_eq!(new_sched.n_out(), 1);
    }
}
