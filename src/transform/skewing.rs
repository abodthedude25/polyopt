//! Loop skewing transformation.
//!
//! Skewing transforms a loop nest to expose parallelism in loops with
//! dependences. It's essential for wavefront parallelization.
//!
//! Example:
//! ```text
//! for i = 0 to N:
//!   for j = 0 to M:
//!     A[i][j] = A[i-1][j] + A[i][j-1]
//! ```
//! After skewing j by i (j' = j + i):
//! ```text
//! for i = 0 to N:
//!   for j' = i to M + i:
//!     A[i][j'-i] = A[i-1][j'-i] + A[i][j'-i-1]
//! ```
//! Now iterations with the same j' value are independent and can run in parallel.

use crate::ir::pir::{PolyProgram, PolyStmt};
use crate::analysis::{Dependence, Direction};
use crate::polyhedral::map::AffineMap;
use crate::polyhedral::expr::AffineExpr;
use crate::transform::Transform;
use anyhow::Result;

/// Loop skewing transformation.
#[derive(Debug, Clone)]
pub struct Skewing {
    /// Target dimension to skew
    pub target_dim: usize,
    /// Source dimension to skew by
    pub source_dim: usize,
    /// Skewing factor (target' = target + factor * source)
    pub factor: i64,
}

impl Skewing {
    /// Create a new skewing transformation.
    pub fn new(target_dim: usize, source_dim: usize, factor: i64) -> Self {
        Self {
            target_dim,
            source_dim,
            factor,
        }
    }

    /// Create skewing for wavefront parallelization.
    /// Skews inner loop by outer loop: j' = j + i
    pub fn wavefront(outer: usize, inner: usize) -> Self {
        Self::new(inner, outer, 1)
    }

    /// Apply skewing to a single statement's schedule.
    fn skew_schedule(&self, stmt: &PolyStmt) -> Option<AffineMap> {
        let n_out = stmt.schedule.n_out();
        
        if self.target_dim >= n_out || self.source_dim >= n_out {
            return None;
        }
        
        Some(stmt.schedule.skew(self.target_dim, self.source_dim, self.factor))
    }

    /// Check if skewing is legal based on dependences.
    /// Skewing preserves dependence directions if done correctly.
    pub fn is_skewing_legal(&self, deps: &[Dependence]) -> bool {
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            // Get directions at source and target dimensions
            let src_dir = dep.direction.get(self.source_dim).copied().unwrap_or(Direction::Star);
            let tgt_dir = dep.direction.get(self.target_dim).copied().unwrap_or(Direction::Star);
            
            // After skewing, the new target direction depends on the combination
            // For factor > 0: if src_dir is Lt and tgt_dir is Gt, result could be negative
            // This is a simplified check - full check requires distance analysis
            if self.factor > 0 {
                match (src_dir, tgt_dir) {
                    (Direction::Gt, Direction::Lt) => return false,
                    _ => {}
                }
            } else if self.factor < 0 {
                match (src_dir, tgt_dir) {
                    (Direction::Lt, Direction::Gt) => return false,
                    _ => {}
                }
            }
        }
        true
    }

    /// Compute the optimal skewing factor for wavefront parallelization.
    pub fn compute_optimal_factor(deps: &[Dependence], outer: usize, inner: usize) -> Option<i64> {
        let mut max_factor = 0i64;
        
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            if let Some(ref dist) = dep.distance {
                let d_outer = dist.get(outer).copied().unwrap_or(0);
                let d_inner = dist.get(inner).copied().unwrap_or(0);
                
                // Need factor such that d_inner + factor * d_outer >= 0
                if d_outer > 0 && d_inner < 0 {
                    let needed = (-d_inner + d_outer - 1) / d_outer;
                    max_factor = max_factor.max(needed);
                }
            }
        }
        
        if max_factor > 0 {
            Some(max_factor)
        } else {
            Some(1) // Default factor for wavefront
        }
    }
}

impl Transform for Skewing {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        let mut changed = false;
        
        for stmt in &mut program.statements {
            if let Some(new_schedule) = self.skew_schedule(stmt) {
                stmt.schedule = new_schedule;
                changed = true;
            }
        }
        
        Ok(changed)
    }

    fn is_legal(&self, _program: &PolyProgram, deps: &[Dependence]) -> bool {
        self.is_skewing_legal(deps)
    }

    fn name(&self) -> &str {
        "skewing"
    }
}

/// Automatically compute and apply wavefront skewing.
pub fn auto_wavefront(
    program: &mut PolyProgram,
    deps: &[Dependence],
    outer: usize,
    inner: usize,
) -> Result<bool> {
    if let Some(factor) = Skewing::compute_optimal_factor(deps, outer, inner) {
        let skewing = Skewing::new(inner, outer, factor);
        if skewing.is_legal(program, deps) {
            return skewing.apply(program);
        }
    }
    Ok(false)
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
            body: crate::ir::pir::StmtBody::Assignment {
                target: crate::ir::pir::AccessExpr { array: "A".to_string(), indices: vec![] },
                expr: crate::ir::pir::ComputeExpr::Int(0),
            },
            span: crate::utils::location::Span::default(),
        }
    }

    #[test]
    fn test_skewing_schedule() {
        let stmt = make_test_stmt(0, 2);
        let skewing = Skewing::new(1, 0, 1);
        
        let new_sched = skewing.skew_schedule(&stmt).unwrap();
        
        // Original: [i, j] -> [i, j]
        // After skew(1, 0, 1): [i, j] -> [i, j+i]
        assert_eq!(new_sched.apply(&[2, 3], &[]), vec![2, 5]);
    }

    #[test]
    fn test_wavefront_skewing() {
        let stmt = make_test_stmt(0, 2);
        let skewing = Skewing::wavefront(0, 1);
        
        let new_sched = skewing.skew_schedule(&stmt).unwrap();
        
        // [i, j] -> [i, j+i]
        assert_eq!(new_sched.apply(&[3, 4], &[]), vec![3, 7]);
    }

    #[test]
    fn test_skewing_legal() {
        let deps = vec![Dependence {
            source: StmtId::new(0),
            target: StmtId::new(0),
            kind: DependenceKind::Flow,
            relation: DependenceRelation::universe(2, 2, 0),
            distance: Some(vec![1, 0]),
            direction: vec![Direction::Lt, Direction::Eq],
            array: "A".to_string(),
            level: Some(0),
            is_loop_independent: false,
        }];
        
        let skewing = Skewing::new(1, 0, 1);
        assert!(skewing.is_skewing_legal(&deps));
    }
}
