//! Loop interchange transformation.
//!
//! Interchange swaps two loops in a loop nest, which can improve
//! memory access patterns or expose parallelism.
//!
//! Example:
//! ```text
//! for i = 0 to N:
//!   for j = 0 to M:
//!     A[i][j] = B[j][i]
//! ```
//! becomes (after interchange(0, 1)):
//! ```text
//! for j = 0 to M:
//!   for i = 0 to N:
//!     A[i][j] = B[j][i]
//! ```

use crate::ir::pir::{PolyProgram, PolyStmt};
use crate::analysis::{Dependence, Direction};
use crate::polyhedral::map::AffineMap;
use crate::transform::Transform;
use anyhow::Result;

/// Loop interchange transformation.
#[derive(Debug, Clone)]
pub struct Interchange {
    /// First dimension to swap
    pub dim1: usize,
    /// Second dimension to swap
    pub dim2: usize,
}

impl Interchange {
    /// Create a new interchange transformation.
    pub fn new(dim1: usize, dim2: usize) -> Self {
        Self { dim1, dim2 }
    }

    /// Apply interchange to a single statement's schedule.
    fn interchange_schedule(&self, stmt: &PolyStmt) -> Option<AffineMap> {
        let n_out = stmt.schedule.n_out();
        
        // Check dimensions are valid
        if self.dim1 >= n_out || self.dim2 >= n_out {
            return None;
        }
        
        if self.dim1 == self.dim2 {
            return Some(stmt.schedule.clone());
        }
        
        // Swap the two dimensions
        Some(stmt.schedule.interchange(self.dim1, self.dim2))
    }

    /// Check if interchange is legal based on dependence analysis.
    /// 
    /// Interchange is legal if, for all dependences, swapping the direction
    /// vector entries doesn't create a backward (negative) dependence in
    /// an outer dimension.
    pub fn is_interchange_legal_for(&self, deps: &[Dependence]) -> bool {
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            if !self.check_dependence_after_interchange(dep) {
                return false;
            }
        }
        true
    }

    /// Check if a single dependence remains valid after interchange.
    fn check_dependence_after_interchange(&self, dep: &Dependence) -> bool {
        let dir = &dep.direction;
        
        // Get directions at the two dimensions (use Star if out of bounds)
        let d1 = dir.get(self.dim1).copied().unwrap_or(Direction::Star);
        let d2 = dir.get(self.dim2).copied().unwrap_or(Direction::Star);
        
        // After interchange, d2 will be at position dim1 (outer) and
        // d1 will be at position dim2 (inner), assuming dim1 < dim2
        let (outer_pos, inner_pos) = if self.dim1 < self.dim2 {
            (self.dim1, self.dim2)
        } else {
            (self.dim2, self.dim1)
        };
        
        let (outer_dir, inner_dir) = if self.dim1 < self.dim2 {
            (d2, d1)
        } else {
            (d1, d2)
        };
        
        // Build the direction vector after interchange
        let mut new_dir = dir.clone();
        if outer_pos < new_dir.len() {
            new_dir[outer_pos] = outer_dir;
        }
        if inner_pos < new_dir.len() {
            new_dir[inner_pos] = inner_dir;
        }
        
        // Check lexicographic positivity
        self.is_lexicographically_positive(&new_dir)
    }

    /// Check if a direction vector is lexicographically positive.
    /// A valid dependence must be lexicographically positive (first non-zero
    /// entry must be positive/forward).
    fn is_lexicographically_positive(&self, dir: &[Direction]) -> bool {
        for d in dir {
            match d {
                Direction::Lt => return true,  // Forward - valid
                Direction::Gt => return false, // Backward - invalid
                Direction::Eq => continue,     // Equal - check next dimension
                Direction::Le => return true,  // Forward or equal - valid (≤)
                Direction::Ge => return false, // Backward or equal - could be backward
                Direction::Star => continue,   // Unknown - assume ok, check next
            }
        }
        true // All equal or unknown - valid (loop independent)
    }
}

impl Transform for Interchange {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        let mut changed = false;
        
        for stmt in &mut program.statements {
            if let Some(new_schedule) = self.interchange_schedule(stmt) {
                if new_schedule.coefficient_matrix() != stmt.schedule.coefficient_matrix() {
                    stmt.schedule = new_schedule;
                    changed = true;
                }
            }
        }
        
        Ok(changed)
    }

    fn is_legal(&self, _program: &PolyProgram, deps: &[Dependence]) -> bool {
        self.is_interchange_legal_for(deps)
    }

    fn name(&self) -> &str {
        "interchange"
    }
}

/// Create a permutation of loop dimensions.
#[derive(Debug, Clone)]
pub struct Permutation {
    /// New order of dimensions: permutation[i] is the original dimension
    /// that should be at position i
    pub permutation: Vec<usize>,
}

impl Permutation {
    /// Create a new permutation.
    pub fn new(permutation: Vec<usize>) -> Self {
        Self { permutation }
    }

    /// Create a permutation that reverses the order of dimensions.
    pub fn reverse(n_dim: usize) -> Self {
        Self {
            permutation: (0..n_dim).rev().collect(),
        }
    }

    /// Apply permutation to a schedule.
    fn permute_schedule(&self, stmt: &PolyStmt) -> AffineMap {
        let perm_map = AffineMap::permutation(&self.permutation);
        perm_map.compose(&stmt.schedule)
    }
}

impl Transform for Permutation {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        let mut changed = false;
        
        for stmt in &mut program.statements {
            let new_schedule = self.permute_schedule(stmt);
            stmt.schedule = new_schedule;
            changed = true;
        }
        
        Ok(changed)
    }

    fn is_legal(&self, _program: &PolyProgram, deps: &[Dependence]) -> bool {
        // Check each pair of swapped dimensions
        for i in 0..self.permutation.len() {
            for j in i+1..self.permutation.len() {
                if self.permutation[i] > self.permutation[j] {
                    // These dimensions are swapped
                    let interchange = Interchange::new(self.permutation[j], self.permutation[i]);
                    if !interchange.is_legal(_program, deps) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn name(&self) -> &str {
        "permutation"
    }
}

/// Convenience function to interchange two loops if legal.
pub fn interchange_if_legal(
    program: &mut PolyProgram,
    dim1: usize,
    dim2: usize,
    deps: &[Dependence],
) -> Result<bool> {
    let transform = Interchange::new(dim1, dim2);
    
    if transform.is_legal(program, deps) {
        transform.apply(program)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedral::set::IntegerSet;
    use crate::polyhedral::expr::AffineExpr;
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
    fn test_interchange_schedule() {
        let stmt = make_test_stmt(0, 3);
        let interchange = Interchange::new(0, 2);
        
        let new_sched = interchange.interchange_schedule(&stmt).unwrap();
        
        // Original: [i0, i1, i2] -> [i0, i1, i2]
        // After swap(0,2): [i0, i1, i2] -> [i2, i1, i0]
        assert_eq!(new_sched.apply(&[1, 2, 3], &[]), vec![3, 2, 1]);
    }

    #[test]
    fn test_interchange_legal_no_deps() {
        let deps = vec![];
        let interchange = Interchange::new(0, 1);
        
        assert!(interchange.is_interchange_legal_for(&deps));
    }

    #[test]
    fn test_interchange_legal_forward_deps() {
        // Dependence: (i, j) -> (i, j+1), direction = (=, <)
        let deps = vec![Dependence {
            source: StmtId::new(0),
            target: StmtId::new(0),
            kind: DependenceKind::Flow,
            relation: DependenceRelation::universe(2, 2, 0),
            distance: Some(vec![0, 1]),
            direction: vec![Direction::Eq, Direction::Lt],
            array: "A".to_string(),
            level: Some(1),
            is_loop_independent: false,
        }];
        
        let interchange = Interchange::new(0, 1);
        
        // After interchange: (=, <) -> (<, =) - still valid
        assert!(interchange.is_interchange_legal_for(&deps));
    }

    #[test]
    fn test_interchange_illegal() {
        // Dependence: (i, j) -> (i+1, j-1), direction = (<, >)
        // After interchange(0,1): (>, <) - illegal!
        let deps = vec![Dependence {
            source: StmtId::new(0),
            target: StmtId::new(0),
            kind: DependenceKind::Flow,
            relation: DependenceRelation::universe(2, 2, 0),
            distance: Some(vec![1, -1]),
            direction: vec![Direction::Lt, Direction::Gt],
            array: "A".to_string(),
            level: Some(0),
            is_loop_independent: false,
        }];
        
        let interchange = Interchange::new(0, 1);
        
        // After interchange: (<, >) -> (>, <) - first entry is backward!
        assert!(!interchange.is_interchange_legal_for(&deps));
    }
    
    #[test]
    fn test_permutation() {
        let stmt = make_test_stmt(0, 3);
        let perm = Permutation::new(vec![2, 0, 1]);
        
        let new_sched = perm.permute_schedule(&stmt);
        
        // Original: [i0, i1, i2] -> [i0, i1, i2]
        // Permutation [2,0,1]: position 0 gets dim 2, position 1 gets dim 0, position 2 gets dim 1
        // Result: [i2, i0, i1]
        assert_eq!(new_sched.apply(&[1, 2, 3], &[]), vec![3, 1, 2]);
    }
}
