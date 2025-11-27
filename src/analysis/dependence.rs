//! Dependence analysis for polyhedral programs.

use crate::ir::pir::{PolyProgram, PolyStmt, StmtId, AccessKind};
use crate::polyhedral::set::IntegerSet;
use crate::polyhedral::map::AffineMap;
use serde::{Serialize, Deserialize};
use anyhow::Result;

/// A data dependence between two statement instances.
#[derive(Debug, Clone)]
pub struct Dependence {
    /// Source statement
    pub source: StmtId,
    /// Target statement  
    pub target: StmtId,
    /// Kind of dependence
    pub kind: DependenceKind,
    /// Dependence polyhedron (pairs of iterations with dependence)
    pub relation: DependenceRelation,
    /// Distance vector (if uniform)
    pub distance: Option<Vec<i64>>,
    /// Direction vector
    pub direction: Vec<Direction>,
}

/// Kind of data dependence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependenceKind {
    /// Read-after-write (true/flow dependence)
    Flow,
    /// Write-after-read (anti dependence)
    Anti,
    /// Write-after-write (output dependence)
    Output,
    /// Read-after-read (input dependence, not a true dependence)
    Input,
}

impl DependenceKind {
    pub fn is_true_dependence(&self) -> bool {
        !matches!(self, DependenceKind::Input)
    }
}

/// Direction of a dependence in one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    /// < (forward)
    Lt,
    /// = (same iteration)
    Eq,
    /// > (backward)
    Gt,
    /// <= 
    Le,
    /// >=
    Ge,
    /// Unknown
    Star,
}

/// A dependence relation between iterations.
#[derive(Debug, Clone)]
pub struct DependenceRelation {
    /// The set of (source_iter, target_iter) pairs
    pub pairs: IntegerSet,
}

impl DependenceRelation {
    pub fn empty(src_dim: usize, tgt_dim: usize) -> Self {
        Self {
            pairs: IntegerSet::empty(src_dim + tgt_dim),
        }
    }

    pub fn universe(src_dim: usize, tgt_dim: usize) -> Self {
        Self {
            pairs: IntegerSet::universe(src_dim + tgt_dim),
        }
    }
}

/// Dependence analyzer.
pub struct DependenceAnalysis {
    /// Computed dependencies
    dependencies: Vec<Dependence>,
}

impl DependenceAnalysis {
    pub fn new() -> Self {
        Self {
            dependencies: Vec::new(),
        }
    }

    /// Analyze all dependencies in a program.
    pub fn analyze(&self, program: &PolyProgram) -> Result<Vec<Dependence>> {
        let mut deps = Vec::new();

        // For each pair of statements
        for (i, s1) in program.statements.iter().enumerate() {
            for s2 in program.statements.iter().skip(i) {
                // Check for dependencies between s1 and s2
                deps.extend(self.analyze_pair(s1, s2)?);
                
                // Also check s2 -> s1 if different
                if s1.id != s2.id {
                    deps.extend(self.analyze_pair(s2, s1)?);
                }
            }
        }

        Ok(deps)
    }

    /// Analyze dependencies between two statements.
    fn analyze_pair(&self, src: &PolyStmt, tgt: &PolyStmt) -> Result<Vec<Dependence>> {
        let mut deps = Vec::new();

        // Check all pairs of accesses
        for write in &src.writes {
            for read in &tgt.reads {
                if write.array == read.array {
                    // Potential flow dependence: write -> read
                    if let Some(dep) = self.compute_dependence(
                        src, tgt, write, read, DependenceKind::Flow
                    )? {
                        deps.push(dep);
                    }
                }
            }

            for other_write in &tgt.writes {
                if write.array == other_write.array && src.id != tgt.id {
                    // Potential output dependence: write -> write
                    if let Some(dep) = self.compute_dependence(
                        src, tgt, write, other_write, DependenceKind::Output
                    )? {
                        deps.push(dep);
                    }
                }
            }
        }

        for read in &src.reads {
            for write in &tgt.writes {
                if read.array == write.array {
                    // Potential anti dependence: read -> write
                    if let Some(dep) = self.compute_dependence(
                        src, tgt, read, write, DependenceKind::Anti
                    )? {
                        deps.push(dep);
                    }
                }
            }
        }

        Ok(deps)
    }

    /// Compute a specific dependence.
    fn compute_dependence(
        &self,
        src: &PolyStmt,
        tgt: &PolyStmt,
        src_access: &crate::ir::pir::AccessRelation,
        tgt_access: &crate::ir::pir::AccessRelation,
        kind: DependenceKind,
    ) -> Result<Option<Dependence>> {
        // The dependence exists if:
        // 1. Both iterations are in their respective domains
        // 2. They access the same memory location
        // 3. The source executes before the target (lexicographically)

        let src_dim = src.depth();
        let tgt_dim = tgt.depth();

        // Placeholder: assume dependence exists
        // Real implementation would solve the dependence equations
        let relation = DependenceRelation::universe(src_dim, tgt_dim);
        
        // Compute direction vector (placeholder)
        let direction = vec![Direction::Star; src_dim.max(tgt_dim)];

        Ok(Some(Dependence {
            source: src.id,
            target: tgt.id,
            kind,
            relation,
            distance: None,
            direction,
        }))
    }

    /// Check if a transformation preserves all dependencies.
    pub fn is_legal_transformation(
        &self,
        deps: &[Dependence],
        transform: &AffineMap,
    ) -> bool {
        // A transformation is legal if it preserves the lexicographic
        // positivity of all dependence distance vectors
        for dep in deps {
            if dep.kind.is_true_dependence() {
                // Check if transformed dependence is still lexicographically positive
                // Placeholder: assume legal
            }
        }
        true
    }
}

impl Default for DependenceAnalysis {
    fn default() -> Self { Self::new() }
}

/// Compute the GCD test for dependence.
pub fn gcd_test(coeffs: &[i64], constant: i64) -> bool {
    use num_integer::Integer;
    let g = coeffs.iter().fold(0i64, |acc, &c| acc.gcd(&c));
    if g == 0 {
        constant == 0
    } else {
        constant % g == 0
    }
}

/// Compute the Banerjee bounds test.
pub fn banerjee_test(
    coeffs: &[i64],
    constant: i64,
    lower_bounds: &[i64],
    upper_bounds: &[i64],
) -> bool {
    // Compute min and max possible values
    let mut min_val = constant;
    let mut max_val = constant;

    for (i, &c) in coeffs.iter().enumerate() {
        let lb = lower_bounds.get(i).copied().unwrap_or(0);
        let ub = upper_bounds.get(i).copied().unwrap_or(100);
        
        if c > 0 {
            min_val += c * lb;
            max_val += c * ub;
        } else {
            min_val += c * ub;
            max_val += c * lb;
        }
    }

    // Dependence possible if 0 is in [min_val, max_val]
    min_val <= 0 && max_val >= 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd_test() {
        // 2i - 2j = 1 has no integer solution
        assert!(!gcd_test(&[2, -2], 1));
        // 2i - 2j = 0 has solutions
        assert!(gcd_test(&[2, -2], 0));
    }

    #[test]
    fn test_banerjee() {
        // i - j = 0, with 0 <= i,j < 10
        assert!(banerjee_test(&[1, -1], 0, &[0, 0], &[9, 9]));
    }
}
