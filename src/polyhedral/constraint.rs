//! Linear constraints for polyhedral representation.
//!
//! A constraint is a linear inequality or equality:
//! - Inequality: expr >= 0
//! - Equality: expr = 0

use crate::polyhedral::expr::AffineExpr;
use serde::{Serialize, Deserialize};
use std::fmt;

/// A linear constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraint {
    /// The affine expression (constraint is: expr >= 0 or expr = 0)
    pub expr: AffineExpr,
    /// Kind of constraint
    pub kind: ConstraintKind,
}

/// Kind of constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// Greater than or equal: expr >= 0
    Inequality,
    /// Equal: expr = 0
    Equality,
}

impl Constraint {
    /// Create a new constraint.
    pub fn new(expr: AffineExpr, kind: ConstraintKind) -> Self {
        Self { expr, kind }
    }

    /// Create an inequality constraint: expr >= 0
    pub fn ge_zero(expr: AffineExpr) -> Self {
        Self::new(expr, ConstraintKind::Inequality)
    }

    /// Create an equality constraint: expr = 0
    pub fn eq_zero(expr: AffineExpr) -> Self {
        Self::new(expr, ConstraintKind::Equality)
    }

    /// Create a constraint: lhs >= rhs
    pub fn ge(lhs: AffineExpr, rhs: AffineExpr) -> Self {
        Self::ge_zero(lhs - rhs)
    }

    /// Create a constraint: lhs <= rhs
    pub fn le(lhs: AffineExpr, rhs: AffineExpr) -> Self {
        Self::ge_zero(rhs - lhs)
    }

    /// Create a constraint: lhs = rhs
    pub fn eq(lhs: AffineExpr, rhs: AffineExpr) -> Self {
        Self::eq_zero(lhs - rhs)
    }

    /// Create a lower bound constraint: var >= lower
    pub fn lower_bound(dim: usize, lower: i64, n_dim: usize, n_param: usize) -> Self {
        // var - lower >= 0
        let mut expr = AffineExpr::var(dim, n_dim, n_param);
        expr.constant = -lower;
        Self::ge_zero(expr)
    }

    /// Create an upper bound constraint: var <= upper (i.e., var < upper+1)
    pub fn upper_bound(dim: usize, upper: i64, n_dim: usize, n_param: usize) -> Self {
        // upper - var >= 0
        let mut expr = AffineExpr::var(dim, n_dim, n_param);
        expr = -expr;
        expr.constant = upper;
        Self::ge_zero(expr)
    }

    /// Create a strict upper bound constraint: var < upper
    /// In integers, this is equivalent to: var <= upper - 1
    pub fn strict_upper_bound(dim: usize, upper: i64, n_dim: usize, n_param: usize) -> Self {
        Self::upper_bound(dim, upper - 1, n_dim, n_param)
    }

    /// Check if this is an equality constraint.
    pub fn is_equality(&self) -> bool {
        matches!(self.kind, ConstraintKind::Equality)
    }

    /// Check if this is an inequality constraint.
    pub fn is_inequality(&self) -> bool {
        matches!(self.kind, ConstraintKind::Inequality)
    }

    /// Check if this constraint is satisfied by the given point.
    pub fn is_satisfied(&self, dim_values: &[i64], param_values: &[i64]) -> bool {
        let value = self.expr.evaluate(dim_values, param_values);
        match self.kind {
            ConstraintKind::Inequality => value >= 0,
            ConstraintKind::Equality => value == 0,
        }
    }

    /// Negate the constraint (flip the inequality).
    pub fn negate(&self) -> Self {
        match self.kind {
            ConstraintKind::Inequality => {
                // expr >= 0 becomes -expr - 1 >= 0 (i.e., expr < 0 becomes expr <= -1)
                let mut neg_expr = -self.expr.clone();
                neg_expr.constant -= 1;
                Self::ge_zero(neg_expr)
            }
            ConstraintKind::Equality => {
                // Can't really negate equality in a single constraint
                // Would need disjunction: expr != 0 means expr > 0 OR expr < 0
                self.clone()
            }
        }
    }

    /// Get the number of dimensions.
    pub fn n_dim(&self) -> usize {
        self.expr.n_dim()
    }

    /// Get the number of parameters.
    pub fn n_param(&self) -> usize {
        self.expr.n_param()
    }

    /// Convert to string with given names.
    pub fn to_string_with_names(&self, dim_names: &[String], param_names: &[String]) -> String {
        let expr_str = self.expr.to_string_with_names(dim_names, param_names);
        match self.kind {
            ConstraintKind::Inequality => format!("{} >= 0", expr_str),
            ConstraintKind::Equality => format!("{} = 0", expr_str),
        }
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dim_names: Vec<String> = (0..self.n_dim()).map(|i| format!("d{}", i)).collect();
        let param_names: Vec<String> = (0..self.n_param()).map(|i| format!("p{}", i)).collect();
        write!(f, "{}", self.to_string_with_names(&dim_names, &param_names))
    }
}

/// A system of constraints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintSystem {
    /// All constraints in the system
    pub constraints: Vec<Constraint>,
    /// Number of dimensions
    pub n_dim: usize,
    /// Number of parameters
    pub n_param: usize,
}

impl ConstraintSystem {
    /// Create an empty constraint system.
    pub fn new(n_dim: usize, n_param: usize) -> Self {
        Self {
            constraints: Vec::new(),
            n_dim,
            n_param,
        }
    }

    /// Add a constraint.
    pub fn add(&mut self, constraint: Constraint) {
        assert_eq!(constraint.n_dim(), self.n_dim);
        assert_eq!(constraint.n_param(), self.n_param);
        self.constraints.push(constraint);
    }

    /// Add multiple constraints.
    pub fn add_all(&mut self, constraints: impl IntoIterator<Item = Constraint>) {
        for c in constraints {
            self.add(c);
        }
    }

    /// Get all equality constraints.
    pub fn equalities(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.iter().filter(|c| c.is_equality())
    }

    /// Get all inequality constraints.
    pub fn inequalities(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.iter().filter(|c| c.is_inequality())
    }

    /// Check if a point satisfies all constraints.
    pub fn is_satisfied(&self, dim_values: &[i64], param_values: &[i64]) -> bool {
        self.constraints.iter().all(|c| c.is_satisfied(dim_values, param_values))
    }

    /// Check if the system is empty (has no constraints).
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Get the number of constraints.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lower_bound() {
        // i >= 0
        let c = Constraint::lower_bound(0, 0, 2, 0);
        assert!(c.is_satisfied(&[0, 0], &[]));
        assert!(c.is_satisfied(&[5, 0], &[]));
        assert!(!c.is_satisfied(&[-1, 0], &[]));
    }

    #[test]
    fn test_upper_bound() {
        // i <= 10
        let c = Constraint::upper_bound(0, 10, 2, 0);
        assert!(c.is_satisfied(&[10, 0], &[]));
        assert!(c.is_satisfied(&[5, 0], &[]));
        assert!(!c.is_satisfied(&[11, 0], &[]));
    }

    #[test]
    fn test_equality() {
        // i = 5
        let mut expr = AffineExpr::var(0, 1, 0);
        expr.constant = -5;
        let c = Constraint::eq_zero(expr);
        assert!(c.is_satisfied(&[5], &[]));
        assert!(!c.is_satisfied(&[4], &[]));
    }

    #[test]
    fn test_constraint_system() {
        let mut sys = ConstraintSystem::new(2, 0);
        // 0 <= i < 10
        sys.add(Constraint::lower_bound(0, 0, 2, 0));
        sys.add(Constraint::strict_upper_bound(0, 10, 2, 0));
        // 0 <= j < 10
        sys.add(Constraint::lower_bound(1, 0, 2, 0));
        sys.add(Constraint::strict_upper_bound(1, 10, 2, 0));

        assert!(sys.is_satisfied(&[0, 0], &[]));
        assert!(sys.is_satisfied(&[5, 5], &[]));
        assert!(sys.is_satisfied(&[9, 9], &[]));
        assert!(!sys.is_satisfied(&[10, 0], &[]));
        assert!(!sys.is_satisfied(&[-1, 0], &[]));
    }
}
