//! Integer sets (polyhedra) for iteration domains.

use crate::polyhedral::space::Space;
use crate::polyhedral::constraint::{Constraint, ConstraintSystem, ConstraintKind};
use crate::polyhedral::expr::AffineExpr;
use serde::{Serialize, Deserialize};
use std::fmt;

/// An integer set defined by affine constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegerSet {
    pub space: Space,
    pub constraints: ConstraintSystem,
}

impl IntegerSet {
    pub fn empty(n_dim: usize) -> Self {
        let space = Space::set(n_dim);
        let mut constraints = ConstraintSystem::new(n_dim, 0);
        let mut expr = AffineExpr::zero(n_dim, 0);
        expr.constant = -1;
        constraints.add(Constraint::ge_zero(expr));
        Self { space, constraints }
    }

    pub fn universe(n_dim: usize) -> Self {
        Self {
            space: Space::set(n_dim),
            constraints: ConstraintSystem::new(n_dim, 0),
        }
    }

    pub fn from_space(space: Space) -> Self {
        let constraints = ConstraintSystem::new(space.n_dim, space.n_param);
        Self { space, constraints }
    }

    pub fn rectangular(bounds: &[i64]) -> Self {
        let n_dim = bounds.len();
        let mut set = Self::universe(n_dim);
        for (i, &bound) in bounds.iter().enumerate() {
            set.add_constraint(Constraint::lower_bound(i, 0, n_dim, 0));
            set.add_constraint(Constraint::strict_upper_bound(i, bound, n_dim, 0));
        }
        set
    }

    pub fn dim(&self) -> usize { self.space.n_dim }
    pub fn n_param(&self) -> usize { self.space.n_param }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.add(constraint);
    }

    pub fn contains(&self, point: &[i64], params: &[i64]) -> bool {
        self.constraints.is_satisfied(point, params)
    }

    pub fn is_obviously_empty(&self) -> bool {
        for c in &self.constraints.constraints {
            if c.expr.is_constant() {
                let val = c.expr.constant;
                match c.kind {
                    ConstraintKind::Inequality if val < 0 => return true,
                    ConstraintKind::Equality if val != 0 => return true,
                    _ => {}
                }
            }
        }
        false
    }

    pub fn intersect(&self, other: &IntegerSet) -> IntegerSet {
        assert_eq!(self.dim(), other.dim());
        let mut result = self.clone();
        for c in &other.constraints.constraints {
            result.add_constraint(c.clone());
        }
        result
    }

    pub fn dim_names(&self) -> Vec<String> { self.space.all_dim_names() }
    pub fn param_names(&self) -> Vec<String> { self.space.all_param_names() }

    pub fn with_dim_names(mut self, names: Vec<String>) -> Self {
        self.space = self.space.with_dim_names(names);
        self
    }

    pub fn with_param_names(mut self, names: Vec<String>) -> Self {
        self.space = self.space.with_param_names(names);
        self
    }
}

impl fmt::Display for IntegerSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dim_names = self.dim_names();
        let param_names = self.param_names();
        write!(f, "{{ [")?;
        for (i, name) in dim_names.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", name)?;
        }
        write!(f, "]")?;
        if !self.constraints.is_empty() {
            write!(f, " : ")?;
            for (i, c) in self.constraints.constraints.iter().enumerate() {
                if i > 0 { write!(f, " and ")?; }
                write!(f, "{}", c.to_string_with_names(&dim_names, &param_names))?;
            }
        }
        write!(f, " }}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular() {
        let set = IntegerSet::rectangular(&[10, 20]);
        assert!(set.contains(&[0, 0], &[]));
        assert!(set.contains(&[9, 19], &[]));
        assert!(!set.contains(&[10, 0], &[]));
    }
}
