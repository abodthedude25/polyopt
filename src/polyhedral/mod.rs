//! Polyhedral data structures and operations.
//!
//! This module provides the mathematical foundation for polyhedral optimization:
//! - Affine expressions and constraints
//! - Integer sets (polyhedra)
//! - Affine maps (transformations)
//! - Operations on polyhedra

pub mod space;
pub mod expr;
pub mod constraint;
pub mod set;
pub mod map;
pub mod operations;

pub use space::Space;
pub use expr::AffineExpr;
pub use constraint::{Constraint, ConstraintKind};
pub use set::IntegerSet;
pub use map::AffineMap;
