//! Intermediate Representations for the polyhedral optimizer.
//!
//! This module defines three levels of IR:
//! - AST: Abstract Syntax Tree (re-exported from frontend)
//! - HIR: High-level IR (structured, close to source)
//! - PIR: Polyhedral IR (polyhedral representation)

pub mod ast;
pub mod hir;
pub mod pir;

pub use hir::*;
pub use pir::*;
