//! Intermediate Representations for the polyhedral optimizer.
//!
//! This module defines three levels of IR:
//! - AST: Abstract Syntax Tree (re-exported from frontend)
//! - HIR: High-level IR (structured, close to source)
//! - PIR: Polyhedral IR (polyhedral representation)
//!
//! And the lowering passes between them:
//! - lower_ast: AST -> HIR
//! - lower_hir: HIR -> PIR

pub mod ast;
pub mod hir;
pub mod pir;
pub mod lower_ast;
pub mod lower_hir;

pub use hir::*;
pub use pir::*;
pub use lower_ast::lower_program;
pub use lower_hir::lower_to_pir;
