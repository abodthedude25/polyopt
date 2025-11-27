//! # PolyOpt - Polyhedral Compiler Optimization Framework
//!
//! A comprehensive framework for polyhedral loop optimization, including:
//! - SCoP detection and extraction
//! - Dependence analysis
//! - Loop transformations (tiling, fusion, interchange, etc.)
//! - Automatic scheduling (Pluto-like)
//! - Multi-target code generation (C, OpenMP, CUDA, etc.)
//!
//! ## Architecture
//!
//! ```text
//! Input → Frontend → IR → Polyhedral Model → Analysis → Transform → CodeGen → Output
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use polyopt::prelude::*;
//!
//! let source = r#"
//!     func matmul(A[N][K], B[K][M], C[N][M]) {
//!         for i = 0 to N {
//!             for j = 0 to M {
//!                 for k = 0 to K {
//!                     C[i][j] = C[i][j] + A[i][k] * B[k][j];
//!                 }
//!             }
//!         }
//!     }
//! "#;
//!
//! let program = polyopt::parse(source)?;
//! let optimized = polyopt::optimize(program, OptimizationConfig::default())?;
//! let code = polyopt::codegen::emit_c(&optimized)?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(dead_code)] // During development

pub mod frontend;
pub mod ir;
pub mod polyhedral;
pub mod analysis;
pub mod transform;
pub mod codegen;
pub mod utils;

// Re-export commonly used types
pub mod prelude {
    //! Convenient re-exports of commonly used types and traits.
    
    pub use crate::frontend::{parse, ParseError};
    pub use crate::ir::ast::*;
    pub use crate::ir::hir::*;
    pub use crate::ir::pir::*;
    pub use crate::ir::{lower_program, lower_to_pir};
    pub use crate::polyhedral::{
        AffineExpr, Constraint, IntegerSet, AffineMap, Space,
    };
    pub use crate::analysis::{
        Dependence, DependenceKind, SCoP, ScoPStmt, ScoPDetector,
        extract_scops, extract_polyhedral,
    };
    pub use crate::transform::Transform;
    pub use crate::codegen::Target;
    pub use crate::utils::errors::*;
}

use anyhow::Result;

/// Main entry point for parsing source code.
pub fn parse(source: &str) -> Result<ir::ast::Program> {
    frontend::parse(source)
}

/// Lower AST to HIR.
pub fn lower_ast(program: &ir::ast::Program) -> Result<ir::HirProgram> {
    ir::lower_program(program)
}

/// Lower HIR to PIR (polyhedral representation).
pub fn lower_hir(hir: &ir::HirProgram) -> Result<Vec<ir::PolyProgram>> {
    ir::lower_to_pir(hir)
}

/// Full pipeline: parse source and lower to polyhedral representation.
pub fn parse_and_lower(source: &str) -> Result<Vec<ir::PolyProgram>> {
    let ast = parse(source)?;
    let hir = lower_ast(&ast)?;
    lower_hir(&hir)
}

/// Configuration for the optimization pipeline.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable loop tiling
    pub enable_tiling: bool,
    /// Tile sizes (if tiling enabled)
    pub tile_sizes: Vec<i64>,
    /// Enable loop fusion
    pub enable_fusion: bool,
    /// Enable loop interchange
    pub enable_interchange: bool,
    /// Enable automatic scheduling
    pub enable_auto_schedule: bool,
    /// Target for code generation
    pub target: codegen::Target,
    /// Enable vectorization hints
    pub enable_vectorization: bool,
    /// Enable parallelization
    pub enable_parallelization: bool,
    /// Verbosity level (0-3)
    pub verbosity: u8,
    /// Enable auto-tuning (stretch goal)
    #[cfg(feature = "autotuning")]
    pub enable_autotuning: bool,
    /// Enable ML-based scheduling (stretch goal)
    #[cfg(feature = "ml")]
    pub enable_ml_scheduling: bool,
    /// Enable sparse iteration space support (stretch goal)
    #[cfg(feature = "sparse")]
    pub enable_sparse: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_tiling: true,
            tile_sizes: vec![32, 32, 32],
            enable_fusion: true,
            enable_interchange: true,
            enable_auto_schedule: true,
            target: codegen::Target::C,
            enable_vectorization: true,
            enable_parallelization: true,
            verbosity: 1,
            #[cfg(feature = "autotuning")]
            enable_autotuning: false,
            #[cfg(feature = "ml")]
            enable_ml_scheduling: false,
            #[cfg(feature = "sparse")]
            enable_sparse: false,
        }
    }
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Run the full optimization pipeline.
pub fn optimize(
    _program: ir::ast::Program,
    _config: OptimizationConfig,
) -> Result<ir::pir::PolyProgram> {
    // This will be implemented as we build out the phases
    todo!("Full optimization pipeline not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}