//! Analysis passes for polyhedral optimization.

pub mod scop;
pub mod dependence;

pub use scop::{ScoPDetector, SCoP, ScoPStmt};
pub use dependence::{DependenceAnalysis, Dependence, DependenceKind};

use crate::frontend::ast::Program;
use crate::ir::pir::PolyProgram;
use anyhow::Result;

/// Extract static control parts from a program.
pub fn extract_scops(program: &Program) -> Result<Vec<SCoP>> {
    let detector = ScoPDetector::new();
    detector.detect(program)
}

/// Analyze dependencies in a polyhedral program.
pub fn analyze_dependencies(program: &PolyProgram) -> Result<Vec<Dependence>> {
    let analyzer = DependenceAnalysis::new();
    analyzer.analyze(program)
}
