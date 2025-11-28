//! Analysis passes for polyhedral optimization.
//!
//! This module provides:
//! - SCoP detection (identifying polyhedral regions)
//! - Dependence analysis (RAW, WAR, WAW dependencies)
//! - Full pipeline from AST to polyhedral representation

pub mod scop;
pub mod dependence;

pub use scop::{ScoPDetector, SCoP, ScoPStmt};
pub use dependence::{
    DependenceAnalysis, Dependence, DependenceKind, Direction,
    DependenceRelation, DependenceEquation, DependenceGraph,
    gcd_test, banerjee_test, extended_gcd, solve_diophantine,
};

// Re-export DependenceGraphSummary if it exists
#[allow(unused_imports)]
pub use dependence::DependenceGraphSummary;

use crate::frontend::ast::Program;
use crate::ir::pir::PolyProgram;
use crate::ir::{lower_program, lower_to_pir};
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

/// Full extraction pipeline: AST -> HIR -> PIR.
/// Returns polyhedral programs for all functions.
pub fn extract_polyhedral(program: &Program) -> Result<Vec<PolyProgram>> {
    // First check if the program forms valid SCoPs
    let detector = ScoPDetector::new();
    let scops = detector.detect(program)?;
    
    // If no valid SCoPs found, we can still try to lower (may fail later)
    if scops.is_empty() {
        anyhow::bail!("No valid SCoPs found in program");
    }
    
    // Lower to HIR
    let hir = lower_program(program)?;
    
    // Lower to PIR
    let pir = lower_to_pir(&hir)?;
    
    Ok(pir)
}

/// Extract and analyze: get polyhedral programs with dependencies.
pub fn extract_and_analyze(program: &Program) -> Result<Vec<(PolyProgram, Vec<Dependence>)>> {
    let pir_programs = extract_polyhedral(program)?;
    
    let mut results = Vec::new();
    for prog in pir_programs {
        let deps = analyze_dependencies(&prog)?;
        results.push((prog, deps));
    }
    
    Ok(results)
}

/// Build a dependence graph for a polyhedral program.
pub fn build_dependence_graph(program: &PolyProgram) -> Result<DependenceGraph> {
    let analyzer = DependenceAnalysis::new();
    analyzer.build_graph(program)
}

/// Check if a loop at the given level can be parallelized.
pub fn is_parallelizable(program: &PolyProgram, level: usize) -> Result<bool> {
    let graph = build_dependence_graph(program)?;
    Ok(graph.is_parallel_at(level))
}
