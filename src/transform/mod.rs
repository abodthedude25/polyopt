//! Loop transformations for polyhedral optimization.
//!
//! This module provides various loop transformations that can be applied
//! to polyhedral programs to optimize for parallelism, locality, or both.
//!
//! # Transformations
//!
//! - **Tiling**: Break loops into smaller tiles for cache locality
//! - **Interchange**: Swap loop order to improve access patterns
//! - **Fusion**: Combine multiple loops for better reuse
//! - **Distribution**: Split loops to enable other optimizations
//! - **Skewing**: Transform loops for wavefront parallelism
//! - **Unrolling**: Replicate loop body for ILP
//! - **Strip-mining**: Single-dimension tiling
//!
//! # Scheduling
//!
//! The scheduler module provides automatic optimization using algorithms like:
//! - Feautrier (minimize latency)
//! - Pluto (maximize parallelism + locality)
//!
//! # Pipeline
//!
//! The pipeline module provides a high-level interface for applying
//! multiple transformations with automatic legality checking.

pub mod tiling;
pub mod interchange;
pub mod fusion;
pub mod scheduler;
pub mod skewing;
pub mod unrolling;
pub mod pipeline;

// Re-export main types from tiling
pub use tiling::{Tiling, tile_rectangular, tile_if_legal};

// Re-export main types from interchange
pub use interchange::{Interchange, Permutation, interchange_if_legal};

// Re-export main types from fusion
pub use fusion::{Fusion, Distribution, maximal_fusion, typed_fusion};

// Re-export main types from scheduler
pub use scheduler::{
    Scheduler, ScheduleAlgorithm, ScheduleOptions,
    auto_schedule, schedule_for_parallelism, schedule_for_locality,
};

// Re-export main types from skewing
pub use skewing::{Skewing, auto_wavefront};

// Re-export main types from unrolling
pub use unrolling::{Unrolling, StripMining, UnrollAndJam};

// Re-export main types from pipeline
pub use pipeline::{
    Pipeline, PipelineConfig, OptLevel, TargetInfo, OptimizationResult,
    quick_optimize, optimize_parallel, optimize_locality, analyze_and_optimize,
};

use crate::ir::pir::PolyProgram;
use crate::analysis::Dependence;
use anyhow::Result;

/// Apply automatic optimization to a program.
pub fn optimize(program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
    let scheduler = Scheduler::new();
    scheduler.schedule(program, deps)
}

/// Transformation pass trait.
pub trait Transform {
    /// Apply the transformation.
    fn apply(&self, program: &mut PolyProgram) -> Result<bool>;
    
    /// Check if transformation is legal given dependencies.
    fn is_legal(&self, program: &PolyProgram, deps: &[Dependence]) -> bool;
    
    /// Get transformation name.
    fn name(&self) -> &str;
}

/// Apply a sequence of transformations.
pub fn apply_transformations(
    program: &mut PolyProgram,
    transforms: &[&dyn Transform],
    deps: &[Dependence],
) -> Result<Vec<String>> {
    let mut applied = Vec::new();
    
    for transform in transforms {
        if transform.is_legal(program, deps) {
            if transform.apply(program)? {
                applied.push(transform.name().to_string());
            }
        }
    }
    
    Ok(applied)
}

/// Optimize a program with tiling and parallelization.
pub fn optimize_for_parallel_tiled(
    program: &mut PolyProgram,
    deps: &[Dependence],
    tile_sizes: Vec<i64>,
) -> Result<()> {
    let scheduler = Scheduler::new()
        .with_algorithm(ScheduleAlgorithm::Pluto)
        .with_parallelism(true)
        .with_tiling(tile_sizes);
    
    scheduler.schedule(program, deps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedral::set::IntegerSet;
    use crate::polyhedral::map::AffineMap;
    use crate::ir::pir::StmtId;

    fn make_test_program() -> PolyProgram {
        let mut program = PolyProgram::new("test".to_string());
        program.statements.push(crate::ir::pir::PolyStmt {
            id: StmtId::new(0),
            name: "S0".to_string(),
            domain: IntegerSet::universe(2),
            schedule: AffineMap::identity(2),
            reads: vec![],
            writes: vec![],
            body: crate::ir::pir::StmtBody::Assignment {
                target: crate::ir::pir::AccessExpr { array: "A".to_string(), indices: vec![] },
                expr: crate::ir::pir::ComputeExpr::Int(0),
            },
            span: crate::utils::location::Span::default(),
        });
        program
    }

    #[test]
    fn test_optimize() {
        let mut program = make_test_program();
        let deps = vec![];
        
        optimize(&mut program, &deps).unwrap();
        
        // Schedule should be modified
        assert!(program.statements[0].schedule.n_out() > 2);
    }
    
    #[test]
    fn test_transform_trait() {
        let mut program = make_test_program();
        let deps = vec![];
        
        let tiling = Tiling::new(vec![32, 32]);
        assert!(tiling.is_legal(&program, &deps));
        assert_eq!(tiling.name(), "tiling");
        
        let changed = tiling.apply(&mut program).unwrap();
        assert!(changed);
    }
}
