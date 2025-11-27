//! Loop transformations for polyhedral optimization.

pub mod tiling;
pub mod interchange;
pub mod fusion;
pub mod scheduler;

pub use scheduler::Scheduler;

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
