//! Loop fusion transformation.

use crate::ir::pir::{PolyProgram, StmtId};
use crate::analysis::Dependence;
use crate::transform::Transform;
use anyhow::Result;

/// Loop fusion transformation.
pub struct Fusion {
    /// Statements to fuse
    pub statements: Vec<StmtId>,
}

impl Fusion {
    pub fn new(statements: Vec<StmtId>) -> Self {
        Self { statements }
    }
}

impl Transform for Fusion {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        // Placeholder implementation
        Ok(true)
    }

    fn is_legal(&self, _program: &PolyProgram, deps: &[Dependence]) -> bool {
        // Check that fusion doesn't violate dependencies
        true
    }

    fn name(&self) -> &str {
        "fusion"
    }
}

/// Loop distribution (opposite of fusion).
pub struct Distribution {
    /// Statement to distribute
    pub statement: StmtId,
}

impl Distribution {
    pub fn new(statement: StmtId) -> Self {
        Self { statement }
    }
}

impl Transform for Distribution {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        Ok(true)
    }

    fn is_legal(&self, _program: &PolyProgram, _deps: &[Dependence]) -> bool {
        true
    }

    fn name(&self) -> &str {
        "distribution"
    }
}
