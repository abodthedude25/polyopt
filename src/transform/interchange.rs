//! Loop interchange transformation.

use crate::ir::pir::PolyProgram;
use crate::analysis::Dependence;
use crate::transform::Transform;
use anyhow::Result;

/// Loop interchange transformation.
pub struct Interchange {
    /// First dimension to swap
    pub dim1: usize,
    /// Second dimension to swap
    pub dim2: usize,
}

impl Interchange {
    pub fn new(dim1: usize, dim2: usize) -> Self {
        Self { dim1, dim2 }
    }
}

impl Transform for Interchange {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        // Placeholder implementation
        Ok(true)
    }

    fn is_legal(&self, _program: &PolyProgram, deps: &[Dependence]) -> bool {
        // Check all dependencies preserve direction
        for dep in deps {
            if dep.kind.is_true_dependence() {
                // Check direction vectors
                let d1 = dep.direction.get(self.dim1);
                let d2 = dep.direction.get(self.dim2);
                // Placeholder: more complex checking needed
            }
        }
        true
    }

    fn name(&self) -> &str {
        "interchange"
    }
}
