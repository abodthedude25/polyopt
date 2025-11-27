//! Loop tiling transformation.

use crate::ir::pir::PolyProgram;
use crate::analysis::Dependence;
use crate::transform::Transform;
use anyhow::Result;

/// Loop tiling transformation.
pub struct Tiling {
    /// Tile sizes for each dimension
    pub tile_sizes: Vec<i64>,
}

impl Tiling {
    pub fn new(tile_sizes: Vec<i64>) -> Self {
        Self { tile_sizes }
    }

    pub fn with_default_size(n_dim: usize, size: i64) -> Self {
        Self {
            tile_sizes: vec![size; n_dim],
        }
    }
}

impl Transform for Tiling {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        // Placeholder implementation
        Ok(true)
    }

    fn is_legal(&self, _program: &PolyProgram, _deps: &[Dependence]) -> bool {
        // Tiling is always legal for permutable loops
        true
    }

    fn name(&self) -> &str {
        "tiling"
    }
}
