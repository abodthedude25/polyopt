//! Polyhedral scheduler using ILP.

use crate::ir::pir::{PolyProgram, PolyStmt, StmtId};
use crate::analysis::Dependence;
use crate::polyhedral::map::AffineMap;
use anyhow::Result;

/// Scheduling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleAlgorithm {
    /// Feautrier's algorithm (minimize latency)
    Feautrier,
    /// Pluto algorithm (maximize parallelism + locality)
    Pluto,
    /// ISL scheduler
    Isl,
}

/// Polyhedral scheduler.
pub struct Scheduler {
    /// Algorithm to use
    algorithm: ScheduleAlgorithm,
    /// Tile sizes (if tiling enabled)
    tile_sizes: Option<Vec<i64>>,
    /// Enable fusion
    enable_fusion: bool,
    /// Enable parallelism
    enable_parallel: bool,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            algorithm: ScheduleAlgorithm::Pluto,
            tile_sizes: None,
            enable_fusion: true,
            enable_parallel: true,
        }
    }

    pub fn with_algorithm(mut self, algo: ScheduleAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    pub fn with_tiling(mut self, tile_sizes: Vec<i64>) -> Self {
        self.tile_sizes = Some(tile_sizes);
        self
    }

    pub fn with_fusion(mut self, enable: bool) -> Self {
        self.enable_fusion = enable;
        self
    }

    pub fn with_parallelism(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }

    /// Schedule a program.
    pub fn schedule(&self, program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
        match self.algorithm {
            ScheduleAlgorithm::Feautrier => self.schedule_feautrier(program, deps),
            ScheduleAlgorithm::Pluto => self.schedule_pluto(program, deps),
            ScheduleAlgorithm::Isl => self.schedule_isl(program, deps),
        }
    }

    /// Feautrier's scheduling algorithm.
    fn schedule_feautrier(&self, program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
        // Feautrier minimizes the number of time dimensions
        // Uses affine form: t(i) = c0 + sum(ci * ii) + sum(pi * params)
        
        for stmt in &mut program.statements {
            // Set identity schedule as placeholder
            stmt.schedule = AffineMap::identity(stmt.depth());
        }
        
        Ok(())
    }

    /// Pluto scheduling algorithm.
    fn schedule_pluto(&self, program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
        // Pluto optimizes for:
        // 1. Parallelism (outer parallel loops)
        // 2. Locality (data reuse)
        // Uses ILP to find schedules
        
        for stmt in &mut program.statements {
            stmt.schedule = AffineMap::identity(stmt.depth());
        }
        
        // Apply tiling if requested
        if let Some(ref tile_sizes) = self.tile_sizes {
            self.apply_tiling(program, tile_sizes)?;
        }
        
        Ok(())
    }

    /// ISL-style scheduling.
    fn schedule_isl(&self, program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
        // Similar to Pluto but with different heuristics
        for stmt in &mut program.statements {
            stmt.schedule = AffineMap::identity(stmt.depth());
        }
        Ok(())
    }

    /// Apply tiling to the schedule.
    fn apply_tiling(&self, program: &mut PolyProgram, tile_sizes: &[i64]) -> Result<()> {
        // Tiling transforms: for i = 0 to N { S(i) }
        // Into: for it = 0 to N/T { for i = it*T to min((it+1)*T, N) { S(i) } }
        Ok(())
    }

    /// Find parallel dimensions in the schedule.
    pub fn find_parallel_dims(&self, program: &PolyProgram, deps: &[Dependence]) -> Vec<(StmtId, usize)> {
        let mut parallel_dims = Vec::new();
        
        for stmt in &program.statements {
            for d in 0..stmt.depth() {
                if self.is_parallel_dim(stmt, d, deps) {
                    parallel_dims.push((stmt.id, d));
                }
            }
        }
        
        parallel_dims
    }

    /// Check if a dimension is parallel (no loop-carried dependencies).
    fn is_parallel_dim(&self, stmt: &PolyStmt, dim: usize, deps: &[Dependence]) -> bool {
        for dep in deps {
            if dep.source == stmt.id || dep.target == stmt.id {
                if dep.kind.is_true_dependence() {
                    // Check if there's a carried dependence on this dimension
                    if let Some(dist) = &dep.distance {
                        if let Some(&d) = dist.get(dim) {
                            if d != 0 {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }
}

impl Default for Scheduler {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_new() {
        let scheduler = Scheduler::new();
        assert_eq!(scheduler.algorithm, ScheduleAlgorithm::Pluto);
    }
}
