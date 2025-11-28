//! Optimization pipeline for polyhedral programs.
//!
//! This module provides a high-level interface for applying multiple
//! transformations in sequence with automatic legality checking.

use crate::ir::pir::PolyProgram;
use crate::analysis::{Dependence, DependenceAnalysis, DependenceGraph};
use crate::transform::{
    Transform, Scheduler, ScheduleAlgorithm,
    Tiling, Interchange, Fusion,
};
use crate::transform::skewing::Skewing;
use crate::transform::unrolling::{Unrolling, StripMining};
use anyhow::Result;

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization (identity schedule)
    O0,
    /// Basic scheduling
    O1,
    /// Scheduling with parallelism
    O2,
    /// Full optimization with tiling
    O3,
}

/// Target architecture hints.
#[derive(Debug, Clone)]
pub struct TargetInfo {
    /// Cache line size in bytes
    pub cache_line: usize,
    /// L1 cache size in bytes
    pub l1_size: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
    /// Number of SIMD lanes
    pub simd_width: usize,
    /// Whether to target parallelism
    pub parallel: bool,
}

impl Default for TargetInfo {
    fn default() -> Self {
        Self {
            cache_line: 64,
            l1_size: 32 * 1024,
            l2_size: 256 * 1024,
            simd_width: 8,
            parallel: true,
        }
    }
}

/// Optimization pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Optimization level
    pub opt_level: OptLevel,
    /// Target architecture info
    pub target: TargetInfo,
    /// Tile sizes (if None, auto-computed)
    pub tile_sizes: Option<Vec<i64>>,
    /// Unroll factor
    pub unroll_factor: Option<usize>,
    /// Enable fusion
    pub enable_fusion: bool,
    /// Enable vectorization hints
    pub enable_vectorization: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O2,
            target: TargetInfo::default(),
            tile_sizes: None,
            unroll_factor: None,
            enable_fusion: true,
            enable_vectorization: true,
        }
    }
}

impl PipelineConfig {
    /// Create configuration for maximum parallelism.
    pub fn for_parallelism() -> Self {
        Self {
            opt_level: OptLevel::O3,
            target: TargetInfo { parallel: true, ..Default::default() },
            enable_fusion: true,
            ..Default::default()
        }
    }

    /// Create configuration for best cache performance.
    pub fn for_locality() -> Self {
        Self {
            opt_level: OptLevel::O3,
            tile_sizes: Some(vec![32, 32, 32]),
            enable_fusion: true,
            ..Default::default()
        }
    }

    /// Create configuration for sequential execution.
    pub fn sequential() -> Self {
        Self {
            opt_level: OptLevel::O1,
            target: TargetInfo { parallel: false, ..Default::default() },
            ..Default::default()
        }
    }
}

/// Result of optimization pipeline.
#[derive(Debug)]
pub struct OptimizationResult {
    /// Transformations that were applied
    pub applied_transforms: Vec<String>,
    /// Whether the program was modified
    pub modified: bool,
    /// Parallel dimensions found
    pub parallel_dims: Vec<(usize, usize)>, // (stmt_index, dim)
    /// Estimated speedup (if computable)
    pub estimated_speedup: Option<f64>,
}

/// Optimization pipeline.
pub struct Pipeline {
    config: PipelineConfig,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Create a pipeline with default configuration.
    pub fn default_pipeline() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Run the optimization pipeline.
    pub fn optimize(
        &self,
        program: &mut PolyProgram,
        deps: &[Dependence],
    ) -> Result<OptimizationResult> {
        let mut result = OptimizationResult {
            applied_transforms: Vec::new(),
            modified: false,
            parallel_dims: Vec::new(),
            estimated_speedup: None,
        };

        match self.config.opt_level {
            OptLevel::O0 => {
                // No optimization
            }
            OptLevel::O1 => {
                self.apply_basic_scheduling(program, deps, &mut result)?;
            }
            OptLevel::O2 => {
                self.apply_basic_scheduling(program, deps, &mut result)?;
                self.apply_parallelism_optimizations(program, deps, &mut result)?;
            }
            OptLevel::O3 => {
                self.apply_basic_scheduling(program, deps, &mut result)?;
                self.apply_parallelism_optimizations(program, deps, &mut result)?;
                self.apply_locality_optimizations(program, deps, &mut result)?;
            }
        }

        // Find parallel dimensions
        result.parallel_dims = self.find_parallel_dims(program, deps);

        Ok(result)
    }

    /// Apply basic scheduling.
    fn apply_basic_scheduling(
        &self,
        program: &mut PolyProgram,
        deps: &[Dependence],
        result: &mut OptimizationResult,
    ) -> Result<()> {
        let scheduler = Scheduler::new()
            .with_algorithm(ScheduleAlgorithm::Pluto)
            .with_fusion(self.config.enable_fusion);

        scheduler.schedule(program, deps)?;
        result.applied_transforms.push("scheduling".to_string());
        result.modified = true;

        Ok(())
    }

    /// Apply parallelism-focused optimizations.
    fn apply_parallelism_optimizations(
        &self,
        program: &mut PolyProgram,
        deps: &[Dependence],
        result: &mut OptimizationResult,
    ) -> Result<()> {
        if !self.config.target.parallel {
            return Ok(());
        }

        // Try to find and expose parallelism
        let scheduler = Scheduler::new()
            .with_parallelism(true)
            .with_target_parallelism(2);

        // Look for skewing opportunities for wavefront parallelism
        let max_depth = program.statements.iter()
            .map(|s| s.depth())
            .max()
            .unwrap_or(0);

        if max_depth >= 2 {
            let skew = Skewing::wavefront(0, 1);
            if skew.is_legal(program, deps) {
                if skew.apply(program)? {
                    result.applied_transforms.push("wavefront_skewing".to_string());
                    result.modified = true;
                }
            }
        }

        Ok(())
    }

    /// Apply locality-focused optimizations.
    fn apply_locality_optimizations(
        &self,
        program: &mut PolyProgram,
        deps: &[Dependence],
        result: &mut OptimizationResult,
    ) -> Result<()> {
        // Apply tiling
        let tile_sizes = self.config.tile_sizes.clone()
            .unwrap_or_else(|| self.compute_tile_sizes(program));

        if !tile_sizes.is_empty() {
            let tiling = Tiling::new(tile_sizes);
            if tiling.is_legal(program, deps) {
                if tiling.apply(program)? {
                    result.applied_transforms.push("tiling".to_string());
                    result.modified = true;
                }
            }
        }

        // Apply unrolling if configured
        if let Some(factor) = self.config.unroll_factor {
            let unroll = Unrolling::new(0, factor);
            if unroll.apply(program)? {
                result.applied_transforms.push("unrolling".to_string());
                result.modified = true;
            }
        }

        Ok(())
    }

    /// Compute appropriate tile sizes based on target.
    fn compute_tile_sizes(&self, program: &PolyProgram) -> Vec<i64> {
        let max_depth = program.statements.iter()
            .map(|s| s.depth())
            .max()
            .unwrap_or(0);

        if max_depth == 0 {
            return vec![];
        }

        // Simple heuristic: use 32 for all dimensions
        // In a real implementation, this would consider cache sizes
        vec![32; max_depth.min(3)]
    }

    /// Find parallel dimensions.
    fn find_parallel_dims(
        &self,
        program: &PolyProgram,
        deps: &[Dependence],
    ) -> Vec<(usize, usize)> {
        let scheduler = Scheduler::new();
        scheduler.find_parallel_dims(program, deps)
            .into_iter()
            .enumerate()
            .map(|(i, (_, d))| (i, d))
            .collect()
    }
}

/// Quick optimization function with default settings.
pub fn quick_optimize(program: &mut PolyProgram, deps: &[Dependence]) -> Result<OptimizationResult> {
    let pipeline = Pipeline::default_pipeline();
    pipeline.optimize(program, deps)
}

/// Optimize for parallelism.
pub fn optimize_parallel(program: &mut PolyProgram, deps: &[Dependence]) -> Result<OptimizationResult> {
    let pipeline = Pipeline::new(PipelineConfig::for_parallelism());
    pipeline.optimize(program, deps)
}

/// Optimize for cache locality.
pub fn optimize_locality(program: &mut PolyProgram, deps: &[Dependence]) -> Result<OptimizationResult> {
    let pipeline = Pipeline::new(PipelineConfig::for_locality());
    pipeline.optimize(program, deps)
}

/// Full analysis and optimization pipeline.
pub fn analyze_and_optimize(program: &mut PolyProgram) -> Result<OptimizationResult> {
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(program)?;
    quick_optimize(program, &deps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedral::set::IntegerSet;
    use crate::polyhedral::map::AffineMap;
    use crate::ir::pir::StmtId;

    fn make_test_program(depth: usize) -> PolyProgram {
        let mut program = PolyProgram::new("test".to_string());
        program.statements.push(crate::ir::pir::PolyStmt {
            id: StmtId::new(0),
            name: "S0".to_string(),
            domain: IntegerSet::universe(depth),
            schedule: AffineMap::identity(depth),
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
    fn test_pipeline_o0() {
        let mut program = make_test_program(2);
        let deps = vec![];
        
        let config = PipelineConfig {
            opt_level: OptLevel::O0,
            ..Default::default()
        };
        let pipeline = Pipeline::new(config);
        let result = pipeline.optimize(&mut program, &deps).unwrap();
        
        assert!(result.applied_transforms.is_empty());
    }

    #[test]
    fn test_pipeline_o1() {
        let mut program = make_test_program(2);
        let deps = vec![];
        
        let config = PipelineConfig {
            opt_level: OptLevel::O1,
            ..Default::default()
        };
        let pipeline = Pipeline::new(config);
        let result = pipeline.optimize(&mut program, &deps).unwrap();
        
        assert!(result.applied_transforms.contains(&"scheduling".to_string()));
    }

    #[test]
    fn test_pipeline_o3() {
        let mut program = make_test_program(2);
        let deps = vec![];
        
        let config = PipelineConfig {
            opt_level: OptLevel::O3,
            tile_sizes: Some(vec![32, 32]),
            ..Default::default()
        };
        let pipeline = Pipeline::new(config);
        let result = pipeline.optimize(&mut program, &deps).unwrap();
        
        // At O3, should at least do scheduling
        assert!(result.modified);
        assert!(result.applied_transforms.contains(&"scheduling".to_string()));
        // Tiling might or might not be applied depending on schedule state
    }

    #[test]
    fn test_quick_optimize() {
        let mut program = make_test_program(2);
        let deps = vec![];
        
        let result = quick_optimize(&mut program, &deps).unwrap();
        assert!(result.modified);
    }
}
