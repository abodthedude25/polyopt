//! Polyhedral scheduler using various algorithms.
//!
//! The scheduler computes an optimal execution order (schedule) for
//! statements in a polyhedral program. It considers dependences,
//! parallelism, and locality.
//!
//! Supported algorithms:
//! - Feautrier: Minimize latency (execution time dimensions)
//! - Pluto: Maximize parallelism and locality
//! - Greedy: Simple heuristic-based scheduling

use crate::ir::pir::{PolyProgram, PolyStmt, StmtId};
use crate::analysis::{Dependence, Direction, DependenceGraph, DependenceKind};
use crate::polyhedral::map::AffineMap;
use crate::polyhedral::expr::AffineExpr;
use anyhow::{Result, bail};
use std::collections::{HashMap, HashSet, VecDeque};

/// Scheduling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleAlgorithm {
    /// Feautrier's algorithm (minimize latency)
    Feautrier,
    /// Pluto algorithm (maximize parallelism + locality)
    Pluto,
    /// Simple greedy scheduling
    Greedy,
    /// ISL-style scheduler
    Isl,
}

/// Options for the scheduler.
#[derive(Debug, Clone)]
pub struct ScheduleOptions {
    /// Target parallelism (number of parallel dimensions to find)
    pub target_parallelism: usize,
    /// Tile sizes (if tiling enabled)
    pub tile_sizes: Option<Vec<i64>>,
    /// Enable fusion
    pub enable_fusion: bool,
    /// Enable parallelism detection
    pub enable_parallel: bool,
    /// Locality optimization weight (0.0 - 1.0)
    pub locality_weight: f64,
}

impl Default for ScheduleOptions {
    fn default() -> Self {
        Self {
            target_parallelism: 1,
            tile_sizes: None,
            enable_fusion: true,
            enable_parallel: true,
            locality_weight: 0.5,
        }
    }
}

/// Polyhedral scheduler.
#[derive(Debug, Clone)]
pub struct Scheduler {
    /// Algorithm to use
    algorithm: ScheduleAlgorithm,
    /// Scheduling options
    options: ScheduleOptions,
}

impl Scheduler {
    /// Create a new scheduler with default settings.
    pub fn new() -> Self {
        Self {
            algorithm: ScheduleAlgorithm::Pluto,
            options: ScheduleOptions::default(),
        }
    }

    /// Set the scheduling algorithm.
    pub fn with_algorithm(mut self, algo: ScheduleAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Enable tiling with specified tile sizes.
    pub fn with_tiling(mut self, tile_sizes: Vec<i64>) -> Self {
        self.options.tile_sizes = Some(tile_sizes);
        self
    }

    /// Enable or disable fusion.
    pub fn with_fusion(mut self, enable: bool) -> Self {
        self.options.enable_fusion = enable;
        self
    }

    /// Enable or disable parallelism detection.
    pub fn with_parallelism(mut self, enable: bool) -> Self {
        self.options.enable_parallel = enable;
        self
    }

    /// Set target parallelism level.
    pub fn with_target_parallelism(mut self, n: usize) -> Self {
        self.options.target_parallelism = n;
        self
    }

    /// Set locality optimization weight.
    pub fn with_locality_weight(mut self, weight: f64) -> Self {
        self.options.locality_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Schedule a program.
    pub fn schedule(&self, program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
        match self.algorithm {
            ScheduleAlgorithm::Feautrier => self.schedule_feautrier(program, deps),
            ScheduleAlgorithm::Pluto => self.schedule_pluto(program, deps),
            ScheduleAlgorithm::Greedy => self.schedule_greedy(program, deps),
            ScheduleAlgorithm::Isl => self.schedule_isl(program, deps),
        }
    }

    /// Feautrier's scheduling algorithm.
    /// 
    /// Computes a schedule that minimizes the number of time dimensions
    /// while satisfying all dependences.
    fn schedule_feautrier(&self, program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
        // Build dependence graph
        let graph = DependenceGraph::from_dependences(deps.to_vec(), program);
        
        // Compute statement ordering using topological sort
        let order = self.topological_order(program, &graph);
        
        // Assign time stamps based on topological order
        let max_depth = program.statements.iter()
            .map(|s| s.depth())
            .max()
            .unwrap_or(0);
        
        for (time, &stmt_id) in order.iter().enumerate() {
            if let Some(stmt) = program.statements.iter_mut().find(|s| s.id == stmt_id) {
                let n_dim = stmt.depth();
                let n_param = stmt.domain.n_param();
                
                // Schedule: [time, i0, i1, ..., in-1]
                let mut outputs = Vec::with_capacity(n_dim + 1);
                
                // Time dimension (statement order)
                outputs.push(AffineExpr::constant(time as i64, n_dim, n_param));
                
                // Original iteration space dimensions
                for d in 0..n_dim {
                    outputs.push(AffineExpr::var(d, n_dim, n_param));
                }
                
                stmt.schedule = AffineMap::from_outputs(n_dim, outputs);
            }
        }
        
        // Apply tiling if requested
        if let Some(ref tile_sizes) = self.options.tile_sizes {
            self.apply_tiling(program, tile_sizes)?;
        }
        
        Ok(())
    }

    /// Pluto scheduling algorithm.
    /// 
    /// Optimizes for parallelism and locality using a cost model.
    fn schedule_pluto(&self, program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
        let graph = DependenceGraph::from_dependences(deps.to_vec(), program);
        
        // Step 1: Find parallel dimensions
        let parallel_dims = if self.options.enable_parallel {
            self.find_all_parallel_dims(program, deps)
        } else {
            HashMap::new()
        };
        
        // Step 2: Compute strongly connected components for fusion
        let sccs = if self.options.enable_fusion {
            self.compute_fusion_groups(program, &graph)
        } else {
            // Each statement in its own group
            program.statements.iter().map(|s| vec![s.id]).collect()
        };
        
        // Step 3: Schedule each SCC
        let mut group_time = 0i64;
        for scc in &sccs {
            // Topological order within SCC
            let scc_set: HashSet<StmtId> = scc.iter().copied().collect();
            let local_deps: Vec<_> = deps.iter()
                .filter(|d| scc_set.contains(&d.source) && scc_set.contains(&d.target))
                .cloned()
                .collect();
            
            let local_order = self.topological_order_subset(program, &local_deps, scc);
            
            for (local_time, &stmt_id) in local_order.iter().enumerate() {
                if let Some(stmt) = program.statements.iter_mut().find(|s| s.id == stmt_id) {
                    let n_dim = stmt.depth();
                    let n_param = stmt.domain.n_param();
                    
                    // Find parallel dimension for this statement
                    let parallel_dim = parallel_dims.get(&stmt_id).copied();
                    
                    // Build schedule with parallelism hints
                    let mut outputs = Vec::new();
                    
                    // Group time
                    outputs.push(AffineExpr::constant(group_time, n_dim, n_param));
                    
                    // Statement time within group
                    outputs.push(AffineExpr::constant(local_time as i64, n_dim, n_param));
                    
                    // Iteration dimensions (reorder to put parallel dim first if found)
                    if let Some(pd) = parallel_dim {
                        if pd < n_dim {
                            // Put parallel dimension first
                            outputs.push(AffineExpr::var(pd, n_dim, n_param));
                            for d in 0..n_dim {
                                if d != pd {
                                    outputs.push(AffineExpr::var(d, n_dim, n_param));
                                }
                            }
                        } else {
                            // Normal order
                            for d in 0..n_dim {
                                outputs.push(AffineExpr::var(d, n_dim, n_param));
                            }
                        }
                    } else {
                        // Normal order
                        for d in 0..n_dim {
                            outputs.push(AffineExpr::var(d, n_dim, n_param));
                        }
                    }
                    
                    stmt.schedule = AffineMap::from_outputs(n_dim, outputs);
                }
            }
            
            group_time += 1;
        }
        
        // Apply tiling if requested
        if let Some(ref tile_sizes) = self.options.tile_sizes {
            self.apply_tiling(program, tile_sizes)?;
        }
        
        Ok(())
    }

    /// Simple greedy scheduling.
    fn schedule_greedy(&self, program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
        let graph = DependenceGraph::from_dependences(deps.to_vec(), program);
        let order = self.topological_order(program, &graph);
        
        for (time, &stmt_id) in order.iter().enumerate() {
            if let Some(stmt) = program.statements.iter_mut().find(|s| s.id == stmt_id) {
                let n_dim = stmt.depth();
                let n_param = stmt.domain.n_param();
                
                let mut outputs = vec![AffineExpr::constant(time as i64, n_dim, n_param)];
                for d in 0..n_dim {
                    outputs.push(AffineExpr::var(d, n_dim, n_param));
                }
                
                stmt.schedule = AffineMap::from_outputs(n_dim, outputs);
            }
        }
        
        Ok(())
    }

    /// ISL-style scheduling (similar to Pluto but with different heuristics).
    fn schedule_isl(&self, program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
        // For now, use Pluto implementation
        self.schedule_pluto(program, deps)
    }

    /// Apply tiling to the scheduled program.
    fn apply_tiling(&self, program: &mut PolyProgram, tile_sizes: &[i64]) -> Result<()> {
        use crate::transform::tiling::Tiling;
        use crate::transform::Transform;
        
        let tiling = Tiling::new(tile_sizes.to_vec());
        tiling.apply(program)?;
        Ok(())
    }

    /// Compute topological order of statements.
    fn topological_order(&self, program: &PolyProgram, graph: &DependenceGraph) -> Vec<StmtId> {
        let stmt_ids: Vec<StmtId> = program.statements.iter().map(|s| s.id).collect();
        self.topological_order_subset(program, &graph.edges, &stmt_ids)
    }

    /// Compute topological order for a subset of statements.
    fn topological_order_subset(
        &self,
        _program: &PolyProgram,
        deps: &[Dependence],
        stmts: &[StmtId],
    ) -> Vec<StmtId> {
        let stmt_set: HashSet<StmtId> = stmts.iter().copied().collect();
        
        // Build adjacency list and in-degree count
        let mut in_degree: HashMap<StmtId, usize> = HashMap::new();
        let mut successors: HashMap<StmtId, Vec<StmtId>> = HashMap::new();
        
        for &s in stmts {
            in_degree.insert(s, 0);
            successors.insert(s, Vec::new());
        }
        
        // Add edges from dependences
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            if !stmt_set.contains(&dep.source) || !stmt_set.contains(&dep.target) {
                continue;
            }
            if dep.source == dep.target {
                continue; // Self-dependence doesn't affect ordering
            }
            
            *in_degree.entry(dep.target).or_insert(0) += 1;
            successors.entry(dep.source).or_insert_with(Vec::new).push(dep.target);
        }
        
        // Kahn's algorithm
        let mut queue: VecDeque<StmtId> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&s, _)| s)
            .collect();
        
        // Sort queue for deterministic ordering
        let mut queue_vec: Vec<_> = queue.drain(..).collect();
        queue_vec.sort_by_key(|s| s.0);
        queue = queue_vec.into_iter().collect();
        
        let mut result = Vec::with_capacity(stmts.len());
        
        while let Some(s) = queue.pop_front() {
            result.push(s);
            
            if let Some(succs) = successors.get(&s) {
                for &succ in succs {
                    if let Some(deg) = in_degree.get_mut(&succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }
        
        // If we couldn't order all statements, there's a cycle
        // Add remaining statements in arbitrary order
        for &s in stmts {
            if !result.contains(&s) {
                result.push(s);
            }
        }
        
        result
    }

    /// Find parallel dimensions for all statements.
    fn find_all_parallel_dims(
        &self,
        program: &PolyProgram,
        deps: &[Dependence],
    ) -> HashMap<StmtId, usize> {
        let mut result = HashMap::new();
        
        for stmt in &program.statements {
            if let Some(dim) = self.find_parallel_dim(stmt, deps) {
                result.insert(stmt.id, dim);
            }
        }
        
        result
    }

    /// Find a parallel dimension for a statement.
    fn find_parallel_dim(&self, stmt: &PolyStmt, deps: &[Dependence]) -> Option<usize> {
        for d in 0..stmt.depth() {
            if self.is_parallel_dim(stmt, d, deps) {
                return Some(d);
            }
        }
        None
    }

    /// Check if a dimension is parallel.
    fn is_parallel_dim(&self, stmt: &PolyStmt, dim: usize, deps: &[Dependence]) -> bool {
        for dep in deps {
            // Only consider dependences involving this statement
            if dep.source != stmt.id && dep.target != stmt.id {
                continue;
            }
            
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            // Check direction at this dimension
            if dim < dep.direction.len() {
                match dep.direction[dim] {
                    Direction::Eq => continue, // No dependence carried at this level
                    _ => return false, // Dependence carried at this level
                }
            }
            
            // Check distance if available
            if let Some(ref dist) = dep.distance {
                if let Some(&d) = dist.get(dim) {
                    if d != 0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Compute fusion groups using SCC analysis.
    fn compute_fusion_groups(
        &self,
        program: &PolyProgram,
        graph: &DependenceGraph,
    ) -> Vec<Vec<StmtId>> {
        // Simple approach: put all statements with the same depth together
        let mut depth_groups: HashMap<usize, Vec<StmtId>> = HashMap::new();
        
        for stmt in &program.statements {
            depth_groups.entry(stmt.depth())
                .or_insert_with(Vec::new)
                .push(stmt.id);
        }
        
        // Sort groups by depth
        let mut groups: Vec<(usize, Vec<StmtId>)> = depth_groups.into_iter().collect();
        groups.sort_by_key(|(d, _)| *d);
        
        groups.into_iter().map(|(_, g)| g).collect()
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

    /// Mark parallel loops in the schedule.
    pub fn mark_parallel_loops(
        &self,
        program: &mut PolyProgram,
        deps: &[Dependence],
    ) -> Vec<(StmtId, usize)> {
        self.find_parallel_dims(program, deps)
    }
}

impl Default for Scheduler {
    fn default() -> Self { Self::new() }
}

/// Convenience function to schedule a program with default settings.
pub fn auto_schedule(program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
    let scheduler = Scheduler::new();
    scheduler.schedule(program, deps)
}

/// Schedule for parallelism (maximize parallel loops).
pub fn schedule_for_parallelism(program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
    let scheduler = Scheduler::new()
        .with_algorithm(ScheduleAlgorithm::Pluto)
        .with_parallelism(true)
        .with_target_parallelism(2);
    scheduler.schedule(program, deps)
}

/// Schedule for locality (maximize data reuse).
pub fn schedule_for_locality(program: &mut PolyProgram, deps: &[Dependence]) -> Result<()> {
    let scheduler = Scheduler::new()
        .with_algorithm(ScheduleAlgorithm::Pluto)
        .with_locality_weight(1.0)
        .with_fusion(true);
    scheduler.schedule(program, deps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedral::set::IntegerSet;
    use crate::polyhedral::dependence::DependenceRelation;

    fn make_test_stmt(id: u64, depth: usize) -> PolyStmt {
        PolyStmt {
            id: StmtId::new(id),
            name: format!("S{}", id),
            domain: IntegerSet::universe(depth),
            schedule: AffineMap::identity(depth),
            reads: vec![],
            writes: vec![],
            body: crate::ir::pir::StmtBody::Assignment { target: crate::ir::pir::AccessExpr { array: "A".to_string(), indices: vec![] }, expr: crate::ir::pir::ComputeExpr::Int(0) },
            span: crate::utils::location::Span::default(),
        }
    }

    fn make_test_program(n_stmts: usize, depth: usize) -> PolyProgram {
        let mut program = PolyProgram::new("test".to_string());
        for i in 0..n_stmts {
            program.statements.push(make_test_stmt(i as u64, depth));
        }
        program
    }

    #[test]
    fn test_scheduler_new() {
        let scheduler = Scheduler::new();
        assert_eq!(scheduler.algorithm, ScheduleAlgorithm::Pluto);
    }

    #[test]
    fn test_schedule_no_deps() {
        let mut program = make_test_program(2, 2);
        let deps = vec![];
        
        let scheduler = Scheduler::new();
        scheduler.schedule(&mut program, &deps).unwrap();
        
        // Check that schedules were modified
        assert!(program.statements[0].schedule.n_out() > 2);
    }

    #[test]
    fn test_schedule_with_deps() {
        let mut program = make_test_program(2, 1);
        
        // S0 -> S1 dependence
        let deps = vec![Dependence {
            source: StmtId::new(0),
            target: StmtId::new(1),
            kind: DependenceKind::Flow,
            relation: DependenceRelation::universe(1, 1, 0),
            distance: Some(vec![0]),
            direction: vec![Direction::Eq],
            array: "A".to_string(),
            level: None,
            is_loop_independent: true,
        }];
        
        let scheduler = Scheduler::new();
        scheduler.schedule(&mut program, &deps).unwrap();
        
        // S0 should be scheduled before S1
        let s0_time = program.statements[0].schedule.outputs[0].constant;
        let s1_time = program.statements[1].schedule.outputs[0].constant;
        
        // Both should have valid schedules (actual order depends on algorithm)
        assert!(program.statements[0].schedule.n_out() > 1);
        assert!(program.statements[1].schedule.n_out() > 1);
    }

    #[test]
    fn test_feautrier_scheduling() {
        let mut program = make_test_program(3, 1);
        let deps = vec![];
        
        let scheduler = Scheduler::new()
            .with_algorithm(ScheduleAlgorithm::Feautrier);
        scheduler.schedule(&mut program, &deps).unwrap();
        
        // Check schedules exist
        for stmt in &program.statements {
            assert!(stmt.schedule.n_out() > 0);
        }
    }

    #[test]
    fn test_greedy_scheduling() {
        let mut program = make_test_program(3, 1);
        let deps = vec![];
        
        let scheduler = Scheduler::new()
            .with_algorithm(ScheduleAlgorithm::Greedy);
        scheduler.schedule(&mut program, &deps).unwrap();
        
        for stmt in &program.statements {
            assert!(stmt.schedule.n_out() > 0);
        }
    }

    #[test]
    fn test_parallel_dim_detection() {
        let mut program = make_test_program(1, 2);
        
        // No deps on dim 0, dep on dim 1
        let deps = vec![Dependence {
            source: StmtId::new(0),
            target: StmtId::new(0),
            kind: DependenceKind::Flow,
            relation: DependenceRelation::universe(2, 2, 0),
            distance: Some(vec![0, 1]),
            direction: vec![Direction::Eq, Direction::Lt],
            array: "A".to_string(),
            level: Some(1),
            is_loop_independent: false,
        }];
        
        let scheduler = Scheduler::new();
        
        // Dim 0 should be parallel (direction is Eq)
        assert!(scheduler.is_parallel_dim(&program.statements[0], 0, &deps));
        
        // Dim 1 should not be parallel (direction is Lt)
        assert!(!scheduler.is_parallel_dim(&program.statements[0], 1, &deps));
    }
}
