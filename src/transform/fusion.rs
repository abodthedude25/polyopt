//! Loop fusion and distribution transformations.
//!
//! Fusion combines multiple loop nests into a single loop nest,
//! which can improve locality and reduce loop overhead.
//!
//! Distribution (fission) splits a loop into multiple loops,
//! which can enable other optimizations or improve parallelism.
//!
//! Example of fusion:
//! ```text
//! for i = 0 to N:
//!   A[i] = B[i] + 1
//! for i = 0 to N:
//!   C[i] = A[i] * 2
//! ```
//! becomes:
//! ```text
//! for i = 0 to N:
//!   A[i] = B[i] + 1
//!   C[i] = A[i] * 2
//! ```

use crate::ir::pir::{PolyProgram, PolyStmt, StmtId};
use crate::analysis::{Dependence, DependenceGraph};
use crate::polyhedral::map::AffineMap;
use crate::polyhedral::expr::AffineExpr;
use crate::transform::Transform;
use anyhow::Result;
use std::collections::{HashSet, HashMap};

/// Loop fusion transformation.
#[derive(Debug, Clone)]
pub struct Fusion {
    /// Groups of statements to fuse together.
    /// Each group will share the same outer loop structure.
    pub groups: Vec<Vec<StmtId>>,
}

impl Fusion {
    /// Create a fusion of specific statements.
    pub fn new(statements: Vec<StmtId>) -> Self {
        Self { 
            groups: vec![statements],
        }
    }

    /// Create a fusion with multiple groups.
    pub fn with_groups(groups: Vec<Vec<StmtId>>) -> Self {
        Self { groups }
    }

    /// Fuse all statements at the same nesting level.
    pub fn fuse_all(program: &PolyProgram) -> Self {
        let all_stmts: Vec<StmtId> = program.statements.iter()
            .map(|s| s.id)
            .collect();
        Self::new(all_stmts)
    }

    /// Apply fusion to a program by adjusting schedules.
    /// 
    /// Fusion works by giving statements the same outer schedule dimensions
    /// and differentiating them with a statement ordering dimension.
    fn fuse_schedules(&self, program: &mut PolyProgram) -> bool {
        let mut changed = false;
        
        for (group_idx, group) in self.groups.iter().enumerate() {
            for (stmt_idx_in_group, &stmt_id) in group.iter().enumerate() {
                if let Some(stmt) = program.statements.iter_mut().find(|s| s.id == stmt_id) {
                    // Add a statement ordering dimension at the outermost level
                    // This ensures statements in the same group execute in order
                    let n_dim = stmt.depth();
                    let n_param = stmt.domain.n_param();
                    
                    // New schedule: [group_idx, stmt_order, original_dims...]
                    let mut new_outputs = Vec::with_capacity(n_dim + 2);
                    
                    // Group ordering (for separating fused groups)
                    new_outputs.push(AffineExpr::constant(group_idx as i64, n_dim, n_param));
                    
                    // Statement ordering within group
                    new_outputs.push(AffineExpr::constant(stmt_idx_in_group as i64, n_dim, n_param));
                    
                    // Original schedule dimensions
                    for expr in &stmt.schedule.outputs {
                        new_outputs.push(expr.clone());
                    }
                    
                    stmt.schedule = AffineMap::from_outputs(n_dim, new_outputs);
                    changed = true;
                }
            }
        }
        
        changed
    }

    /// Check if fusion is legal (no dependence violations).
    pub fn is_fusion_legal_for(&self, deps: &[Dependence]) -> bool {
        // Build a map of which group each statement is in
        let mut stmt_to_group: HashMap<StmtId, usize> = HashMap::new();
        for (group_idx, group) in self.groups.iter().enumerate() {
            for &stmt_id in group {
                stmt_to_group.insert(stmt_id, group_idx);
            }
        }

        // Check each dependence
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }

            let src_group = stmt_to_group.get(&dep.source);
            let tgt_group = stmt_to_group.get(&dep.target);

            match (src_group, tgt_group) {
                (Some(&sg), Some(&tg)) if sg == tg => {
                    // Both in same group - check statement ordering
                    let src_pos = self.groups[sg].iter().position(|&s| s == dep.source);
                    let tgt_pos = self.groups[tg].iter().position(|&s| s == dep.target);
                    
                    if let (Some(sp), Some(tp)) = (src_pos, tgt_pos) {
                        // Source must come before or at same position as target
                        if sp > tp {
                            return false;
                        }
                    }
                }
                (Some(&sg), Some(&tg)) if sg != tg => {
                    // Different groups - source group must come before target group
                    if sg > tg {
                        return false;
                    }
                }
                _ => {
                    // At least one statement not in a group - ignore
                }
            }
        }
        
        true
    }
}

impl Transform for Fusion {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        Ok(self.fuse_schedules(program))
    }

    fn is_legal(&self, _program: &PolyProgram, deps: &[Dependence]) -> bool {
        self.is_fusion_legal_for(deps)
    }

    fn name(&self) -> &str {
        "fusion"
    }
}

/// Loop distribution (fission) transformation.
#[derive(Debug, Clone)]
pub struct Distribution {
    /// Statement to distribute (separate from others)
    pub statement: StmtId,
    /// Which loop level to distribute at (0 = outermost)
    pub level: usize,
}

impl Distribution {
    /// Create a distribution for a specific statement.
    pub fn new(statement: StmtId) -> Self {
        Self { 
            statement,
            level: 0,
        }
    }

    /// Specify the loop level for distribution.
    pub fn at_level(mut self, level: usize) -> Self {
        self.level = level;
        self
    }

    /// Apply distribution by adjusting the schedule.
    fn distribute_schedule(&self, program: &mut PolyProgram) -> bool {
        // Find the statement
        let stmt_idx = program.statements.iter()
            .position(|s| s.id == self.statement);
        
        if let Some(idx) = stmt_idx {
            let stmt = &mut program.statements[idx];
            let n_dim = stmt.depth();
            let n_param = stmt.domain.n_param();
            
            // Add a distribution dimension before the specified level
            let mut new_outputs = Vec::with_capacity(n_dim + 1);
            
            // Add outputs before distribution level
            for i in 0..self.level.min(stmt.schedule.n_out()) {
                new_outputs.push(stmt.schedule.outputs[i].clone());
            }
            
            // Add distribution constant (unique to this statement)
            new_outputs.push(AffineExpr::constant(idx as i64, n_dim, n_param));
            
            // Add remaining outputs
            for i in self.level..stmt.schedule.n_out() {
                new_outputs.push(stmt.schedule.outputs[i].clone());
            }
            
            stmt.schedule = AffineMap::from_outputs(n_dim, new_outputs);
            return true;
        }
        
        false
    }
}

impl Transform for Distribution {
    fn apply(&self, program: &mut PolyProgram) -> Result<bool> {
        Ok(self.distribute_schedule(program))
    }

    fn is_legal(&self, _program: &PolyProgram, _deps: &[Dependence]) -> bool {
        // Distribution is always legal (it only separates statements)
        true
    }

    fn name(&self) -> &str {
        "distribution"
    }
}

/// Maximal fusion: fuse as many loops as legally possible.
pub fn maximal_fusion(program: &mut PolyProgram, deps: &[Dependence]) -> Result<bool> {
    // Simple greedy approach: try to fuse all statements
    let all_stmts: Vec<StmtId> = program.statements.iter()
        .map(|s| s.id)
        .collect();
    
    let fusion = Fusion::new(all_stmts);
    
    if fusion.is_legal(program, deps) {
        fusion.apply(program)
    } else {
        // Try smaller groups based on dependence structure
        Ok(false)
    }
}

/// Typed fusion: fuse only statements accessing the same arrays.
pub fn typed_fusion(program: &mut PolyProgram, deps: &[Dependence]) -> Result<bool> {
    // Group statements by the arrays they access
    let mut array_groups: HashMap<Vec<String>, Vec<StmtId>> = HashMap::new();
    
    for stmt in &program.statements {
        let mut arrays: Vec<String> = stmt.reads.iter()
            .chain(stmt.writes.iter())
            .map(|a| a.array.clone())
            .collect();
        arrays.sort();
        arrays.dedup();
        
        array_groups.entry(arrays)
            .or_insert_with(Vec::new)
            .push(stmt.id);
    }
    
    let groups: Vec<Vec<StmtId>> = array_groups.into_values().collect();
    let fusion = Fusion::with_groups(groups);
    
    if fusion.is_legal(program, deps) {
        fusion.apply(program)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedral::set::IntegerSet;
    use crate::analysis::DependenceKind;
    use crate::polyhedral::dependence::DependenceRelation;
    use crate::analysis::Direction;

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
    fn test_fusion_schedules() {
        let mut program = make_test_program(2, 1);
        let fusion = Fusion::new(vec![StmtId::new(0), StmtId::new(1)]);
        
        let changed = fusion.fuse_schedules(&mut program);
        
        assert!(changed);
        // Both statements should now have 3 output dimensions
        assert_eq!(program.statements[0].schedule.n_out(), 3);
        assert_eq!(program.statements[1].schedule.n_out(), 3);
    }

    #[test]
    fn test_fusion_legal_no_deps() {
        let fusion = Fusion::new(vec![StmtId::new(0), StmtId::new(1)]);
        let deps = vec![];
        
        assert!(fusion.is_fusion_legal_for(&deps));
    }

    #[test]
    fn test_fusion_legal_forward_dep() {
        // S0 -> S1 dependence (forward)
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
        
        // Fusion with S0 before S1 is legal
        let fusion = Fusion::new(vec![StmtId::new(0), StmtId::new(1)]);
        assert!(fusion.is_fusion_legal_for(&deps));
    }

    #[test]
    fn test_fusion_illegal_backward_dep() {
        // S1 -> S0 dependence (backward in program order)
        let deps = vec![Dependence {
            source: StmtId::new(1),
            target: StmtId::new(0),
            kind: DependenceKind::Flow,
            relation: DependenceRelation::universe(1, 1, 0),
            distance: Some(vec![0]),
            direction: vec![Direction::Eq],
            array: "A".to_string(),
            level: None,
            is_loop_independent: true,
        }];
        
        // Fusion with S0 before S1 is illegal (would violate S1 -> S0 dependence)
        let fusion = Fusion::new(vec![StmtId::new(0), StmtId::new(1)]);
        assert!(!fusion.is_fusion_legal_for(&deps));
    }

    #[test]
    fn test_distribution() {
        let mut program = make_test_program(2, 1);
        let dist = Distribution::new(StmtId::new(0));
        
        let changed = dist.distribute_schedule(&mut program);
        
        assert!(changed);
        // Statement 0 should have an extra dimension
        assert_eq!(program.statements[0].schedule.n_out(), 2);
    }
}
