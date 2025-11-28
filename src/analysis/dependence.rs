//! Dependence analysis for polyhedral programs.
//!
//! This module implements data dependence analysis using the polyhedral model:
//! - GCD test for quick independence checks
//! - Banerjee bounds test for tighter analysis
//! - Dependence polyhedra construction
//! - Direction and distance vector computation
//! - RAW/WAR/WAW classification

use crate::ir::pir::{PolyProgram, PolyStmt, StmtId, AccessRelation, AccessKind};
use crate::polyhedral::set::IntegerSet;
use crate::polyhedral::map::AffineMap;
use crate::polyhedral::expr::AffineExpr;
use crate::polyhedral::constraint::{Constraint, ConstraintKind};
use crate::polyhedral::space::Space;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use std::collections::HashMap;

/// A data dependence between two statement instances.
#[derive(Debug, Clone)]
pub struct Dependence {
    /// Source statement
    pub source: StmtId,
    /// Target statement  
    pub target: StmtId,
    /// Kind of dependence
    pub kind: DependenceKind,
    /// Dependence polyhedron (pairs of iterations with dependence)
    pub relation: DependenceRelation,
    /// Distance vector (if uniform)
    pub distance: Option<Vec<i64>>,
    /// Direction vector
    pub direction: Vec<Direction>,
    /// Array involved in this dependence
    pub array: String,
    /// Level at which dependence is carried (outermost loop that carries it)
    pub level: Option<usize>,
    /// Whether dependence is loop-independent
    pub is_loop_independent: bool,
}

impl Dependence {
    /// Check if this dependence can be satisfied by parallelization at given level.
    pub fn is_parallelizable_at(&self, level: usize) -> bool {
        if level < self.direction.len() {
            matches!(self.direction[level], Direction::Eq)
        } else {
            true
        }
    }

    /// Check if this is a loop-carried dependence.
    pub fn is_loop_carried(&self) -> bool {
        !self.is_loop_independent
    }

    /// Get a human-readable description.
    pub fn description(&self) -> String {
        let kind_str = match self.kind {
            DependenceKind::Flow => "flow (RAW)",
            DependenceKind::Anti => "anti (WAR)",
            DependenceKind::Output => "output (WAW)",
            DependenceKind::Input => "input (RAR)",
        };
        let dir_str: String = self.direction.iter()
            .map(|d| d.to_char())
            .collect();
        format!("{} -> {} [{}] on {} dir=<{}>",
            self.source, self.target, kind_str, self.array, dir_str)
    }
}

/// Kind of data dependence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependenceKind {
    /// Read-after-write (true/flow dependence)
    Flow,
    /// Write-after-read (anti dependence)
    Anti,
    /// Write-after-write (output dependence)
    Output,
    /// Read-after-read (input dependence, not a true dependence)
    Input,
}

impl DependenceKind {
    /// Check if this is a "true" dependence that must be respected.
    pub fn is_true_dependence(&self) -> bool {
        !matches!(self, DependenceKind::Input)
    }

    /// Check if this dependence involves a write.
    pub fn involves_write(&self) -> bool {
        !matches!(self, DependenceKind::Input)
    }

    /// Get short name for the dependence kind.
    pub fn short_name(&self) -> &'static str {
        match self {
            DependenceKind::Flow => "RAW",
            DependenceKind::Anti => "WAR",
            DependenceKind::Output => "WAW",
            DependenceKind::Input => "RAR",
        }
    }
}

/// Direction of a dependence in one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// < (forward dependence, positive distance)
    Lt,
    /// = (same iteration, zero distance)
    Eq,
    /// > (backward dependence, negative distance)
    Gt,
    /// <= (forward or same)
    Le,
    /// >= (backward or same)
    Ge,
    /// * (unknown/any direction)
    Star,
}

impl Direction {
    /// Get the character representation.
    pub fn to_char(&self) -> char {
        match self {
            Direction::Lt => '<',
            Direction::Eq => '=',
            Direction::Gt => '>',
            Direction::Le => '≤',
            Direction::Ge => '≥',
            Direction::Star => '*',
        }
    }

    /// Check if this direction allows parallelization.
    pub fn allows_parallel(&self) -> bool {
        matches!(self, Direction::Eq)
    }

    /// Combine two directions (union).
    pub fn union(&self, other: &Direction) -> Direction {
        if self == other {
            *self
        } else {
            match (self, other) {
                (Direction::Lt, Direction::Eq) | (Direction::Eq, Direction::Lt) => Direction::Le,
                (Direction::Gt, Direction::Eq) | (Direction::Eq, Direction::Gt) => Direction::Ge,
                (Direction::Lt, Direction::Gt) | (Direction::Gt, Direction::Lt) => Direction::Star,
                (Direction::Le, Direction::Gt) | (Direction::Gt, Direction::Le) => Direction::Star,
                (Direction::Ge, Direction::Lt) | (Direction::Lt, Direction::Ge) => Direction::Star,
                (Direction::Le, Direction::Ge) | (Direction::Ge, Direction::Le) => Direction::Star,
                _ => Direction::Star,
            }
        }
    }

    /// Compute direction from a distance value.
    pub fn from_distance(dist: i64) -> Direction {
        match dist.cmp(&0) {
            std::cmp::Ordering::Less => Direction::Gt,
            std::cmp::Ordering::Equal => Direction::Eq,
            std::cmp::Ordering::Greater => Direction::Lt,
        }
    }
}

/// A dependence relation between iterations.
#[derive(Debug, Clone)]
pub struct DependenceRelation {
    /// Number of source dimensions
    pub src_dim: usize,
    /// Number of target dimensions
    pub tgt_dim: usize,
    /// Number of parameters
    pub n_param: usize,
    /// The set of (source_iter, target_iter) pairs where dependence exists
    pub pairs: IntegerSet,
}

impl DependenceRelation {
    /// Create an empty dependence relation.
    pub fn empty(src_dim: usize, tgt_dim: usize, n_param: usize) -> Self {
        Self {
            src_dim,
            tgt_dim,
            n_param,
            pairs: IntegerSet::empty(src_dim + tgt_dim),
        }
    }

    /// Create a universe (all pairs) dependence relation.
    pub fn universe(src_dim: usize, tgt_dim: usize, n_param: usize) -> Self {
        let space = Space::set_with_params(src_dim + tgt_dim, n_param);
        Self {
            src_dim,
            tgt_dim,
            n_param,
            pairs: IntegerSet::from_space(space),
        }
    }

    /// Check if the relation is empty (no dependence).
    pub fn is_empty(&self) -> bool {
        self.pairs.is_obviously_empty()
    }

    /// Check if a specific (source, target) pair is in the relation.
    pub fn contains(&self, src_point: &[i64], tgt_point: &[i64], params: &[i64]) -> bool {
        let mut combined = src_point.to_vec();
        combined.extend_from_slice(tgt_point);
        self.pairs.contains(&combined, params)
    }
}

/// Dependence analyzer.
pub struct DependenceAnalysis {
    /// Enable verbose output
    pub verbose: bool,
    /// Use precise (more expensive) analysis
    pub precise: bool,
}

impl DependenceAnalysis {
    /// Create a new dependence analyzer.
    pub fn new() -> Self {
        Self {
            verbose: false,
            precise: true,
        }
    }

    /// Create analyzer with options.
    pub fn with_options(verbose: bool, precise: bool) -> Self {
        Self { verbose, precise }
    }

    /// Analyze all dependencies in a program.
    pub fn analyze(&self, program: &PolyProgram) -> Result<Vec<Dependence>> {
        let mut deps = Vec::new();
        let n_stmts = program.statements.len();

        for i in 0..n_stmts {
            for j in i..n_stmts {
                let s1 = &program.statements[i];
                let s2 = &program.statements[j];

                deps.extend(self.analyze_pair(s1, s2, program, i == j)?);
                
                if i != j {
                    deps.extend(self.analyze_pair(s2, s1, program, false)?);
                }
            }
        }

        Ok(deps)
    }

    /// Analyze dependencies between two statements.
    fn analyze_pair(
        &self,
        src: &PolyStmt,
        tgt: &PolyStmt,
        program: &PolyProgram,
        is_same_stmt: bool,
    ) -> Result<Vec<Dependence>> {
        let mut deps = Vec::new();

        // Flow: write -> read (RAW)
        for write in &src.writes {
            for read in &tgt.reads {
                if write.array == read.array {
                    if let Some(dep) = self.compute_dependence(
                        src, tgt, write, read, DependenceKind::Flow,
                        program, is_same_stmt
                    )? {
                        deps.push(dep);
                    }
                }
            }
        }

        // Anti: read -> write (WAR)
        for read in &src.reads {
            for write in &tgt.writes {
                if read.array == write.array {
                    if let Some(dep) = self.compute_dependence(
                        src, tgt, read, write, DependenceKind::Anti,
                        program, is_same_stmt
                    )? {
                        deps.push(dep);
                    }
                }
            }
        }

        // Output: write -> write (WAW)
        for (i, write1) in src.writes.iter().enumerate() {
            for (j, write2) in tgt.writes.iter().enumerate() {
                if write1.array == write2.array {
                    if is_same_stmt && i == j {
                        continue;
                    }
                    if let Some(dep) = self.compute_dependence(
                        src, tgt, write1, write2, DependenceKind::Output,
                        program, is_same_stmt
                    )? {
                        deps.push(dep);
                    }
                }
            }
        }

        Ok(deps)
    }

    /// Compute a specific dependence between two accesses.
    fn compute_dependence(
        &self,
        src: &PolyStmt,
        tgt: &PolyStmt,
        src_access: &AccessRelation,
        tgt_access: &AccessRelation,
        kind: DependenceKind,
        program: &PolyProgram,
        is_same_stmt: bool,
    ) -> Result<Option<Dependence>> {
        let src_dim = src.depth();
        let tgt_dim = tgt.depth();
        let n_param = program.parameters.len();

        // Build dependence equations
        let dep_equations = self.build_dependence_equations(
            src_access, tgt_access, src_dim, tgt_dim, n_param
        );

        // GCD test
        for eq in &dep_equations {
            if !self.gcd_test_equation(eq) {
                return Ok(None);
            }
        }

        // Banerjee test
        let src_bounds = self.extract_bounds(&src.domain);
        let tgt_bounds = self.extract_bounds(&tgt.domain);
        
        for eq in &dep_equations {
            if !self.banerjee_test_equation(eq, &src_bounds, &tgt_bounds) {
                return Ok(None);
            }
        }

        // Build dependence polyhedron
        let relation = self.build_dependence_polyhedron(
            src, tgt, &dep_equations, n_param
        );

        if relation.is_empty() {
            return Ok(None);
        }

        // Compute direction and distance
        let (direction, distance, level) = self.compute_direction_distance(
            &relation, src_dim, tgt_dim, is_same_stmt
        );

        let is_loop_independent = is_same_stmt && 
            direction.iter().all(|d| matches!(d, Direction::Eq));

        Ok(Some(Dependence {
            source: src.id,
            target: tgt.id,
            kind,
            relation,
            distance,
            direction,
            array: src_access.array.clone(),
            level,
            is_loop_independent,
        }))
    }

    /// Build dependence equations from access functions.
    fn build_dependence_equations(
        &self,
        src_access: &AccessRelation,
        tgt_access: &AccessRelation,
        src_dim: usize,
        tgt_dim: usize,
        n_param: usize,
    ) -> Vec<DependenceEquation> {
        let mut equations = Vec::new();
        
        let n_subscripts = src_access.relation.n_out()
            .min(tgt_access.relation.n_out());

        for k in 0..n_subscripts {
            let src_expr = &src_access.relation.outputs[k];
            let tgt_expr = &tgt_access.relation.outputs[k];

            let mut eq = DependenceEquation {
                src_coeffs: vec![0; src_dim],
                tgt_coeffs: vec![0; tgt_dim],
                param_coeffs: vec![0; n_param],
                constant: 0,
            };

            for i in 0..src_dim.min(src_expr.coeffs.len()) {
                eq.src_coeffs[i] = src_expr.coeffs[i];
            }

            for j in 0..tgt_dim.min(tgt_expr.coeffs.len()) {
                eq.tgt_coeffs[j] = -tgt_expr.coeffs[j];
            }

            for p in 0..n_param.min(src_expr.param_coeffs.len()) {
                eq.param_coeffs[p] += src_expr.param_coeffs[p];
            }
            for p in 0..n_param.min(tgt_expr.param_coeffs.len()) {
                eq.param_coeffs[p] -= tgt_expr.param_coeffs[p];
            }

            eq.constant = src_expr.constant - tgt_expr.constant;

            equations.push(eq);
        }

        equations
    }

    /// GCD test for a single dependence equation.
    fn gcd_test_equation(&self, eq: &DependenceEquation) -> bool {
        use num_integer::Integer;

        let coeffs: Vec<i64> = eq.src_coeffs.iter()
            .chain(eq.tgt_coeffs.iter())
            .copied()
            .collect();

        let g = coeffs.iter().fold(0i64, |acc, &c| acc.gcd(&c));

        if g == 0 {
            eq.constant == 0
        } else {
            eq.constant % g == 0
        }
    }

    /// Banerjee bounds test for a single equation.
    fn banerjee_test_equation(
        &self,
        eq: &DependenceEquation,
        src_bounds: &[(i64, i64)],
        tgt_bounds: &[(i64, i64)],
    ) -> bool {
        let mut min_val = eq.constant as i128;
        let mut max_val = eq.constant as i128;

        for (i, &coeff) in eq.src_coeffs.iter().enumerate() {
            let (lb, ub) = src_bounds.get(i).copied().unwrap_or((0, 1000));
            let c = coeff as i128;
            if c > 0 {
                min_val += c * (lb as i128);
                max_val += c * (ub as i128);
            } else if c < 0 {
                min_val += c * (ub as i128);
                max_val += c * (lb as i128);
            }
        }

        for (j, &coeff) in eq.tgt_coeffs.iter().enumerate() {
            let (lb, ub) = tgt_bounds.get(j).copied().unwrap_or((0, 1000));
            let c = coeff as i128;
            if c > 0 {
                min_val += c * (lb as i128);
                max_val += c * (ub as i128);
            } else if c < 0 {
                min_val += c * (ub as i128);
                max_val += c * (lb as i128);
            }
        }

        min_val <= 0 && max_val >= 0
    }

    /// Extract loop bounds from an iteration domain.
    fn extract_bounds(&self, domain: &IntegerSet) -> Vec<(i64, i64)> {
        let n_dim = domain.dim();
        let mut bounds = vec![(0i64, 1000i64); n_dim];

        for constraint in &domain.constraints.constraints {
            let expr = &constraint.expr;
            
            let non_zero: Vec<_> = expr.coeffs.iter()
                .enumerate()
                .filter(|(_, &c)| c != 0)
                .collect();

            if non_zero.len() == 1 && expr.param_coeffs.iter().all(|&c| c == 0) {
                let (dim, &coeff) = non_zero[0];
                let bound = -expr.constant;

                if constraint.kind == ConstraintKind::Inequality {
                    if coeff > 0 {
                        let lb = (bound + coeff - 1) / coeff;
                        bounds[dim].0 = bounds[dim].0.max(lb);
                    } else {
                        let ub = bound / coeff;
                        bounds[dim].1 = bounds[dim].1.min(ub);
                    }
                }
            }
        }

        bounds
    }

    /// Build the dependence polyhedron.
    fn build_dependence_polyhedron(
        &self,
        src: &PolyStmt,
        tgt: &PolyStmt,
        equations: &[DependenceEquation],
        n_param: usize,
    ) -> DependenceRelation {
        let src_dim = src.depth();
        let tgt_dim = tgt.depth();
        let total_dim = src_dim + tgt_dim;

        let space = Space::set_with_params(total_dim, n_param);
        let mut result = IntegerSet::from_space(space);

        // Add source domain constraints
        for constraint in &src.domain.constraints.constraints {
            let lifted = self.lift_constraint(constraint, true, src_dim, tgt_dim, n_param);
            result.add_constraint(lifted);
        }

        // Add target domain constraints
        for constraint in &tgt.domain.constraints.constraints {
            let lifted = self.lift_constraint(constraint, false, src_dim, tgt_dim, n_param);
            result.add_constraint(lifted);
        }

        // Add dependence equations
        for eq in equations {
            let constraint = self.equation_to_constraint(eq, src_dim, tgt_dim, n_param);
            result.add_constraint(constraint);
        }

        DependenceRelation {
            src_dim,
            tgt_dim,
            n_param,
            pairs: result,
        }
    }

    /// Lift a constraint to the combined space.
    fn lift_constraint(
        &self,
        constraint: &Constraint,
        is_source: bool,
        src_dim: usize,
        tgt_dim: usize,
        n_param: usize,
    ) -> Constraint {
        let total_dim = src_dim + tgt_dim;
        let mut new_coeffs = vec![0i64; total_dim];
        
        let offset = if is_source { 0 } else { src_dim };
        let orig_dim = if is_source { src_dim } else { tgt_dim };

        for i in 0..orig_dim.min(constraint.expr.coeffs.len()) {
            new_coeffs[offset + i] = constraint.expr.coeffs[i];
        }

        let mut new_expr = AffineExpr::zero(total_dim, n_param);
        new_expr.coeffs = new_coeffs;
        new_expr.constant = constraint.expr.constant;
        
        for i in 0..n_param.min(constraint.expr.param_coeffs.len()) {
            new_expr.param_coeffs[i] = constraint.expr.param_coeffs[i];
        }

        Constraint {
            expr: new_expr,
            kind: constraint.kind,
        }
    }

    /// Convert a dependence equation to a constraint.
    fn equation_to_constraint(
        &self,
        eq: &DependenceEquation,
        src_dim: usize,
        tgt_dim: usize,
        n_param: usize,
    ) -> Constraint {
        let total_dim = src_dim + tgt_dim;
        let mut coeffs = vec![0i64; total_dim];

        for (i, &c) in eq.src_coeffs.iter().enumerate() {
            if i < src_dim {
                coeffs[i] = c;
            }
        }

        for (j, &c) in eq.tgt_coeffs.iter().enumerate() {
            if j < tgt_dim {
                coeffs[src_dim + j] = c;
            }
        }

        let mut expr = AffineExpr::zero(total_dim, n_param);
        expr.coeffs = coeffs;
        expr.constant = eq.constant;
        
        for (i, &c) in eq.param_coeffs.iter().enumerate() {
            if i < n_param {
                expr.param_coeffs[i] = c;
            }
        }

        Constraint::eq_zero(expr)
    }

    /// Compute direction and distance vectors.
    fn compute_direction_distance(
        &self,
        relation: &DependenceRelation,
        src_dim: usize,
        tgt_dim: usize,
        is_same_stmt: bool,
    ) -> (Vec<Direction>, Option<Vec<i64>>, Option<usize>) {
        let depth = src_dim.max(tgt_dim);
        let mut direction = vec![Direction::Star; depth];
        let mut distance = None;
        let mut level = None;

        if is_same_stmt && src_dim == tgt_dim {
            let mut dist_vec = vec![0i64; depth];
            let mut has_distance = false;
            
            for constraint in &relation.pairs.constraints.constraints {
                if constraint.kind == ConstraintKind::Equality {
                    let expr = &constraint.expr;
                    
                    for k in 0..depth {
                        let src_coeff = expr.coeff(k);
                        let tgt_coeff = if k + src_dim < expr.coeffs.len() {
                            expr.coeffs[k + src_dim]
                        } else {
                            0
                        };

                        if src_coeff == 1 && tgt_coeff == -1 {
                            let d = -expr.constant;
                            dist_vec[k] = d;
                            direction[k] = Direction::from_distance(d);
                            has_distance = true;
                            
                            if level.is_none() && d != 0 {
                                level = Some(k);
                            }
                        } else if src_coeff == -1 && tgt_coeff == 1 {
                            let d = expr.constant;
                            dist_vec[k] = d;
                            direction[k] = Direction::from_distance(d);
                            has_distance = true;
                            
                            if level.is_none() && d != 0 {
                                level = Some(k);
                            }
                        }
                    }
                }
            }

            if has_distance {
                distance = Some(dist_vec);
            }
        } else if !is_same_stmt {
            direction[0] = Direction::Lt;
            level = Some(0);
        }

        (direction, distance, level)
    }

    /// Check if a transformation preserves all dependencies.
    pub fn is_legal_transformation(
        &self,
        deps: &[Dependence],
        transform: &AffineMap,
    ) -> bool {
        for dep in deps {
            if dep.kind.is_true_dependence() {
                if !self.check_transformed_direction(dep, transform) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if a transformed dependence is lexicographically positive.
    fn check_transformed_direction(&self, dep: &Dependence, transform: &AffineMap) -> bool {
        if let Some(ref dist) = dep.distance {
            let transformed = transform.apply(dist, &[]);
            
            for &d in &transformed {
                if d > 0 {
                    return true;
                } else if d < 0 {
                    return false;
                }
            }
            return true;
        }
        true
    }

    /// Build a dependence graph for the program.
    pub fn build_graph(&self, program: &PolyProgram) -> Result<DependenceGraph> {
        let deps = self.analyze(program)?;
        Ok(DependenceGraph::from_dependences(deps, program))
    }

    /// Check if loop interchange at levels i and j is legal.
    pub fn is_interchange_legal(&self, deps: &[Dependence], level_i: usize, level_j: usize) -> bool {
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            // Get direction at both levels
            let dir_i = dep.direction.get(level_i).copied().unwrap_or(Direction::Star);
            let dir_j = dep.direction.get(level_j).copied().unwrap_or(Direction::Star);
            
            // After interchange, level_i becomes level_j and vice versa
            // Check if the transformed direction vector is still lexicographically positive
            
            // If the outer loop (level_i) has a positive direction, we're fine
            // But after interchange, level_j becomes the outer loop
            if dir_j == Direction::Gt {
                // Would become negative in outer position - illegal
                return false;
            }
            
            // If directions at both levels could be negative together
            if matches!(dir_i, Direction::Gt | Direction::Ge | Direction::Star) &&
               matches!(dir_j, Direction::Gt | Direction::Ge | Direction::Star) {
                // Need more careful analysis - conservative: reject
                if dir_i == Direction::Gt || dir_j == Direction::Gt {
                    // At least one is definitely negative
                    return false;
                }
            }
        }
        true
    }

    /// Check if tiling at given level with given tile size is legal.
    pub fn is_tiling_legal(&self, deps: &[Dependence], level: usize) -> bool {
        // Tiling is legal if all dependences at that level are non-negative
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            let dir = dep.direction.get(level).copied().unwrap_or(Direction::Star);
            
            // Tiling requires forward or zero dependences
            if matches!(dir, Direction::Gt) {
                return false;
            }
        }
        true
    }

    /// Check if loop fusion of two statement groups is legal.
    pub fn is_fusion_legal(&self, deps: &[Dependence], group1: &[StmtId], group2: &[StmtId]) -> bool {
        // Check that no dependence goes from group2 to group1
        // (i.e., group2 doesn't need to execute before group1)
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            let src_in_g2 = group2.contains(&dep.source);
            let tgt_in_g1 = group1.contains(&dep.target);
            
            if src_in_g2 && tgt_in_g1 {
                // Dependence from group2 to group1 would be violated by fusion
                return false;
            }
        }
        true
    }

    /// Check if a loop at given level can be parallelized.
    pub fn is_parallelizable(&self, deps: &[Dependence], level: usize) -> bool {
        for dep in deps {
            if !dep.kind.is_true_dependence() {
                continue;
            }
            
            // Check if dependence is carried at this level
            if let Some(carried_level) = dep.level {
                if carried_level == level {
                    // This loop carries the dependence - not parallelizable
                    return false;
                }
            }
            
            // Also check direction vector
            let dir = dep.direction.get(level).copied().unwrap_or(Direction::Star);
            if !matches!(dir, Direction::Eq) {
                // There's a cross-iteration dependence
                return false;
            }
        }
        true
    }

    /// Find the outermost parallelizable loop level.
    pub fn find_parallel_level(&self, deps: &[Dependence], max_depth: usize) -> Option<usize> {
        for level in 0..max_depth {
            if self.is_parallelizable(deps, level) {
                return Some(level);
            }
        }
        None
    }

    /// Compute the dependence distance at a specific level.
    pub fn get_distance_at_level(&self, dep: &Dependence, level: usize) -> Option<i64> {
        dep.distance.as_ref().and_then(|d| d.get(level).copied())
    }

    /// Check if all dependences are uniform (constant distance).
    pub fn are_dependences_uniform(&self, deps: &[Dependence]) -> bool {
        deps.iter().all(|d| d.distance.is_some())
    }

    /// Get the minimum positive distance at a level across all dependences.
    pub fn min_positive_distance(&self, deps: &[Dependence], level: usize) -> Option<i64> {
        let mut min_dist: Option<i64> = None;
        
        for dep in deps {
            if let Some(dist) = self.get_distance_at_level(dep, level) {
                if dist > 0 {
                    min_dist = Some(min_dist.map_or(dist, |m| m.min(dist)));
                }
            }
        }
        
        min_dist
    }
}

impl Default for DependenceAnalysis {
    fn default() -> Self { Self::new() }
}

/// A dependence equation.
#[derive(Debug, Clone)]
pub struct DependenceEquation {
    pub src_coeffs: Vec<i64>,
    pub tgt_coeffs: Vec<i64>,
    pub param_coeffs: Vec<i64>,
    pub constant: i64,
}

impl DependenceEquation {
    pub fn has_variables(&self) -> bool {
        self.src_coeffs.iter().any(|&c| c != 0) ||
        self.tgt_coeffs.iter().any(|&c| c != 0)
    }

    pub fn all_coeffs(&self) -> Vec<i64> {
        let mut result = self.src_coeffs.clone();
        result.extend_from_slice(&self.tgt_coeffs);
        result
    }
}

/// A dependence graph for a program.
#[derive(Debug, Clone)]
pub struct DependenceGraph {
    pub statements: Vec<StmtId>,
    pub edges: Vec<Dependence>,
    pub successors: HashMap<StmtId, Vec<usize>>,
    pub predecessors: HashMap<StmtId, Vec<usize>>,
}

impl DependenceGraph {
    pub fn from_dependences(deps: Vec<Dependence>, program: &PolyProgram) -> Self {
        let statements: Vec<StmtId> = program.statements.iter()
            .map(|s| s.id)
            .collect();
        
        let mut successors: HashMap<StmtId, Vec<usize>> = HashMap::new();
        let mut predecessors: HashMap<StmtId, Vec<usize>> = HashMap::new();

        for stmt in &statements {
            successors.insert(*stmt, Vec::new());
            predecessors.insert(*stmt, Vec::new());
        }

        for (i, dep) in deps.iter().enumerate() {
            if let Some(v) = successors.get_mut(&dep.source) {
                v.push(i);
            }
            if let Some(v) = predecessors.get_mut(&dep.target) {
                v.push(i);
            }
        }

        Self { statements, edges: deps, successors, predecessors }
    }

    pub fn get_outgoing(&self, stmt: StmtId) -> Vec<&Dependence> {
        self.successors.get(&stmt)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    pub fn get_incoming(&self, stmt: StmtId) -> Vec<&Dependence> {
        self.predecessors.get(&stmt)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    pub fn has_dependence(&self, from: StmtId, to: StmtId) -> bool {
        self.get_outgoing(from).iter().any(|d| d.target == to)
    }

    pub fn true_dependences(&self) -> Vec<&Dependence> {
        self.edges.iter()
            .filter(|d| d.kind.is_true_dependence())
            .collect()
    }

    pub fn is_parallel_at(&self, level: usize) -> bool {
        self.true_dependences().iter()
            .all(|d| d.is_parallelizable_at(level))
    }

    /// Get all dependences of a specific kind.
    pub fn dependences_of_kind(&self, kind: DependenceKind) -> Vec<&Dependence> {
        self.edges.iter()
            .filter(|d| d.kind == kind)
            .collect()
    }

    /// Get flow (RAW) dependences only.
    pub fn flow_dependences(&self) -> Vec<&Dependence> {
        self.dependences_of_kind(DependenceKind::Flow)
    }

    /// Get anti (WAR) dependences only.
    pub fn anti_dependences(&self) -> Vec<&Dependence> {
        self.dependences_of_kind(DependenceKind::Anti)
    }

    /// Get output (WAW) dependences only.
    pub fn output_dependences(&self) -> Vec<&Dependence> {
        self.dependences_of_kind(DependenceKind::Output)
    }

    /// Check if there's a cycle in the dependence graph.
    pub fn has_cycle(&self) -> bool {
        let sccs = self.strongly_connected_components();
        sccs.iter().any(|scc| scc.len() > 1 || self.has_self_loop(scc[0]))
    }

    /// Check if a statement has a self-loop.
    fn has_self_loop(&self, stmt: StmtId) -> bool {
        self.get_outgoing(stmt).iter().any(|d| d.target == stmt && !d.is_loop_independent)
    }

    /// Get all loop-carried dependences.
    pub fn loop_carried_dependences(&self) -> Vec<&Dependence> {
        self.edges.iter()
            .filter(|d| d.is_loop_carried())
            .collect()
    }

    /// Get all loop-independent dependences.
    pub fn loop_independent_dependences(&self) -> Vec<&Dependence> {
        self.edges.iter()
            .filter(|d| d.is_loop_independent)
            .collect()
    }

    /// Get dependences carried at a specific level.
    pub fn dependences_at_level(&self, level: usize) -> Vec<&Dependence> {
        self.edges.iter()
            .filter(|d| d.level == Some(level))
            .collect()
    }

    /// Compute a topological sort of statements (if acyclic).
    pub fn topological_sort(&self) -> Option<Vec<StmtId>> {
        let mut in_degree: HashMap<StmtId, usize> = HashMap::new();
        for stmt in &self.statements {
            in_degree.insert(*stmt, 0);
        }

        for dep in &self.edges {
            if dep.kind.is_true_dependence() && dep.source != dep.target {
                *in_degree.get_mut(&dep.target).unwrap() += 1;
            }
        }

        let mut queue: Vec<StmtId> = self.statements.iter()
            .filter(|s| in_degree[s] == 0)
            .copied()
            .collect();

        let mut result = Vec::new();

        while let Some(stmt) = queue.pop() {
            result.push(stmt);

            for dep in self.get_outgoing(stmt) {
                if dep.kind.is_true_dependence() && dep.source != dep.target {
                    let count = in_degree.get_mut(&dep.target).unwrap();
                    *count -= 1;
                    if *count == 0 {
                        queue.push(dep.target);
                    }
                }
            }
        }

        if result.len() == self.statements.len() {
            Some(result)
        } else {
            None // Has cycle
        }
    }

    /// Compute the maximum depth of any statement.
    pub fn max_depth(&self) -> usize {
        self.edges.iter()
            .map(|d| d.direction.len())
            .max()
            .unwrap_or(0)
    }

    /// Get a summary of the dependence graph.
    pub fn summary(&self) -> DependenceGraphSummary {
        DependenceGraphSummary {
            num_statements: self.statements.len(),
            num_dependences: self.edges.len(),
            num_flow: self.flow_dependences().len(),
            num_anti: self.anti_dependences().len(),
            num_output: self.output_dependences().len(),
            num_loop_carried: self.loop_carried_dependences().len(),
            num_loop_independent: self.loop_independent_dependences().len(),
            has_cycle: self.has_cycle(),
            max_depth: self.max_depth(),
        }
    }

    /// Get strongly connected components using Tarjan's algorithm.
    pub fn strongly_connected_components(&self) -> Vec<Vec<StmtId>> {
        let mut sccs = Vec::new();
        let mut index_counter = 0usize;
        let mut stack = Vec::new();
        let mut indices: HashMap<StmtId, usize> = HashMap::new();
        let mut lowlinks: HashMap<StmtId, usize> = HashMap::new();
        let mut on_stack: HashMap<StmtId, bool> = HashMap::new();

        for &v in &self.statements {
            if !indices.contains_key(&v) {
                self.strongconnect(
                    v, &mut index_counter, &mut stack,
                    &mut indices, &mut lowlinks, &mut on_stack, &mut sccs
                );
            }
        }

        sccs
    }

    fn strongconnect(
        &self,
        v: StmtId,
        index_counter: &mut usize,
        stack: &mut Vec<StmtId>,
        indices: &mut HashMap<StmtId, usize>,
        lowlinks: &mut HashMap<StmtId, usize>,
        on_stack: &mut HashMap<StmtId, bool>,
        sccs: &mut Vec<Vec<StmtId>>,
    ) {
        indices.insert(v, *index_counter);
        lowlinks.insert(v, *index_counter);
        *index_counter += 1;
        stack.push(v);
        on_stack.insert(v, true);

        for dep in self.get_outgoing(v) {
            let w = dep.target;
            if !indices.contains_key(&w) {
                self.strongconnect(w, index_counter, stack, indices, lowlinks, on_stack, sccs);
                let lowlink_v = *lowlinks.get(&v).unwrap();
                let lowlink_w = *lowlinks.get(&w).unwrap();
                lowlinks.insert(v, lowlink_v.min(lowlink_w));
            } else if *on_stack.get(&w).unwrap_or(&false) {
                let lowlink_v = *lowlinks.get(&v).unwrap();
                let index_w = *indices.get(&w).unwrap();
                lowlinks.insert(v, lowlink_v.min(index_w));
            }
        }

        if lowlinks.get(&v) == indices.get(&v) {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.insert(w, false);
                scc.push(w);
                if w == v { break; }
            }
            sccs.push(scc);
        }
    }
}

/// Summary of a dependence graph.
#[derive(Debug, Clone)]
pub struct DependenceGraphSummary {
    pub num_statements: usize,
    pub num_dependences: usize,
    pub num_flow: usize,
    pub num_anti: usize,
    pub num_output: usize,
    pub num_loop_carried: usize,
    pub num_loop_independent: usize,
    pub has_cycle: bool,
    pub max_depth: usize,
}

impl std::fmt::Display for DependenceGraphSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dependence Graph Summary:")?;
        writeln!(f, "  Statements: {}", self.num_statements)?;
        writeln!(f, "  Total dependences: {}", self.num_dependences)?;
        writeln!(f, "    Flow (RAW): {}", self.num_flow)?;
        writeln!(f, "    Anti (WAR): {}", self.num_anti)?;
        writeln!(f, "    Output (WAW): {}", self.num_output)?;
        writeln!(f, "  Loop-carried: {}", self.num_loop_carried)?;
        writeln!(f, "  Loop-independent: {}", self.num_loop_independent)?;
        writeln!(f, "  Has cycle: {}", self.has_cycle)?;
        writeln!(f, "  Max depth: {}", self.max_depth)?;
        Ok(())
    }
}

// Standalone functions for backward compatibility

/// GCD test for dependence.
pub fn gcd_test(coeffs: &[i64], constant: i64) -> bool {
    use num_integer::Integer;
    let g = coeffs.iter().fold(0i64, |acc, &c| acc.gcd(&c));
    if g == 0 {
        constant == 0
    } else {
        constant % g == 0
    }
}

/// Banerjee bounds test.
pub fn banerjee_test(
    coeffs: &[i64],
    constant: i64,
    lower_bounds: &[i64],
    upper_bounds: &[i64],
) -> bool {
    let mut min_val = constant;
    let mut max_val = constant;

    for (i, &c) in coeffs.iter().enumerate() {
        let lb = lower_bounds.get(i).copied().unwrap_or(0);
        let ub = upper_bounds.get(i).copied().unwrap_or(100);
        
        if c > 0 {
            min_val += c * lb;
            max_val += c * ub;
        } else {
            min_val += c * ub;
            max_val += c * lb;
        }
    }

    min_val <= 0 && max_val >= 0
}

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a.abs(), a.signum(), 0)
    } else {
        let (g, x, y) = extended_gcd(b, a % b);
        (g, y, x - (a / b) * y)
    }
}

/// Solve linear Diophantine equation: a*x + b*y = c.
pub fn solve_diophantine(a: i64, b: i64, c: i64) -> Option<(i64, i64)> {
    if a == 0 && b == 0 {
        return if c == 0 { Some((0, 0)) } else { None };
    }
    
    let (g, x0, y0) = extended_gcd(a, b);
    
    if c % g != 0 {
        return None;
    }
    
    let scale = c / g;
    Some((x0 * scale, y0 * scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::pir::*;

    #[test]
    fn test_gcd_test() {
        assert!(!gcd_test(&[2, -2], 1));
        assert!(gcd_test(&[2, -2], 0));
        assert!(gcd_test(&[3, -6], 9));
        assert!(!gcd_test(&[3, -6], 10));
    }

    #[test]
    fn test_banerjee() {
        assert!(banerjee_test(&[1, -1], 0, &[0, 0], &[9, 9]));
        assert!(!banerjee_test(&[1, -1], 20, &[0, 0], &[9, 9]));
    }

    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(12, 8);
        assert_eq!(g, 4);
        assert_eq!(12 * x + 8 * y, 4);
    }

    #[test]
    fn test_solve_diophantine() {
        let (x, y) = solve_diophantine(3, 5, 1).unwrap();
        assert_eq!(3 * x + 5 * y, 1);
        assert!(solve_diophantine(2, 4, 3).is_none());
    }

    #[test]
    fn test_direction_from_distance() {
        assert_eq!(Direction::from_distance(1), Direction::Lt);
        assert_eq!(Direction::from_distance(0), Direction::Eq);
        assert_eq!(Direction::from_distance(-1), Direction::Gt);
    }

    #[test]
    fn test_direction_union() {
        assert_eq!(Direction::Lt.union(&Direction::Eq), Direction::Le);
        assert_eq!(Direction::Gt.union(&Direction::Eq), Direction::Ge);
        assert_eq!(Direction::Lt.union(&Direction::Gt), Direction::Star);
    }

    #[test]
    fn test_dependence_kind() {
        assert!(DependenceKind::Flow.is_true_dependence());
        assert!(DependenceKind::Anti.is_true_dependence());
        assert!(DependenceKind::Output.is_true_dependence());
        assert!(!DependenceKind::Input.is_true_dependence());
    }

    #[test]
    fn test_dependence_relation() {
        let rel = DependenceRelation::empty(2, 2, 1);
        assert!(rel.is_empty());

        let rel2 = DependenceRelation::universe(2, 2, 1);
        assert!(!rel2.is_empty());
    }

    #[test]
    fn test_analyze_simple() {
        use crate::polyhedral::space::Space;

        let mut program = PolyProgram::new("test".to_string());
        program.parameters.push("N".to_string());

        let space = Space::set_with_params(1, 1);
        let mut domain = IntegerSet::from_space(space);
        let mut lb = AffineExpr::zero(1, 1);
        lb.coeffs[0] = 1;
        domain.add_constraint(Constraint::ge_zero(lb));
        let mut ub = AffineExpr::zero(1, 1);
        ub.coeffs[0] = -1;
        ub.param_coeffs[0] = 1;
        ub.constant = -1;
        domain.add_constraint(Constraint::ge_zero(ub));

        let mut read_map = AffineMap::identity(1);
        read_map.outputs[0].constant = -1;
        let read = AccessRelation::new("A".to_string(), read_map, AccessKind::Read);

        let write_map = AffineMap::identity(1);
        let write = AccessRelation::new("A".to_string(), write_map, AccessKind::Write);

        let body = StmtBody::Assignment {
            target: AccessExpr {
                array: "A".to_string(),
                indices: vec![AffineExprStr("i".to_string())],
            },
            expr: ComputeExpr::Binary {
                op: BinaryComputeOp::Add,
                left: Box::new(ComputeExpr::Access(AccessExpr {
                    array: "A".to_string(),
                    indices: vec![AffineExprStr("i-1".to_string())],
                })),
                right: Box::new(ComputeExpr::Int(1)),
            },
        };

        let stmt = PolyStmt {
            id: StmtId::new(0),
            name: "S0".to_string(),
            domain,
            schedule: AffineMap::identity(1),
            reads: vec![read],
            writes: vec![write],
            body,
            span: crate::utils::location::Span::dummy(),
        };

        program.statements.push(stmt);

        let analyzer = DependenceAnalysis::new();
        let deps = analyzer.analyze(&program).unwrap();

        assert!(!deps.is_empty());
        let flow_deps: Vec<_> = deps.iter()
            .filter(|d| d.kind == DependenceKind::Flow)
            .collect();
        assert!(!flow_deps.is_empty());
    }

    #[test]
    fn test_dependence_graph() {
        let deps = vec![
            Dependence {
                source: StmtId::new(0),
                target: StmtId::new(1),
                kind: DependenceKind::Flow,
                relation: DependenceRelation::universe(1, 1, 0),
                distance: Some(vec![0]),
                direction: vec![Direction::Eq],
                array: "A".to_string(),
                level: None,
                is_loop_independent: true,
            },
        ];

        let mut program = PolyProgram::new("test".to_string());
        program.statements.push(PolyStmt {
            id: StmtId::new(0),
            name: "S0".to_string(),
            domain: IntegerSet::universe(1),
            schedule: AffineMap::identity(1),
            reads: vec![],
            writes: vec![],
            body: StmtBody::Assignment {
                target: AccessExpr { array: "A".to_string(), indices: vec![] },
                expr: ComputeExpr::Int(0),
            },
            span: crate::utils::location::Span::dummy(),
        });
        program.statements.push(PolyStmt {
            id: StmtId::new(1),
            name: "S1".to_string(),
            domain: IntegerSet::universe(1),
            schedule: AffineMap::identity(1),
            reads: vec![],
            writes: vec![],
            body: StmtBody::Assignment {
                target: AccessExpr { array: "B".to_string(), indices: vec![] },
                expr: ComputeExpr::Int(0),
            },
            span: crate::utils::location::Span::dummy(),
        });

        let graph = DependenceGraph::from_dependences(deps, &program);
        
        assert!(graph.has_dependence(StmtId::new(0), StmtId::new(1)));
        assert!(!graph.has_dependence(StmtId::new(1), StmtId::new(0)));
        assert_eq!(graph.true_dependences().len(), 1);
    }

    #[test]
    fn test_scc() {
        let mut program = PolyProgram::new("test".to_string());
        for i in 0..3 {
            program.statements.push(PolyStmt {
                id: StmtId::new(i),
                name: format!("S{}", i),
                domain: IntegerSet::universe(1),
                schedule: AffineMap::identity(1),
                reads: vec![],
                writes: vec![],
                body: StmtBody::Assignment {
                    target: AccessExpr { array: "A".to_string(), indices: vec![] },
                    expr: ComputeExpr::Int(0),
                },
                span: crate::utils::location::Span::dummy(),
            });
        }

        // Create a cycle: S0 -> S1 -> S2 -> S0
        let deps = vec![
            Dependence {
                source: StmtId::new(0),
                target: StmtId::new(1),
                kind: DependenceKind::Flow,
                relation: DependenceRelation::universe(1, 1, 0),
                distance: None,
                direction: vec![Direction::Lt],
                array: "A".to_string(),
                level: Some(0),
                is_loop_independent: false,
            },
            Dependence {
                source: StmtId::new(1),
                target: StmtId::new(2),
                kind: DependenceKind::Flow,
                relation: DependenceRelation::universe(1, 1, 0),
                distance: None,
                direction: vec![Direction::Lt],
                array: "A".to_string(),
                level: Some(0),
                is_loop_independent: false,
            },
            Dependence {
                source: StmtId::new(2),
                target: StmtId::new(0),
                kind: DependenceKind::Flow,
                relation: DependenceRelation::universe(1, 1, 0),
                distance: None,
                direction: vec![Direction::Lt],
                array: "A".to_string(),
                level: Some(0),
                is_loop_independent: false,
            },
        ];

        let graph = DependenceGraph::from_dependences(deps, &program);
        let sccs = graph.strongly_connected_components();
        
        // All three statements should be in one SCC due to the cycle
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }
}
