//! Static Control Part (SCoP) detection.

use crate::frontend::ast::{Program, Function, Stmt, StmtKind, Expr, ExprKind};
use crate::polyhedral::set::IntegerSet;
use crate::utils::location::Span;
use anyhow::Result;

/// A Static Control Part - a maximal region with affine loop bounds and accesses.
#[derive(Debug, Clone)]
pub struct SCoP {
    /// Name of the containing function
    pub function: String,
    /// Statements in the SCoP
    pub statements: Vec<ScoPStmt>,
    /// Parameters (symbolic constants)
    pub parameters: Vec<String>,
    /// Context constraints (e.g., N > 0)
    pub context: IntegerSet,
    /// Source span
    pub span: Span,
}

impl SCoP {
    pub fn new(function: String) -> Self {
        Self {
            function,
            statements: Vec::new(),
            parameters: Vec::new(),
            context: IntegerSet::universe(0),
            span: Span::dummy(),
        }
    }
}

/// A statement in a SCoP.
#[derive(Debug, Clone)]
pub struct ScoPStmt {
    /// Unique ID within the SCoP
    pub id: usize,
    /// Iteration domain
    pub domain: IntegerSet,
    /// Loop iterators (in order from outer to inner)
    pub iterators: Vec<String>,
    /// Source span
    pub span: Span,
}

/// SCoP detector.
pub struct ScoPDetector {
    /// Current parameters found
    parameters: Vec<String>,
    /// Current loop nest depth
    loop_depth: usize,
    /// Current iterators
    iterators: Vec<String>,
}

impl ScoPDetector {
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            loop_depth: 0,
            iterators: Vec::new(),
        }
    }

    /// Detect all SCoPs in a program.
    pub fn detect(&self, program: &Program) -> Result<Vec<SCoP>> {
        let mut scops = Vec::new();
        for func in &program.functions {
            if let Some(scop) = self.detect_in_function(func)? {
                scops.push(scop);
            }
        }
        Ok(scops)
    }

    /// Detect SCoP in a function.
    fn detect_in_function(&self, func: &Function) -> Result<Option<SCoP>> {
        let mut scop = SCoP::new(func.name.clone());
        scop.span = func.span;
        
        // Collect parameters from function signature
        for param in &func.params {
            for dim in &param.dimensions {
                self.collect_params_from_expr(dim, &mut scop.parameters);
            }
        }

        // Check if body forms a valid SCoP
        let mut detector = ScoPDetector::new();
        detector.parameters = scop.parameters.clone();
        
        if detector.is_scop_block(&func.body.statements) {
            scop.parameters = detector.parameters;
            Ok(Some(scop))
        } else {
            Ok(None)
        }
    }

    /// Check if a block of statements forms a valid SCoP.
    fn is_scop_block(&mut self, stmts: &[Stmt]) -> bool {
        for stmt in stmts {
            if !self.is_scop_stmt(stmt) {
                return false;
            }
        }
        true
    }

    /// Check if a statement is valid in a SCoP.
    fn is_scop_stmt(&mut self, stmt: &Stmt) -> bool {
        match &stmt.kind {
            StmtKind::For { iterator, start, end, step, body, .. } => {
                // Check bounds are affine
                if !self.is_affine_expr(start) || !self.is_affine_expr(end) {
                    return false;
                }
                if let Some(s) = step {
                    if !self.is_constant_expr(s) {
                        return false;
                    }
                }
                
                self.iterators.push(iterator.clone());
                self.loop_depth += 1;
                let result = self.is_scop_block(&body.statements);
                self.loop_depth -= 1;
                self.iterators.pop();
                result
            }
            
            StmtKind::If { condition, then_branch, else_branch } => {
                if !self.is_affine_condition(condition) {
                    return false;
                }
                if !self.is_scop_block(&then_branch.statements) {
                    return false;
                }
                if let Some(else_b) = else_branch {
                    if !self.is_scop_block(&else_b.statements) {
                        return false;
                    }
                }
                true
            }
            
            StmtKind::Assignment { target, value, .. } => {
                self.is_affine_access_target(target) && self.is_valid_rhs(value)
            }
            
            StmtKind::Declaration { value, .. } => {
                if let Some(v) = value {
                    self.is_valid_rhs(v)
                } else {
                    true
                }
            }
            
            StmtKind::Empty => true,
            
            _ => false, // While loops, returns, etc. break SCoP
        }
    }

    /// Check if an expression is affine in loop iterators and parameters.
    fn is_affine_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::IntLiteral(_) => true,
            ExprKind::Variable(name) => {
                self.iterators.contains(name) || self.parameters.contains(name)
            }
            ExprKind::Binary { op, left, right } => {
                use crate::frontend::ast::BinaryOp;
                match op {
                    BinaryOp::Add | BinaryOp::Sub => {
                        self.is_affine_expr(left) && self.is_affine_expr(right)
                    }
                    BinaryOp::Mul => {
                        (self.is_constant_expr(left) && self.is_affine_expr(right)) ||
                        (self.is_affine_expr(left) && self.is_constant_expr(right))
                    }
                    BinaryOp::Div | BinaryOp::Mod => {
                        self.is_affine_expr(left) && self.is_constant_expr(right)
                    }
                    _ => false,
                }
            }
            ExprKind::Unary { op, operand } => {
                use crate::frontend::ast::UnaryOp;
                matches!(op, UnaryOp::Neg) && self.is_affine_expr(operand)
            }
            ExprKind::Grouped(inner) => self.is_affine_expr(inner),
            ExprKind::Min(a, b) | ExprKind::Max(a, b) => {
                self.is_affine_expr(a) && self.is_affine_expr(b)
            }
            _ => false,
        }
    }

    /// Check if an expression is a compile-time constant.
    fn is_constant_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::IntLiteral(_) | ExprKind::FloatLiteral(_) => true,
            ExprKind::Variable(name) => self.parameters.contains(name),
            ExprKind::Binary { left, right, .. } => {
                self.is_constant_expr(left) && self.is_constant_expr(right)
            }
            ExprKind::Unary { operand, .. } => self.is_constant_expr(operand),
            ExprKind::Grouped(inner) => self.is_constant_expr(inner),
            _ => false,
        }
    }

    /// Check if a condition is affine (can be represented as affine constraints).
    fn is_affine_condition(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Binary { op, left, right } => {
                use crate::frontend::ast::BinaryOp;
                match op {
                    BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge |
                    BinaryOp::Eq | BinaryOp::Ne => {
                        self.is_affine_expr(left) && self.is_affine_expr(right)
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        self.is_affine_condition(left) && self.is_affine_condition(right)
                    }
                    _ => false,
                }
            }
            ExprKind::Unary { op, operand } => {
                use crate::frontend::ast::UnaryOp;
                matches!(op, UnaryOp::Not) && self.is_affine_condition(operand)
            }
            ExprKind::Grouped(inner) => self.is_affine_condition(inner),
            _ => false,
        }
    }

    /// Check if an assignment target has affine subscripts.
    fn is_affine_access_target(&self, target: &crate::frontend::ast::AssignTarget) -> bool {
        use crate::frontend::ast::AssignTarget;
        match target {
            AssignTarget::Variable(_) => true,
            AssignTarget::ArrayAccess { indices, .. } => {
                indices.iter().all(|idx| self.is_affine_expr(idx))
            }
        }
    }

    /// Check if an RHS expression is valid (all array accesses are affine).
    fn is_valid_rhs(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::ArrayAccess { indices, array } => {
                indices.iter().all(|idx| self.is_affine_expr(idx)) &&
                self.is_valid_rhs(array)
            }
            ExprKind::Binary { left, right, .. } => {
                self.is_valid_rhs(left) && self.is_valid_rhs(right)
            }
            ExprKind::Unary { operand, .. } => self.is_valid_rhs(operand),
            ExprKind::Call { args, .. } => args.iter().all(|a| self.is_valid_rhs(a)),
            ExprKind::Grouped(inner) => self.is_valid_rhs(inner),
            ExprKind::Min(a, b) | ExprKind::Max(a, b) => {
                self.is_valid_rhs(a) && self.is_valid_rhs(b)
            }
            _ => true, // Literals, variables are fine
        }
    }

    /// Collect parameter names from an expression.
    fn collect_params_from_expr(&self, expr: &Expr, params: &mut Vec<String>) {
        match &expr.kind {
            ExprKind::Variable(name) => {
                if !params.contains(name) {
                    params.push(name.clone());
                }
            }
            ExprKind::Binary { left, right, .. } => {
                self.collect_params_from_expr(left, params);
                self.collect_params_from_expr(right, params);
            }
            ExprKind::Unary { operand, .. } => {
                self.collect_params_from_expr(operand, params);
            }
            _ => {}
        }
    }
}

impl Default for ScoPDetector {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parse;

    #[test]
    fn test_simple_scop() {
        let source = "func test(A[N]) { for i = 0 to N { A[i] = i; } }";
        let program = parse(source).unwrap();
        let scops = ScoPDetector::new().detect(&program).unwrap();
        assert!(!scops.is_empty());
    }
}
