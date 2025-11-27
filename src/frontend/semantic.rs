//! Semantic analysis for the polyhedral DSL.

use crate::frontend::ast::*;
use crate::utils::errors::{SemanticError, SemanticErrorKind};
use crate::utils::location::Span;
use anyhow::{Result, bail};
use std::collections::HashMap;

/// Perform semantic analysis on a program.
pub fn analyze(program: &mut Program) -> Result<()> {
    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze_program(program)?;
    Ok(())
}

/// Symbol information.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub ty: Type,
    pub is_mutable: bool,
    pub is_param: bool,
    pub is_loop_var: bool,
    pub span: Span,
}

/// A scope containing symbol definitions.
#[derive(Debug, Default)]
pub struct Scope {
    symbols: HashMap<String, SymbolInfo>,
    parent: Option<usize>,
}

/// Semantic analyzer.
pub struct SemanticAnalyzer {
    scopes: Vec<Scope>,
    current_scope: usize,
    errors: Vec<SemanticError>,
    current_function: Option<String>,
    loop_vars: Vec<String>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self {
            scopes: vec![Scope::default()],
            current_scope: 0,
            errors: Vec::new(),
            current_function: None,
            loop_vars: Vec::new(),
        }
    }

    pub fn analyze_program(&mut self, program: &mut Program) -> Result<()> {
        // First pass: collect globals and function signatures
        for global in &program.globals {
            self.declare_symbol(SymbolInfo {
                name: global.name.clone(),
                ty: global.ty.clone(),
                is_mutable: false,
                is_param: global.is_param,
                is_loop_var: false,
                span: global.span,
            })?;
        }

        for func in &program.functions {
            self.declare_symbol(SymbolInfo {
                name: func.name.clone(),
                ty: Type::Void,
                is_mutable: false,
                is_param: false,
                is_loop_var: false,
                span: func.span,
            })?;
        }

        // Second pass: analyze function bodies
        for func in &mut program.functions {
            self.analyze_function(func)?;
        }

        if !self.errors.is_empty() {
            bail!("Semantic errors: {:?}", self.errors);
        }
        Ok(())
    }

    fn analyze_function(&mut self, func: &mut Function) -> Result<()> {
        self.current_function = Some(func.name.clone());
        self.push_scope();

        for param in &func.params {
            let ty = if param.is_array() {
                Type::Array {
                    element: Box::new(Type::Unknown),
                    dimensions: param.dimensions.iter().map(|_| None).collect(),
                }
            } else {
                param.ty.clone()
            };

            self.declare_symbol(SymbolInfo {
                name: param.name.clone(),
                ty,
                is_mutable: true,
                is_param: true,
                is_loop_var: false,
                span: param.span,
            })?;

            for dim_expr in &param.dimensions {
                self.collect_parameters(dim_expr);
            }
        }

        self.analyze_block(&mut func.body)?;
        self.pop_scope();
        self.current_function = None;
        Ok(())
    }

    fn collect_parameters(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Variable(name) => {
                if self.lookup_symbol(name).is_none() {
                    let _ = self.declare_symbol(SymbolInfo {
                        name: name.clone(),
                        ty: Type::Int,
                        is_mutable: false,
                        is_param: true,
                        is_loop_var: false,
                        span: expr.span,
                    });
                }
            }
            ExprKind::Binary { left, right, .. } => {
                self.collect_parameters(left);
                self.collect_parameters(right);
            }
            ExprKind::Unary { operand, .. } => {
                self.collect_parameters(operand);
            }
            _ => {}
        }
    }

    fn analyze_block(&mut self, block: &mut Block) -> Result<()> {
        for stmt in &mut block.statements {
            self.analyze_statement(stmt)?;
        }
        Ok(())
    }

    fn analyze_statement(&mut self, stmt: &mut Stmt) -> Result<()> {
        match &mut stmt.kind {
            StmtKind::Declaration { name, ty, value, is_mutable } => {
                let inferred_ty = if let Some(expr) = value {
                    self.analyze_expression(expr)?;
                    expr.ty.clone()
                } else {
                    Type::Unknown
                };

                let final_ty = ty.clone().unwrap_or(inferred_ty);

                self.declare_symbol(SymbolInfo {
                    name: name.clone(),
                    ty: final_ty,
                    is_mutable: *is_mutable,
                    is_param: false,
                    is_loop_var: false,
                    span: stmt.span,
                })?;
            }

            StmtKind::Assignment { target, op: _, value } => {
                match target {
                    AssignTarget::Variable(name) => {
                        if let Some(info) = self.lookup_symbol(name) {
                            if !info.is_mutable && !info.is_param {
                                self.error(SemanticErrorKind::TypeMismatch,
                                    format!("Cannot assign to immutable variable '{}'", name), stmt.span);
                            }
                        } else {
                            self.error(SemanticErrorKind::UndefinedVariable,
                                format!("Undefined variable '{}'", name), stmt.span);
                        }
                    }
                    AssignTarget::ArrayAccess { array, indices } => {
                        if self.lookup_symbol(array).is_none() {
                            self.error(SemanticErrorKind::UndefinedArray,
                                format!("Undefined array '{}'", array), stmt.span);
                        }
                        for idx in indices {
                            self.analyze_expression(idx)?;
                        }
                    }
                }
                self.analyze_expression(value)?;
            }

            StmtKind::For { iterator, start, end, step, body, .. } => {
                self.analyze_expression(start)?;
                self.analyze_expression(end)?;
                if let Some(s) = step { self.analyze_expression(s)?; }

                self.push_scope();
                self.declare_symbol(SymbolInfo {
                    name: iterator.clone(),
                    ty: Type::Int,
                    is_mutable: false,
                    is_param: false,
                    is_loop_var: true,
                    span: stmt.span,
                })?;

                self.loop_vars.push(iterator.clone());
                self.analyze_block(body)?;
                self.loop_vars.pop();
                self.pop_scope();
            }

            StmtKind::If { condition, then_branch, else_branch } => {
                self.analyze_expression(condition)?;
                self.push_scope();
                self.analyze_block(then_branch)?;
                self.pop_scope();
                if let Some(else_b) = else_branch {
                    self.push_scope();
                    self.analyze_block(else_b)?;
                    self.pop_scope();
                }
            }

            StmtKind::While { condition, body } => {
                self.analyze_expression(condition)?;
                self.push_scope();
                self.analyze_block(body)?;
                self.pop_scope();
            }

            StmtKind::Return { value } => {
                if let Some(v) = value { self.analyze_expression(v)?; }
            }

            StmtKind::Expression { expr } => { self.analyze_expression(expr)?; }

            StmtKind::Block { block } => {
                self.push_scope();
                self.analyze_block(block)?;
                self.pop_scope();
            }

            StmtKind::Empty => {}
        }
        Ok(())
    }

    fn analyze_expression(&mut self, expr: &mut Expr) -> Result<()> {
        match &mut expr.kind {
            ExprKind::IntLiteral(_) => { expr.ty = Type::Int; }
            ExprKind::FloatLiteral(_) => { expr.ty = Type::Float; }
            ExprKind::BoolLiteral(_) => { expr.ty = Type::Bool; }
            ExprKind::StringLiteral(_) => { expr.ty = Type::Unknown; }
            
            ExprKind::Variable(name) => {
                if let Some(info) = self.lookup_symbol(name) {
                    expr.ty = info.ty.clone();
                } else {
                    self.declare_symbol(SymbolInfo {
                        name: name.clone(),
                        ty: Type::Int,
                        is_mutable: false,
                        is_param: true,
                        is_loop_var: false,
                        span: expr.span,
                    })?;
                    expr.ty = Type::Int;
                }
            }

            ExprKind::ArrayAccess { array, indices } => {
                self.analyze_expression(array)?;
                for idx in indices { self.analyze_expression(idx)?; }
                if let Type::Array { element, .. } = &array.ty {
                    expr.ty = (**element).clone();
                } else {
                    expr.ty = Type::Unknown;
                }
            }

            ExprKind::Binary { op, left, right } => {
                self.analyze_expression(left)?;
                self.analyze_expression(right)?;
                expr.ty = match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                        if left.ty == Type::Float || right.ty == Type::Float { Type::Float }
                        else if left.ty == Type::Double || right.ty == Type::Double { Type::Double }
                        else { Type::Int }
                    }
                    _ => Type::Bool,
                };
            }

            ExprKind::Unary { op, operand } => {
                self.analyze_expression(operand)?;
                expr.ty = match op {
                    UnaryOp::Neg => operand.ty.clone(),
                    UnaryOp::Not => Type::Bool,
                };
            }

            ExprKind::Call { args, .. } => {
                for arg in args { self.analyze_expression(arg)?; }
                expr.ty = Type::Unknown;
            }

            ExprKind::Ternary { condition, then_expr, else_expr } => {
                self.analyze_expression(condition)?;
                self.analyze_expression(then_expr)?;
                self.analyze_expression(else_expr)?;
                expr.ty = then_expr.ty.clone();
            }

            ExprKind::Min(a, b) | ExprKind::Max(a, b) => {
                self.analyze_expression(a)?;
                self.analyze_expression(b)?;
                expr.ty = a.ty.clone();
            }

            ExprKind::FloorDiv { dividend, divisor } | ExprKind::CeilDiv { dividend, divisor } => {
                self.analyze_expression(dividend)?;
                self.analyze_expression(divisor)?;
                expr.ty = Type::Int;
            }

            ExprKind::Cast { target_type, expr: inner } => {
                self.analyze_expression(inner)?;
                expr.ty = target_type.clone();
            }

            ExprKind::Grouped(inner) => {
                self.analyze_expression(inner)?;
                expr.ty = inner.ty.clone();
            }
        }
        Ok(())
    }

    fn push_scope(&mut self) {
        self.scopes.push(Scope { symbols: HashMap::new(), parent: Some(self.current_scope) });
        self.current_scope = self.scopes.len() - 1;
    }

    fn pop_scope(&mut self) {
        if let Some(parent) = self.scopes[self.current_scope].parent {
            self.current_scope = parent;
        }
    }

    fn declare_symbol(&mut self, info: SymbolInfo) -> Result<()> {
        let already_exists = self.scopes[self.current_scope].symbols.contains_key(&info.name);
        if already_exists {
            self.error(SemanticErrorKind::DuplicateDefinition,
                format!("Duplicate definition of '{}'", info.name), info.span);
        }
        self.scopes[self.current_scope].symbols.insert(info.name.clone(), info);
        Ok(())
    }

    fn lookup_symbol(&self, name: &str) -> Option<SymbolInfo> {
        let mut scope_idx = Some(self.current_scope);
        while let Some(idx) = scope_idx {
            if let Some(info) = self.scopes[idx].symbols.get(name) {
                return Some(info.clone());
            }
            scope_idx = self.scopes[idx].parent;
        }
        None
    }

    fn error(&mut self, kind: SemanticErrorKind, message: String, span: Span) {
        self.errors.push(SemanticError { message, span, kind });
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parse;

    #[test]
    fn test_basic_analysis() {
        let source = "func test() { let x = 1; let y = x + 2; }";
        let mut program = parse(source).unwrap();
        assert!(analyze(&mut program).is_ok());
    }

    #[test]
    fn test_for_loop_analysis() {
        let source = "func test(A[N]) { for i = 0 to N { A[i] = i; } }";
        let mut program = parse(source).unwrap();
        assert!(analyze(&mut program).is_ok());
    }
}
