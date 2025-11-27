//! AST to HIR lowering.
//!
//! This module converts the parsed AST into the High-level IR (HIR).
//! The lowering process:
//! - Desugars compound assignments (+=, -= etc.) into simple assignments
//! - Normalizes loop bounds (step is always explicit)
//! - Resolves variable names to unique IDs
//! - Propagates type information

use std::collections::HashMap;
use crate::frontend::ast::{
    self, Program, Function, Stmt, StmtKind, Expr, ExprKind,
    Block, Parameter, Type, BinaryOp, UnaryOp, AssignTarget, AssignOp,
};
use crate::ir::hir::*;
use crate::utils::location::Span;
use anyhow::{Result, bail};

/// Context for lowering, tracking variable bindings.
struct LoweringContext {
    /// ID generator for HIR nodes
    id_gen: HirIdGen,
    /// Map from variable names to their HIR IDs
    var_ids: HashMap<String, HirId>,
    /// Map from variable names to their types
    var_types: HashMap<String, HirType>,
    /// Parameters (symbolic constants)
    parameters: Vec<(String, HirId)>,
    /// Stack of loop variable names (for detecting loop variables)
    loop_vars: Vec<String>,
}

impl LoweringContext {
    fn new() -> Self {
        Self {
            id_gen: HirIdGen::new(),
            var_ids: HashMap::new(),
            var_types: HashMap::new(),
            parameters: Vec::new(),
            loop_vars: Vec::new(),
        }
    }

    fn fresh_id(&mut self) -> HirId {
        self.id_gen.next()
    }

    fn define_var(&mut self, name: &str, ty: HirType) -> HirId {
        let id = self.fresh_id();
        self.var_ids.insert(name.to_string(), id);
        self.var_types.insert(name.to_string(), ty);
        id
    }

    fn lookup_var(&self, name: &str) -> Option<HirId> {
        self.var_ids.get(name).copied()
    }

    fn lookup_type(&self, name: &str) -> HirType {
        self.var_types.get(name).cloned().unwrap_or(HirType::Unknown)
    }

    fn is_parameter(&self, name: &str) -> bool {
        self.parameters.iter().any(|(n, _)| n == name)
    }

    fn is_loop_var(&self, name: &str) -> bool {
        self.loop_vars.contains(&name.to_string())
    }

    fn get_param_id(&self, name: &str) -> Option<HirId> {
        self.parameters.iter().find(|(n, _)| n == name).map(|(_, id)| *id)
    }
}

/// Lower an AST program to HIR.
pub fn lower_program(program: &Program) -> Result<HirProgram> {
    let mut hir = HirProgram::new();
    hir.span = program.span;

    for func in &program.functions {
        let hir_func = lower_function(func)?;
        hir.functions.push(hir_func);
    }

    Ok(hir)
}

/// Lower a function to HIR.
fn lower_function(func: &Function) -> Result<HirFunction> {
    let mut ctx = LoweringContext::new();
    let func_id = ctx.fresh_id();

    // First, collect parameters from array dimensions
    // These are the symbolic constants like N, M, K
    collect_parameters(func, &mut ctx);

    // Lower function parameters
    let mut hir_params = Vec::new();
    for param in &func.params {
        let param_id = ctx.define_var(&param.name, lower_type(&param.ty));
        let dims = param.dimensions.iter()
            .map(|d| lower_expr(d, &mut ctx))
            .collect::<Result<Vec<_>>>()?;
        
        hir_params.push(HirFuncParam {
            id: param_id,
            name: param.name.clone(),
            ty: lower_type(&param.ty),
            dimensions: dims,
            span: param.span,
        });
    }

    // Lower body
    let hir_body = lower_block(&func.body, &mut ctx)?;

    Ok(HirFunction {
        id: func_id,
        name: func.name.clone(),
        params: hir_params,
        body: hir_body,
        span: func.span,
    })
}

/// Collect symbolic parameters from function signature.
fn collect_parameters(func: &Function, ctx: &mut LoweringContext) {
    for param in &func.params {
        for dim_expr in &param.dimensions {
            collect_params_from_expr(dim_expr, ctx);
        }
    }
}

/// Recursively collect parameter names from an expression.
fn collect_params_from_expr(expr: &Expr, ctx: &mut LoweringContext) {
    match &expr.kind {
        ExprKind::Variable(name) => {
            // If it's not already defined and not a known function, it's a parameter
            if ctx.lookup_var(name).is_none() && !ctx.is_parameter(name) {
                let id = ctx.fresh_id();
                ctx.parameters.push((name.clone(), id));
                ctx.var_ids.insert(name.clone(), id);
                ctx.var_types.insert(name.clone(), HirType::Int);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            collect_params_from_expr(left, ctx);
            collect_params_from_expr(right, ctx);
        }
        ExprKind::Unary { operand, .. } => {
            collect_params_from_expr(operand, ctx);
        }
        ExprKind::Grouped(inner) => {
            collect_params_from_expr(inner, ctx);
        }
        ExprKind::Min(a, b) | ExprKind::Max(a, b) => {
            collect_params_from_expr(a, ctx);
            collect_params_from_expr(b, ctx);
        }
        _ => {}
    }
}

/// Lower a block of statements.
fn lower_block(block: &Block, ctx: &mut LoweringContext) -> Result<HirBlock> {
    let mut stmts = Vec::new();
    for stmt in &block.statements {
        let hir_stmts = lower_stmt(stmt, ctx)?;
        stmts.extend(hir_stmts);
    }
    Ok(HirBlock {
        statements: stmts,
        span: block.span,
    })
}

/// Lower a statement (may produce multiple HIR statements for desugaring).
fn lower_stmt(stmt: &Stmt, ctx: &mut LoweringContext) -> Result<Vec<HirStmt>> {
    let span = stmt.span;
    
    match &stmt.kind {
        StmtKind::Declaration { name, ty, value, is_mutable: _ } => {
            let hir_ty = ty.as_ref().map(lower_type).unwrap_or(HirType::Unknown);
            let var_id = ctx.define_var(name, hir_ty.clone());
            let init = value.as_ref().map(|v| lower_expr(v, ctx)).transpose()?;
            
            Ok(vec![HirStmt {
                id: ctx.fresh_id(),
                kind: HirStmtKind::Let {
                    var_id,
                    name: name.clone(),
                    ty: hir_ty,
                    init,
                },
                span,
            }])
        }

        StmtKind::Assignment { target, op, value } => {
            lower_assignment(target, *op, value, ctx, span)
        }

        StmtKind::For { iterator, start, end, step, body, is_parallel } => {
            let var_id = ctx.define_var(iterator, HirType::Int);
            ctx.loop_vars.push(iterator.clone());

            let lower = lower_expr(start, ctx)?;
            let upper = lower_expr(end, ctx)?;
            let step_expr = match step {
                Some(s) => lower_expr(s, ctx)?,
                None => HirExpr::int(1),
            };
            let hir_body = lower_block(body, ctx)?;

            ctx.loop_vars.pop();

            Ok(vec![HirStmt {
                id: ctx.fresh_id(),
                kind: HirStmtKind::For {
                    var_id,
                    var_name: iterator.clone(),
                    lower,
                    upper,
                    step: step_expr,
                    body: hir_body,
                    is_parallel: *is_parallel,
                },
                span,
            }])
        }

        StmtKind::If { condition, then_branch, else_branch } => {
            let cond = lower_expr(condition, ctx)?;
            let then_body = lower_block(then_branch, ctx)?;
            let else_body = else_branch.as_ref()
                .map(|b| lower_block(b, ctx))
                .transpose()?;

            Ok(vec![HirStmt {
                id: ctx.fresh_id(),
                kind: HirStmtKind::If {
                    condition: cond,
                    then_body,
                    else_body,
                },
                span,
            }])
        }

        StmtKind::Return { value } => {
            let val = value.as_ref().map(|v| lower_expr(v, ctx)).transpose()?;
            Ok(vec![HirStmt {
                id: ctx.fresh_id(),
                kind: HirStmtKind::Return { value: val },
                span,
            }])
        }

        StmtKind::Expression { expr } => {
            let e = lower_expr(expr, ctx)?;
            Ok(vec![HirStmt {
                id: ctx.fresh_id(),
                kind: HirStmtKind::Expr { expr: e },
                span,
            }])
        }

        StmtKind::Block { block } => {
            // Flatten nested blocks
            let hir_block = lower_block(block, ctx)?;
            Ok(hir_block.statements)
        }

        StmtKind::While { .. } => {
            // While loops are not supported in SCoPs but we can lower them
            bail!("While loops are not supported in polyhedral representation")
        }

        StmtKind::Empty => Ok(vec![]),
    }
}

/// Lower an assignment, desugaring compound assignments.
fn lower_assignment(
    target: &AssignTarget,
    op: AssignOp,
    value: &Expr,
    ctx: &mut LoweringContext,
    span: Span,
) -> Result<Vec<HirStmt>> {
    let hir_target = lower_assign_target(target, ctx)?;
    let hir_value = lower_expr(value, ctx)?;

    // Desugar compound assignments: a += b becomes a = a + b
    let final_value = match op {
        AssignOp::Assign => hir_value,
        AssignOp::AddAssign | AssignOp::SubAssign | AssignOp::MulAssign | AssignOp::DivAssign => {
            let target_expr = lvalue_to_expr(&hir_target);
            let bin_op = match op {
                AssignOp::AddAssign => HirBinaryOp::Add,
                AssignOp::SubAssign => HirBinaryOp::Sub,
                AssignOp::MulAssign => HirBinaryOp::Mul,
                AssignOp::DivAssign => HirBinaryOp::Div,
                _ => unreachable!(),
            };
            HirExpr {
                kind: HirExprKind::Binary {
                    op: bin_op,
                    left: Box::new(target_expr),
                    right: Box::new(hir_value),
                },
                ty: HirType::Unknown,
                span: Span::dummy(),
            }
        }
    };

    Ok(vec![HirStmt {
        id: ctx.fresh_id(),
        kind: HirStmtKind::Assign {
            target: hir_target,
            value: final_value,
        },
        span,
    }])
}

/// Convert an l-value back to an expression (for desugaring compound assignments).
fn lvalue_to_expr(lvalue: &HirLValue) -> HirExpr {
    match &lvalue.kind {
        HirLValueKind::Var { id, name } => HirExpr {
            kind: HirExprKind::Var { id: *id, name: name.clone() },
            ty: HirType::Unknown,
            span: lvalue.span,
        },
        HirLValueKind::ArrayElem { array_id, array_name, indices } => HirExpr {
            kind: HirExprKind::ArrayAccess {
                array_id: *array_id,
                array_name: array_name.clone(),
                indices: indices.clone(),
            },
            ty: HirType::Unknown,
            span: lvalue.span,
        },
    }
}

/// Lower an assignment target.
fn lower_assign_target(target: &AssignTarget, ctx: &mut LoweringContext) -> Result<HirLValue> {
    match target {
        AssignTarget::Variable(name) => {
            let id = ctx.lookup_var(name).unwrap_or_else(|| ctx.define_var(name, HirType::Unknown));
            Ok(HirLValue {
                kind: HirLValueKind::Var { id, name: name.clone() },
                span: Span::dummy(),
            })
        }
        AssignTarget::ArrayAccess { array, indices } => {
            let array_id = ctx.lookup_var(array)
                .unwrap_or_else(|| ctx.define_var(array, HirType::Unknown));
            let hir_indices = indices.iter()
                .map(|i| lower_expr(i, ctx))
                .collect::<Result<Vec<_>>>()?;
            Ok(HirLValue {
                kind: HirLValueKind::ArrayElem {
                    array_id,
                    array_name: array.clone(),
                    indices: hir_indices,
                },
                span: Span::dummy(),
            })
        }
    }
}

/// Lower an expression.
fn lower_expr(expr: &Expr, ctx: &mut LoweringContext) -> Result<HirExpr> {
    let span = expr.span;
    
    let kind = match &expr.kind {
        ExprKind::IntLiteral(v) => HirExprKind::IntLit(*v),
        ExprKind::FloatLiteral(v) => HirExprKind::FloatLit(*v),
        ExprKind::BoolLiteral(v) => HirExprKind::BoolLit(*v),

        ExprKind::Variable(name) => {
            if ctx.is_parameter(name) && !ctx.is_loop_var(name) {
                let id = ctx.get_param_id(name).unwrap();
                HirExprKind::Param { id, name: name.clone() }
            } else {
                let id = ctx.lookup_var(name)
                    .unwrap_or_else(|| ctx.define_var(name, HirType::Unknown));
                HirExprKind::Var { id, name: name.clone() }
            }
        }

        ExprKind::ArrayAccess { array, indices } => {
            // The array should be a variable
            let (array_id, array_name) = match &array.kind {
                ExprKind::Variable(name) => {
                    let id = ctx.lookup_var(name)
                        .unwrap_or_else(|| ctx.define_var(name, HirType::Unknown));
                    (id, name.clone())
                }
                _ => bail!("Complex array expressions not supported"),
            };
            let hir_indices = indices.iter()
                .map(|i| lower_expr(i, ctx))
                .collect::<Result<Vec<_>>>()?;
            HirExprKind::ArrayAccess {
                array_id,
                array_name,
                indices: hir_indices,
            }
        }

        ExprKind::Binary { op, left, right } => {
            let hir_op = lower_binary_op(*op);
            let l = Box::new(lower_expr(left, ctx)?);
            let r = Box::new(lower_expr(right, ctx)?);
            HirExprKind::Binary { op: hir_op, left: l, right: r }
        }

        ExprKind::Unary { op, operand } => {
            let hir_op = lower_unary_op(*op);
            let o = Box::new(lower_expr(operand, ctx)?);
            HirExprKind::Unary { op: hir_op, operand: o }
        }

        ExprKind::Call { function, args } => {
            let hir_args = args.iter()
                .map(|a| lower_expr(a, ctx))
                .collect::<Result<Vec<_>>>()?;
            HirExprKind::Call { func: function.clone(), args: hir_args }
        }

        ExprKind::Min(a, b) => {
            let ha = Box::new(lower_expr(a, ctx)?);
            let hb = Box::new(lower_expr(b, ctx)?);
            HirExprKind::Min(ha, hb)
        }

        ExprKind::Max(a, b) => {
            let ha = Box::new(lower_expr(a, ctx)?);
            let hb = Box::new(lower_expr(b, ctx)?);
            HirExprKind::Max(ha, hb)
        }

        ExprKind::FloorDiv { dividend, divisor } => {
            let d = Box::new(lower_expr(dividend, ctx)?);
            let v = Box::new(lower_expr(divisor, ctx)?);
            HirExprKind::FloorDiv { dividend: d, divisor: v }
        }

        ExprKind::CeilDiv { dividend, divisor } => {
            let d = Box::new(lower_expr(dividend, ctx)?);
            let v = Box::new(lower_expr(divisor, ctx)?);
            HirExprKind::CeilDiv { dividend: d, divisor: v }
        }

        ExprKind::Grouped(inner) => {
            return lower_expr(inner, ctx);
        }

        ExprKind::Ternary { condition, then_expr, else_expr } => {
            // Lower ternary to min/max if possible, otherwise error
            bail!("Ternary expressions not yet supported in HIR")
        }

        ExprKind::Cast { target_type, expr } => {
            // For now, just lower the inner expression
            return lower_expr(expr, ctx);
        }

        ExprKind::StringLiteral(_) => bail!("String literals not supported"),
    };

    Ok(HirExpr {
        kind,
        ty: infer_type(&expr.kind),
        span,
    })
}

/// Lower AST type to HIR type.
fn lower_type(ty: &Type) -> HirType {
    match ty {
        Type::Int => HirType::Int,
        Type::Float => HirType::Float,
        Type::Double => HirType::Double,
        Type::Bool => HirType::Bool,
        Type::Array { element, dimensions } => HirType::Array {
            element: Box::new(lower_type(element)),
            ndims: dimensions.len(),
        },
        Type::Void | Type::Unknown => HirType::Unknown,
    }
}

/// Lower binary operator.
fn lower_binary_op(op: BinaryOp) -> HirBinaryOp {
    match op {
        BinaryOp::Add => HirBinaryOp::Add,
        BinaryOp::Sub => HirBinaryOp::Sub,
        BinaryOp::Mul => HirBinaryOp::Mul,
        BinaryOp::Div => HirBinaryOp::Div,
        BinaryOp::Mod => HirBinaryOp::Mod,
        BinaryOp::Eq => HirBinaryOp::Eq,
        BinaryOp::Ne => HirBinaryOp::Ne,
        BinaryOp::Lt => HirBinaryOp::Lt,
        BinaryOp::Le => HirBinaryOp::Le,
        BinaryOp::Gt => HirBinaryOp::Gt,
        BinaryOp::Ge => HirBinaryOp::Ge,
        BinaryOp::And => HirBinaryOp::And,
        BinaryOp::Or => HirBinaryOp::Or,
    }
}

/// Lower unary operator.
fn lower_unary_op(op: UnaryOp) -> HirUnaryOp {
    match op {
        UnaryOp::Neg => HirUnaryOp::Neg,
        UnaryOp::Not => HirUnaryOp::Not,
    }
}

/// Basic type inference from expression kind.
fn infer_type(kind: &ExprKind) -> HirType {
    match kind {
        ExprKind::IntLiteral(_) => HirType::Int,
        ExprKind::FloatLiteral(_) => HirType::Float,
        ExprKind::BoolLiteral(_) => HirType::Bool,
        _ => HirType::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parse;

    #[test]
    fn test_lower_simple_function() {
        let source = "func test(A[N]) { for i = 0 to N { A[i] = i; } }";
        let program = parse(source).unwrap();
        let hir = lower_program(&program).unwrap();
        
        assert_eq!(hir.functions.len(), 1);
        assert_eq!(hir.functions[0].name, "test");
        assert_eq!(hir.functions[0].params.len(), 1);
    }

    #[test]
    fn test_lower_compound_assignment() {
        let source = "func test(A[N]) { for i = 0 to N { A[i] += 1; } }";
        let program = parse(source).unwrap();
        let hir = lower_program(&program).unwrap();
        
        // Check that += was desugared
        let func = &hir.functions[0];
        let for_stmt = &func.body.statements[0];
        if let HirStmtKind::For { body, .. } = &for_stmt.kind {
            let assign_stmt = &body.statements[0];
            if let HirStmtKind::Assign { value, .. } = &assign_stmt.kind {
                // Value should be a binary expression (A[i] + 1)
                assert!(matches!(value.kind, HirExprKind::Binary { .. }));
            } else {
                panic!("Expected assignment statement");
            }
        } else {
            panic!("Expected for statement");
        }
    }

    #[test]
    fn test_lower_nested_loops() {
        let source = r#"
            func matmul(A[N][K], B[K][M], C[N][M]) {
                for i = 0 to N {
                    for j = 0 to M {
                        for k = 0 to K {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        "#;
        let program = parse(source).unwrap();
        let hir = lower_program(&program).unwrap();
        
        assert_eq!(hir.functions.len(), 1);
        let func = &hir.functions[0];
        assert_eq!(func.params.len(), 3);
    }
}