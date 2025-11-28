//! HIR to PIR lowering.
//!
//! This module converts the High-level IR (HIR) into the Polyhedral IR (PIR).
//! The lowering process:
//! - Builds iteration domains from loop nests
//! - Extracts access relations from array accesses
//! - Creates statement bodies for code generation
//! - Builds initial identity schedules

use std::collections::HashMap;
use crate::ir::hir::*;
use crate::ir::pir::*;
use crate::polyhedral::{
    Space, IntegerSet, AffineMap, AffineExpr,
    constraint::{Constraint, ConstraintSystem},
};
use crate::utils::location::Span;
use anyhow::{Result, bail, Context};

/// Context for HIR to PIR lowering.
struct PirLoweringContext {
    /// Statement ID generator
    stmt_id_gen: StmtIdGen,
    /// Current loop nest (iterator name -> depth, with 0 being outermost)
    loop_nest: Vec<LoopInfo>,
    /// Parameters (symbolic constants)
    parameters: Vec<String>,
    /// Arrays in the program
    arrays: HashMap<String, ArrayInfo>,
    /// Generated statements
    statements: Vec<PolyStmt>,
    /// Current statement counter (for naming)
    stmt_counter: usize,
}

/// Information about a loop in the current nest.
#[derive(Debug, Clone)]
struct LoopInfo {
    /// Iterator variable name
    var_name: String,
    /// Lower bound expression
    lower: HirExpr,
    /// Upper bound expression  
    upper: HirExpr,
    /// Step (must be a constant for polyhedral)
    step: i64,
    /// Is this loop marked parallel?
    is_parallel: bool,
}

impl PirLoweringContext {
    fn new() -> Self {
        Self {
            stmt_id_gen: StmtIdGen::new(),
            loop_nest: Vec::new(),
            parameters: Vec::new(),
            arrays: HashMap::new(),
            statements: Vec::new(),
            stmt_counter: 0,
        }
    }

    fn fresh_stmt_id(&mut self) -> StmtId {
        self.stmt_id_gen.next()
    }

    /// Get current loop depth.
    fn depth(&self) -> usize {
        self.loop_nest.len()
    }

    /// Get iterator names in order.
    fn iterator_names(&self) -> Vec<String> {
        self.loop_nest.iter().map(|l| l.var_name.clone()).collect()
    }

    /// Get the index of an iterator variable, if it exists.
    fn iterator_index(&self, name: &str) -> Option<usize> {
        self.loop_nest.iter().position(|l| l.var_name == name)
    }

    /// Get the index of a parameter, if it exists.
    fn param_index(&self, name: &str) -> Option<usize> {
        self.parameters.iter().position(|p| p == name)
    }
}

/// Lower an HIR program to PIR.
pub fn lower_to_pir(hir: &HirProgram) -> Result<Vec<PolyProgram>> {
    let mut programs = Vec::new();
    
    for func in &hir.functions {
        let program = lower_function(func)?;
        programs.push(program);
    }
    
    Ok(programs)
}

/// Lower a single function to a PolyProgram.
fn lower_function(func: &HirFunction) -> Result<PolyProgram> {
    let mut ctx = PirLoweringContext::new();
    
    // Collect parameters from function signature
    for param in &func.params {
        // Collect parameters from array dimensions
        for dim in &param.dimensions {
            collect_params_from_hir_expr(dim, &mut ctx.parameters);
        }
        
        // Register array parameters
        if param.is_array() {
            let array_info = ArrayInfo {
                name: param.name.clone(),
                ndims: param.dimensions.len(),
                element_type: hir_type_to_element_type(&param.ty),
                sizes: param.dimensions.iter()
                    .map(|d| hir_expr_to_string(d))
                    .collect(),
            };
            ctx.arrays.insert(param.name.clone(), array_info);
        }
    }

    // Lower the function body
    lower_block(&func.body, &mut ctx)?;

    // Build context constraints (parameters > 0)
    let n_param = ctx.parameters.len();
    let context = if n_param > 0 {
        let space = Space::set_with_params(0, n_param)
            .with_param_names(ctx.parameters.clone());
        let mut constraints = ConstraintSystem::new(0, n_param);
        for i in 0..n_param {
            // param_i >= 1 (positive)
            let mut expr = AffineExpr::param(i, 0, n_param);
            expr.constant = -1;
            constraints.add(Constraint::ge_zero(expr));
        }
        IntegerSet { space, constraints }
    } else {
        IntegerSet::universe(0)
    };

    Ok(PolyProgram {
        name: func.name.clone(),
        parameters: ctx.parameters,
        statements: ctx.statements,
        arrays: ctx.arrays.into_values().collect(),
        context,
    })
}

/// Lower a block of HIR statements.
fn lower_block(block: &HirBlock, ctx: &mut PirLoweringContext) -> Result<()> {
    for stmt in &block.statements {
        lower_stmt(stmt, ctx)?;
    }
    Ok(())
}

/// Lower a single HIR statement.
fn lower_stmt(stmt: &HirStmt, ctx: &mut PirLoweringContext) -> Result<()> {
    match &stmt.kind {
        HirStmtKind::For { var_name, lower, upper, step, body, is_parallel, .. } => {
            // Extract step value (must be constant)
            let step_val = eval_constant_expr(step)
                .context("Loop step must be a constant")?;
            
            // Push loop onto the nest
            ctx.loop_nest.push(LoopInfo {
                var_name: var_name.clone(),
                lower: lower.clone(),
                upper: upper.clone(),
                step: step_val,
                is_parallel: *is_parallel,
            });
            
            // Lower the body
            lower_block(body, ctx)?;
            
            // Pop loop from nest
            ctx.loop_nest.pop();
        }

        HirStmtKind::Assign { target, value } => {
            // Only create polyhedral statements inside loops
            if ctx.depth() > 0 {
                create_poly_stmt(target, value, stmt.span, ctx)?;
            }
        }

        HirStmtKind::If { condition, then_body, else_body } => {
            // For now, we handle simple conditionals by incorporating them
            // into the iteration domain. This is a simplification.
            // A full implementation would split the domain.
            lower_block(then_body, ctx)?;
            if let Some(else_b) = else_body {
                lower_block(else_b, ctx)?;
            }
        }

        HirStmtKind::Let { .. } => {
            // Local declarations don't create polyhedral statements
        }

        HirStmtKind::Return { .. } => {
            // Returns don't create polyhedral statements
        }

        HirStmtKind::Expr { .. } => {
            // Expression statements (like function calls) are not supported
        }
    }
    
    Ok(())
}

/// Create a polyhedral statement from an assignment.
fn create_poly_stmt(
    target: &HirLValue,
    value: &HirExpr,
    span: Span,
    ctx: &mut PirLoweringContext,
) -> Result<()> {
    let stmt_id = ctx.fresh_stmt_id();
    let stmt_name = format!("S{}", ctx.stmt_counter);
    ctx.stmt_counter += 1;
    
    let n_dim = ctx.depth();
    let n_param = ctx.parameters.len();
    let iterators = ctx.iterator_names();
    
    // Build iteration domain
    let domain = build_iteration_domain(ctx)?;
    
    // Build schedule (identity schedule: execute in original order)
    let schedule = build_identity_schedule(n_dim, n_param, ctx);
    
    // Collect accesses
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    
    // Extract write access from target
    if let Some(access) = extract_access(target, AccessKind::Write, ctx)? {
        writes.push(access);
    }
    
    // Extract read accesses from value
    extract_reads_from_expr(value, &mut reads, ctx)?;
    
    // Build statement body
    let body = build_stmt_body(target, value, ctx)?;
    
    ctx.statements.push(PolyStmt {
        id: stmt_id,
        name: stmt_name,
        domain,
        schedule,
        reads,
        writes,
        body,
        span,
    });
    
    Ok(())
}

/// Build the iteration domain from the current loop nest.
fn build_iteration_domain(ctx: &PirLoweringContext) -> Result<IntegerSet> {
    let n_dim = ctx.depth();
    let n_param = ctx.parameters.len();
    
    let mut space = Space::set_with_params(n_dim, n_param);
    space = space.with_dim_names(ctx.iterator_names());
    space = space.with_param_names(ctx.parameters.clone());
    
    let mut constraints = ConstraintSystem::new(n_dim, n_param);
    
    for (depth, loop_info) in ctx.loop_nest.iter().enumerate() {
        // Lower bound: iterator >= lower
        // Translates to: iterator - lower >= 0
        let lower_expr = hir_expr_to_affine(&loop_info.lower, depth, ctx)?;
        let mut lb_constraint = AffineExpr::var(depth, n_dim, n_param);
        lb_constraint = lb_constraint - lower_expr;
        constraints.add(Constraint::ge_zero(lb_constraint));
        
        // Upper bound: iterator < upper (for i = 0 to N means i < N)
        // Translates to: upper - 1 - iterator >= 0
        let upper_expr = hir_expr_to_affine(&loop_info.upper, depth, ctx)?;
        let mut ub_constraint = upper_expr;
        ub_constraint.constant -= 1;  // < becomes <=
        let iter_var = AffineExpr::var(depth, n_dim, n_param);
        ub_constraint = ub_constraint - iter_var;
        constraints.add(Constraint::ge_zero(ub_constraint));
    }
    
    Ok(IntegerSet { space, constraints })
}

/// Convert an HIR expression to an affine expression.
fn hir_expr_to_affine(
    expr: &HirExpr,
    current_depth: usize,
    ctx: &PirLoweringContext,
) -> Result<AffineExpr> {
    let n_dim = ctx.depth();
    let n_param = ctx.parameters.len();
    
    match &expr.kind {
        HirExprKind::IntLit(v) => Ok(AffineExpr::constant(*v, n_dim, n_param)),
        
        HirExprKind::Var { name, .. } => {
            if let Some(idx) = ctx.iterator_index(name) {
                Ok(AffineExpr::var(idx, n_dim, n_param))
            } else if let Some(idx) = ctx.param_index(name) {
                Ok(AffineExpr::param(idx, n_dim, n_param))
            } else {
                bail!("Unknown variable in affine expression: {}", name)
            }
        }
        
        HirExprKind::Param { name, .. } => {
            if let Some(idx) = ctx.param_index(name) {
                Ok(AffineExpr::param(idx, n_dim, n_param))
            } else {
                bail!("Unknown parameter: {}", name)
            }
        }
        
        HirExprKind::Binary { op, left, right } => {
            let l = hir_expr_to_affine(left, current_depth, ctx)?;
            let r = hir_expr_to_affine(right, current_depth, ctx)?;
            
            match op {
                HirBinaryOp::Add => Ok(l + r),
                HirBinaryOp::Sub => Ok(l - r),
                HirBinaryOp::Mul => {
                    // One side must be constant for affine
                    if let Some(c) = l.as_constant() {
                        Ok(r.scale(c))
                    } else if let Some(c) = r.as_constant() {
                        Ok(l.scale(c))
                    } else {
                        bail!("Non-affine multiplication")
                    }
                }
                _ => bail!("Non-affine operator: {:?}", op),
            }
        }
        
        HirExprKind::Unary { op, operand } => {
            let o = hir_expr_to_affine(operand, current_depth, ctx)?;
            match op {
                HirUnaryOp::Neg => Ok(-o),
                _ => bail!("Non-affine unary operator"),
            }
        }
        
        HirExprKind::Min(a, b) => {
            // For bounds, we can handle min/max specially
            // but for now, error out
            bail!("Min/Max in loop bounds not yet fully supported")
        }
        
        HirExprKind::Max(a, b) => {
            bail!("Min/Max in loop bounds not yet fully supported")
        }
        
        _ => bail!("Expression is not affine"),
    }
}

/// Build identity schedule.
fn build_identity_schedule(
    n_dim: usize,
    n_param: usize,
    ctx: &PirLoweringContext,
) -> AffineMap {
    // Identity schedule: [i, j, k] -> [0, i, 1, j, 2, k]
    // This interleaves statement ordering with loop dimensions
    // For simplicity, we just use [i, j, k] -> [i, j, k] for now
    let mut map = AffineMap::identity(n_dim);
    map.space = map.space.with_param_names(ctx.parameters.clone());
    map
}

/// Extract an access relation from an l-value.
fn extract_access(
    lvalue: &HirLValue,
    kind: AccessKind,
    ctx: &PirLoweringContext,
) -> Result<Option<AccessRelation>> {
    match &lvalue.kind {
        HirLValueKind::Var { name, .. } => {
            // Scalar access - not an array access
            Ok(None)
        }
        HirLValueKind::ArrayElem { array_name, indices, .. } => {
            let relation = build_access_map(array_name, indices, ctx)?;
            Ok(Some(AccessRelation {
                array: array_name.clone(),
                relation,
                kind,
            }))
        }
    }
}

/// Build an access map from array indices.
fn build_access_map(
    array_name: &str,
    indices: &[HirExpr],
    ctx: &PirLoweringContext,
) -> Result<AffineMap> {
    let n_in = ctx.depth();
    let n_out = indices.len();
    let n_param = ctx.parameters.len();
    
    let space = Space::map_with_params(n_in, n_out, n_param)
        .with_param_names(ctx.parameters.clone());
    
    let mut outputs = Vec::new();
    for idx in indices {
        let expr = hir_expr_to_affine(idx, 0, ctx)?;
        outputs.push(expr);
    }
    
    Ok(AffineMap { space, outputs })
}

/// Extract read accesses from an expression.
fn extract_reads_from_expr(
    expr: &HirExpr,
    reads: &mut Vec<AccessRelation>,
    ctx: &PirLoweringContext,
) -> Result<()> {
    match &expr.kind {
        HirExprKind::ArrayAccess { array_name, indices, .. } => {
            let relation = build_access_map(array_name, indices, ctx)?;
            reads.push(AccessRelation {
                array: array_name.clone(),
                relation,
                kind: AccessKind::Read,
            });
            // Also check indices for reads
            for idx in indices {
                extract_reads_from_expr(idx, reads, ctx)?;
            }
        }
        HirExprKind::Binary { left, right, .. } => {
            extract_reads_from_expr(left, reads, ctx)?;
            extract_reads_from_expr(right, reads, ctx)?;
        }
        HirExprKind::Unary { operand, .. } => {
            extract_reads_from_expr(operand, reads, ctx)?;
        }
        HirExprKind::Call { args, .. } => {
            for arg in args {
                extract_reads_from_expr(arg, reads, ctx)?;
            }
        }
        HirExprKind::Min(a, b) | HirExprKind::Max(a, b) => {
            extract_reads_from_expr(a, reads, ctx)?;
            extract_reads_from_expr(b, reads, ctx)?;
        }
        _ => {}
    }
    Ok(())
}

/// Build statement body for code generation.
fn build_stmt_body(
    target: &HirLValue,
    value: &HirExpr,
    ctx: &PirLoweringContext,
) -> Result<StmtBody> {
    let target_access = build_access_expr(target, ctx)?;
    let compute_expr = build_compute_expr(value, ctx)?;
    
    Ok(StmtBody::Assignment {
        target: target_access,
        expr: compute_expr,
    })
}

/// Build access expression for code generation.
fn build_access_expr(lvalue: &HirLValue, ctx: &PirLoweringContext) -> Result<AccessExpr> {
    match &lvalue.kind {
        HirLValueKind::Var { name, .. } => {
            Ok(AccessExpr {
                array: name.clone(),
                indices: vec![],
            })
        }
        HirLValueKind::ArrayElem { array_name, indices, .. } => {
            let idx_strs = indices.iter()
                .map(|i| AffineExprStr(hir_expr_to_string(i)))
                .collect();
            Ok(AccessExpr {
                array: array_name.clone(),
                indices: idx_strs,
            })
        }
    }
}

/// Build compute expression for code generation.
fn build_compute_expr(expr: &HirExpr, ctx: &PirLoweringContext) -> Result<ComputeExpr> {
    match &expr.kind {
        HirExprKind::IntLit(v) => Ok(ComputeExpr::Int(*v)),
        HirExprKind::FloatLit(v) => Ok(ComputeExpr::Float(*v)),
        
        HirExprKind::Var { name, .. } | HirExprKind::Param { name, .. } => {
            Ok(ComputeExpr::Var(name.clone()))
        }
        
        HirExprKind::ArrayAccess { array_name, indices, .. } => {
            let idx_strs = indices.iter()
                .map(|i| AffineExprStr(hir_expr_to_string(i)))
                .collect();
            Ok(ComputeExpr::Access(AccessExpr {
                array: array_name.clone(),
                indices: idx_strs,
            }))
        }
        
        HirExprKind::Binary { op, left, right } => {
            let l = Box::new(build_compute_expr(left, ctx)?);
            let r = Box::new(build_compute_expr(right, ctx)?);
            let bin_op = match op {
                HirBinaryOp::Add => BinaryComputeOp::Add,
                HirBinaryOp::Sub => BinaryComputeOp::Sub,
                HirBinaryOp::Mul => BinaryComputeOp::Mul,
                HirBinaryOp::Div => BinaryComputeOp::Div,
                HirBinaryOp::Mod => BinaryComputeOp::Mod,
                _ => bail!("Unsupported binary operator in compute expression"),
            };
            Ok(ComputeExpr::Binary { op: bin_op, left: l, right: r })
        }
        
        HirExprKind::Unary { op, operand } => {
            let o = Box::new(build_compute_expr(operand, ctx)?);
            let un_op = match op {
                HirUnaryOp::Neg => UnaryComputeOp::Neg,
                _ => bail!("Unsupported unary operator"),
            };
            Ok(ComputeExpr::Unary { op: un_op, operand: o })
        }
        
        HirExprKind::Call { func, args } => {
            let call_args = args.iter()
                .map(|a| build_compute_expr(a, ctx))
                .collect::<Result<Vec<_>>>()?;
            Ok(ComputeExpr::Call {
                func: func.clone(),
                args: call_args,
            })
        }
        
        _ => bail!("Unsupported expression kind in compute expression"),
    }
}

/// Convert HIR expression to a string (for code generation).
fn hir_expr_to_string(expr: &HirExpr) -> String {
    match &expr.kind {
        HirExprKind::IntLit(v) => v.to_string(),
        HirExprKind::FloatLit(v) => v.to_string(),
        HirExprKind::BoolLit(v) => v.to_string(),
        HirExprKind::Var { name, .. } => name.clone(),
        HirExprKind::Param { name, .. } => name.clone(),
        HirExprKind::ArrayAccess { array_name, indices, .. } => {
            let idx_str = indices.iter()
                .map(hir_expr_to_string)
                .collect::<Vec<_>>()
                .join("][");
            format!("{}[{}]", array_name, idx_str)
        }
        HirExprKind::Binary { op, left, right } => {
            let op_str = match op {
                HirBinaryOp::Add => "+",
                HirBinaryOp::Sub => "-",
                HirBinaryOp::Mul => "*",
                HirBinaryOp::Div => "/",
                HirBinaryOp::Mod => "%",
                HirBinaryOp::Eq => "==",
                HirBinaryOp::Ne => "!=",
                HirBinaryOp::Lt => "<",
                HirBinaryOp::Le => "<=",
                HirBinaryOp::Gt => ">",
                HirBinaryOp::Ge => ">=",
                HirBinaryOp::And => "&&",
                HirBinaryOp::Or => "||",
            };
            format!("({} {} {})", hir_expr_to_string(left), op_str, hir_expr_to_string(right))
        }
        HirExprKind::Unary { op, operand } => {
            let op_str = match op {
                HirUnaryOp::Neg => "-",
                HirUnaryOp::Not => "!",
            };
            format!("{}{}", op_str, hir_expr_to_string(operand))
        }
        HirExprKind::Call { func, args } => {
            let args_str = args.iter()
                .map(hir_expr_to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}({})", func, args_str)
        }
        HirExprKind::Min(a, b) => format!("min({}, {})", hir_expr_to_string(a), hir_expr_to_string(b)),
        HirExprKind::Max(a, b) => format!("max({}, {})", hir_expr_to_string(a), hir_expr_to_string(b)),
        HirExprKind::FloorDiv { dividend, divisor } => {
            format!("floord({}, {})", hir_expr_to_string(dividend), hir_expr_to_string(divisor))
        }
        HirExprKind::CeilDiv { dividend, divisor } => {
            format!("ceild({}, {})", hir_expr_to_string(dividend), hir_expr_to_string(divisor))
        }
    }
}

/// Evaluate a constant expression.
fn eval_constant_expr(expr: &HirExpr) -> Result<i64> {
    match &expr.kind {
        HirExprKind::IntLit(v) => Ok(*v),
        HirExprKind::Binary { op, left, right } => {
            let l = eval_constant_expr(left)?;
            let r = eval_constant_expr(right)?;
            match op {
                HirBinaryOp::Add => Ok(l + r),
                HirBinaryOp::Sub => Ok(l - r),
                HirBinaryOp::Mul => Ok(l * r),
                HirBinaryOp::Div => Ok(l / r),
                HirBinaryOp::Mod => Ok(l % r),
                _ => bail!("Non-arithmetic operator in constant expression"),
            }
        }
        HirExprKind::Unary { op, operand } => {
            let o = eval_constant_expr(operand)?;
            match op {
                HirUnaryOp::Neg => Ok(-o),
                _ => bail!("Non-arithmetic unary operator in constant expression"),
            }
        }
        _ => bail!("Expression is not a constant"),
    }
}

/// Collect parameter names from an HIR expression.
fn collect_params_from_hir_expr(expr: &HirExpr, params: &mut Vec<String>) {
    match &expr.kind {
        HirExprKind::Var { name, .. } | HirExprKind::Param { name, .. } => {
            if !params.contains(name) {
                params.push(name.clone());
            }
        }
        HirExprKind::Binary { left, right, .. } => {
            collect_params_from_hir_expr(left, params);
            collect_params_from_hir_expr(right, params);
        }
        HirExprKind::Unary { operand, .. } => {
            collect_params_from_hir_expr(operand, params);
        }
        HirExprKind::Min(a, b) | HirExprKind::Max(a, b) => {
            collect_params_from_hir_expr(a, params);
            collect_params_from_hir_expr(b, params);
        }
        _ => {}
    }
}

/// Convert HIR type to PIR element type.
fn hir_type_to_element_type(ty: &HirType) -> ElementType {
    match ty {
        HirType::Int => ElementType::Int,
        HirType::Float => ElementType::Float,
        HirType::Double => ElementType::Double,
        HirType::Array { element, .. } => hir_type_to_element_type(element),
        _ => ElementType::Double, // Default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::parse;
    use crate::ir::lower_ast::lower_program;

    #[test]
    fn test_lower_simple_loop() {
        let source = "func test(A[N]) { for i = 0 to N { A[i] = i; } }";
        let ast = parse(source).unwrap();
        let hir = lower_program(&ast).unwrap();
        let pir = lower_to_pir(&hir).unwrap();
        
        assert_eq!(pir.len(), 1);
        let prog = &pir[0];
        assert_eq!(prog.name, "test");
        assert_eq!(prog.statements.len(), 1);
        assert_eq!(prog.parameters, vec!["N"]);
        
        let stmt = &prog.statements[0];
        assert_eq!(stmt.domain.dim(), 1);
    }

    #[test]
    fn test_lower_nested_loops() {
        let source = r#"
            func matmul(A[N][K], B[K][M], C[N][M]) {
                for i = 0 to N {
                    for j = 0 to M {
                        for k = 0 to K {
                            C[i][j] = C[i][j] + A[i][k] * B[k][j];
                        }
                    }
                }
            }
        "#;
        let ast = parse(source).unwrap();
        let hir = lower_program(&ast).unwrap();
        let pir = lower_to_pir(&hir).unwrap();
        
        assert_eq!(pir.len(), 1);
        let prog = &pir[0];
        assert_eq!(prog.statements.len(), 1);
        
        let stmt = &prog.statements[0];
        assert_eq!(stmt.domain.dim(), 3); // i, j, k
        
        // Check we have the right parameters
        assert!(prog.parameters.contains(&"N".to_string()));
        assert!(prog.parameters.contains(&"M".to_string()));
        assert!(prog.parameters.contains(&"K".to_string()));
    }

    #[test]
    fn test_access_relations() {
        let source = "func test(A[N], B[N]) { for i = 0 to N { B[i] = A[i] + 1; } }";
        let ast = parse(source).unwrap();
        let hir = lower_program(&ast).unwrap();
        let pir = lower_to_pir(&hir).unwrap();
        
        let stmt = &pir[0].statements[0];
        
        // Should have 1 write (B[i]) and 1 read (A[i])
        assert_eq!(stmt.writes.len(), 1);
        assert_eq!(stmt.writes[0].array, "B");
        
        assert_eq!(stmt.reads.len(), 1);
        assert_eq!(stmt.reads[0].array, "A");
    }
}
