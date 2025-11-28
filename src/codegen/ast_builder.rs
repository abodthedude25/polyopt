//! AST builder for code generation.
//!
//! This module provides an intermediate AST representation that bridges
//! the polyhedral representation and the final code output.

use crate::ir::pir::StmtId;

/// AST node for generated code.
#[derive(Debug, Clone)]
pub enum AstNode {
    /// A for loop
    Loop {
        /// Loop iterator variable
        iterator: String,
        /// Lower bound expression
        lower: AstExpr,
        /// Upper bound expression
        upper: AstExpr,
        /// Loop step
        step: i64,
        /// Loop body
        body: Vec<AstNode>,
        /// Is this loop parallel?
        is_parallel: bool,
    },
    /// Conditional
    If {
        /// Condition expression
        condition: AstExpr,
        /// Then branch
        then_body: Vec<AstNode>,
        /// Optional else branch
        else_body: Option<Vec<AstNode>>,
    },
    /// Statement invocation
    Statement {
        /// Statement ID
        id: StmtId,
        /// Iterator values
        iterators: Vec<AstExpr>,
    },
    /// Block of statements
    Block {
        /// Statements in block
        statements: Vec<AstNode>,
    },
    /// Raw code (for custom insertions)
    Raw(String),
}

/// Expression in the generated AST.
#[derive(Debug, Clone)]
pub enum AstExpr {
    /// Integer constant
    Int(i64),
    /// Variable reference
    Var(String),
    /// Binary operation
    Binary {
        /// Operator
        op: AstBinOp,
        /// Left operand
        left: Box<AstExpr>,
        /// Right operand
        right: Box<AstExpr>,
    },
    /// Floor division
    FloorDiv(Box<AstExpr>, Box<AstExpr>),
    /// Ceiling division
    CeilDiv(Box<AstExpr>, Box<AstExpr>),
    /// Minimum of two expressions
    Min(Box<AstExpr>, Box<AstExpr>),
    /// Maximum of two expressions
    Max(Box<AstExpr>, Box<AstExpr>),
    /// Modulo operation
    Mod(Box<AstExpr>, Box<AstExpr>),
}

impl AstExpr {
    /// Create integer constant.
    pub fn int(v: i64) -> Self { Self::Int(v) }
    
    /// Create variable reference.
    pub fn var(name: &str) -> Self { Self::Var(name.to_string()) }

    /// Create addition.
    pub fn add(self, other: Self) -> Self {
        Self::Binary { op: AstBinOp::Add, left: Box::new(self), right: Box::new(other) }
    }

    /// Create subtraction.
    pub fn sub(self, other: Self) -> Self {
        Self::Binary { op: AstBinOp::Sub, left: Box::new(self), right: Box::new(other) }
    }

    /// Create multiplication.
    pub fn mul(self, other: Self) -> Self {
        Self::Binary { op: AstBinOp::Mul, left: Box::new(self), right: Box::new(other) }
    }

    /// Create division.
    pub fn div(self, other: Self) -> Self {
        Self::Binary { op: AstBinOp::Div, left: Box::new(self), right: Box::new(other) }
    }

    /// Create floor division.
    pub fn floordiv(self, other: Self) -> Self {
        Self::FloorDiv(Box::new(self), Box::new(other))
    }

    /// Create ceiling division.
    pub fn ceildiv(self, other: Self) -> Self {
        Self::CeilDiv(Box::new(self), Box::new(other))
    }

    /// Create minimum.
    pub fn min(self, other: Self) -> Self {
        Self::Min(Box::new(self), Box::new(other))
    }

    /// Create maximum.
    pub fn max(self, other: Self) -> Self {
        Self::Max(Box::new(self), Box::new(other))
    }

    /// Check if this is a constant.
    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Int(_))
    }

    /// Try to evaluate as constant.
    pub fn eval_constant(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Binary { op, left, right } => {
                let l = left.eval_constant()?;
                let r = right.eval_constant()?;
                Some(match op {
                    AstBinOp::Add => l + r,
                    AstBinOp::Sub => l - r,
                    AstBinOp::Mul => l * r,
                    AstBinOp::Div => l / r,
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    /// Simplify the expression.
    pub fn simplify(self) -> Self {
        match self {
            Self::Binary { op, left, right } => {
                let l = left.simplify();
                let r = right.simplify();
                
                // Constant folding
                if let (Some(lv), Some(rv)) = (l.eval_constant(), r.eval_constant()) {
                    let result = match op {
                        AstBinOp::Add => lv + rv,
                        AstBinOp::Sub => lv - rv,
                        AstBinOp::Mul => lv * rv,
                        AstBinOp::Div if rv != 0 => lv / rv,
                        _ => return Self::Binary { op, left: Box::new(l), right: Box::new(r) },
                    };
                    return Self::Int(result);
                }

                // Identity simplifications
                match (&op, l.eval_constant(), r.eval_constant()) {
                    (AstBinOp::Add, Some(0), _) => return r,
                    (AstBinOp::Add, _, Some(0)) => return l,
                    (AstBinOp::Sub, _, Some(0)) => return l,
                    (AstBinOp::Mul, Some(1), _) => return r,
                    (AstBinOp::Mul, _, Some(1)) => return l,
                    (AstBinOp::Mul, Some(0), _) | (AstBinOp::Mul, _, Some(0)) => return Self::Int(0),
                    (AstBinOp::Div, _, Some(1)) => return l,
                    _ => {}
                }

                Self::Binary { op, left: Box::new(l), right: Box::new(r) }
            }
            Self::Min(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if let (Some(av), Some(bv)) = (a.eval_constant(), b.eval_constant()) {
                    return Self::Int(av.min(bv));
                }
                Self::Min(Box::new(a), Box::new(b))
            }
            Self::Max(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if let (Some(av), Some(bv)) = (a.eval_constant(), b.eval_constant()) {
                    return Self::Int(av.max(bv));
                }
                Self::Max(Box::new(a), Box::new(b))
            }
            other => other,
        }
    }
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstBinOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Modulo
    Mod,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Logical and
    And,
    /// Logical or
    Or,
}

/// AST builder for constructing code AST from schedules.
pub struct AstBuilder {
    nodes: Vec<AstNode>,
}

impl AstBuilder {
    /// Create a new AST builder.
    pub fn new() -> Self {
        Self { nodes: vec![] }
    }

    /// Add a loop to the AST.
    pub fn add_loop(
        &mut self,
        iterator: &str,
        lower: AstExpr,
        upper: AstExpr,
        step: i64,
        is_parallel: bool,
    ) -> &mut Self {
        self.nodes.push(AstNode::Loop {
            iterator: iterator.to_string(),
            lower,
            upper,
            step,
            body: vec![],
            is_parallel,
        });
        self
    }

    /// Add a statement invocation.
    pub fn add_statement(&mut self, id: StmtId, iterators: Vec<AstExpr>) -> &mut Self {
        self.nodes.push(AstNode::Statement { id, iterators });
        self
    }

    /// Add raw code.
    pub fn add_raw(&mut self, code: &str) -> &mut Self {
        self.nodes.push(AstNode::Raw(code.to_string()));
        self
    }

    /// Build the final AST.
    pub fn build(self) -> Vec<AstNode> {
        self.nodes
    }
}

impl Default for AstBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert AST expression to C code string.
pub fn expr_to_c(expr: &AstExpr) -> String {
    match expr {
        AstExpr::Int(v) => v.to_string(),
        AstExpr::Var(name) => name.clone(),
        AstExpr::Binary { op, left, right } => {
            let l = expr_to_c(left);
            let r = expr_to_c(right);
            let op_str = match op {
                AstBinOp::Add => "+",
                AstBinOp::Sub => "-",
                AstBinOp::Mul => "*",
                AstBinOp::Div => "/",
                AstBinOp::Mod => "%",
                AstBinOp::Lt => "<",
                AstBinOp::Le => "<=",
                AstBinOp::Gt => ">",
                AstBinOp::Ge => ">=",
                AstBinOp::Eq => "==",
                AstBinOp::Ne => "!=",
                AstBinOp::And => "&&",
                AstBinOp::Or => "||",
            };
            format!("({} {} {})", l, op_str, r)
        }
        AstExpr::FloorDiv(a, b) => {
            format!("FLOOR_DIV({}, {})", expr_to_c(a), expr_to_c(b))
        }
        AstExpr::CeilDiv(a, b) => {
            format!("CEIL_DIV({}, {})", expr_to_c(a), expr_to_c(b))
        }
        AstExpr::Min(a, b) => {
            format!("MIN({}, {})", expr_to_c(a), expr_to_c(b))
        }
        AstExpr::Max(a, b) => {
            format!("MAX({}, {})", expr_to_c(a), expr_to_c(b))
        }
        AstExpr::Mod(a, b) => {
            format!("({} % {})", expr_to_c(a), expr_to_c(b))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_expr_simplify() {
        // 0 + x = x
        let expr = AstExpr::int(0).add(AstExpr::var("x"));
        let simplified = expr.simplify();
        assert!(matches!(simplified, AstExpr::Var(ref s) if s == "x"));

        // x * 1 = x
        let expr = AstExpr::var("x").mul(AstExpr::int(1));
        let simplified = expr.simplify();
        assert!(matches!(simplified, AstExpr::Var(ref s) if s == "x"));

        // 2 + 3 = 5
        let expr = AstExpr::int(2).add(AstExpr::int(3));
        let simplified = expr.simplify();
        assert!(matches!(simplified, AstExpr::Int(5)));
    }

    #[test]
    fn test_expr_to_c() {
        let expr = AstExpr::var("i").add(AstExpr::int(1));
        assert_eq!(expr_to_c(&expr), "(i + 1)");

        let expr = AstExpr::var("i").floordiv(AstExpr::int(32));
        assert_eq!(expr_to_c(&expr), "FLOOR_DIV(i, 32)");
    }

    #[test]
    fn test_ast_builder() {
        let mut builder = AstBuilder::new();
        builder
            .add_loop("i", AstExpr::int(0), AstExpr::var("N"), 1, true)
            .add_statement(StmtId::new(0), vec![AstExpr::var("i")]);
        
        let ast = builder.build();
        assert_eq!(ast.len(), 2);
    }
}
