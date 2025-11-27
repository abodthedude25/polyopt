//! High-level Intermediate Representation (HIR).
//!
//! The HIR is a simplified, normalized form of the AST that:
//! - Has all types resolved
//! - Has all names resolved to unique IDs
//! - Has compound assignments desugared
//! - Has control flow normalized

use crate::utils::location::Span;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A unique identifier for HIR nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HirId(pub u64);

impl HirId {
    pub fn new(id: u64) -> Self { Self(id) }
}

/// High-level IR for a complete program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirProgram {
    /// All functions in the program
    pub functions: Vec<HirFunction>,
    /// Global parameters (symbolic constants like N, M, K)
    pub parameters: Vec<HirParameter>,
    /// Span of the original program
    pub span: Span,
}

impl HirProgram {
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            parameters: Vec::new(),
            span: Span::dummy(),
        }
    }
}

impl Default for HirProgram {
    fn default() -> Self { Self::new() }
}

/// A symbolic parameter (like N, M, K in array bounds).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirParameter {
    pub id: HirId,
    pub name: String,
    pub span: Span,
}

/// A function in HIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirFunction {
    pub id: HirId,
    pub name: String,
    pub params: Vec<HirFuncParam>,
    pub body: HirBlock,
    pub span: Span,
}

/// A function parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirFuncParam {
    pub id: HirId,
    pub name: String,
    pub ty: HirType,
    /// Array dimensions (empty for scalars)
    pub dimensions: Vec<HirExpr>,
    pub span: Span,
}

impl HirFuncParam {
    pub fn is_array(&self) -> bool { !self.dimensions.is_empty() }
}

/// HIR type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HirType {
    Int,
    Float,
    Double,
    Bool,
    Array { element: Box<HirType>, ndims: usize },
    Unknown,
}

/// A block of statements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirBlock {
    pub statements: Vec<HirStmt>,
    pub span: Span,
}

impl HirBlock {
    pub fn empty() -> Self {
        Self { statements: Vec::new(), span: Span::dummy() }
    }
}

/// A statement in HIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirStmt {
    pub id: HirId,
    pub kind: HirStmtKind,
    pub span: Span,
}

/// Statement kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HirStmtKind {
    /// Variable declaration
    Let {
        var_id: HirId,
        name: String,
        ty: HirType,
        init: Option<HirExpr>,
    },
    
    /// Simple assignment (all compound assignments are desugared)
    Assign {
        target: HirLValue,
        value: HirExpr,
    },
    
    /// For loop (normalized form)
    For {
        var_id: HirId,
        var_name: String,
        lower: HirExpr,
        upper: HirExpr,
        step: HirExpr,
        body: HirBlock,
        is_parallel: bool,
    },
    
    /// If statement
    If {
        condition: HirExpr,
        then_body: HirBlock,
        else_body: Option<HirBlock>,
    },
    
    /// Return statement
    Return {
        value: Option<HirExpr>,
    },
    
    /// Expression statement
    Expr {
        expr: HirExpr,
    },
}

/// An l-value (assignment target).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirLValue {
    pub kind: HirLValueKind,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HirLValueKind {
    /// Simple variable
    Var { id: HirId, name: String },
    /// Array element
    ArrayElem { array_id: HirId, array_name: String, indices: Vec<HirExpr> },
}

/// An expression in HIR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: HirType,
    pub span: Span,
}

impl HirExpr {
    pub fn int(value: i64) -> Self {
        Self {
            kind: HirExprKind::IntLit(value),
            ty: HirType::Int,
            span: Span::dummy(),
        }
    }

    pub fn var(id: HirId, name: String, ty: HirType) -> Self {
        Self {
            kind: HirExprKind::Var { id, name },
            ty,
            span: Span::dummy(),
        }
    }
}

/// Expression kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HirExprKind {
    /// Integer literal
    IntLit(i64),
    /// Float literal
    FloatLit(f64),
    /// Boolean literal
    BoolLit(bool),
    
    /// Variable reference
    Var { id: HirId, name: String },
    
    /// Parameter reference (symbolic constant)
    Param { id: HirId, name: String },
    
    /// Array access
    ArrayAccess {
        array_id: HirId,
        array_name: String,
        indices: Vec<HirExpr>,
    },
    
    /// Binary operation
    Binary {
        op: HirBinaryOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
    },
    
    /// Unary operation
    Unary {
        op: HirUnaryOp,
        operand: Box<HirExpr>,
    },
    
    /// Function call
    Call {
        func: String,
        args: Vec<HirExpr>,
    },
    
    /// Min of two values
    Min(Box<HirExpr>, Box<HirExpr>),
    
    /// Max of two values
    Max(Box<HirExpr>, Box<HirExpr>),
    
    /// Floor division
    FloorDiv { dividend: Box<HirExpr>, divisor: Box<HirExpr> },
    
    /// Ceiling division
    CeilDiv { dividend: Box<HirExpr>, divisor: Box<HirExpr> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HirBinaryOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HirUnaryOp {
    Neg,
    Not,
}

/// ID generator for HIR nodes.
#[derive(Debug, Default)]
pub struct HirIdGen {
    next: u64,
}

impl HirIdGen {
    pub fn new() -> Self { Self { next: 0 } }
    
    pub fn next(&mut self) -> HirId {
        let id = HirId(self.next);
        self.next += 1;
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hir_id_gen() {
        let mut gen = HirIdGen::new();
        assert_eq!(gen.next(), HirId(0));
        assert_eq!(gen.next(), HirId(1));
        assert_eq!(gen.next(), HirId(2));
    }
}
