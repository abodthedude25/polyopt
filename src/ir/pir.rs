//! Polyhedral Intermediate Representation (PIR).
//!
//! The PIR represents programs in polyhedral form:
//! - Statements with iteration domains
//! - Access relations for memory references
//! - Schedules for execution ordering
//! - Dependencies between statements

use crate::utils::location::Span;
use crate::polyhedral::{IntegerSet, AffineMap, Space};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A unique identifier for PIR statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StmtId(pub u64);

impl StmtId {
    pub fn new(id: u64) -> Self { Self(id) }
}

impl std::fmt::Display for StmtId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "S{}", self.0)
    }
}

/// A complete polyhedral program.
#[derive(Debug, Clone)]
pub struct PolyProgram {
    /// Name of the program/function
    pub name: String,
    /// Symbolic parameters (N, M, K, etc.)
    pub parameters: Vec<String>,
    /// Statements in the program
    pub statements: Vec<PolyStmt>,
    /// Arrays used in the program
    pub arrays: Vec<ArrayInfo>,
    /// Context constraints (e.g., N > 0)
    pub context: IntegerSet,
}

impl PolyProgram {
    pub fn new(name: String) -> Self {
        Self {
            name,
            parameters: Vec::new(),
            statements: Vec::new(),
            arrays: Vec::new(),
            context: IntegerSet::universe(0),
        }
    }

    /// Get a statement by ID.
    pub fn get_stmt(&self, id: StmtId) -> Option<&PolyStmt> {
        self.statements.iter().find(|s| s.id == id)
    }

    /// Get a mutable statement by ID.
    pub fn get_stmt_mut(&mut self, id: StmtId) -> Option<&mut PolyStmt> {
        self.statements.iter_mut().find(|s| s.id == id)
    }
}

/// Information about an array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayInfo {
    /// Array name
    pub name: String,
    /// Number of dimensions
    pub ndims: usize,
    /// Element type
    pub element_type: ElementType,
    /// Dimension sizes (symbolic expressions)
    pub sizes: Vec<String>,
}

/// Element type of arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementType {
    Int,
    Float,
    Double,
}

/// A polyhedral statement.
#[derive(Debug, Clone)]
pub struct PolyStmt {
    /// Unique identifier
    pub id: StmtId,
    /// Human-readable name
    pub name: String,
    /// Iteration domain: { [i,j,...] : constraints }
    pub domain: IntegerSet,
    /// Schedule: maps iteration point to logical time
    pub schedule: AffineMap,
    /// Read accesses
    pub reads: Vec<AccessRelation>,
    /// Write accesses
    pub writes: Vec<AccessRelation>,
    /// The computation body (for code generation)
    pub body: StmtBody,
    /// Original source location
    pub span: Span,
}

impl PolyStmt {
    /// Get the dimensionality of the iteration space.
    pub fn depth(&self) -> usize {
        self.domain.dim()
    }

    /// Get all memory accesses.
    pub fn accesses(&self) -> impl Iterator<Item = &AccessRelation> {
        self.reads.iter().chain(self.writes.iter())
    }

    /// Check if this statement has any writes to the given array.
    pub fn writes_to(&self, array: &str) -> bool {
        self.writes.iter().any(|a| a.array == array)
    }

    /// Check if this statement reads from the given array.
    pub fn reads_from(&self, array: &str) -> bool {
        self.reads.iter().any(|a| a.array == array)
    }
}

/// An access relation maps iteration points to memory locations.
/// { [i,j] -> A[i][j+1] } means at iteration (i,j), we access A[i,j+1].
#[derive(Debug, Clone)]
pub struct AccessRelation {
    /// Array being accessed
    pub array: String,
    /// The access map: domain -> array indices
    pub relation: AffineMap,
    /// Access type (read or write)
    pub kind: AccessKind,
}

impl AccessRelation {
    /// Create a new access relation.
    pub fn new(array: String, relation: AffineMap, kind: AccessKind) -> Self {
        Self { array, relation, kind }
    }
}

/// Kind of memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessKind {
    Read,
    Write,
}

/// The computation body of a statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StmtBody {
    /// Assignment: target = expr
    Assignment {
        target: AccessExpr,
        expr: ComputeExpr,
    },
    /// Compound assignment: target op= expr (e.g., +=, -=)
    CompoundAssign {
        target: AccessExpr,
        op: CompoundOp,
        expr: ComputeExpr,
    },
}

/// An access expression (for code generation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessExpr {
    pub array: String,
    pub indices: Vec<AffineExprStr>,
}

/// String representation of an affine expression for code generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineExprStr(pub String);

/// Compound assignment operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompoundOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// A compute expression (for code generation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeExpr {
    /// Integer constant
    Int(i64),
    /// Float constant
    Float(f64),
    /// Variable reference
    Var(String),
    /// Array access
    Access(AccessExpr),
    /// Binary operation
    Binary {
        op: BinaryComputeOp,
        left: Box<ComputeExpr>,
        right: Box<ComputeExpr>,
    },
    /// Unary operation
    Unary {
        op: UnaryComputeOp,
        operand: Box<ComputeExpr>,
    },
    /// Function call
    Call {
        func: String,
        args: Vec<ComputeExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryComputeOp {
    Add, Sub, Mul, Div, Mod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryComputeOp {
    Neg,
}

/// Statement ID generator.
#[derive(Debug, Default)]
pub struct StmtIdGen {
    next: u64,
}

impl StmtIdGen {
    pub fn new() -> Self { Self { next: 0 } }
    
    pub fn next(&mut self) -> StmtId {
        let id = StmtId(self.next);
        self.next += 1;
        id
    }
}

/// Builder for constructing polyhedral statements.
#[derive(Debug)]
pub struct PolyStmtBuilder {
    id: StmtId,
    name: String,
    domain: Option<IntegerSet>,
    schedule: Option<AffineMap>,
    reads: Vec<AccessRelation>,
    writes: Vec<AccessRelation>,
    body: Option<StmtBody>,
    span: Span,
}

impl PolyStmtBuilder {
    pub fn new(id: StmtId, name: String) -> Self {
        Self {
            id,
            name,
            domain: None,
            schedule: None,
            reads: Vec::new(),
            writes: Vec::new(),
            body: None,
            span: Span::dummy(),
        }
    }

    pub fn domain(mut self, domain: IntegerSet) -> Self {
        self.domain = Some(domain);
        self
    }

    pub fn schedule(mut self, schedule: AffineMap) -> Self {
        self.schedule = Some(schedule);
        self
    }

    pub fn add_read(mut self, access: AccessRelation) -> Self {
        self.reads.push(access);
        self
    }

    pub fn add_write(mut self, access: AccessRelation) -> Self {
        self.writes.push(access);
        self
    }

    pub fn body(mut self, body: StmtBody) -> Self {
        self.body = Some(body);
        self
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn build(self) -> Option<PolyStmt> {
        let domain = self.domain?;
        let dim = domain.dim();
        let schedule = self.schedule.unwrap_or_else(|| {
            // Default identity schedule
            AffineMap::identity(dim)
        });
        Some(PolyStmt {
            id: self.id,
            name: self.name,
            schedule,
            domain,
            reads: self.reads,
            writes: self.writes,
            body: self.body?,
            span: self.span,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stmt_id() {
        let mut gen = StmtIdGen::new();
        assert_eq!(gen.next().to_string(), "S0");
        assert_eq!(gen.next().to_string(), "S1");
    }

    #[test]
    fn test_poly_program() {
        let program = PolyProgram::new("test".to_string());
        assert_eq!(program.name, "test");
        assert!(program.statements.is_empty());
    }
}
