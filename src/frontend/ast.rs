//! Abstract Syntax Tree (AST) for the polyhedral DSL.
//!
//! The AST represents the parsed structure of the input program.
//! It preserves the high-level structure including loops, statements,
//! and expressions.

use crate::utils::location::Span;
use crate::utils::intern::Symbol;
use serde::{Serialize, Deserialize};
use std::fmt;

/// A complete program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    /// Functions in the program
    pub functions: Vec<Function>,
    /// Global constants/parameters
    pub globals: Vec<GlobalDecl>,
    /// Source span
    pub span: Span,
}

impl Program {
    /// Create a new empty program.
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            globals: Vec::new(),
            span: Span::dummy(),
        }
    }

    /// Find a function by name.
    pub fn find_function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

/// A global declaration (constant or parameter).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalDecl {
    /// Name of the global
    pub name: String,
    /// Type
    pub ty: Type,
    /// Initial value (if constant)
    pub value: Option<Expr>,
    /// Is this a parameter (symbolic constant)?
    pub is_param: bool,
    /// Source span
    pub span: Span,
}

/// A function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    /// Function name
    pub name: String,
    /// Parameters
    pub params: Vec<Parameter>,
    /// Return type (None for void)
    pub return_type: Option<Type>,
    /// Function body
    pub body: Block,
    /// Annotations (like @parallel)
    pub annotations: Vec<Annotation>,
    /// Source span
    pub span: Span,
}

impl Function {
    /// Get array parameters.
    pub fn array_params(&self) -> impl Iterator<Item = &Parameter> {
        self.params.iter().filter(|p| p.is_array())
    }

    /// Get scalar parameters.
    pub fn scalar_params(&self) -> impl Iterator<Item = &Parameter> {
        self.params.iter().filter(|p| !p.is_array())
    }
}

/// A function parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub ty: Type,
    /// Array dimensions (if array)
    pub dimensions: Vec<Expr>,
    /// Source span
    pub span: Span,
}

impl Parameter {
    /// Check if this parameter is an array.
    pub fn is_array(&self) -> bool {
        !self.dimensions.is_empty()
    }

    /// Get the number of dimensions (0 for scalars).
    pub fn ndims(&self) -> usize {
        self.dimensions.len()
    }
}

/// A type in the language.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    /// 32-bit integer
    Int,
    /// 32-bit floating point
    Float,
    /// 64-bit floating point
    Double,
    /// Boolean
    Bool,
    /// Void (for functions)
    Void,
    /// Array type with element type and dimensions
    Array {
        element: Box<Type>,
        dimensions: Vec<Option<i64>>, // None means symbolic
    },
    /// Unknown type (before type inference)
    Unknown,
}

impl Type {
    /// Check if this is a numeric type.
    pub fn is_numeric(&self) -> bool {
        matches!(self, Type::Int | Type::Float | Type::Double)
    }

    /// Check if this is an array type.
    pub fn is_array(&self) -> bool {
        matches!(self, Type::Array { .. })
    }

    /// Get the element type if this is an array.
    pub fn element_type(&self) -> Option<&Type> {
        match self {
            Type::Array { element, .. } => Some(element),
            _ => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::Double => write!(f, "double"),
            Type::Bool => write!(f, "bool"),
            Type::Void => write!(f, "void"),
            Type::Array { element, dimensions } => {
                write!(f, "{}", element)?;
                for dim in dimensions {
                    match dim {
                        Some(n) => write!(f, "[{}]", n)?,
                        None => write!(f, "[]")?,
                    }
                }
                Ok(())
            }
            Type::Unknown => write!(f, "?"),
        }
    }
}

/// An annotation (e.g., @parallel).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Annotation name
    pub name: String,
    /// Arguments (if any)
    pub args: Vec<Expr>,
    /// Source span
    pub span: Span,
}

/// A block of statements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Statements in the block
    pub statements: Vec<Stmt>,
    /// Source span
    pub span: Span,
}

impl Block {
    /// Create an empty block.
    pub fn empty() -> Self {
        Self {
            statements: Vec::new(),
            span: Span::dummy(),
        }
    }

    /// Check if the block is empty.
    pub fn is_empty(&self) -> bool {
        self.statements.is_empty()
    }
}

/// A statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stmt {
    /// The kind of statement
    pub kind: StmtKind,
    /// Source span
    pub span: Span,
    /// Annotations
    pub annotations: Vec<Annotation>,
}

/// The kind of a statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StmtKind {
    /// Variable declaration: `let x = expr;` or `var x: type = expr;`
    Declaration {
        name: String,
        ty: Option<Type>,
        value: Option<Expr>,
        is_mutable: bool,
    },

    /// Assignment: `x = expr;` or `a[i][j] = expr;`
    Assignment {
        target: AssignTarget,
        op: AssignOp,
        value: Expr,
    },

    /// For loop: `for i = start to end step s { body }`
    For {
        iterator: String,
        start: Expr,
        end: Expr,
        step: Option<Expr>,
        body: Block,
        is_parallel: bool,
    },

    /// If statement: `if cond { then } else { else }`
    If {
        condition: Expr,
        then_branch: Block,
        else_branch: Option<Block>,
    },

    /// While loop: `while cond { body }`
    While {
        condition: Expr,
        body: Block,
    },

    /// Return statement: `return expr;`
    Return {
        value: Option<Expr>,
    },

    /// Expression statement: `expr;`
    Expression {
        expr: Expr,
    },

    /// Block statement: `{ stmts }`
    Block {
        block: Block,
    },

    /// Empty statement (just a semicolon)
    Empty,
}

/// An assignment target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssignTarget {
    /// Simple variable
    Variable(String),
    /// Array element
    ArrayAccess {
        array: String,
        indices: Vec<Expr>,
    },
}

impl AssignTarget {
    /// Get the variable name.
    pub fn name(&self) -> &str {
        match self {
            AssignTarget::Variable(name) => name,
            AssignTarget::ArrayAccess { array, .. } => array,
        }
    }
}

/// An assignment operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssignOp {
    /// `=`
    Assign,
    /// `+=`
    AddAssign,
    /// `-=`
    SubAssign,
    /// `*=`
    MulAssign,
    /// `/=`
    DivAssign,
}

impl fmt::Display for AssignOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssignOp::Assign => write!(f, "="),
            AssignOp::AddAssign => write!(f, "+="),
            AssignOp::SubAssign => write!(f, "-="),
            AssignOp::MulAssign => write!(f, "*="),
            AssignOp::DivAssign => write!(f, "/="),
        }
    }
}

/// An expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expr {
    /// The kind of expression
    pub kind: ExprKind,
    /// Inferred type (filled in during semantic analysis)
    pub ty: Type,
    /// Source span
    pub span: Span,
}

impl Expr {
    /// Create a new expression with unknown type.
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self {
            kind,
            ty: Type::Unknown,
            span,
        }
    }

    /// Create an integer literal.
    pub fn int_lit(value: i64, span: Span) -> Self {
        Self::new(ExprKind::IntLiteral(value), span)
    }

    /// Create a float literal.
    pub fn float_lit(value: f64, span: Span) -> Self {
        Self::new(ExprKind::FloatLiteral(value), span)
    }

    /// Create a variable reference.
    pub fn var(name: String, span: Span) -> Self {
        Self::new(ExprKind::Variable(name), span)
    }

    /// Check if this is a constant expression.
    pub fn is_constant(&self) -> bool {
        match &self.kind {
            ExprKind::IntLiteral(_) | ExprKind::FloatLiteral(_) | ExprKind::BoolLiteral(_) => true,
            ExprKind::Unary { operand, .. } => operand.is_constant(),
            ExprKind::Binary { left, right, .. } => left.is_constant() && right.is_constant(),
            _ => false,
        }
    }

    /// Check if this is an affine expression (for polyhedral analysis).
    /// An affine expression is a linear combination of variables plus a constant.
    pub fn is_affine(&self, loop_vars: &[&str]) -> bool {
        match &self.kind {
            ExprKind::IntLiteral(_) => true,
            ExprKind::Variable(name) => {
                // Loop variables and parameters are affine
                loop_vars.contains(&name.as_str())
            }
            ExprKind::Binary { op, left, right } => {
                match op {
                    BinaryOp::Add | BinaryOp::Sub => {
                        left.is_affine(loop_vars) && right.is_affine(loop_vars)
                    }
                    BinaryOp::Mul => {
                        // One side must be constant
                        (left.is_constant() && right.is_affine(loop_vars)) ||
                        (right.is_constant() && left.is_affine(loop_vars))
                    }
                    BinaryOp::Div | BinaryOp::Mod => {
                        // Division/mod by constant
                        left.is_affine(loop_vars) && right.is_constant()
                    }
                    _ => false,
                }
            }
            ExprKind::Unary { op, operand } => {
                match op {
                    UnaryOp::Neg => operand.is_affine(loop_vars),
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

/// The kind of an expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExprKind {
    /// Integer literal
    IntLiteral(i64),
    /// Floating-point literal
    FloatLiteral(f64),
    /// Boolean literal
    BoolLiteral(bool),
    /// String literal
    StringLiteral(String),

    /// Variable reference
    Variable(String),

    /// Array access: `a[i][j]`
    ArrayAccess {
        array: Box<Expr>,
        indices: Vec<Expr>,
    },

    /// Binary operation: `left op right`
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operation: `op operand`
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    /// Function call: `func(args)`
    Call {
        function: String,
        args: Vec<Expr>,
    },

    /// Ternary conditional: `cond ? then : else`
    Ternary {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },

    /// Cast: `(type) expr`
    Cast {
        target_type: Type,
        expr: Box<Expr>,
    },

    /// Min function: `min(a, b)`
    Min(Box<Expr>, Box<Expr>),

    /// Max function: `max(a, b)`
    Max(Box<Expr>, Box<Expr>),

    /// Floor division (for polyhedral): `floor(a / b)`
    FloorDiv {
        dividend: Box<Expr>,
        divisor: Box<Expr>,
    },

    /// Ceiling division: `ceil(a / b)`
    CeilDiv {
        dividend: Box<Expr>,
        divisor: Box<Expr>,
    },

    /// Grouped expression (parenthesized)
    Grouped(Box<Expr>),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,
}

impl BinaryOp {
    /// Get the precedence of this operator (higher binds tighter).
    pub fn precedence(&self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Eq | BinaryOp::Ne => 3,
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => 4,
            BinaryOp::Add | BinaryOp::Sub => 5,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 6,
        }
    }

    /// Check if this operator is left-associative.
    pub fn is_left_assoc(&self) -> bool {
        true // All our operators are left-associative
    }

    /// Check if this is a comparison operator.
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
        )
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::Ne => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::Le => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::Ge => write!(f, ">="),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
        }
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Negation: `-x`
    Neg,
    /// Logical not: `!x`
    Not,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
        }
    }
}

/// A node ID for unique identification of AST nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Create a new node ID.
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Visitor trait for traversing the AST.
pub trait AstVisitor {
    /// Visit a program.
    fn visit_program(&mut self, program: &Program) {
        for func in &program.functions {
            self.visit_function(func);
        }
    }

    /// Visit a function.
    fn visit_function(&mut self, func: &Function) {
        self.visit_block(&func.body);
    }

    /// Visit a block.
    fn visit_block(&mut self, block: &Block) {
        for stmt in &block.statements {
            self.visit_stmt(stmt);
        }
    }

    /// Visit a statement.
    fn visit_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Declaration { value, .. } => {
                if let Some(expr) = value {
                    self.visit_expr(expr);
                }
            }
            StmtKind::Assignment { value, target, .. } => {
                if let AssignTarget::ArrayAccess { indices, .. } = target {
                    for idx in indices {
                        self.visit_expr(idx);
                    }
                }
                self.visit_expr(value);
            }
            StmtKind::For { start, end, step, body, .. } => {
                self.visit_expr(start);
                self.visit_expr(end);
                if let Some(s) = step {
                    self.visit_expr(s);
                }
                self.visit_block(body);
            }
            StmtKind::If { condition, then_branch, else_branch } => {
                self.visit_expr(condition);
                self.visit_block(then_branch);
                if let Some(else_b) = else_branch {
                    self.visit_block(else_b);
                }
            }
            StmtKind::While { condition, body } => {
                self.visit_expr(condition);
                self.visit_block(body);
            }
            StmtKind::Return { value } => {
                if let Some(v) = value {
                    self.visit_expr(v);
                }
            }
            StmtKind::Expression { expr } => {
                self.visit_expr(expr);
            }
            StmtKind::Block { block } => {
                self.visit_block(block);
            }
            StmtKind::Empty => {}
        }
    }

    /// Visit an expression.
    fn visit_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Binary { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            ExprKind::Unary { operand, .. } => {
                self.visit_expr(operand);
            }
            ExprKind::ArrayAccess { array, indices } => {
                self.visit_expr(array);
                for idx in indices {
                    self.visit_expr(idx);
                }
            }
            ExprKind::Call { args, .. } => {
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::Ternary { condition, then_expr, else_expr } => {
                self.visit_expr(condition);
                self.visit_expr(then_expr);
                self.visit_expr(else_expr);
            }
            ExprKind::Min(a, b) | ExprKind::Max(a, b) => {
                self.visit_expr(a);
                self.visit_expr(b);
            }
            ExprKind::FloorDiv { dividend, divisor } | ExprKind::CeilDiv { dividend, divisor } => {
                self.visit_expr(dividend);
                self.visit_expr(divisor);
            }
            ExprKind::Cast { expr, .. } => {
                self.visit_expr(expr);
            }
            ExprKind::Grouped(inner) => {
                self.visit_expr(inner);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_display() {
        assert_eq!(Type::Int.to_string(), "int");
        assert_eq!(Type::Float.to_string(), "float");
        
        let arr_ty = Type::Array {
            element: Box::new(Type::Int),
            dimensions: vec![Some(10), None],
        };
        assert_eq!(arr_ty.to_string(), "int[10][]");
    }

    #[test]
    fn test_expr_is_affine() {
        let loop_vars = vec!["i", "j"];
        
        // i + j is affine
        let expr = Expr::new(
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: Box::new(Expr::var("i".to_string(), Span::dummy())),
                right: Box::new(Expr::var("j".to_string(), Span::dummy())),
            },
            Span::dummy(),
        );
        assert!(expr.is_affine(&loop_vars));

        // i * j is NOT affine
        let expr = Expr::new(
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expr::var("i".to_string(), Span::dummy())),
                right: Box::new(Expr::var("j".to_string(), Span::dummy())),
            },
            Span::dummy(),
        );
        assert!(!expr.is_affine(&loop_vars));

        // 2 * i is affine
        let expr = Expr::new(
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left: Box::new(Expr::int_lit(2, Span::dummy())),
                right: Box::new(Expr::var("i".to_string(), Span::dummy())),
            },
            Span::dummy(),
        );
        assert!(expr.is_affine(&loop_vars));
    }
}
