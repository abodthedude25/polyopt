//! Error types for the polyhedral optimizer.
//!
//! This module defines all error types used throughout the framework,
//! organized by the phase that produces them.

use thiserror::Error;
use crate::utils::location::Span;
use std::fmt;

/// Top-level error type for the optimizer.
#[derive(Error, Debug)]
pub enum PolyOptError {
    /// Error during lexing/tokenization
    #[error("Lexer error: {0}")]
    Lexer(#[from] LexerError),

    /// Error during parsing
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),

    /// Error during semantic analysis
    #[error("Semantic error: {0}")]
    Semantic(#[from] SemanticError),

    /// Error during SCoP detection
    #[error("SCoP detection error: {0}")]
    ScopDetection(#[from] ScoPError),

    /// Error during dependence analysis
    #[error("Dependence analysis error: {0}")]
    Dependence(#[from] DependenceError),

    /// Error during transformation
    #[error("Transformation error: {0}")]
    Transform(#[from] TransformError),

    /// Error during code generation
    #[error("Code generation error: {0}")]
    Codegen(#[from] CodegenError),

    /// Internal compiler error
    #[error("Internal error: {0}")]
    Internal(String),

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Error during lexical analysis.
#[derive(Error, Debug, Clone)]
pub struct LexerError {
    /// The error message
    pub message: String,
    /// Location in source
    pub span: Span,
    /// The kind of lexer error
    pub kind: LexerErrorKind,
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.message, self.span)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexerErrorKind {
    /// Unexpected character
    UnexpectedChar,
    /// Unterminated string literal
    UnterminatedString,
    /// Invalid number literal
    InvalidNumber,
    /// Invalid escape sequence
    InvalidEscape,
    /// Unexpected end of file
    UnexpectedEof,
}

/// Error during parsing.
#[derive(Error, Debug, Clone)]
pub struct ParseError {
    /// The error message
    pub message: String,
    /// Location in source
    pub span: Span,
    /// The kind of parse error
    pub kind: ParseErrorKind,
    /// Expected tokens (if applicable)
    pub expected: Vec<String>,
    /// What was found
    pub found: Option<String>,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.message, self.span)?;
        if !self.expected.is_empty() {
            write!(f, " (expected: {})", self.expected.join(", "))?;
        }
        if let Some(ref found) = self.found {
            write!(f, " (found: {})", found)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseErrorKind {
    /// Unexpected token
    UnexpectedToken,
    /// Expected a specific token
    ExpectedToken,
    /// Expected an expression
    ExpectedExpression,
    /// Expected a statement
    ExpectedStatement,
    /// Expected an identifier
    ExpectedIdentifier,
    /// Expected a type
    ExpectedType,
    /// Invalid syntax
    InvalidSyntax,
    /// Mismatched brackets/braces
    MismatchedDelimiter,
    /// Unexpected end of file
    UnexpectedEof,
}

/// Error during semantic analysis.
#[derive(Error, Debug, Clone)]
pub struct SemanticError {
    /// The error message
    pub message: String,
    /// Location in source
    pub span: Span,
    /// The kind of semantic error
    pub kind: SemanticErrorKind,
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.message, self.span)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticErrorKind {
    /// Undefined variable
    UndefinedVariable,
    /// Undefined function
    UndefinedFunction,
    /// Undefined array
    UndefinedArray,
    /// Duplicate definition
    DuplicateDefinition,
    /// Type mismatch
    TypeMismatch,
    /// Invalid array dimensions
    InvalidDimensions,
    /// Invalid loop bounds
    InvalidBounds,
    /// Division by zero
    DivisionByZero,
}

/// Error during SCoP detection.
#[derive(Error, Debug, Clone)]
pub struct ScoPError {
    /// The error message
    pub message: String,
    /// Location in source (if available)
    pub span: Option<Span>,
    /// The kind of SCoP error
    pub kind: ScoPErrorKind,
}

impl fmt::Display for ScoPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref span) = self.span {
            write!(f, "{} at {}", self.message, span)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoPErrorKind {
    /// Non-affine loop bound
    NonAffineBound,
    /// Non-affine array subscript
    NonAffineSubscript,
    /// Non-affine conditional
    NonAffineCondition,
    /// Function call in SCoP
    FunctionCall,
    /// Irregular control flow
    IrregularControlFlow,
    /// No SCoP regions found
    NoScoPFound,
    /// Unsupported construct
    Unsupported,
}

/// Error during dependence analysis.
#[derive(Error, Debug, Clone)]
pub struct DependenceError {
    /// The error message
    pub message: String,
    /// The kind of dependence error
    pub kind: DependenceErrorKind,
}

impl fmt::Display for DependenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependenceErrorKind {
    /// Failed to compute dependence polyhedron
    ComputationFailed,
    /// Cyclic dependencies detected
    CyclicDependencies,
    /// Dependence analysis timeout
    Timeout,
    /// Unsupported access pattern
    UnsupportedPattern,
}

/// Error during transformation.
#[derive(Error, Debug, Clone)]
pub struct TransformError {
    /// The error message
    pub message: String,
    /// The kind of transformation error
    pub kind: TransformErrorKind,
    /// The transformation that failed
    pub transform: String,
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} in {}", self.message, self.transform)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformErrorKind {
    /// Transformation would violate dependencies
    IllegalTransform,
    /// Invalid tile size
    InvalidTileSize,
    /// Loops cannot be fused
    IncompatibleFusion,
    /// Scheduling failed
    SchedulingFailed,
    /// ILP solver failed
    IlpFailed,
    /// Transformation not applicable
    NotApplicable,
}

/// Error during code generation.
#[derive(Error, Debug, Clone)]
pub struct CodegenError {
    /// The error message
    pub message: String,
    /// The kind of codegen error
    pub kind: CodegenErrorKind,
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodegenErrorKind {
    /// Failed to generate loop bounds
    BoundGeneration,
    /// Failed to generate array subscripts
    SubscriptGeneration,
    /// Target not supported
    UnsupportedTarget,
    /// Feature not supported on target
    UnsupportedFeature,
    /// Code too complex to generate
    ComplexityLimit,
}

/// A diagnostic message with severity level.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Severity level
    pub severity: DiagnosticSeverity,
    /// Message
    pub message: String,
    /// Primary span
    pub span: Option<Span>,
    /// Additional notes
    pub notes: Vec<String>,
    /// Suggested fix (if any)
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    /// Error - compilation cannot continue
    Error,
    /// Warning - compilation continues but result may be suboptimal
    Warning,
    /// Note - informational message
    Note,
    /// Help - suggestion for fixing the issue
    Help,
}

impl Diagnostic {
    /// Create a new error diagnostic.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Error,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            suggestion: None,
        }
    }

    /// Create a new warning diagnostic.
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Warning,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            suggestion: None,
        }
    }

    /// Add a span to the diagnostic.
    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a note to the diagnostic.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Add a suggestion to the diagnostic.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// Result type using PolyOptError.
pub type PolyResult<T> = Result<T, PolyOptError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ParseError {
            message: "Unexpected token".to_string(),
            span: Span::new(1, 5, 1, 10),
            kind: ParseErrorKind::UnexpectedToken,
            expected: vec!["identifier".to_string()],
            found: Some("number".to_string()),
        };
        let s = format!("{}", err);
        assert!(s.contains("Unexpected token"));
        assert!(s.contains("identifier"));
    }
}
