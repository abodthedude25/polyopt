//! Frontend: Lexer, Parser, and AST for the polyhedral DSL.
//!
//! This module handles parsing of the input language into an AST.
//!
//! ## Language Overview
//!
//! The input language is a simple C-like DSL focused on loop nests:
//!
//! ```text
//! func matmul(A[N][K], B[K][M], C[N][M]) {
//!     for i = 0 to N {
//!         for j = 0 to M {
//!             for k = 0 to K {
//!                 C[i][j] = C[i][j] + A[i][k] * B[k][j];
//!             }
//!         }
//!     }
//! }
//! ```

pub mod token;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod semantic;

// Re-exports
pub use lexer::Lexer;
pub use parser::Parser;
pub use ast::*;
pub use token::{Token, TokenKind};
pub use crate::utils::errors::ParseError;

use anyhow::Result;

/// Parse source code into an AST.
pub fn parse(source: &str) -> Result<ast::Program> {
    let lexer = Lexer::new(source);
    let mut parser = Parser::new(lexer)?;
    parser.parse_program()
}

/// Parse and perform semantic analysis.
pub fn parse_and_analyze(source: &str) -> Result<ast::Program> {
    let mut program = parse(source)?;
    semantic::analyze(&mut program)?;
    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let source = r#"
            func test() {
                for i = 0 to 10 {
                    A[i] = i;
                }
            }
        "#;
        let result = parse(source);
        assert!(result.is_ok());
    }
}
