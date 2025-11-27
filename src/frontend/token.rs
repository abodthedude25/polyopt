//! Token types for the polyhedral DSL.
//!
//! This module defines all token types produced by the lexer.

use crate::utils::location::Span;
use std::fmt;

/// A token in the source code.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The kind of token
    pub kind: TokenKind,
    /// The source span
    pub span: Span,
    /// The lexeme (raw text)
    pub lexeme: String,
}

impl Token {
    /// Create a new token.
    pub fn new(kind: TokenKind, span: Span, lexeme: String) -> Self {
        Self { kind, span, lexeme }
    }

    /// Check if this is an EOF token.
    pub fn is_eof(&self) -> bool {
        matches!(self.kind, TokenKind::Eof)
    }

    /// Check if this token is a keyword.
    pub fn is_keyword(&self) -> bool {
        self.kind.is_keyword()
    }

    /// Check if this token is an operator.
    pub fn is_operator(&self) -> bool {
        self.kind.is_operator()
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({})", self.kind, self.lexeme)
    }
}

/// The kind of a token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Literals
    /// Integer literal
    Integer,
    /// Floating-point literal
    Float,
    /// String literal
    String,

    // Identifiers
    /// Identifier (variable, function name, etc.)
    Identifier,

    // Keywords
    /// `func` keyword
    Func,
    /// `for` keyword
    For,
    /// `while` keyword
    While,
    /// `if` keyword
    If,
    /// `else` keyword
    Else,
    /// `return` keyword
    Return,
    /// `to` keyword
    To,
    /// `step` keyword
    Step,
    /// `let` keyword
    Let,
    /// `var` keyword
    Var,
    /// `const` keyword
    Const,
    /// `parallel` keyword (annotation)
    Parallel,
    /// `reduce` keyword (annotation)
    Reduce,
    /// `min` keyword
    Min,
    /// `max` keyword
    Max,
    /// `and` keyword
    And,
    /// `or` keyword
    Or,
    /// `not` keyword
    Not,
    /// `true` keyword
    True,
    /// `false` keyword
    False,
    /// `mod` keyword
    Mod,

    // Type keywords
    /// `int` type
    Int,
    /// `float` type
    FloatType,
    /// `double` type
    Double,
    /// `bool` type
    Bool,

    // Arithmetic operators
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,

    // Comparison operators
    /// `==`
    EqualEqual,
    /// `!=`
    BangEqual,
    /// `<`
    Less,
    /// `<=`
    LessEqual,
    /// `>`
    Greater,
    /// `>=`
    GreaterEqual,

    // Assignment operators
    /// `=`
    Equal,
    /// `+=`
    PlusEqual,
    /// `-=`
    MinusEqual,
    /// `*=`
    StarEqual,
    /// `/=`
    SlashEqual,

    // Logical operators
    /// `&&`
    AmpAmp,
    /// `||`
    PipePipe,
    /// `!`
    Bang,

    // Delimiters
    /// `(`
    LeftParen,
    /// `)`
    RightParen,
    /// `[`
    LeftBracket,
    /// `]`
    RightBracket,
    /// `{`
    LeftBrace,
    /// `}`
    RightBrace,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `:`
    Colon,
    /// `.`
    Dot,
    /// `->`
    Arrow,

    // Special
    /// End of file
    Eof,
    /// Error token
    Error,
    /// Newline (for line-sensitive parsing)
    Newline,
    /// Comment (usually skipped)
    Comment,
    /// Annotation marker `@`
    At,
}

impl TokenKind {
    /// Check if this is a keyword.
    pub fn is_keyword(&self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            Func | For | If | Else | Return | To | Step | Let | Var | Const |
            Parallel | Reduce | Min | Max | And | Or | Not | True | False | Mod |
            Int | FloatType | Double | Bool
        )
    }

    /// Check if this is an operator.
    pub fn is_operator(&self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            Plus | Minus | Star | Slash | Percent |
            EqualEqual | BangEqual | Less | LessEqual | Greater | GreaterEqual |
            Equal | PlusEqual | MinusEqual | StarEqual | SlashEqual |
            AmpAmp | PipePipe | Bang
        )
    }

    /// Check if this is a comparison operator.
    pub fn is_comparison(&self) -> bool {
        use TokenKind::*;
        matches!(
            self,
            EqualEqual | BangEqual | Less | LessEqual | Greater | GreaterEqual
        )
    }

    /// Check if this is an assignment operator.
    pub fn is_assignment(&self) -> bool {
        use TokenKind::*;
        matches!(self, Equal | PlusEqual | MinusEqual | StarEqual | SlashEqual)
    }

    /// Get the keyword for a string, if it is a keyword.
    pub fn keyword(s: &str) -> Option<TokenKind> {
        match s {
            "func" => Some(TokenKind::Func),
            "for" => Some(TokenKind::For),
            "while" => Some(TokenKind::While),
            "if" => Some(TokenKind::If),
            "else" => Some(TokenKind::Else),
            "return" => Some(TokenKind::Return),
            "to" => Some(TokenKind::To),
            "step" => Some(TokenKind::Step),
            "let" => Some(TokenKind::Let),
            "var" => Some(TokenKind::Var),
            "const" => Some(TokenKind::Const),
            "parallel" => Some(TokenKind::Parallel),
            "reduce" => Some(TokenKind::Reduce),
            "min" => Some(TokenKind::Min),
            "max" => Some(TokenKind::Max),
            "and" => Some(TokenKind::And),
            "or" => Some(TokenKind::Or),
            "not" => Some(TokenKind::Not),
            "true" => Some(TokenKind::True),
            "false" => Some(TokenKind::False),
            "mod" => Some(TokenKind::Mod),
            "int" => Some(TokenKind::Int),
            "float" => Some(TokenKind::FloatType),
            "double" => Some(TokenKind::Double),
            "bool" => Some(TokenKind::Bool),
            _ => None,
        }
    }

    /// Get a human-readable name for this token kind.
    pub fn name(&self) -> &'static str {
        use TokenKind::*;
        match self {
            Integer => "integer",
            Float => "float",
            String => "string",
            Identifier => "identifier",
            Func => "func",
            For => "for",
            While => "while",
            If => "if",
            Else => "else",
            Return => "return",
            To => "to",
            Step => "step",
            Let => "let",
            Var => "var",
            Const => "const",
            Parallel => "parallel",
            Reduce => "reduce",
            Min => "min",
            Max => "max",
            And => "and",
            Or => "or",
            Not => "not",
            True => "true",
            False => "false",
            Mod => "mod",
            Int => "int",
            FloatType => "float",
            Double => "double",
            Bool => "bool",
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",
            EqualEqual => "==",
            BangEqual => "!=",
            Less => "<",
            LessEqual => "<=",
            Greater => ">",
            GreaterEqual => ">=",
            Equal => "=",
            PlusEqual => "+=",
            MinusEqual => "-=",
            StarEqual => "*=",
            SlashEqual => "/=",
            AmpAmp => "&&",
            PipePipe => "||",
            Bang => "!",
            LeftParen => "(",
            RightParen => ")",
            LeftBracket => "[",
            RightBracket => "]",
            LeftBrace => "{",
            RightBrace => "}",
            Comma => ",",
            Semicolon => ";",
            Colon => ":",
            Dot => ".",
            Arrow => "->",
            Eof => "end of file",
            Error => "error",
            Newline => "newline",
            Comment => "comment",
            At => "@",
        }
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keyword_lookup() {
        assert_eq!(TokenKind::keyword("for"), Some(TokenKind::For));
        assert_eq!(TokenKind::keyword("func"), Some(TokenKind::Func));
        assert_eq!(TokenKind::keyword("foobar"), None);
    }

    #[test]
    fn test_is_keyword() {
        assert!(TokenKind::For.is_keyword());
        assert!(TokenKind::Func.is_keyword());
        assert!(!TokenKind::Plus.is_keyword());
    }

    #[test]
    fn test_is_operator() {
        assert!(TokenKind::Plus.is_operator());
        assert!(TokenKind::EqualEqual.is_operator());
        assert!(!TokenKind::For.is_operator());
    }
}
