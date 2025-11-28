//! Lexer for the polyhedral DSL.
//!
//! The lexer converts source text into a stream of tokens.

use crate::frontend::token::{Token, TokenKind};
use crate::utils::location::{Span, SourceLocation, SourceMap};
use crate::utils::errors::{LexerError, LexerErrorKind};
use unicode_xid::UnicodeXID;
use std::iter::Peekable;
use std::str::Chars;

/// A lexer for tokenizing source code.
pub struct Lexer<'a> {
    /// The source text
    source: &'a str,
    /// Character iterator
    chars: Peekable<Chars<'a>>,
    /// Current byte offset
    offset: usize,
    /// Current line number (1-indexed)
    line: usize,
    /// Current column number (1-indexed)
    column: usize,
    /// Start of current token
    token_start: SourceLocation,
    /// Source map for location lookups
    source_map: SourceMap,
    /// Whether we've hit EOF
    at_eof: bool,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer for the given source.
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            chars: source.chars().peekable(),
            offset: 0,
            line: 1,
            column: 1,
            token_start: SourceLocation::start(),
            source_map: SourceMap::new(source.to_string()),
            at_eof: false,
        }
    }

    /// Get the source map.
    pub fn source_map(&self) -> &SourceMap {
        &self.source_map
    }

    /// Get the current location.
    fn current_location(&self) -> SourceLocation {
        SourceLocation::new(self.line, self.column, self.offset)
    }

    /// Mark the start of a new token.
    fn mark_token_start(&mut self) {
        self.token_start = self.current_location();
    }

    /// Create a span from token start to current location.
    fn make_span(&self) -> Span {
        Span::from_locations(self.token_start, self.current_location())
    }

    /// Peek at the current character without consuming it.
    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    /// Peek at the next character (one ahead).
    fn peek_next(&self) -> Option<char> {
        let mut chars = self.source[self.offset..].chars();
        chars.next();
        chars.next()
    }

    /// Consume and return the current character.
    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        self.offset += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    /// Consume the current character if it matches.
    fn match_char(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Skip whitespace and comments.
    fn skip_whitespace(&mut self) {
        loop {
            match self.peek() {
                Some(' ') | Some('\t') | Some('\r') | Some('\n') => {
                    self.advance();
                }
                Some('/') => {
                    if self.peek_next() == Some('/') {
                        // Line comment
                        while self.peek().is_some() && self.peek() != Some('\n') {
                            self.advance();
                        }
                    } else if self.peek_next() == Some('*') {
                        // Block comment
                        self.advance(); // /
                        self.advance(); // *
                        let mut depth = 1;
                        while depth > 0 {
                            match self.advance() {
                                Some('*') if self.peek() == Some('/') => {
                                    self.advance();
                                    depth -= 1;
                                }
                                Some('/') if self.peek() == Some('*') => {
                                    self.advance();
                                    depth += 1;
                                }
                                None => break,
                                _ => {}
                            }
                        }
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
    }

    /// Create a token with the given kind.
    fn make_token(&self, kind: TokenKind) -> Token {
        let span = self.make_span();
        let lexeme = self.source[span.start_offset..span.end_offset].to_string();
        Token::new(kind, span, lexeme)
    }

    /// Create an error.
    fn make_error(&self, message: &str, kind: LexerErrorKind) -> LexerError {
        LexerError {
            message: message.to_string(),
            span: self.make_span(),
            kind,
        }
    }

    /// Scan a number literal.
    fn scan_number(&mut self) -> Result<Token, LexerError> {
        while self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
            self.advance();
        }

        let mut is_float = false;

        // Check for decimal point
        if self.peek() == Some('.') && self.peek_next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
            is_float = true;
            self.advance(); // consume '.'
            while self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                self.advance();
            }
        }
        
        // Check for exponent (can appear with or without decimal point)
        if self.peek() == Some('e') || self.peek() == Some('E') {
            is_float = true;
            self.advance();
            if self.peek() == Some('+') || self.peek() == Some('-') {
                self.advance();
            }
            if !self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                return Err(self.make_error(
                    "Invalid floating-point exponent",
                    LexerErrorKind::InvalidNumber,
                ));
            }
            while self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                self.advance();
            }
        }
        
        if is_float {
            Ok(self.make_token(TokenKind::Float))
        } else {
            Ok(self.make_token(TokenKind::Integer))
        }
    }

    /// Scan an identifier or keyword.
    fn scan_identifier(&mut self) -> Token {
        while self.peek().map(|c| c.is_xid_continue() || c == '_').unwrap_or(false) {
            self.advance();
        }

        let span = self.make_span();
        let lexeme = &self.source[span.start_offset..span.end_offset];
        
        let kind = TokenKind::keyword(lexeme).unwrap_or(TokenKind::Identifier);
        Token::new(kind, span, lexeme.to_string())
    }

    /// Scan a string literal.
    fn scan_string(&mut self) -> Result<Token, LexerError> {
        // Opening quote already consumed
        let mut value = String::new();
        
        loop {
            match self.advance() {
                Some('"') => break,
                Some('\\') => {
                    match self.advance() {
                        Some('n') => value.push('\n'),
                        Some('t') => value.push('\t'),
                        Some('r') => value.push('\r'),
                        Some('\\') => value.push('\\'),
                        Some('"') => value.push('"'),
                        Some('0') => value.push('\0'),
                        Some(c) => {
                            return Err(self.make_error(
                                &format!("Invalid escape sequence: \\{}", c),
                                LexerErrorKind::InvalidEscape,
                            ));
                        }
                        None => {
                            return Err(self.make_error(
                                "Unterminated string literal",
                                LexerErrorKind::UnterminatedString,
                            ));
                        }
                    }
                }
                Some('\n') => {
                    return Err(self.make_error(
                        "Unterminated string literal (newline in string)",
                        LexerErrorKind::UnterminatedString,
                    ));
                }
                Some(c) => value.push(c),
                None => {
                    return Err(self.make_error(
                        "Unterminated string literal",
                        LexerErrorKind::UnterminatedString,
                    ));
                }
            }
        }
        
        Ok(self.make_token(TokenKind::String))
    }

    /// Scan the next token.
    pub fn next_token(&mut self) -> Result<Token, LexerError> {
        self.skip_whitespace();
        self.mark_token_start();

        let c = match self.advance() {
            Some(c) => c,
            None => {
                self.at_eof = true;
                return Ok(self.make_token(TokenKind::Eof));
            }
        };

        match c {
            // Single-character tokens
            '(' => Ok(self.make_token(TokenKind::LeftParen)),
            ')' => Ok(self.make_token(TokenKind::RightParen)),
            '[' => Ok(self.make_token(TokenKind::LeftBracket)),
            ']' => Ok(self.make_token(TokenKind::RightBracket)),
            '{' => Ok(self.make_token(TokenKind::LeftBrace)),
            '}' => Ok(self.make_token(TokenKind::RightBrace)),
            ',' => Ok(self.make_token(TokenKind::Comma)),
            ';' => Ok(self.make_token(TokenKind::Semicolon)),
            ':' => Ok(self.make_token(TokenKind::Colon)),
            '.' => Ok(self.make_token(TokenKind::Dot)),
            '@' => Ok(self.make_token(TokenKind::At)),

            // Operators (potentially multi-character)
            '+' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::PlusEqual))
                } else {
                    Ok(self.make_token(TokenKind::Plus))
                }
            }
            '-' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::MinusEqual))
                } else if self.match_char('>') {
                    Ok(self.make_token(TokenKind::Arrow))
                } else {
                    Ok(self.make_token(TokenKind::Minus))
                }
            }
            '*' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::StarEqual))
                } else {
                    Ok(self.make_token(TokenKind::Star))
                }
            }
            '/' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::SlashEqual))
                } else {
                    Ok(self.make_token(TokenKind::Slash))
                }
            }
            '%' => Ok(self.make_token(TokenKind::Percent)),

            '=' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::EqualEqual))
                } else {
                    Ok(self.make_token(TokenKind::Equal))
                }
            }
            '!' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::BangEqual))
                } else {
                    Ok(self.make_token(TokenKind::Bang))
                }
            }
            '<' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::LessEqual))
                } else {
                    Ok(self.make_token(TokenKind::Less))
                }
            }
            '>' => {
                if self.match_char('=') {
                    Ok(self.make_token(TokenKind::GreaterEqual))
                } else {
                    Ok(self.make_token(TokenKind::Greater))
                }
            }
            '&' => {
                if self.match_char('&') {
                    Ok(self.make_token(TokenKind::AmpAmp))
                } else {
                    Err(self.make_error(
                        "Expected '&&', found single '&'",
                        LexerErrorKind::UnexpectedChar,
                    ))
                }
            }
            '|' => {
                if self.match_char('|') {
                    Ok(self.make_token(TokenKind::PipePipe))
                } else {
                    Err(self.make_error(
                        "Expected '||', found single '|'",
                        LexerErrorKind::UnexpectedChar,
                    ))
                }
            }

            // String literals
            '"' => self.scan_string(),

            // Numbers
            c if c.is_ascii_digit() => self.scan_number(),

            // Identifiers and keywords
            c if c.is_xid_start() || c == '_' => Ok(self.scan_identifier()),

            // Unknown character
            _ => Err(self.make_error(
                &format!("Unexpected character: '{}'", c),
                LexerErrorKind::UnexpectedChar,
            )),
        }
    }

    /// Check if we've reached EOF.
    pub fn is_at_end(&self) -> bool {
        self.at_eof
    }

    /// Collect all tokens into a vector.
    pub fn tokenize(mut self) -> Result<Vec<Token>, LexerError> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            let is_eof = token.is_eof();
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token, LexerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.at_eof {
            None
        } else {
            let result = self.next_token();
            if result.as_ref().map(|t| t.is_eof()).unwrap_or(false) {
                self.at_eof = true;
            }
            Some(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(source: &str) -> Vec<Token> {
        Lexer::new(source).tokenize().unwrap()
    }

    fn token_kinds(source: &str) -> Vec<TokenKind> {
        lex(source).into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_empty() {
        let tokens = lex("");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn test_whitespace() {
        let tokens = lex("   \t\n\r\n   ");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TokenKind::Eof);
    }

    #[test]
    fn test_keywords() {
        let kinds = token_kinds("func for if else return to step");
        assert_eq!(kinds, vec![
            TokenKind::Func,
            TokenKind::For,
            TokenKind::If,
            TokenKind::Else,
            TokenKind::Return,
            TokenKind::To,
            TokenKind::Step,
            TokenKind::Eof,
        ]);
    }

    #[test]
    fn test_identifiers() {
        let tokens = lex("foo bar _test x123");
        assert_eq!(tokens[0].kind, TokenKind::Identifier);
        assert_eq!(tokens[0].lexeme, "foo");
        assert_eq!(tokens[1].lexeme, "bar");
        assert_eq!(tokens[2].lexeme, "_test");
        assert_eq!(tokens[3].lexeme, "x123");
    }

    #[test]
    fn test_numbers() {
        let tokens = lex("123 45.67 1e10 3.14e-2");
        assert_eq!(tokens[0].kind, TokenKind::Integer);
        assert_eq!(tokens[0].lexeme, "123");
        assert_eq!(tokens[1].kind, TokenKind::Float);
        assert_eq!(tokens[1].lexeme, "45.67");
        assert_eq!(tokens[2].kind, TokenKind::Float);
        assert_eq!(tokens[3].kind, TokenKind::Float);
    }

    #[test]
    fn test_operators() {
        let kinds = token_kinds("+ - * / % = == != < <= > >= += -= *= /=");
        assert!(kinds.contains(&TokenKind::Plus));
        assert!(kinds.contains(&TokenKind::EqualEqual));
        assert!(kinds.contains(&TokenKind::PlusEqual));
    }

    #[test]
    fn test_delimiters() {
        let kinds = token_kinds("( ) [ ] { } , ; :");
        assert_eq!(kinds, vec![
            TokenKind::LeftParen,
            TokenKind::RightParen,
            TokenKind::LeftBracket,
            TokenKind::RightBracket,
            TokenKind::LeftBrace,
            TokenKind::RightBrace,
            TokenKind::Comma,
            TokenKind::Semicolon,
            TokenKind::Colon,
            TokenKind::Eof,
        ]);
    }

    #[test]
    fn test_comments() {
        let tokens = lex("foo // comment\nbar");
        assert_eq!(tokens[0].lexeme, "foo");
        assert_eq!(tokens[1].lexeme, "bar");
    }

    #[test]
    fn test_block_comments() {
        let tokens = lex("foo /* block comment */ bar");
        assert_eq!(tokens[0].lexeme, "foo");
        assert_eq!(tokens[1].lexeme, "bar");
    }

    #[test]
    fn test_string_literal() {
        let tokens = lex(r#""hello world""#);
        assert_eq!(tokens[0].kind, TokenKind::String);
    }

    #[test]
    fn test_location_tracking() {
        let tokens = lex("foo\nbar");
        assert_eq!(tokens[0].span.start_line, 1);
        assert_eq!(tokens[1].span.start_line, 2);
    }

    #[test]
    fn test_complex_program() {
        let source = r#"
            func matmul(A[N][M], B[M][K], C[N][K]) {
                for i = 0 to N {
                    for j = 0 to K {
                        C[i][j] = 0;
                        for k = 0 to M {
                            C[i][j] = C[i][j] + A[i][k] * B[k][j];
                        }
                    }
                }
            }
        "#;
        let result = Lexer::new(source).tokenize();
        assert!(result.is_ok());
    }
}
