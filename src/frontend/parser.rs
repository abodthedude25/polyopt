//! Parser for the polyhedral DSL.
//!
//! This module implements a recursive descent parser that converts
//! a stream of tokens into an AST.

use crate::frontend::lexer::Lexer;
use crate::frontend::token::{Token, TokenKind};
use crate::frontend::ast::*;
use crate::utils::location::Span;
use crate::utils::errors::{ParseError, ParseErrorKind};
use anyhow::{Result, bail, anyhow};

/// A parser for the polyhedral DSL.
pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    previous: Token,
    errors: Vec<ParseError>,
    panic_mode: bool,
}

impl<'a> Parser<'a> {
    /// Create a new parser from a lexer.
    pub fn new(mut lexer: Lexer<'a>) -> Result<Self> {
        let first_token = lexer.next_token()
            .map_err(|e| anyhow!("Lexer error: {}", e))?;
        
        Ok(Self {
            lexer,
            current: first_token.clone(),
            previous: first_token,
            errors: Vec::new(),
            panic_mode: false,
        })
    }

    /// Parse a complete program.
    pub fn parse_program(&mut self) -> Result<Program> {
        let start = self.current.span;
        let mut program = Program::new();

        while !self.is_at_end() {
            match self.parse_top_level() {
                Ok(TopLevel::Function(func)) => program.functions.push(func),
                Ok(TopLevel::Global(global)) => program.globals.push(global),
                Err(e) => {
                    self.errors.push(self.make_parse_error(&e.to_string()));
                    self.synchronize();
                }
            }
        }

        program.span = start.merge(&self.previous.span);

        if !self.errors.is_empty() {
            bail!("Parse errors: {:?}", self.errors);
        }

        Ok(program)
    }

    fn parse_top_level(&mut self) -> Result<TopLevel> {
        let annotations = self.parse_annotations()?;

        if self.check(TokenKind::Func) {
            Ok(TopLevel::Function(self.parse_function(annotations)?))
        } else if self.check(TokenKind::Const) {
            if !annotations.is_empty() {
                bail!("Annotations not allowed on global constants");
            }
            Ok(TopLevel::Global(self.parse_global()?))
        } else {
            bail!("Expected 'func' or 'const', found {:?}", self.current.kind);
        }
    }

    fn parse_annotations(&mut self) -> Result<Vec<Annotation>> {
        let mut annotations = Vec::new();

        while self.check(TokenKind::At) {
            let start = self.current.span;
            self.advance()?;

            // Annotation name can be an identifier OR the "parallel" keyword
            let name = if self.check(TokenKind::Identifier) {
                let n = self.current.lexeme.clone();
                self.advance()?;
                n
            } else if self.check(TokenKind::Parallel) {
                self.advance()?;
                "parallel".to_string()
            } else if self.check(TokenKind::Reduce) {
                self.advance()?;
                "reduce".to_string()
            } else {
                bail!("Expected annotation name, found {:?}", self.current.kind);
            };
            
            let args = if self.match_token(TokenKind::LeftParen)? {
                let args = self.parse_args()?;
                self.consume(TokenKind::RightParen, "Expected ')' after annotation arguments")?;
                args
            } else {
                Vec::new()
            };

            annotations.push(Annotation {
                name,
                args,
                span: start.merge(&self.previous.span),
            });
        }

        Ok(annotations)
    }

    fn parse_global(&mut self) -> Result<GlobalDecl> {
        let start = self.current.span;
        self.consume(TokenKind::Const, "Expected 'const'")?;
        
        let name = self.consume_identifier("Expected constant name")?;
        
        let ty = if self.match_token(TokenKind::Colon)? {
            self.parse_type()?
        } else {
            Type::Unknown
        };

        self.consume(TokenKind::Equal, "Expected '=' in constant declaration")?;
        let value = self.parse_expression()?;
        self.consume(TokenKind::Semicolon, "Expected ';' after constant declaration")?;

        Ok(GlobalDecl {
            name,
            ty,
            value: Some(value),
            is_param: false,
            span: start.merge(&self.previous.span),
        })
    }

    fn parse_function(&mut self, annotations: Vec<Annotation>) -> Result<Function> {
        let start = self.current.span;
        self.consume(TokenKind::Func, "Expected 'func'")?;
        
        let name = self.consume_identifier("Expected function name")?;
        
        self.consume(TokenKind::LeftParen, "Expected '(' after function name")?;
        let params = self.parse_parameters()?;
        self.consume(TokenKind::RightParen, "Expected ')' after parameters")?;

        let return_type = if self.match_token(TokenKind::Arrow)? {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;

        Ok(Function {
            name,
            params,
            return_type,
            body,
            annotations,
            span: start.merge(&self.previous.span),
        })
    }

    fn parse_parameters(&mut self) -> Result<Vec<Parameter>> {
        let mut params = Vec::new();

        if !self.check(TokenKind::RightParen) {
            loop {
                params.push(self.parse_parameter()?);
                if !self.match_token(TokenKind::Comma)? {
                    break;
                }
            }
        }

        Ok(params)
    }

    fn parse_parameter(&mut self) -> Result<Parameter> {
        let start = self.current.span;
        let name = self.consume_identifier("Expected parameter name")?;

        let mut dimensions = Vec::new();
        while self.match_token(TokenKind::LeftBracket)? {
            let dim = self.parse_expression()?;
            dimensions.push(dim);
            self.consume(TokenKind::RightBracket, "Expected ']' after array dimension")?;
        }

        let ty = if self.match_token(TokenKind::Colon)? {
            self.parse_type()?
        } else if dimensions.is_empty() {
            Type::Unknown
        } else {
            Type::Array {
                element: Box::new(Type::Unknown),
                dimensions: dimensions.iter().map(|_| None).collect(),
            }
        };

        Ok(Parameter {
            name,
            ty,
            dimensions,
            span: start.merge(&self.previous.span),
        })
    }

    fn parse_type(&mut self) -> Result<Type> {
        let base_type = match self.current.kind {
            TokenKind::Int => { self.advance()?; Type::Int }
            TokenKind::FloatType => { self.advance()?; Type::Float }
            TokenKind::Double => { self.advance()?; Type::Double }
            TokenKind::Bool => { self.advance()?; Type::Bool }
            _ => bail!("Expected type, found {:?}", self.current.kind),
        };

        if self.check(TokenKind::LeftBracket) {
            let mut dimensions = Vec::new();
            while self.match_token(TokenKind::LeftBracket)? {
                if self.check(TokenKind::RightBracket) {
                    dimensions.push(None);
                } else if let TokenKind::Integer = self.current.kind {
                    let size: i64 = self.current.lexeme.parse()
                        .map_err(|_| anyhow!("Invalid array size"))?;
                    self.advance()?;
                    dimensions.push(Some(size));
                } else {
                    dimensions.push(None);
                }
                self.consume(TokenKind::RightBracket, "Expected ']'")?;
            }
            Ok(Type::Array { element: Box::new(base_type), dimensions })
        } else {
            Ok(base_type)
        }
    }

    fn parse_block(&mut self) -> Result<Block> {
        let start = self.current.span;
        self.consume(TokenKind::LeftBrace, "Expected '{'")?;

        let mut statements = Vec::new();
        while !self.check(TokenKind::RightBrace) && !self.is_at_end() {
            match self.parse_statement() {
                Ok(stmt) => statements.push(stmt),
                Err(e) => {
                    self.errors.push(self.make_parse_error(&e.to_string()));
                    self.synchronize_statement();
                }
            }
        }

        self.consume(TokenKind::RightBrace, "Expected '}'")?;

        Ok(Block {
            statements,
            span: start.merge(&self.previous.span),
        })
    }

    fn parse_statement(&mut self) -> Result<Stmt> {
        let annotations = self.parse_annotations()?;
        let start = self.current.span;

        let kind = match self.current.kind {
            TokenKind::For => self.parse_for_statement(annotations.iter().any(|a| a.name == "parallel"))?,
            TokenKind::If => {
                if !annotations.is_empty() { bail!("Annotations not allowed on if statements"); }
                self.parse_if_statement()?
            }
            TokenKind::While => {
                if !annotations.is_empty() { bail!("Annotations not allowed on while statements"); }
                self.parse_while_statement()?
            }
            TokenKind::Let | TokenKind::Var => {
                if !annotations.is_empty() { bail!("Annotations not allowed on declarations"); }
                self.parse_declaration()?
            }
            TokenKind::Return => {
                if !annotations.is_empty() { bail!("Annotations not allowed on return statements"); }
                self.parse_return_statement()?
            }
            TokenKind::LeftBrace => {
                if !annotations.is_empty() { bail!("Annotations not allowed on blocks"); }
                StmtKind::Block { block: self.parse_block()? }
            }
            TokenKind::Semicolon => {
                self.advance()?;
                StmtKind::Empty
            }
            _ => {
                if !annotations.is_empty() { bail!("Annotations not allowed on expression statements"); }
                self.parse_assignment_or_expr_statement()?
            }
        };

        Ok(Stmt {
            kind,
            span: start.merge(&self.previous.span),
            annotations,
        })
    }

    fn parse_for_statement(&mut self, is_parallel: bool) -> Result<StmtKind> {
        self.consume(TokenKind::For, "Expected 'for'")?;
        
        let iterator = self.consume_identifier("Expected loop variable")?;
        self.consume(TokenKind::Equal, "Expected '=' after loop variable")?;
        
        let start = self.parse_expression()?;
        self.consume(TokenKind::To, "Expected 'to' in for loop")?;
        let end = self.parse_expression()?;

        let step = if self.match_token(TokenKind::Step)? {
            Some(self.parse_expression()?)
        } else {
            None
        };

        let body = self.parse_block()?;

        Ok(StmtKind::For { iterator, start, end, step, body, is_parallel })
    }

    fn parse_if_statement(&mut self) -> Result<StmtKind> {
        self.consume(TokenKind::If, "Expected 'if'")?;
        let condition = self.parse_expression()?;
        let then_branch = self.parse_block()?;
        let else_branch = if self.match_token(TokenKind::Else)? {
            Some(self.parse_block()?)
        } else {
            None
        };
        Ok(StmtKind::If { condition, then_branch, else_branch })
    }

    fn parse_while_statement(&mut self) -> Result<StmtKind> {
        self.consume(TokenKind::While, "Expected 'while'")?;
        let condition = self.parse_expression()?;
        let body = self.parse_block()?;
        Ok(StmtKind::While { condition, body })
    }

    fn parse_declaration(&mut self) -> Result<StmtKind> {
        let is_mutable = self.current.kind == TokenKind::Var;
        self.advance()?;

        let name = self.consume_identifier("Expected variable name")?;
        let ty = if self.match_token(TokenKind::Colon)? { Some(self.parse_type()?) } else { None };
        let value = if self.match_token(TokenKind::Equal)? { Some(self.parse_expression()?) } else { None };
        self.consume(TokenKind::Semicolon, "Expected ';' after declaration")?;

        Ok(StmtKind::Declaration { name, ty, value, is_mutable })
    }

    fn parse_return_statement(&mut self) -> Result<StmtKind> {
        self.consume(TokenKind::Return, "Expected 'return'")?;
        let value = if !self.check(TokenKind::Semicolon) { Some(self.parse_expression()?) } else { None };
        self.consume(TokenKind::Semicolon, "Expected ';' after return")?;
        Ok(StmtKind::Return { value })
    }

    fn parse_assignment_or_expr_statement(&mut self) -> Result<StmtKind> {
        let expr = self.parse_expression()?;

        if let Some(op) = self.match_assign_op()? {
            let target = self.expr_to_assign_target(expr)?;
            let value = self.parse_expression()?;
            self.consume(TokenKind::Semicolon, "Expected ';' after assignment")?;
            Ok(StmtKind::Assignment { target, op, value })
        } else {
            self.consume(TokenKind::Semicolon, "Expected ';' after expression")?;
            Ok(StmtKind::Expression { expr })
        }
    }

    fn expr_to_assign_target(&self, expr: Expr) -> Result<AssignTarget> {
        match expr.kind {
            ExprKind::Variable(name) => Ok(AssignTarget::Variable(name)),
            ExprKind::ArrayAccess { array, indices } => {
                if let ExprKind::Variable(name) = array.kind {
                    Ok(AssignTarget::ArrayAccess { array: name, indices })
                } else {
                    bail!("Invalid assignment target")
                }
            }
            _ => bail!("Invalid assignment target"),
        }
    }

    fn match_assign_op(&mut self) -> Result<Option<AssignOp>> {
        let op = match self.current.kind {
            TokenKind::Equal => Some(AssignOp::Assign),
            TokenKind::PlusEqual => Some(AssignOp::AddAssign),
            TokenKind::MinusEqual => Some(AssignOp::SubAssign),
            TokenKind::StarEqual => Some(AssignOp::MulAssign),
            TokenKind::SlashEqual => Some(AssignOp::DivAssign),
            _ => None,
        };
        if op.is_some() { self.advance()?; }
        Ok(op)
    }

    // Expression parsing with precedence climbing
    fn parse_expression(&mut self) -> Result<Expr> { self.parse_or_expr() }

    fn parse_or_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_and_expr()?;
        while self.match_token(TokenKind::PipePipe)? || self.match_token(TokenKind::Or)? {
            let right = self.parse_and_expr()?;
            let span = left.span.merge(&right.span);
            left = Expr::new(ExprKind::Binary { op: BinaryOp::Or, left: Box::new(left), right: Box::new(right) }, span);
        }
        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_equality_expr()?;
        while self.match_token(TokenKind::AmpAmp)? || self.match_token(TokenKind::And)? {
            let right = self.parse_equality_expr()?;
            let span = left.span.merge(&right.span);
            left = Expr::new(ExprKind::Binary { op: BinaryOp::And, left: Box::new(left), right: Box::new(right) }, span);
        }
        Ok(left)
    }

    fn parse_equality_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_comparison_expr()?;
        loop {
            let op = match self.current.kind {
                TokenKind::EqualEqual => BinaryOp::Eq,
                TokenKind::BangEqual => BinaryOp::Ne,
                _ => break,
            };
            self.advance()?;
            let right = self.parse_comparison_expr()?;
            let span = left.span.merge(&right.span);
            left = Expr::new(ExprKind::Binary { op, left: Box::new(left), right: Box::new(right) }, span);
        }
        Ok(left)
    }

    fn parse_comparison_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_additive_expr()?;
        loop {
            let op = match self.current.kind {
                TokenKind::Less => BinaryOp::Lt,
                TokenKind::LessEqual => BinaryOp::Le,
                TokenKind::Greater => BinaryOp::Gt,
                TokenKind::GreaterEqual => BinaryOp::Ge,
                _ => break,
            };
            self.advance()?;
            let right = self.parse_additive_expr()?;
            let span = left.span.merge(&right.span);
            left = Expr::new(ExprKind::Binary { op, left: Box::new(left), right: Box::new(right) }, span);
        }
        Ok(left)
    }

    fn parse_additive_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_multiplicative_expr()?;
        loop {
            let op = match self.current.kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance()?;
            let right = self.parse_multiplicative_expr()?;
            let span = left.span.merge(&right.span);
            left = Expr::new(ExprKind::Binary { op, left: Box::new(left), right: Box::new(right) }, span);
        }
        Ok(left)
    }

    fn parse_multiplicative_expr(&mut self) -> Result<Expr> {
        let mut left = self.parse_unary_expr()?;
        loop {
            let op = match self.current.kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent | TokenKind::Mod => BinaryOp::Mod,
                _ => break,
            };
            self.advance()?;
            let right = self.parse_unary_expr()?;
            let span = left.span.merge(&right.span);
            left = Expr::new(ExprKind::Binary { op, left: Box::new(left), right: Box::new(right) }, span);
        }
        Ok(left)
    }

    fn parse_unary_expr(&mut self) -> Result<Expr> {
        let start = self.current.span;
        match self.current.kind {
            TokenKind::Minus => {
                self.advance()?;
                let operand = self.parse_unary_expr()?;
                let span = start.merge(&operand.span);
                Ok(Expr::new(ExprKind::Unary { op: UnaryOp::Neg, operand: Box::new(operand) }, span))
            }
            TokenKind::Bang | TokenKind::Not => {
                self.advance()?;
                let operand = self.parse_unary_expr()?;
                let span = start.merge(&operand.span);
                Ok(Expr::new(ExprKind::Unary { op: UnaryOp::Not, operand: Box::new(operand) }, span))
            }
            _ => self.parse_primary_expr(),
        }
    }

    fn parse_primary_expr(&mut self) -> Result<Expr> {
        let start = self.current.span;

        match self.current.kind {
            TokenKind::Integer => {
                let value: i64 = self.current.lexeme.parse().map_err(|_| anyhow!("Invalid integer"))?;
                self.advance()?;
                Ok(Expr::int_lit(value, start))
            }
            TokenKind::Float => {
                let value: f64 = self.current.lexeme.parse().map_err(|_| anyhow!("Invalid float"))?;
                self.advance()?;
                Ok(Expr::float_lit(value, start))
            }
            TokenKind::True => { self.advance()?; Ok(Expr::new(ExprKind::BoolLiteral(true), start)) }
            TokenKind::False => { self.advance()?; Ok(Expr::new(ExprKind::BoolLiteral(false), start)) }
            TokenKind::String => {
                let value = self.current.lexeme[1..self.current.lexeme.len()-1].to_string();
                self.advance()?;
                Ok(Expr::new(ExprKind::StringLiteral(value), start))
            }
            TokenKind::Min => {
                self.advance()?;
                self.consume(TokenKind::LeftParen, "Expected '('")?;
                let a = self.parse_expression()?;
                self.consume(TokenKind::Comma, "Expected ','")?;
                let b = self.parse_expression()?;
                self.consume(TokenKind::RightParen, "Expected ')'")?;
                Ok(Expr::new(ExprKind::Min(Box::new(a), Box::new(b)), start.merge(&self.previous.span)))
            }
            TokenKind::Max => {
                self.advance()?;
                self.consume(TokenKind::LeftParen, "Expected '('")?;
                let a = self.parse_expression()?;
                self.consume(TokenKind::Comma, "Expected ','")?;
                let b = self.parse_expression()?;
                self.consume(TokenKind::RightParen, "Expected ')'")?;
                Ok(Expr::new(ExprKind::Max(Box::new(a), Box::new(b)), start.merge(&self.previous.span)))
            }
            TokenKind::Identifier => {
                let name = self.current.lexeme.clone();
                self.advance()?;
                if self.check(TokenKind::LeftParen) {
                    self.advance()?;
                    let args = self.parse_args()?;
                    self.consume(TokenKind::RightParen, "Expected ')'")?;
                    Ok(Expr::new(ExprKind::Call { function: name, args }, start.merge(&self.previous.span)))
                } else if self.check(TokenKind::LeftBracket) {
                    let mut indices = Vec::new();
                    while self.match_token(TokenKind::LeftBracket)? {
                        indices.push(self.parse_expression()?);
                        self.consume(TokenKind::RightBracket, "Expected ']'")?;
                    }
                    Ok(Expr::new(ExprKind::ArrayAccess { array: Box::new(Expr::var(name, start)), indices }, start.merge(&self.previous.span)))
                } else {
                    Ok(Expr::var(name, start))
                }
            }
            TokenKind::LeftParen => {
                self.advance()?;
                let inner = self.parse_expression()?;
                self.consume(TokenKind::RightParen, "Expected ')'")?;
                Ok(Expr::new(ExprKind::Grouped(Box::new(inner)), start.merge(&self.previous.span)))
            }
            _ => bail!("Unexpected token: {:?}", self.current.kind),
        }
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>> {
        let mut args = Vec::new();
        if !self.check(TokenKind::RightParen) {
            loop {
                args.push(self.parse_expression()?);
                if !self.match_token(TokenKind::Comma)? { break; }
            }
        }
        Ok(args)
    }

    // Helper methods
    fn check(&self, kind: TokenKind) -> bool { self.current.kind == kind }
    fn is_at_end(&self) -> bool { self.current.kind == TokenKind::Eof }

    fn advance(&mut self) -> Result<&Token> {
        self.previous = self.current.clone();
        self.current = self.lexer.next_token().map_err(|e| anyhow!("Lexer error: {}", e))?;
        Ok(&self.previous)
    }

    fn consume(&mut self, kind: TokenKind, message: &str) -> Result<&Token> {
        if self.check(kind) { self.advance() }
        else { bail!("{}: expected {:?}, found {:?}", message, kind, self.current.kind) }
    }

    fn consume_identifier(&mut self, message: &str) -> Result<String> {
        if self.check(TokenKind::Identifier) {
            let name = self.current.lexeme.clone();
            self.advance()?;
            Ok(name)
        } else {
            bail!("{}: expected identifier, found {:?}", message, self.current.kind)
        }
    }

    fn match_token(&mut self, kind: TokenKind) -> Result<bool> {
        if self.check(kind) { self.advance()?; Ok(true) } else { Ok(false) }
    }

    fn make_parse_error(&self, message: &str) -> ParseError {
        ParseError {
            message: message.to_string(),
            span: self.current.span,
            kind: ParseErrorKind::UnexpectedToken,
            expected: Vec::new(),
            found: Some(format!("{:?}", self.current.kind)),
        }
    }

    fn synchronize(&mut self) {
        self.panic_mode = false;
        
        // Always advance at least once to avoid infinite loops
        if !self.is_at_end() {
            let _ = self.advance();
        }
        
        while !self.is_at_end() {
            // Stop at semicolon or closing brace
            if self.previous.kind == TokenKind::Semicolon { return; }
            if self.previous.kind == TokenKind::RightBrace { return; }
            
            // Stop at top-level keywords (function/const declarations)
            match self.current.kind {
                TokenKind::Func | TokenKind::Const => return,
                _ => {}
            }
            let _ = self.advance();
        }
    }

    fn synchronize_statement(&mut self) {
        while !self.is_at_end() && !self.check(TokenKind::RightBrace) {
            if self.previous.kind == TokenKind::Semicolon { return; }
            match self.current.kind {
                TokenKind::For | TokenKind::If | TokenKind::While |
                TokenKind::Return | TokenKind::Let | TokenKind::Var => return,
                _ => {}
            }
            let _ = self.advance();
        }
    }
}

enum TopLevel { Function(Function), Global(GlobalDecl) }

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Result<Program> {
        let lexer = Lexer::new(source);
        let mut parser = Parser::new(lexer)?;
        parser.parse_program()
    }

    #[test]
    fn test_empty_function() {
        let program = parse("func test() {}").unwrap();
        assert_eq!(program.functions.len(), 1);
        assert_eq!(program.functions[0].name, "test");
    }

    #[test]
    fn test_function_with_params() {
        let program = parse("func test(x, y, z) {}").unwrap();
        assert_eq!(program.functions[0].params.len(), 3);
    }

    #[test]
    fn test_array_params() {
        let program = parse("func test(A[N][M], B[K]) {}").unwrap();
        assert_eq!(program.functions[0].params[0].dimensions.len(), 2);
        assert_eq!(program.functions[0].params[1].dimensions.len(), 1);
    }

    #[test]
    fn test_for_loop() {
        let source = "func test() { for i = 0 to N { A[i] = i; } }";
        parse(source).unwrap();
    }

    #[test]
    fn test_nested_loops() {
        let source = "func test() { for i = 0 to N { for j = 0 to M { A[i][j] = 0; } } }";
        parse(source).unwrap();
    }

    #[test]
    fn test_matmul() {
        let source = r#"
            func matmul(A[N][K], B[K][M], C[N][M]) {
                for i = 0 to N {
                    for j = 0 to M {
                        C[i][j] = 0;
                        for k = 0 to K {
                            C[i][j] = C[i][j] + A[i][k] * B[k][j];
                        }
                    }
                }
            }
        "#;
        let program = parse(source).unwrap();
        assert_eq!(program.functions[0].name, "matmul");
    }

    #[test]
    fn test_parallel_annotation() {
        let source = "func test() { @parallel for i = 0 to N { A[i] = i; } }";
        let program = parse(source).unwrap();
        if let StmtKind::For { is_parallel, .. } = &program.functions[0].body.statements[0].kind {
            assert!(*is_parallel);
        } else {
            panic!("Expected for loop");
        }
    }
}
