# Frontend Module

## Overview

The **frontend** is the first stage of any compiler. It takes human-readable source code and transforms it into a structured representation (Abstract Syntax Tree) that the rest of the compiler can work with.

```
Source Code (.poly file)
        │
        ▼
    ┌────────┐
    │ Lexer  │  ──► Breaks text into tokens
    └────────┘
        │
        ▼
    ┌────────┐
    │ Parser │  ──► Builds tree structure from tokens
    └────────┘
        │
        ▼
    ┌──────────┐
    │ Semantic │  ──► Checks types, scopes, validity
    │ Analysis │
    └──────────┘
        │
        ▼
   Abstract Syntax Tree (AST)
```

---

## Files in This Module

| File | Purpose |
|------|---------|
| `token.rs` | Defines all token types (keywords, operators, etc.) |
| `lexer.rs` | Converts source text into a stream of tokens |
| `ast.rs` | Defines the Abstract Syntax Tree data structures |
| `parser.rs` | Builds AST from token stream |
| `semantic.rs` | Type checking and scope analysis |
| `mod.rs` | Module exports |

---

## 1. Tokens (`token.rs`)

### What is a Token?

A **token** is the smallest meaningful unit in source code. The lexer breaks source code into tokens.

**Example:**
```
for (i = 0; i < N; i++)
```
becomes:
```
[FOR] [LPAREN] [IDENT "i"] [ASSIGN] [INT 0] [SEMICOLON] 
[IDENT "i"] [LT] [IDENT "N"] [SEMICOLON] [IDENT "i"] [PLUSPLUS] [RPAREN]
```

### Token Categories

```rust
pub enum TokenKind {
    // === Literals ===
    Integer(i64),      // 42, -7, 1000
    Float(f64),        // 3.14, -0.5
    Identifier(String), // i, N, matrix, myVar
    
    // === Keywords ===
    For,        // for
    If,         // if
    Else,       // else
    While,      // while
    Return,     // return
    Int,        // int
    Float,      // float (type)
    Double,     // double
    Void,       // void
    
    // === Operators ===
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    
    // === Comparison ===
    Lt,         // <
    Le,         // <=
    Gt,         // >
    Ge,         // >=
    Eq,         // ==
    Ne,         // !=
    
    // === Assignment ===
    Assign,     // =
    PlusAssign, // +=
    MinusAssign,// -=
    
    // === Delimiters ===
    LParen,     // (
    RParen,     // )
    LBrace,     // {
    RBrace,     // }
    LBracket,   // [
    RBracket,   // ]
    Semicolon,  // ;
    Comma,      // ,
    
    // === Special ===
    Eof,        // End of file
}
```

### Why Tokens Matter

Tokens abstract away irrelevant details:
- Whitespace is ignored (except to separate tokens)
- Comments are stripped out
- The parser doesn't care if you wrote `for` or `  for  `

---

## 2. Lexer (`lexer.rs`)

### What is a Lexer?

A **lexer** (also called scanner or tokenizer) reads source code character by character and groups them into tokens.

### How It Works

```
Input: "for (i = 0; i < N; i++)"

Lexer state machine:
┌──────────────────────────────────────────────────────┐
│ Read 'f' → might be identifier or keyword            │
│ Read 'o' → still building                            │
│ Read 'r' → still building                            │
│ Read ' ' → word complete! "for" = FOR keyword        │
│ Read '(' → single char token = LPAREN                │
│ Read 'i' → start identifier                          │
│ Read ' ' → identifier complete = IDENT("i")          │
│ Read '=' → could be = or ==                          │
│ Read ' ' → just = → ASSIGN                           │
│ ... and so on                                        │
└──────────────────────────────────────────────────────┘
```

### Key Lexer Concepts

#### 1. Lookahead
Sometimes we need to peek at the next character to decide what token we have:

```rust
// Is this '=' or '==' ?
fn scan_equals(&mut self) -> Token {
    if self.peek() == '=' {
        self.advance();
        Token::Eq  // ==
    } else {
        Token::Assign  // =
    }
}
```

#### 2. Longest Match
The lexer always tries to make the longest possible token:
- `<=` is one token (LE), not two tokens (LT, ASSIGN)
- `for` is a keyword, not three identifiers

#### 3. Keyword vs Identifier
After scanning a word, check if it's a reserved keyword:

```rust
fn identifier_or_keyword(&self, word: &str) -> TokenKind {
    match word {
        "for" => TokenKind::For,
        "if" => TokenKind::If,
        "while" => TokenKind::While,
        "int" => TokenKind::Int,
        // ... more keywords
        _ => TokenKind::Identifier(word.to_string()),
    }
}
```

### Error Handling

The lexer catches:
- Invalid characters: `@`, `$` (not in our language)
- Unterminated strings: `"hello` (missing closing quote)
- Invalid numbers: `3.14.15` (two decimal points)

---

## 3. Abstract Syntax Tree (`ast.rs`)

### What is an AST?

An **Abstract Syntax Tree** represents the hierarchical structure of source code. Unlike the flat token stream, an AST shows how parts relate to each other.

### Example

Source code:
```c
C[i][j] = A[i][k] + B[k][j];
```

AST representation:
```
        Assignment
        /        \
    Access      BinaryOp(+)
    /    \      /        \
   C    [i,j] Access    Access
              /    \    /    \
             A   [i,k] B   [k,j]
```

### Key AST Node Types

```rust
/// A statement in the program
pub enum Stmt {
    /// Variable declaration: int x = 5;
    Declaration {
        dtype: DataType,
        name: String,
        init: Option<Expr>,
    },
    
    /// Assignment: x = expr;
    Assignment {
        target: Expr,
        value: Expr,
    },
    
    /// For loop: for (init; cond; update) body
    For {
        init: Box<Stmt>,
        condition: Expr,
        update: Box<Stmt>,
        body: Vec<Stmt>,
    },
    
    /// If statement: if (cond) then_body else else_body
    If {
        condition: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
    },
    
    /// Block of statements: { stmt1; stmt2; }
    Block(Vec<Stmt>),
}

/// An expression that produces a value
pub enum Expr {
    /// Integer literal: 42
    IntLit(i64),
    
    /// Float literal: 3.14
    FloatLit(f64),
    
    /// Variable reference: x
    Var(String),
    
    /// Array access: A[i][j]
    ArrayAccess {
        array: Box<Expr>,
        indices: Vec<Expr>,
    },
    
    /// Binary operation: a + b
    BinaryOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    
    /// Unary operation: -x, !flag
    UnaryOp {
        op: UnOp,
        operand: Box<Expr>,
    },
    
    /// Function call: foo(a, b)
    Call {
        name: String,
        args: Vec<Expr>,
    },
}
```

### Why Trees?

Trees naturally represent:
- **Nesting**: Inner loops inside outer loops
- **Precedence**: `a + b * c` parses as `a + (b * c)`
- **Scope**: Variables declared in a block

---

## 4. Parser (`parser.rs`)

### What is a Parser?

A **parser** reads tokens and builds an AST according to the grammar rules of the language.

### Grammar Notation

We describe languages using **grammar rules**:

```
program     → statement*
statement   → for_stmt | if_stmt | assignment | declaration
for_stmt    → 'for' '(' init ';' condition ';' update ')' block
block       → '{' statement* '}'
assignment  → expr '=' expr ';'
expr        → term (('+' | '-') term)*
term        → factor (('*' | '/') factor)*
factor      → INTEGER | IDENTIFIER | '(' expr ')' | array_access
```

### Recursive Descent Parsing

Our parser uses **recursive descent**: each grammar rule becomes a function.

```rust
impl Parser {
    /// Parse a for loop
    fn parse_for(&mut self) -> Result<Stmt> {
        self.expect(Token::For)?;       // consume 'for'
        self.expect(Token::LParen)?;    // consume '('
        
        let init = self.parse_stmt()?;  // parse initialization
        self.expect(Token::Semicolon)?;
        
        let cond = self.parse_expr()?;  // parse condition
        self.expect(Token::Semicolon)?;
        
        let update = self.parse_stmt()?; // parse update
        self.expect(Token::RParen)?;     // consume ')'
        
        let body = self.parse_block()?;  // parse body
        
        Ok(Stmt::For { init, cond, update, body })
    }
}
```

### Operator Precedence

How do we parse `1 + 2 * 3` correctly as `1 + (2 * 3)`?

**Precedence climbing** assigns levels to operators:

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 (lowest) | `\|\|` | left |
| 2 | `&&` | left |
| 3 | `== !=` | left |
| 4 | `< <= > >=` | left |
| 5 | `+ -` | left |
| 6 (highest) | `* / %` | left |

```rust
fn parse_expr(&mut self) -> Expr {
    self.parse_precedence(1)  // Start at lowest precedence
}

fn parse_precedence(&mut self, min_prec: u8) -> Expr {
    let mut left = self.parse_primary();  // Get first operand
    
    while let Some(op) = self.current_binary_op() {
        let prec = op.precedence();
        if prec < min_prec {
            break;  // Operator has lower precedence, stop
        }
        
        self.advance();  // Consume operator
        let right = self.parse_precedence(prec + 1);  // Parse right side
        left = Expr::BinaryOp { op, left, right };
    }
    
    left
}
```

### Error Recovery

Good parsers don't just fail on the first error. They try to recover and find more errors:

```rust
fn parse_stmt(&mut self) -> Result<Stmt> {
    match self.parse_stmt_inner() {
        Ok(stmt) => Ok(stmt),
        Err(e) => {
            self.errors.push(e);
            self.synchronize();  // Skip to next statement
            self.parse_stmt()    // Try again
        }
    }
}

fn synchronize(&mut self) {
    // Skip tokens until we find a statement boundary
    while !self.is_at_end() {
        if self.previous() == Token::Semicolon {
            return;
        }
        match self.current() {
            Token::For | Token::If | Token::While => return,
            _ => self.advance(),
        }
    }
}
```

---

## 5. Semantic Analysis (`semantic.rs`)

### What is Semantic Analysis?

The parser checks **syntax** (structure), but semantic analysis checks **meaning**:

- Are variables declared before use?
- Do types match in operations?
- Are array indices valid?

### Symbol Table

A **symbol table** tracks all declared variables:

```rust
struct SymbolTable {
    scopes: Vec<HashMap<String, Symbol>>,
}

struct Symbol {
    name: String,
    dtype: DataType,
    kind: SymbolKind,  // Variable, Parameter, Array
    dimensions: Vec<Expr>,  // For arrays: [N][M]
}

impl SymbolTable {
    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    
    fn exit_scope(&mut self) {
        self.scopes.pop();
    }
    
    fn lookup(&self, name: &str) -> Option<&Symbol> {
        // Search from innermost to outermost scope
        for scope in self.scopes.iter().rev() {
            if let Some(sym) = scope.get(name) {
                return Some(sym);
            }
        }
        None
    }
}
```

### Type Checking

```rust
fn check_assignment(&mut self, target: &Expr, value: &Expr) -> Result<()> {
    let target_type = self.type_of(target)?;
    let value_type = self.type_of(value)?;
    
    if !target_type.compatible_with(&value_type) {
        return Err(SemanticError::TypeMismatch {
            expected: target_type,
            found: value_type,
        });
    }
    
    Ok(())
}

fn type_of(&self, expr: &Expr) -> Result<DataType> {
    match expr {
        Expr::IntLit(_) => Ok(DataType::Int),
        Expr::FloatLit(_) => Ok(DataType::Double),
        Expr::Var(name) => {
            self.symbols.lookup(name)
                .map(|s| s.dtype.clone())
                .ok_or(SemanticError::UndeclaredVariable(name.clone()))
        }
        Expr::BinaryOp { op, left, right } => {
            let lt = self.type_of(left)?;
            let rt = self.type_of(right)?;
            // Return the "wider" type (double > int)
            Ok(lt.promote(&rt))
        }
        // ...
    }
}
```

### Common Semantic Errors

| Error | Example | Message |
|-------|---------|---------|
| Undeclared variable | `x = 5;` (no declaration) | "Undeclared variable 'x'" |
| Type mismatch | `int x = "hello";` | "Cannot assign string to int" |
| Redeclaration | `int x; int x;` | "Variable 'x' already declared" |
| Wrong argument count | `foo(1, 2)` when foo takes 3 | "Expected 3 arguments, got 2" |

---

## Complete Pipeline Example

Let's trace `for (i = 0; i < N; i++) { A[i] = 0; }` through the frontend:

### 1. Lexing
```
[FOR] [LPAREN] [IDENT "i"] [ASSIGN] [INT 0] [SEMICOLON]
[IDENT "i"] [LT] [IDENT "N"] [SEMICOLON] 
[IDENT "i"] [PLUSPLUS] [RPAREN]
[LBRACE] [IDENT "A"] [LBRACKET] [IDENT "i"] [RBRACKET] 
[ASSIGN] [INT 0] [SEMICOLON] [RBRACE]
```

### 2. Parsing
```
For {
    init: Assignment { target: Var("i"), value: IntLit(0) }
    condition: BinaryOp { op: Lt, left: Var("i"), right: Var("N") }
    update: UnaryOp { op: PostIncrement, operand: Var("i") }
    body: [
        Assignment {
            target: ArrayAccess { array: Var("A"), indices: [Var("i")] }
            value: IntLit(0)
        }
    ]
}
```

### 3. Semantic Analysis
- ✓ Check `i` is declared as loop variable
- ✓ Check `N` is a declared parameter
- ✓ Check `A` is a declared array
- ✓ Check index `i` is integer type
- ✓ Check assignment type (int to double[])

---

## Key Takeaways

1. **Lexer** converts text to tokens (simplifies parsing)
2. **Parser** builds AST using grammar rules (structure)
3. **Semantic analysis** validates meaning (types, scopes)
4. Each phase catches different errors
5. The AST is the foundation for all later compilation phases

## Further Reading

- [Crafting Interpreters](https://craftinginterpreters.com/) - Excellent free book
- [Dragon Book](https://en.wikipedia.org/wiki/Compilers:_Principles,_Techniques,_and_Tools) - Classic compiler textbook
- [LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/) - Building a simple language