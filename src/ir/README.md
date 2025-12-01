# Intermediate Representation (IR) Module

## Overview

The **IR module** transforms the frontend AST into representations suitable for polyhedral analysis and optimization. This is where we move from "what the programmer wrote" to "what operations happen and in what order."

```
Frontend AST
     │
     ▼
┌─────────┐
│   HIR   │  High-level IR (close to source)
└─────────┘
     │
     ▼
┌─────────┐
│   PIR   │  Polyhedral IR (mathematical representation)
└─────────┘
     │
     ▼
Polyhedral Analysis & Transformation
```

---

## Why Multiple IRs?

Different representations are good for different things:

| IR | Good For | Structure |
|----|----------|-----------|
| **AST** | Parsing, error messages | Tree (syntax-focused) |
| **HIR** | Basic optimization, inlining | Tree (semantics-focused) |
| **PIR** | Loop optimization, parallelization | Mathematical (sets & maps) |

---

## Files in This Module

| File | Purpose |
|------|---------|
| `hir.rs` | High-level IR definition |
| `pir.rs` | Polyhedral IR definition |
| `lower_ast.rs` | AST → HIR transformation |
| `lower_hir.rs` | HIR → PIR transformation |
| `mod.rs` | Module exports |

---

## 1. High-Level IR (`hir.rs`)

### What is HIR?

HIR is a **simplified, normalized** version of the AST:
- Syntactic sugar is removed
- Complex expressions are broken down
- Control flow is explicit

### HIR Structure

```rust
/// A complete HIR program
pub struct HirProgram {
    pub name: String,
    pub parameters: Vec<Parameter>,  // N, M, etc.
    pub arrays: Vec<ArrayDecl>,      // A[N][M], B[N], etc.
    pub statements: Vec<HirStmt>,    // The actual computation
}

/// A statement in HIR
pub enum HirStmt {
    /// A loop: for i = lb to ub { body }
    Loop {
        iterator: String,
        lower_bound: HirExpr,
        upper_bound: HirExpr,
        body: Vec<HirStmt>,
    },
    
    /// An assignment: target = value
    Assignment {
        target: ArrayRef,
        value: HirExpr,
    },
    
    /// Conditional: if (cond) { then } else { else }
    If {
        condition: HirExpr,
        then_body: Vec<HirStmt>,
        else_body: Vec<HirStmt>,
    },
}

/// Array reference: A[i][j+1]
pub struct ArrayRef {
    pub name: String,
    pub indices: Vec<HirExpr>,
}

/// HIR expressions (simplified from AST)
pub enum HirExpr {
    Constant(i64),
    Parameter(String),      // N, M
    Iterator(String),       // i, j, k
    ArrayAccess(ArrayRef),
    BinaryOp {
        op: BinOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
    },
}
```

### Why HIR?

HIR normalizes different ways of writing the same thing:

```c
// All of these become the same HIR:
for (i = 0; i < N; i++)      // C-style
for (int i = 0; i < N; i++)  // C99 declaration
for i = 0 to N-1             // Our DSL style
```

### HIR Example

Source:
```c
for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}
```

HIR:
```
Loop {
    iterator: "i"
    lower_bound: Constant(0)
    upper_bound: Parameter("N")
    body: [
        Loop {
            iterator: "j"
            lower_bound: Constant(0)
            upper_bound: Parameter("M")
            body: [
                Assignment {
                    target: ArrayRef("C", [Iterator("i"), Iterator("j")])
                    value: BinaryOp(Add,
                        ArrayAccess("A", [Iterator("i"), Iterator("j")]),
                        ArrayAccess("B", [Iterator("i"), Iterator("j")])
                    )
                }
            ]
        }
    ]
}
```

---

## 2. Polyhedral IR (`pir.rs`)

### What is Polyhedral IR?

PIR represents loops and array accesses **mathematically**:
- Loop bounds become **integer sets** (polyhedra)
- Array accesses become **affine maps**
- Execution order becomes a **schedule**

This mathematical representation enables powerful analysis and optimization.

### Core Concepts

#### Iteration Domain
The set of all loop iterations, described by constraints:

```
for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
```

Iteration domain: `{ [i, j] : 0 ≤ i < N ∧ 0 ≤ j < M }`

This is a **polyhedron** (a shape bounded by linear constraints).

#### Access Function
How array indices relate to loop iterators:

```c
A[i][j+1]  →  access function: (i, j) → (i, j+1)
A[2*i][j]  →  access function: (i, j) → (2i, j)
```

Access functions must be **affine** (linear + constant):
- ✓ `A[i+1][2*j+5]` — affine
- ✗ `A[i*j]` — not affine (product of iterators)
- ✗ `A[i^2]` — not affine (exponent)

#### Schedule
When each operation executes. A schedule maps iterations to time:

```
Original: S(i,j) → (i, j)      // Execute in original order
Tiled:    S(i,j) → (i/32, j/32, i%32, j%32)  // Execute in tiles
```

### PIR Data Structures

```rust
/// A complete polyhedral program
pub struct PolyProgram {
    pub name: String,
    pub parameters: Vec<String>,        // Symbolic constants: N, M
    pub param_constraints: Vec<Constraint>,  // N > 0, M > 0
    pub statements: Vec<Statement>,     // Computational statements
}

/// A single statement in the program
pub struct Statement {
    pub id: StmtId,
    pub domain: IterationDomain,        // When this executes
    pub schedule: Schedule,             // In what order
    pub accesses: Vec<Access>,          // What arrays it touches
    pub expression: String,             // The computation
}

/// Iteration domain: the set of all iterations
pub struct IterationDomain {
    pub iterators: Vec<String>,         // [i, j, k]
    pub constraints: Vec<Constraint>,   // 0 <= i < N, etc.
}

/// An array access
pub struct Access {
    pub kind: AccessKind,               // Read or Write
    pub array: String,                  // Array name
    pub subscripts: Vec<AffineExpr>,    // Index expressions
}

/// Schedule: maps iterations to execution time
pub struct Schedule {
    pub dimensions: Vec<ScheduleDim>,
}

pub enum ScheduleDim {
    /// Follow a loop iterator
    Iterator(String),
    /// Constant offset (for statement ordering)
    Constant(i64),
}
```

### PIR Example

Source:
```c
for (i = 0; i < N; i++) {
    S0: A[i] = B[i] + 1;
}
```

PIR:
```
Statement S0:
    Domain: { [i] : 0 ≤ i < N }
    Schedule: [i] → (i)
    Accesses:
        - Write: A[i]  →  (i) → (i)
        - Read:  B[i]  →  (i) → (i)
    Expression: "A[i] = B[i] + 1"
```

---

## 3. AST to HIR Lowering (`lower_ast.rs`)

### What is Lowering?

**Lowering** transforms a higher-level representation into a lower-level one. AST → HIR lowering:
- Normalizes loop forms
- Extracts array declarations
- Simplifies expressions

### Key Transformations

#### 1. Loop Normalization

Convert all loop forms to standard form:

```c
// Input (various forms)
for (i = 5; i < N+3; i += 2)    // Start at 5, step by 2
for (int i = N-1; i >= 0; i--)  // Countdown

// Output (normalized)
Loop { iterator: "i", lb: 0, ub: (N-5+1)/2, step: 1 }  // Normalized range
Loop { iterator: "i", lb: 0, ub: N, step: 1 }          // Reversed and normalized
```

#### 2. Expression Simplification

Break complex expressions into simpler forms:

```c
// Input
A[i][j] = B[i][j] * C[i][j] + D[i][j] * E[i][j];

// Could be lowered to (for clarity, though we keep it as one):
t1 = B[i][j] * C[i][j];
t2 = D[i][j] * E[i][j];
A[i][j] = t1 + t2;
```

#### 3. Array Declaration Extraction

```rust
fn lower_program(ast: &Program) -> HirProgram {
    let mut arrays = Vec::new();
    let mut parameters = Vec::new();
    
    // Extract all array declarations
    for decl in &ast.declarations {
        match decl {
            Declaration::Array { name, dims, dtype } => {
                arrays.push(ArrayDecl { name, dims, dtype });
            }
            Declaration::Parameter { name, dtype } => {
                parameters.push(Parameter { name, dtype });
            }
        }
    }
    
    // Lower statements
    let statements = ast.body.iter()
        .map(|stmt| lower_stmt(stmt))
        .collect();
    
    HirProgram { parameters, arrays, statements }
}
```

---

## 4. HIR to PIR Lowering (`lower_hir.rs`)

### The Key Transformation

This is where we extract the **polyhedral model** from imperative code.

### Step 1: Extract Iteration Domain

For each loop nest, build the constraint system:

```rust
fn extract_domain(loop_nest: &[HirLoop]) -> IterationDomain {
    let mut iterators = Vec::new();
    let mut constraints = Vec::new();
    
    for loop_info in loop_nest {
        iterators.push(loop_info.iterator.clone());
        
        // Lower bound: iterator >= lower_bound
        constraints.push(Constraint::ge(
            AffineExpr::var(&loop_info.iterator),
            lower_to_affine(&loop_info.lower_bound)
        ));
        
        // Upper bound: iterator < upper_bound
        constraints.push(Constraint::lt(
            AffineExpr::var(&loop_info.iterator),
            lower_to_affine(&loop_info.upper_bound)
        ));
    }
    
    IterationDomain { iterators, constraints }
}
```

### Step 2: Extract Access Functions

For each array access, extract the affine subscript functions:

```rust
fn extract_access(array_ref: &ArrayRef, iterators: &[String]) -> Access {
    let subscripts: Vec<AffineExpr> = array_ref.indices.iter()
        .map(|idx| hir_expr_to_affine(idx, iterators))
        .collect();
    
    Access {
        array: array_ref.name.clone(),
        subscripts,
    }
}

fn hir_expr_to_affine(expr: &HirExpr, iterators: &[String]) -> AffineExpr {
    match expr {
        HirExpr::Constant(c) => AffineExpr::constant(*c),
        HirExpr::Iterator(name) => AffineExpr::var(name),
        HirExpr::Parameter(name) => AffineExpr::param(name),
        HirExpr::BinaryOp { op: Add, left, right } => {
            hir_expr_to_affine(left, iterators) + hir_expr_to_affine(right, iterators)
        }
        HirExpr::BinaryOp { op: Mul, left, right } => {
            // One operand must be constant for affine
            // e.g., 2*i is affine, i*j is not
            extract_affine_multiply(left, right, iterators)
        }
        _ => panic!("Non-affine expression"),
    }
}
```

### Step 3: Build Initial Schedule

The initial schedule just reflects the original execution order:

```rust
fn build_initial_schedule(domain: &IterationDomain, stmt_index: usize) -> Schedule {
    let mut dims = Vec::new();
    
    for iterator in &domain.iterators {
        dims.push(ScheduleDim::Iterator(iterator.clone()));
    }
    
    // Add statement index to order statements at same iteration
    dims.push(ScheduleDim::Constant(stmt_index as i64));
    
    Schedule { dimensions: dims }
}
```

### Complete Lowering Example

Input HIR:
```
Loop(i, 0, N) {
    Loop(j, 0, M) {
        S0: C[i][j] = A[i][j] + B[i][j]
    }
}
```

Output PIR:
```
Statement S0:
    Domain: { [i,j] : 0 ≤ i < N ∧ 0 ≤ j < M }
    
    Schedule: (i, j) → (i, j, 0)
    
    Accesses:
        Write C: (i,j) → (i, j)     -- C[i][j]
        Read  A: (i,j) → (i, j)     -- A[i][j]  
        Read  B: (i,j) → (i, j)     -- B[i][j]
    
    Expression: "C[i][j] = A[i][j] + B[i][j]"
```

---

## Why the Polyhedral Model?

### Traditional Compiler View
```
for i = 0 to N:
    for j = 0 to M:
        S1
        S2
```
The compiler sees: "Execute S1, then S2, for each (i,j) in order."

### Polyhedral View
```
S1: Domain = {[i,j] : 0≤i<N, 0≤j<M}, Schedule = (i,j,0)
S2: Domain = {[i,j] : 0≤i<N, 0≤j<M}, Schedule = (i,j,1)
```
The compiler sees: "S1 and S2 execute on these integer points, in this order."

### What This Enables

1. **Loop interchange**: Change schedule from (i,j) to (j,i)
2. **Tiling**: Change schedule from (i,j) to (i/32, j/32, i%32, j%32)
3. **Fusion**: Merge two loops by unifying their schedules
4. **Parallelization**: Find which iterations can run simultaneously
5. **Dependence analysis**: Precisely compute what must happen before what

---

## Affine Expressions Deep Dive

### What Makes Something Affine?

An expression is **affine** if it's a linear combination of variables plus a constant:

```
f(x₁, x₂, ..., xₙ) = a₁x₁ + a₂x₂ + ... + aₙxₙ + c
```

Where `aᵢ` and `c` are constants (or symbolic parameters).

### Examples

| Expression | Affine? | Why? |
|------------|---------|------|
| `i + j` | ✓ | Linear combination |
| `2*i + 3*j - 5` | ✓ | Linear with constant |
| `i + N` | ✓ | N is a parameter (constant per execution) |
| `i * j` | ✗ | Product of two variables |
| `i / 2` | ✓* | Integer division by constant (*with floor) |
| `i % 32` | ✓* | Modulo by constant (*requires special handling) |
| `i * i` | ✗ | Quadratic |
| `A[i]` | ✗ | Array value (unknown) |

### Why Only Affine?

Polyhedral analysis relies on **Integer Linear Programming (ILP)**:
- Affine constraints define polyhedra
- ILP can solve systems of affine constraints exactly
- Non-affine → undecidable in general

---

## Iteration Domain Visualization

Consider:
```c
for (i = 0; i < 4; i++)
    for (j = i; j < 4; j++)
        S(i, j);
```

Domain: `{ [i,j] : 0 ≤ i < 4 ∧ i ≤ j < 4 }`

Visualized:
```
j
4 │        
3 │  ●  ●  ●  ●
2 │  ●  ●  ●
1 │  ●  ●
0 │  ●
  └──────────── i
     0  1  2  3
```

Each `●` is one iteration of the loop. The shape is a triangle (a polyhedron!).

---

## Key Takeaways

1. **HIR** simplifies and normalizes the AST
2. **PIR** represents loops mathematically as integer sets
3. **Lowering** extracts the polyhedral model from imperative code
4. **Affine expressions** are required for exact analysis
5. The polyhedral representation enables powerful loop transformations

## Further Reading

- [The Polyhedral Model](https://polyhedral.info/) - Community resources
- [Polly - LLVM's Polyhedral Optimizer](https://polly.llvm.org/)
- [Integer Set Library (ISL)](https://libisl.sourceforge.io/manual.pdf) - Mathematical foundation
- "Optimizing Compilers for Modern Architectures" by Allen & Kennedy