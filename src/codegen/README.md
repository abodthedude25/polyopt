# Code Generation Module

## Overview

The **codegen module** transforms the optimized polyhedral representation back into executable code. This is the final stage of the compiler — turning mathematical schedules and domains back into loops and statements.

```
Optimized Polyhedral IR
         │
         ▼
┌─────────────────────┐
│   AST Builder       │  ← Reconstruct loop structure
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   C Code Generator  │  ← Emit C source code
└─────────────────────┘
         │
         ▼
    C Source Code
    (with OpenMP)
```

---

## Files in This Module

| File | Purpose |
|------|---------|
| `ast_builder.rs` | Converts polyhedral representation to loop AST |
| `c.rs` | Generates C code from loop AST |
| `mod.rs` | Module exports |

---

## 1. The Code Generation Problem

### The Challenge

Given:
- Statements with iteration domains (integer sets)
- A schedule (execution order mapping)

Generate:
- Nested loops that execute each statement at the right time

### Why It's Tricky

```
Statement S1: domain = { [i,j] : 0 <= i < N, 0 <= j < M }
Schedule: S1[i,j] → (i+j, i)  // Skewed schedule

How do we generate loops for this?
```

The schedule maps 2D domain to 2D time. We need to:
1. Find loop bounds for the time dimensions
2. Map back from time to original iterators
3. Generate the loop nest

---

## 2. AST Builder (`ast_builder.rs`)

### The AST Structure

```rust
/// A node in the loop AST
pub enum LoopAst {
    /// A for loop
    For {
        iterator: String,
        lower: AstExpr,
        upper: AstExpr,
        step: i64,
        body: Vec<LoopAst>,
    },
    
    /// A statement to execute
    Statement {
        id: StmtId,
        /// Expressions for original iterators in terms of loop variables
        iterator_map: HashMap<String, AstExpr>,
    },
    
    /// Conditional execution
    If {
        condition: AstExpr,
        then_body: Vec<LoopAst>,
        else_body: Vec<LoopAst>,
    },
    
    /// OpenMP parallel region
    Parallel {
        body: Vec<LoopAst>,
    },
}
```

### The Generation Algorithm

The classic approach is **CLooG-style generation**:

```rust
pub fn generate_ast(
    statements: &[Statement],
    schedule: &Schedule,
) -> LoopAst {
    // 1. Combine all statement domains with their schedules
    let mut scheduled_domain = UnionSet::empty();
    for stmt in statements {
        let domain = stmt.domain.apply(&stmt.schedule);
        scheduled_domain = scheduled_domain.union(&domain);
    }
    
    // 2. Generate loops for the scheduled domain
    generate_loops_recursive(&scheduled_domain, statements, 0)
}

fn generate_loops_recursive(
    domain: &UnionSet,
    statements: &[Statement],
    depth: usize,
) -> LoopAst {
    if domain.is_point() {
        // Base case: single point, emit statement
        return emit_statement(domain, statements);
    }
    
    // Find bounds for current dimension
    let (lower, upper) = compute_bounds(domain, depth);
    let iterator = format!("c{}", depth);
    
    // Project out this dimension to get inner domain
    let inner_domain = domain.project_out(depth);
    
    // Recursively generate body
    let body = generate_loops_recursive(&inner_domain, statements, depth + 1);
    
    LoopAst::For {
        iterator,
        lower,
        upper,
        step: 1,
        body: vec![body],
    }
}
```

### Bound Computation

Computing loop bounds from constraints:

```rust
fn compute_bounds(domain: &IntegerSet, dim: usize) -> (AstExpr, AstExpr) {
    let mut lower_bounds = Vec::new();
    let mut upper_bounds = Vec::new();
    
    for constraint in &domain.constraints {
        let coeff = constraint.expr.iter_coeffs[dim];
        
        if coeff > 0 {
            // Constraint: coeff * x + rest >= 0
            // Lower bound: x >= -rest / coeff
            lower_bounds.push(extract_bound(constraint, dim, true));
        } else if coeff < 0 {
            // Constraint: coeff * x + rest >= 0
            // Upper bound: x <= -rest / |coeff|
            upper_bounds.push(extract_bound(constraint, dim, false));
        }
    }
    
    // Take max of lower bounds, min of upper bounds
    let lower = if lower_bounds.is_empty() {
        AstExpr::Constant(0)
    } else {
        AstExpr::Max(lower_bounds)
    };
    
    let upper = if upper_bounds.is_empty() {
        AstExpr::Parameter("N".to_string())
    } else {
        AstExpr::Min(upper_bounds)
    };
    
    (lower, upper)
}
```

### Example: Simple Case

```
Domain: { [i,j] : 0 <= i < N, 0 <= j < M }
Schedule: (i, j) → (i, j)  // Identity

Generated AST:
For(i, 0, N,
    For(j, 0, M,
        Statement(S0, {i→i, j→j})
    )
)
```

### Example: Tiled Schedule

```
Domain: { [i,j] : 0 <= i < N, 0 <= j < M }
Schedule: (i, j) → (i/32, j/32, i%32, j%32)

Generated AST:
For(c0, 0, ceil(N/32),           // Tile row
    For(c1, 0, ceil(M/32),       // Tile column
        For(c2, 0, min(32, N-c0*32),    // Point row
            For(c3, 0, min(32, M-c1*32),  // Point column
                Statement(S0, {i→c0*32+c2, j→c1*32+c3})
            )
        )
    )
)
```

---

## 3. C Code Generator (`c.rs`)

### Overview

The C generator takes the loop AST and produces C source code with optional OpenMP annotations.

```rust
pub struct CCodeGen {
    options: CodeGenOptions,
}

pub struct CodeGenOptions {
    /// Generate OpenMP pragmas
    pub openmp: bool,
    /// Generate timing code
    pub generate_timing: bool,
    /// Generate allocation helpers
    pub generate_alloc: bool,
    /// Indentation string
    pub indent: String,
}
```

### Code Formatter

```rust
pub struct CodeFormatter {
    buffer: String,
    indent_level: usize,
    indent_string: String,
}

impl CodeFormatter {
    pub fn writeln(&mut self, line: &str) {
        for _ in 0..self.indent_level {
            self.buffer.push_str(&self.indent_string);
        }
        self.buffer.push_str(line);
        self.buffer.push('\n');
    }
    
    pub fn indent(&mut self) {
        self.indent_level += 1;
    }
    
    pub fn dedent(&mut self) {
        self.indent_level -= 1;
    }
}
```

### Generating Loops

```rust
impl CCodeGen {
    fn generate_for(&self, f: &mut CodeFormatter, node: &LoopAst) {
        if let LoopAst::For { iterator, lower, upper, step, body } = node {
            // OpenMP pragma for parallel loops
            if self.options.openmp && self.is_parallel_loop(node) {
                f.writeln("#pragma omp parallel for");
            }
            
            // Loop header
            let lower_str = self.expr_to_c(lower);
            let upper_str = self.expr_to_c(upper);
            
            if *step == 1 {
                f.writeln(&format!(
                    "for (int {} = {}; {} < {}; {}++) {{",
                    iterator, lower_str, iterator, upper_str, iterator
                ));
            } else {
                f.writeln(&format!(
                    "for (int {} = {}; {} < {}; {} += {}) {{",
                    iterator, lower_str, iterator, upper_str, iterator, step
                ));
            }
            
            // Body
            f.indent();
            for child in body {
                self.generate_node(f, child);
            }
            f.dedent();
            
            f.writeln("}");
        }
    }
}
```

### Generating Statements

```rust
fn generate_statement(&self, f: &mut CodeFormatter, stmt: &Statement, iter_map: &HashMap<String, AstExpr>) {
    // Substitute loop variables into original statement
    let mut expr = stmt.expression.clone();
    
    for (orig_iter, loop_expr) in iter_map {
        expr = expr.replace(orig_iter, &self.expr_to_c(loop_expr));
    }
    
    f.writeln(&format!("{};", expr));
}
```

### Expression Generation

```rust
fn expr_to_c(&self, expr: &AstExpr) -> String {
    match expr {
        AstExpr::Constant(n) => n.to_string(),
        
        AstExpr::Variable(name) => name.clone(),
        
        AstExpr::Parameter(name) => name.clone(),
        
        AstExpr::BinaryOp { op, left, right } => {
            let l = self.expr_to_c(left);
            let r = self.expr_to_c(right);
            let op_str = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Mod => "%",
            };
            format!("({} {} {})", l, op_str, r)
        }
        
        AstExpr::Min(exprs) => {
            let parts: Vec<_> = exprs.iter().map(|e| self.expr_to_c(e)).collect();
            self.generate_min(&parts)
        }
        
        AstExpr::Max(exprs) => {
            let parts: Vec<_> = exprs.iter().map(|e| self.expr_to_c(e)).collect();
            self.generate_max(&parts)
        }
        
        AstExpr::FloorDiv { num, denom } => {
            format!("FLOOR_DIV({}, {})", self.expr_to_c(num), denom)
        }
        
        AstExpr::CeilDiv { num, denom } => {
            format!("CEIL_DIV({}, {})", self.expr_to_c(num), denom)
        }
    }
}
```

### OpenMP Generation

```rust
impl CCodeGen {
    fn generate_parallel(&self, f: &mut CodeFormatter, body: &[LoopAst]) {
        if self.options.openmp {
            f.writeln("#pragma omp parallel");
            f.writeln("{");
            f.indent();
            
            for node in body {
                self.generate_node(f, node);
            }
            
            f.dedent();
            f.writeln("}");
        } else {
            // Without OpenMP, just generate the body
            for node in body {
                self.generate_node(f, node);
            }
        }
    }
    
    fn is_parallel_loop(&self, node: &LoopAst) -> bool {
        // Check if loop was marked for parallelization
        // (from analysis phase)
        node.metadata.get("parallel").is_some()
    }
}
```

### Generated Headers

```rust
fn generate_headers(&self, f: &mut CodeFormatter) {
    f.writeln("// Generated by PolyOpt - Polyhedral Optimizer");
    f.writeln("");
    f.writeln("#include <stdio.h>");
    f.writeln("#include <stdlib.h>");
    f.writeln("#include <string.h>");
    f.writeln("#include <math.h>");
    
    if self.options.openmp {
        f.writeln("#include <omp.h>");
    }
    
    if self.options.generate_timing {
        f.writeln("#include <time.h>");
        f.writeln("#include <sys/time.h>");
    }
    
    f.writeln("");
    
    // Utility macros
    f.writeln("// Utility macros");
    f.writeln("#define MIN(a, b) ((a) < (b) ? (a) : (b))");
    f.writeln("#define MAX(a, b) ((a) > (b) ? (a) : (b))");
    f.writeln("#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))");
    f.writeln("#define FLOOR_DIV(a, b) ((a) / (b))");
    f.writeln("");
}
```

---

## 4. Complete Example

### Input

```
kernel matmul(N: int, A: double[N][N], B: double[N][N], C: double[N][N]) {
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}
```

### After Tiling + Parallelization

Schedule: `(i,j,k) → (i/32, j/32, k/32, i%32, j%32, k%32)`
Parallel: Outer i-tile loop

### Generated C Code

```c
// Generated by PolyOpt - Polyhedral Optimizer

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define FLOOR_DIV(a, b) ((a) / (b))

void matmul(int N, double* A, double* B, double* C) {
    #pragma omp parallel for
    for (int ii = 0; ii < CEIL_DIV(N, 32); ii++) {
        for (int jj = 0; jj < CEIL_DIV(N, 32); jj++) {
            for (int kk = 0; kk < CEIL_DIV(N, 32); kk++) {
                for (int i = ii * 32; i < MIN((ii + 1) * 32, N); i++) {
                    for (int j = jj * 32; j < MIN((jj + 1) * 32, N); j++) {
                        for (int k = kk * 32; k < MIN((kk + 1) * 32, N); k++) {
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
```

---

## 5. Special Cases

### Triangular Loops

```
Domain: { [i,j] : 0 <= i < N, i <= j < N }

Generated:
for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {  // Note: j starts at i
        ...
    }
}
```

### Conditional Code

When statements have different domains:

```
S1: domain = { [i] : 0 <= i < N, i % 2 == 0 }  // Even i
S2: domain = { [i] : 0 <= i < N, i % 2 == 1 }  // Odd i

Generated:
for (int i = 0; i < N; i++) {
    if (i % 2 == 0) {
        S1(i);
    } else {
        S2(i);
    }
}

// Or better, separate loops:
for (int i = 0; i < N; i += 2) {
    S1(i);
}
for (int i = 1; i < N; i += 2) {
    S2(i);
}
```

### Non-Unit Strides

```
Domain: { [i] : 0 <= i < N, i % 4 == 0 }

Generated:
for (int i = 0; i < N; i += 4) {
    ...
}
```

---

## 6. Optimization Opportunities

### Loop Bound Simplification

```c
// Before simplification
for (i = MAX(0, j-N); i < MIN(N, j+1); i++)

// After simplification (when we know j >= 0 and j < N)
for (i = 0; i < j+1; i++)
```

### Dead Code Elimination

If a loop's bounds result in zero iterations, don't generate it:

```rust
fn generate_loop(&self, ...) -> Option<LoopAst> {
    let (lower, upper) = compute_bounds(...);
    
    // Check if loop is always empty
    if self.is_always_empty(&lower, &upper) {
        return None;
    }
    
    // Generate loop...
}
```

### Hoisting Invariants

```c
// Before hoisting
for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
        A[i][j] = B[i] * C[j];  // B[i] invariant in inner loop

// After hoisting
for (i = 0; i < N; i++) {
    double temp = B[i];  // Hoisted
    for (j = 0; j < M; j++)
        A[i][j] = temp * C[j];
}
```

---

## Key Takeaways

1. **AST Builder** reconstructs loops from polyhedral domains and schedules
2. **Bound computation** extracts loop limits from constraints
3. **C generator** produces readable, compilable C code
4. **OpenMP pragmas** are inserted for parallel loops
5. **Tiled code** uses floor/ceiling division for tile bounds
6. **Special cases** (triangular, conditional) require careful handling

## Generated Code Quality

Good generated code should be:
- **Correct**: Executes all iterations in valid order
- **Efficient**: Minimal overhead, good cache behavior
- **Readable**: Can be inspected and debugged
- **Compilable**: Standard C with optional OpenMP

## Further Reading

- CLooG: Code generator library for polyhedral scanning
- ISL: Code generation facilities
- "Code Generation in the Polyhedral Model Is Easier Than You Think" (Bastoul)