# Analysis Module

## Overview

The **analysis module** examines the polyhedral IR to extract information needed for safe and profitable transformations. The two main analyses are:

1. **SCoP Detection** — Finding which parts of the code can be optimized
2. **Dependence Analysis** — Determining what execution order constraints exist

```
        Polyhedral IR
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌────────────┐   ┌──────────────┐
│    SCoP    │   │  Dependence  │
│  Detection │   │   Analysis   │
└────────────┘   └──────────────┘
    │                   │
    ▼                   ▼
  "Can we          "What order
  optimize this?"   must be kept?"
```

---

## Files in This Module

| File | Purpose |
|------|---------|
| `scop.rs` | Static Control Part (SCoP) detection |
| `dependence.rs` | Data dependence analysis |
| `mod.rs` | Module exports |

---

## 1. SCoP Detection (`scop.rs`)

### What is a SCoP?

A **Static Control Part (SCoP)** is a maximal program region where:
1. Loop bounds are **affine** functions of outer iterators and parameters
2. Array subscripts are **affine** functions
3. Conditionals have **affine** conditions
4. No function calls (except pure math functions)
5. No irregular control flow (goto, break, early return)

### Why SCoPs Matter

Only SCoPs can be analyzed and transformed with polyhedral techniques:

```c
// ✓ This IS a SCoP
for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
        C[i][j] = A[i][j] + B[i][j];

// ✗ This is NOT a SCoP (non-affine bound)
for (i = 0; i < A[0]; i++)  // bound depends on array value!
    B[i] = 0;

// ✗ This is NOT a SCoP (non-affine subscript)
for (i = 0; i < N; i++)
    A[B[i]] = 0;  // index depends on array value!

// ✗ This is NOT a SCoP (function call with side effects)
for (i = 0; i < N; i++)
    A[i] = read_input();  // external call
```

### SCoP Detection Algorithm

```rust
pub struct ScopDetector {
    /// Current loop depth
    depth: usize,
    /// Parameters (symbolic constants)
    parameters: HashSet<String>,
    /// Current loop iterators
    iterators: Vec<String>,
}

impl ScopDetector {
    /// Check if a statement is part of a valid SCoP
    pub fn is_valid_scop(&mut self, stmt: &HirStmt) -> bool {
        match stmt {
            HirStmt::Loop { iterator, lower, upper, body } => {
                // 1. Check bounds are affine
                if !self.is_affine_expr(lower) || !self.is_affine_expr(upper) {
                    return false;
                }
                
                // 2. Add iterator to scope
                self.iterators.push(iterator.clone());
                self.depth += 1;
                
                // 3. Check body recursively
                let body_valid = body.iter().all(|s| self.is_valid_scop(s));
                
                // 4. Restore scope
                self.iterators.pop();
                self.depth -= 1;
                
                body_valid
            }
            
            HirStmt::Assignment { target, value } => {
                // Check array subscripts are affine
                self.is_affine_array_ref(target) && 
                self.is_affine_expr(value)
            }
            
            HirStmt::If { condition, then_body, else_body } => {
                // Condition must be affine
                self.is_affine_condition(condition) &&
                then_body.iter().all(|s| self.is_valid_scop(s)) &&
                else_body.iter().all(|s| self.is_valid_scop(s))
            }
        }
    }
    
    /// Check if an expression is affine
    fn is_affine_expr(&self, expr: &HirExpr) -> bool {
        match expr {
            HirExpr::Constant(_) => true,
            HirExpr::Parameter(name) => self.parameters.contains(name),
            HirExpr::Iterator(name) => self.iterators.contains(name),
            HirExpr::BinaryOp { op, left, right } => {
                match op {
                    // Addition/subtraction preserve affine-ness
                    Add | Sub => {
                        self.is_affine_expr(left) && self.is_affine_expr(right)
                    }
                    // Multiplication: one side must be constant
                    Mul => {
                        (self.is_constant(left) && self.is_affine_expr(right)) ||
                        (self.is_affine_expr(left) && self.is_constant(right))
                    }
                    // Division/modulo: divisor must be constant
                    Div | Mod => {
                        self.is_affine_expr(left) && self.is_constant(right)
                    }
                    _ => false,
                }
            }
            // Array values are NOT affine (we don't know them statically)
            HirExpr::ArrayAccess(_) => false,
            // Function calls are NOT affine (except allowlisted math)
            HirExpr::Call { name, .. } => {
                matches!(name.as_str(), "sqrt" | "sin" | "cos" | "exp" | "log")
            }
        }
    }
}
```

### SCoP Regions

Often, only part of a program is a SCoP:

```c
void mixed_function(int N, double* A, double* B) {
    // NOT in SCoP: dynamic input
    int size = read_size();
    
    // START of SCoP
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N + j] = B[i*N + j] * 2.0;
        }
    }
    // END of SCoP
    
    // NOT in SCoP: output
    print_array(A, N);
}
```

The SCoP detector identifies these maximal regions.

---

## 2. Dependence Analysis (`dependence.rs`)

### What is a Dependence?

A **dependence** exists when two operations access the same memory location and at least one is a write. The earlier operation must complete before the later one.

### Types of Dependences

| Type | Symbol | Description | Example |
|------|--------|-------------|---------|
| Flow (True) | δᶠ | Read After Write | `A[i]=..; ..=A[i]` |
| Anti | δᵃ | Write After Read | `..=A[i]; A[i]=..` |
| Output | δᵒ | Write After Write | `A[i]=..; A[i]=..` |
| Input | δⁱ | Read After Read | `..=A[i]; ..=A[i]` (no constraint) |

```
Flow (RAW):        Anti (WAR):        Output (WAW):
   S1: A = 5         S1: x = A          S1: A = 5
   S2: B = A         S2: A = 10         S2: A = 10
       ↓                  ↓                  ↓
   S1 before S2      S1 before S2       S1 before S2
```

### Dependence Distance

The **dependence distance** tells us how far apart dependent iterations are:

```c
for (i = 1; i < N; i++)
    A[i] = A[i-1] + 1;   // Distance = 1
```

Here, iteration `i` reads `A[i-1]`, written by iteration `i-1`. The distance is 1.

### Dependence Direction

For multi-dimensional loops, we use **direction vectors**:

| Direction | Meaning |
|-----------|---------|
| `<` | Source before sink in this dimension |
| `=` | Same value in this dimension |
| `>` | Source after sink (backward dep) |
| `*` | Any (unknown or multiple) |

Example: `(<, =)` means "earlier in outer loop, same in inner loop"

### Polyhedral Dependence Analysis

The key insight: dependence exists iff the access functions "collide":

```
Statement S1: A[f₁(i₁)] at iteration i₁
Statement S2: A[f₂(i₂)] at iteration i₂

Dependence exists iff:
  1. f₁(i₁) = f₂(i₂)           (same memory location)
  2. i₁ is lexicographically before i₂  (S1 executes first)
  3. Both i₁ and i₂ are in their domains
```

### The Dependence Polyhedron

We build a polyhedron of all (source, sink) iteration pairs that are dependent:

```rust
pub struct Dependence {
    /// Source statement
    pub source: StmtId,
    /// Sink statement  
    pub sink: StmtId,
    /// Type of dependence
    pub kind: DependenceKind,
    /// The dependence polyhedron
    /// Contains pairs (i_src, i_snk) where dependence exists
    pub polyhedron: IntegerSet,
    /// Distance vector (if uniform)
    pub distance: Option<Vec<i64>>,
    /// Direction vector
    pub direction: Vec<Direction>,
}
```

### Building the Dependence Polyhedron

```rust
pub fn compute_dependence(
    source: &Statement,
    sink: &Statement,
    source_access: &Access,
    sink_access: &Access,
) -> Option<Dependence> {
    // 1. Check same array
    if source_access.array != sink_access.array {
        return None;
    }
    
    // 2. Check at least one write
    let kind = match (source_access.kind, sink_access.kind) {
        (Write, Read) => DependenceKind::Flow,
        (Read, Write) => DependenceKind::Anti,
        (Write, Write) => DependenceKind::Output,
        (Read, Read) => return None,  // No dependence needed
    };
    
    // 3. Build the combined space: [i_src, i_snk]
    let src_dims = source.domain.iterators.len();
    let snk_dims = sink.domain.iterators.len();
    
    // 4. Add domain constraints for both
    let mut constraints = Vec::new();
    constraints.extend(embed_constraints(&source.domain, 0));
    constraints.extend(embed_constraints(&sink.domain, src_dims));
    
    // 5. Add access equality: f_src(i_src) = f_snk(i_snk)
    for (src_sub, snk_sub) in source_access.subscripts.iter()
                                 .zip(&sink_access.subscripts) {
        let src_embedded = embed_expr(src_sub, 0, src_dims + snk_dims);
        let snk_embedded = embed_expr(snk_sub, src_dims, src_dims + snk_dims);
        constraints.push(Constraint::eq(src_embedded, snk_embedded));
    }
    
    // 6. Add temporal ordering: i_src << i_snk (lexicographically before)
    constraints.extend(lexicographic_before_constraints(
        src_dims, snk_dims, &source.schedule, &sink.schedule
    ));
    
    // 7. Check if polyhedron is non-empty
    let polyhedron = IntegerSet { constraints, .. };
    if polyhedron.is_empty() {
        return None;  // No dependence!
    }
    
    // 8. Extract distance/direction
    let distance = extract_distance(&polyhedron);
    let direction = extract_direction(&polyhedron);
    
    Some(Dependence { source, sink, kind, polyhedron, distance, direction })
}
```

### Example: Matrix Multiply

```c
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
```

**Accesses:**
- Write C: `(i,j,k) → (i, j)`
- Read C:  `(i,j,k) → (i, j)`
- Read A:  `(i,j,k) → (i, k)`
- Read B:  `(i,j,k) → (k, j)`

**Flow dependence C → C:**
```
Same location: (i₁,j₁) = (i₂,j₂)
Order: (i₁,j₁,k₁) << (i₂,j₂,k₂)

Since i₁=i₂ and j₁=j₂:
  Need k₁ < k₂

Dependence: (=, =, <) with distance (0, 0, 1)
```

This means: **k loop carries the dependence, i and j are parallelizable!**

### Dependence Graph

All dependences form a **Dependence Graph**:

```rust
pub struct DependenceGraph {
    /// All statements
    pub statements: Vec<StmtId>,
    /// All dependences
    pub dependences: Vec<Dependence>,
}

impl DependenceGraph {
    /// Get all dependences from a statement
    pub fn dependences_from(&self, stmt: StmtId) -> Vec<&Dependence> {
        self.dependences.iter()
            .filter(|d| d.source == stmt)
            .collect()
    }
    
    /// Get all dependences to a statement
    pub fn dependences_to(&self, stmt: StmtId) -> Vec<&Dependence> {
        self.dependences.iter()
            .filter(|d| d.sink == stmt)
            .collect()
    }
    
    /// Check if a transformation is legal
    pub fn is_legal_schedule(&self, new_schedule: &Schedule) -> bool {
        // A schedule is legal iff all dependence distances are non-negative
        self.dependences.iter().all(|dep| {
            let new_dist = compute_distance_under_schedule(dep, new_schedule);
            is_lexicographically_positive(&new_dist)
        })
    }
}
```

### Visualizing Dependences

```c
for (i = 0; i < 4; i++)
    for (j = 1; j < 4; j++)
        A[i][j] = A[i][j-1] + 1;
```

Dependence: `(i, j-1) → (i, j)` with direction `(=, <)`

```
     j
   4 ┤
   3 ┤  ●──→●──→●──→●
   2 ┤  ●──→●──→●──→●
   1 ┤  ●──→●──→●──→●
   0 ┤  (no writes here)
     └────────────────── i
        0   1   2   3

Arrows show dependence direction: must go left-to-right in j
```

---

## Dependence-Based Legality

### When Can We Transform?

A transformation is **legal** if it preserves all dependences:

```
Original order: S1 before S2 (dependence S1 → S2)
After transform: S1 must STILL be before S2
```

### Example: Loop Interchange

```c
// Original: (i, j) order
for (i...) 
    for (j...)
        A[i][j] = A[i][j-1] + 1;  // dep: (=, <)

// Interchanged: (j, i) order  
for (j...)
    for (i...)
        A[i][j] = A[i][j-1] + 1;
```

Is this legal?

Original dependence: `(i, j-1) → (i, j)` means same `i`, increasing `j`.

After interchange:
- Original: `(i=2, j=1) → (i=2, j=2)`, executed as `(2,1) → (2,2)` ✓
- Interchanged: `(j=1, i=2) → (j=2, i=2)`, executed as `(1,2) → (2,2)` ✓

Still legal! The dependence `(=, <)` becomes `(<, =)`.

### Example: Illegal Interchange

```c
// Original
for (i = 1; i < N; i++)
    for (j...)
        A[i][j] = A[i-1][j] + 1;  // dep: (<, =)

// Interchanged - ILLEGAL!
for (j...)
    for (i = 1; i < N; i++)
        A[i][j] = A[i-1][j] + 1;
```

Original: `(i-1, j) → (i, j)`, i.e., `(1, 5) → (2, 5)` executes as `(1,5)→(2,5)` ✓
Interchanged: same points, but `(5, 1) → (5, 2)` — now `(5,2)` is AFTER `(5,1)` ✗

The dependence is violated!

---

## Complexity and Optimizations

### Exact Analysis is Expensive

Dependence analysis requires solving integer linear programs, which is NP-hard in general. Optimizations:

1. **Quick rejection**: If arrays differ, no dependence
2. **GCD test**: If subscripts can't match (modular arithmetic), no dependence
3. **Banerjee bounds**: Approximate bounds test
4. **Range test**: Check if iteration ranges overlap
5. **Full ILP**: Only if quick tests inconclusive

```rust
pub fn quick_independence_test(
    access1: &Access, 
    access2: &Access
) -> Option<bool> {
    // Different arrays → definitely independent
    if access1.array != access2.array {
        return Some(true);
    }
    
    // GCD test for each dimension
    for (sub1, sub2) in access1.subscripts.iter().zip(&access2.subscripts) {
        if let Some(independent) = gcd_test(sub1, sub2) {
            if independent {
                return Some(true);  // Definitely independent
            }
        }
    }
    
    // Can't determine quickly
    None
}

/// GCD test: if c₁i + c₂j = c₃ has no integer solution
fn gcd_test(sub1: &AffineExpr, sub2: &AffineExpr) -> Option<bool> {
    // Check if constant difference is divisible by GCD of coefficients
    let diff = sub1.constant - sub2.constant;
    let gcd = compute_gcd_of_coefficients(sub1, sub2);
    
    if diff % gcd != 0 {
        Some(true)  // No integer solution → independent
    } else {
        None  // Might be dependent, need full analysis
    }
}
```

---

## Key Takeaways

1. **SCoP detection** finds which code regions can use polyhedral optimization
2. **Dependence analysis** finds what order constraints must be preserved
3. **Dependences** are computed by finding when access functions collide
4. **Legal transformations** must preserve all dependence orderings
5. **Direction/distance vectors** summarize dependence patterns

## Further Reading

- "Optimizing Compilers for Modern Architectures" (Allen & Kennedy) - Chapter on dependence
- "High Performance Compilers for Parallel Computing" (Wolfe)
- [Omega Library](https://github.com/davewathaern/the-omega-project) - Classic dependence analysis
- Feautrier, "Dataflow Analysis of Array and Scalar References"