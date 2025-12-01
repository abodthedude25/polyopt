# Polyhedral Module

## Overview

The **polyhedral module** provides the mathematical foundation for the entire compiler. It implements the core data structures and operations for working with **polyhedra** — geometric shapes defined by linear constraints.

This is where the math happens.

```
┌────────────────────────────────────────────────────────┐
│                  POLYHEDRAL MODEL                       │
│                                                        │
│   Iteration Space    Access Functions    Schedules     │
│   (Integer Sets)     (Affine Maps)       (Time Maps)   │
│         │                  │                 │         │
│         ▼                  ▼                 ▼         │
│    ┌─────────┐       ┌──────────┐      ┌──────────┐   │
│    │   Set   │       │   Map    │      │ Schedule │   │
│    └─────────┘       └──────────┘      └──────────┘   │
│         │                  │                 │         │
│         └──────────────────┴─────────────────┘         │
│                          │                             │
│                   Constraint System                    │
│                          │                             │
│                    Affine Expressions                  │
└────────────────────────────────────────────────────────┘
```

---

## Files in This Module

| File | Purpose |
|------|---------|
| `space.rs` | Defines the "space" (dimensions) of polyhedra |
| `constraint.rs` | Linear constraints (equalities and inequalities) |
| `expr.rs` | Affine expressions (linear combinations) |
| `set.rs` | Integer sets (iteration domains) |
| `map.rs` | Affine maps (access functions, schedules) |
| `operations.rs` | Set operations (union, intersection, etc.) |
| `mod.rs` | Module exports |

---

## 1. What is a Polyhedron?

A **polyhedron** is a shape defined by linear inequalities. In 2D, it's a polygon. In higher dimensions, it's a polytope.

### Visual Example (2D)

```
      y
      │
    4 ┤     ┌───────┐
      │     │///////│
    3 ┤     │///////│
      │     │///////│
    2 ┤     │///////│
      │     └───────┘
    1 ┤
      │
    0 ┼─────┬───┬───┬────► x
          1   2   3   4

Constraints:
  1 ≤ x ≤ 4
  2 ≤ y ≤ 4

This is a polyhedron (a rectangle)!
```

### In Loop Terms

```c
for (i = 1; i <= 4; i++)
    for (j = 2; j <= 4; j++)
        S(i, j);
```

The iteration domain is exactly the rectangle above!

---

## 2. Spaces (`space.rs`)

### What is a Space?

A **space** defines the dimensions we're working in:
- **Set space**: Just dimensions (for iteration domains)
- **Map space**: Input and output dimensions (for transformations)

```rust
/// Defines the dimensionality and structure of a polyhedral object
pub struct Space {
    /// Parameter names (symbolic constants like N, M)
    pub params: Vec<String>,
    
    /// Input dimension names (for maps)
    pub in_dims: Vec<String>,
    
    /// Output dimension names (for sets, or map outputs)
    pub out_dims: Vec<String>,
}

impl Space {
    /// Create a set space with given dimensions
    /// Example: set_space(["N"], ["i", "j"]) for { [i,j] : ... } with param N
    pub fn set_space(params: Vec<String>, dims: Vec<String>) -> Self {
        Space {
            params,
            in_dims: vec![],
            out_dims: dims,
        }
    }
    
    /// Create a map space
    /// Example: map_space([], ["i","j"], ["x","y"]) for { [i,j] -> [x,y] }
    pub fn map_space(params: Vec<String>, in_dims: Vec<String>, out_dims: Vec<String>) -> Self {
        Space { params, in_dims, out_dims }
    }
}
```

### Why Spaces Matter

Spaces ensure we don't accidentally mix incompatible objects:
- Can't intersect a 2D set with a 3D set
- Can't compose maps with incompatible dimensions

---

## 3. Affine Expressions (`expr.rs`)

### What is an Affine Expression?

An **affine expression** is a linear combination of variables plus a constant:

```
expr = a₁·v₁ + a₂·v₂ + ... + aₙ·vₙ + c
```

### Structure

```rust
/// An affine expression over iterators and parameters
#[derive(Clone, Debug)]
pub struct AffineExpr {
    /// Coefficients for each iterator (loop variable)
    /// Example: 2i + 3j has coefficients [2, 3] for [i, j]
    pub iter_coeffs: Vec<i64>,
    
    /// Coefficients for each parameter (symbolic constant)
    /// Example: i + N - 1 has param_coeffs [1] for [N]
    pub param_coeffs: Vec<i64>,
    
    /// Constant term
    /// Example: i + 5 has constant = 5
    pub constant: i64,
}
```

### Examples

| Expression | iter_coeffs | param_coeffs | constant |
|------------|-------------|--------------|----------|
| `i` | [1, 0] | [0] | 0 |
| `i + j` | [1, 1] | [0] | 0 |
| `2*i - j + 5` | [2, -1] | [0] | 5 |
| `i + N - 1` | [1, 0] | [1] | -1 |
| `N` | [0, 0] | [1] | 0 |
| `42` | [0, 0] | [0] | 42 |

### Operations

```rust
impl AffineExpr {
    /// Create expression for a single iterator
    pub fn iterator(index: usize, num_iters: usize) -> Self {
        let mut coeffs = vec![0; num_iters];
        coeffs[index] = 1;
        AffineExpr {
            iter_coeffs: coeffs,
            param_coeffs: vec![],
            constant: 0,
        }
    }
    
    /// Create a constant expression
    pub fn constant(value: i64) -> Self {
        AffineExpr {
            iter_coeffs: vec![],
            param_coeffs: vec![],
            constant: value,
        }
    }
    
    /// Add two expressions: (a₁i + b₁) + (a₂i + b₂) = (a₁+a₂)i + (b₁+b₂)
    pub fn add(&self, other: &AffineExpr) -> AffineExpr {
        AffineExpr {
            iter_coeffs: self.iter_coeffs.iter()
                .zip(&other.iter_coeffs)
                .map(|(a, b)| a + b)
                .collect(),
            param_coeffs: self.param_coeffs.iter()
                .zip(&other.param_coeffs)
                .map(|(a, b)| a + b)
                .collect(),
            constant: self.constant + other.constant,
        }
    }
    
    /// Multiply by a scalar: k(ai + b) = (ka)i + kb
    pub fn scale(&self, factor: i64) -> AffineExpr {
        AffineExpr {
            iter_coeffs: self.iter_coeffs.iter().map(|c| c * factor).collect(),
            param_coeffs: self.param_coeffs.iter().map(|c| c * factor).collect(),
            constant: self.constant * factor,
        }
    }
    
    /// Evaluate with concrete values
    pub fn evaluate(&self, iters: &[i64], params: &[i64]) -> i64 {
        let iter_sum: i64 = self.iter_coeffs.iter()
            .zip(iters)
            .map(|(c, v)| c * v)
            .sum();
        let param_sum: i64 = self.param_coeffs.iter()
            .zip(params)
            .map(|(c, v)| c * v)
            .sum();
        iter_sum + param_sum + self.constant
    }
}
```

---

## 4. Constraints (`constraint.rs`)

### What is a Constraint?

A **constraint** is a linear inequality or equality that defines the boundary of a polyhedron.

### Types of Constraints

```rust
/// A linear constraint
#[derive(Clone, Debug)]
pub struct Constraint {
    /// The type of constraint
    pub kind: ConstraintKind,
    
    /// Left-hand side expression
    pub expr: AffineExpr,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstraintKind {
    /// expr >= 0
    GreaterEq,
    /// expr == 0
    Equal,
    /// expr > 0 (strict, less common)
    Greater,
}
```

### Converting Bounds to Constraints

All bounds are normalized to the form `expr >= 0`:

| Original | Normalized |
|----------|------------|
| `i >= 0` | `i >= 0` |
| `i < N` | `N - i - 1 >= 0` |
| `i <= N` | `N - i >= 0` |
| `i == j` | `i - j == 0` |
| `i > j` | `i - j - 1 >= 0` |

### Example: Loop Bounds

```c
for (i = 0; i < N; i++)
    for (j = i; j < M; j++)
```

Constraints:
```
i >= 0           →  i >= 0
i < N            →  N - i - 1 >= 0
j >= i           →  j - i >= 0
j < M            →  M - j - 1 >= 0
```

### Constraint Operations

```rust
impl Constraint {
    /// Create constraint: left >= right
    pub fn ge(left: AffineExpr, right: AffineExpr) -> Self {
        // Normalize to: left - right >= 0
        Constraint {
            kind: ConstraintKind::GreaterEq,
            expr: left.sub(&right),
        }
    }
    
    /// Create constraint: left == right
    pub fn eq(left: AffineExpr, right: AffineExpr) -> Self {
        Constraint {
            kind: ConstraintKind::Equal,
            expr: left.sub(&right),
        }
    }
    
    /// Check if a point satisfies this constraint
    pub fn is_satisfied(&self, iters: &[i64], params: &[i64]) -> bool {
        let value = self.expr.evaluate(iters, params);
        match self.kind {
            ConstraintKind::GreaterEq => value >= 0,
            ConstraintKind::Equal => value == 0,
            ConstraintKind::Greater => value > 0,
        }
    }
}
```

---

## 5. Integer Sets (`set.rs`)

### What is an Integer Set?

An **integer set** is the collection of all integer points inside a polyhedron. This represents iteration domains.

```rust
/// A set of integer points defined by constraints
#[derive(Clone, Debug)]
pub struct IntegerSet {
    /// The space this set lives in
    pub space: Space,
    
    /// Constraints defining the set
    /// A point is in the set iff ALL constraints are satisfied
    pub constraints: Vec<Constraint>,
}
```

### ISL Notation

The Integer Set Library (ISL) uses a standard notation:

```
{ [i, j] : 0 <= i < N and 0 <= j < M }
  └──┬──┘  └────────────┬───────────┘
   dims      constraints (conjunction)
```

With parameters:
```
[N, M] -> { [i, j] : 0 <= i < N and 0 <= j < M }
└──┬──┘
 params
```

### Set Operations

```rust
impl IntegerSet {
    /// Check if a point is in the set
    pub fn contains(&self, point: &[i64], params: &[i64]) -> bool {
        self.constraints.iter()
            .all(|c| c.is_satisfied(point, params))
    }
    
    /// Intersect with another set (AND of constraints)
    pub fn intersect(&self, other: &IntegerSet) -> IntegerSet {
        let mut constraints = self.constraints.clone();
        constraints.extend(other.constraints.clone());
        IntegerSet {
            space: self.space.clone(),
            constraints,
        }
    }
    
    /// Union of sets (OR - more complex, may need multiple polyhedra)
    pub fn union(&self, other: &IntegerSet) -> UnionSet {
        UnionSet {
            sets: vec![self.clone(), other.clone()],
        }
    }
    
    /// Project out (eliminate) a dimension
    pub fn project_out(&self, dim: usize) -> IntegerSet {
        // Fourier-Motzkin elimination
        // ... complex algorithm
    }
    
    /// Enumerate all integer points (for small sets)
    pub fn enumerate(&self, params: &[i64]) -> Vec<Vec<i64>> {
        // Find bounds and iterate
        let bounds = self.compute_bounds(params);
        let mut points = Vec::new();
        
        self.enumerate_recursive(&bounds, params, &mut vec![], &mut points);
        points
    }
}
```

### Visualization

```
Set: { [i, j] : 0 <= i < 3 and 0 <= j < 3 }

j
3 │
2 │  ●  ●  ●
1 │  ●  ●  ●
0 │  ●  ●  ●
  └──────────── i
     0  1  2

Contains 9 points: (0,0), (0,1), (0,2), (1,0), ...
```

---

## 6. Affine Maps (`map.rs`)

### What is an Affine Map?

An **affine map** transforms points from one space to another using affine expressions. Maps represent:
- **Access functions**: `(i, j) → (i, j+1)` for `A[i][j+1]`
- **Schedules**: `(i, j) → (j, i)` for loop interchange

```rust
/// An affine map from input space to output space
#[derive(Clone, Debug)]
pub struct AffineMap {
    /// The space (defines input and output dimensions)
    pub space: Space,
    
    /// Output expressions (one per output dimension)
    /// Each output is an affine function of inputs and parameters
    pub outputs: Vec<AffineExpr>,
    
    /// Domain constraints (which inputs are valid)
    pub domain: Vec<Constraint>,
}
```

### Map Notation

```
{ [i, j] -> [i, j + 1] }
  └──┬──┘   └───┬────┘
  input      output

{ [i, j] -> [j, i] }  -- Loop interchange
{ [i, j] -> [i/32, j/32, i%32, j%32] }  -- Tiling
```

### Access Functions

For array access `A[i][j+1]`:

```rust
AffineMap {
    space: map_space([], ["i", "j"], ["d0", "d1"]),
    outputs: [
        AffineExpr::iterator(0),           // d0 = i
        AffineExpr::iterator(1).add_const(1), // d1 = j + 1
    ],
    domain: [],  // Inherits from statement's domain
}
```

### Schedule Maps

For schedule `S[i,j] → (i, j, 0)`:

```rust
AffineMap {
    space: map_space([], ["i", "j"], ["t0", "t1", "t2"]),
    outputs: [
        AffineExpr::iterator(0),  // t0 = i
        AffineExpr::iterator(1),  // t1 = j
        AffineExpr::constant(0),  // t2 = 0 (statement order)
    ],
    domain: [],
}
```

### Map Operations

```rust
impl AffineMap {
    /// Apply the map to a point
    pub fn apply(&self, input: &[i64], params: &[i64]) -> Vec<i64> {
        self.outputs.iter()
            .map(|expr| expr.evaluate(input, params))
            .collect()
    }
    
    /// Compose two maps: (A ∘ B)(x) = A(B(x))
    pub fn compose(&self, other: &AffineMap) -> AffineMap {
        // Substitute other's outputs into self's expressions
        // ...
    }
    
    /// Inverse map (if it exists)
    pub fn inverse(&self) -> Option<AffineMap> {
        // Only invertible if outputs form a basis
        // ...
    }
    
    /// Apply map to a set (image)
    pub fn apply_to_set(&self, set: &IntegerSet) -> IntegerSet {
        // Transform all constraints through the map
        // ...
    }
}
```

---

## 7. Set Operations (`operations.rs`)

### Key Operations

```rust
/// Intersection: points in BOTH sets
pub fn intersect(a: &IntegerSet, b: &IntegerSet) -> IntegerSet {
    // Combine all constraints
    IntegerSet {
        space: a.space.clone(),
        constraints: [a.constraints.clone(), b.constraints.clone()].concat(),
    }
}

/// Union: points in EITHER set (returns a union type)
pub fn union(a: &IntegerSet, b: &IntegerSet) -> UnionSet {
    UnionSet { sets: vec![a.clone(), b.clone()] }
}

/// Difference: points in A but NOT in B
pub fn subtract(a: &IntegerSet, b: &IntegerSet) -> UnionSet {
    // A - B = A ∩ ¬B
    // Requires splitting into multiple polyhedra
    // ...
}

/// Empty check
pub fn is_empty(set: &IntegerSet, params: &[i64]) -> bool {
    // Try to find any satisfying point (ILP)
    // ...
}

/// Subset check: is A ⊆ B?
pub fn is_subset(a: &IntegerSet, b: &IntegerSet) -> bool {
    // A ⊆ B iff (A - B) is empty
    is_empty(&subtract(a, b))
}
```

### Fourier-Motzkin Elimination

To project out a variable, we use **Fourier-Motzkin elimination**:

```
Given: 0 <= i < N, i <= j < M
Project out i to get constraints on j:

Lower bounds on i: i >= 0
Upper bounds on i: i < N, i <= j

Combine each lower with each upper:
  0 < N     → (always true if N > 0)
  0 <= j    → j >= 0

Result: j >= 0, j < M
```

Algorithm:
1. Partition constraints into: lower bounds on x, upper bounds on x, others
2. For each (lower, upper) pair, combine to eliminate x
3. Keep the "others" unchanged

---

## Why This Math Matters

### Traditional Compiler Analysis

```c
for (i = 0; i < N; i++)
    A[i+1] = A[i] + 1;
```

Traditional: "There's a dependence because A is read and written"

### Polyhedral Analysis

```
Write A: (i) → (i+1)
Read A:  (i) → (i)

Dependence exists when:
  - Same array (A) ✓
  - Write happens before Read
  - They access the same element: i₁ + 1 = i₂

From write at i₁ to read at i₂:
  i₁ + 1 = i₂
  i₁ < i₂ (write before read)
  
  Solution: i₂ = i₁ + 1, so dependence distance = 1
```

This **exact** analysis enables transformations that traditional compilers can't do safely.

---

## Common Patterns

### Pattern 1: Triangular Loop

```c
for (i = 0; i < N; i++)
    for (j = 0; j <= i; j++)
```

Domain: `{ [i,j] : 0 <= i < N ∧ 0 <= j <= i }`

```
    j
    │
  N │────────────●
    │          ●/│
    │        ●/  │
    │      ●/    │
    │    ●/      │
    │  ●/        │
  0 │●───────────┼─── i
    0            N
```

### Pattern 2: Skewed Domain

```c
for (i = 0; i < N; i++)
    for (j = i; j < i + M; j++)
```

Domain: `{ [i,j] : 0 <= i < N ∧ i <= j < i + M }`

```
    j
    │     /────/
    │    /    /
    │   /    /
    │  /    /
    │ /    /
    │/────/
    └─────────── i
```

### Pattern 3: Union of Polyhedra

Sometimes we need multiple polyhedra:

```c
if (i < j) 
    S1;
else 
    S2;
```

S1's domain: `{ [i,j] : i < j ∧ ... }`
S2's domain: `{ [i,j] : i >= j ∧ ... }`

---

## Key Takeaways

1. **Affine expressions** are linear combinations (required for exact analysis)
2. **Constraints** define the boundaries of polyhedra
3. **Sets** represent iteration domains (which iterations execute)
4. **Maps** represent transformations (access functions, schedules)
5. **Operations** let us analyze relationships between sets

The polyhedral model turns loop optimization into **geometry**!

## Further Reading

- [Polyhedral Compilation Tutorial](https://www.cs.colostate.edu/~pouchet/doc/ics-tutorial.pdf)
- [ISL Manual](https://libisl.sourceforge.io/manual.pdf) - Mathematical details
- Schrijver, "Theory of Linear and Integer Programming"
- "Loop Transformations for Restructuring Compilers" by Banerjee