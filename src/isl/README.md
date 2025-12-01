# ISL Module

## Overview

The **ISL module** provides integration with the Integer Set Library (ISL), the gold standard for polyhedral mathematics. This module offers:

1. **Native ISL bindings** (when ISL is installed)
2. **Pure Rust simulation** (fallback when ISL is unavailable)

```
┌─────────────────────────────────────────────────────────┐
│                    ISL Module                            │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────────────┐    │
│  │  Native ISL     │    │  Pure Rust Simulation   │    │
│  │  (via C FFI)    │    │  (no dependencies)      │    │
│  └─────────────────┘    └─────────────────────────┘    │
│           │                        │                    │
│           └────────┬───────────────┘                    │
│                    │                                    │
│              Common API                                 │
└─────────────────────────────────────────────────────────┘
```

---

## Files in This Module

| File | Purpose |
|------|---------|
| `context.rs` | ISL context management |
| `set.rs` | Integer set operations |
| `map.rs` | Affine map operations |
| `schedule.rs` | Scheduling algorithms |
| `codegen.rs` | Polyhedral code generation |
| `simulation.rs` | Pure Rust fallback implementation |
| `mod.rs` | Module exports and API |

---

## 1. What is ISL?

### The Integer Set Library

**ISL** (Integer Set Library) is a C library for manipulating sets and relations of integer points bounded by linear constraints. It's used by:

- **Polly** (LLVM's polyhedral optimizer)
- **PPCG** (polyhedral parallel code generator)
- **Pluto** (automatic parallelizer)
- Most academic polyhedral research

### Key Capabilities

1. **Set operations**: union, intersection, subtraction
2. **Map operations**: composition, inverse, domain/range
3. **Scheduling**: Feautrier/Pluto algorithms
4. **Code generation**: Loop generation from polyhedra
5. **Dependence analysis**: Exact dataflow analysis

---

## 2. ISL Notation

ISL uses a standardized string notation for polyhedra:

### Sets

```
// Basic set: all points (i,j) where 0 ≤ i < 10 and 0 ≤ j < 10
"{ [i, j] : 0 <= i < 10 and 0 <= j < 10 }"

// Named set
"{ S[i, j] : 0 <= i < N and 0 <= j < M }"

// With parameters
"[N, M] -> { [i, j] : 0 <= i < N and 0 <= j < M }"

// Triangular domain
"[N] -> { [i, j] : 0 <= i < N and i <= j < N }"

// Union of sets
"{ [i] : 0 <= i < 10 } union { [i] : 20 <= i < 30 }"
```

### Maps

```
// Identity map
"{ [i, j] -> [i, j] }"

// Access function for A[i][j+1]
"{ S[i, j] -> A[i, j + 1] }"

// Loop interchange
"{ [i, j] -> [j, i] }"

// Tiling (conceptual)
"{ [i, j] -> [floor(i/32), floor(j/32), i mod 32, j mod 32] }"

// Schedule
"{ S1[i] -> [i, 0]; S2[i] -> [i, 1] }"  // S1 before S2 at each i
```

### Parsing ISL Notation

```rust
/// Parse an ISL set string into a PolySet
pub fn parse_set(input: &str) -> Result<PolySet, ParseError> {
    // Tokenize
    let tokens = tokenize(input)?;
    
    // Parse structure: [params] -> { [dims] : constraints }
    let (params, dims, constraints) = parse_set_structure(&tokens)?;
    
    Ok(PolySet {
        parameters: params,
        dimensions: dims,
        constraints,
    })
}
```

---

## 3. ISL Context (`context.rs`)

### What is a Context?

An **ISL context** manages memory and settings for ISL operations. All ISL objects belong to a context.

```rust
pub struct IslContext {
    /// Pointer to native ISL context (if available)
    native: Option<*mut isl_ctx>,
    /// Options for ISL operations
    options: IslOptions,
}

impl IslContext {
    pub fn new() -> Self {
        #[cfg(feature = "native-isl")]
        {
            let ctx = unsafe { isl_ctx_alloc() };
            IslContext {
                native: Some(ctx),
                options: IslOptions::default(),
            }
        }
        
        #[cfg(not(feature = "native-isl"))]
        {
            IslContext {
                native: None,
                options: IslOptions::default(),
            }
        }
    }
}

impl Drop for IslContext {
    fn drop(&mut self) {
        #[cfg(feature = "native-isl")]
        if let Some(ctx) = self.native {
            unsafe { isl_ctx_free(ctx) };
        }
    }
}
```

### Context Options

```rust
pub struct IslOptions {
    /// Scheduling algorithm
    pub schedule_algorithm: ScheduleAlgorithm,
    /// Maximize parallelism vs locality
    pub schedule_maximize_parallelism: bool,
    /// Code generation style
    pub codegen_style: CodeGenStyle,
}

pub enum ScheduleAlgorithm {
    /// Feautrier's algorithm (maximize parallelism)
    Feautrier,
    /// ISL's default algorithm (balanced)
    Isl,
    /// Pluto-like (locality-focused)
    Pluto,
}
```

---

## 4. Integer Sets (`set.rs`)

### Set Operations

```rust
pub struct IslSet {
    context: Arc<IslContext>,
    #[cfg(feature = "native-isl")]
    native: *mut isl_set,
    #[cfg(not(feature = "native-isl"))]
    simulation: PolySet,
}

impl IslSet {
    /// Parse from ISL notation
    pub fn from_str(ctx: &IslContext, s: &str) -> Result<Self, Error> {
        // ...
    }
    
    /// Check if set is empty
    pub fn is_empty(&self) -> bool {
        // ...
    }
    
    /// Intersection of two sets
    pub fn intersect(&self, other: &IslSet) -> IslSet {
        // ...
    }
    
    /// Union of two sets
    pub fn union(&self, other: &IslSet) -> IslSet {
        // ...
    }
    
    /// Subtract: self - other
    pub fn subtract(&self, other: &IslSet) -> IslSet {
        // ...
    }
    
    /// Project out dimensions
    pub fn project_out(&self, dim_type: DimType, first: u32, n: u32) -> IslSet {
        // ...
    }
    
    /// Enumerate all integer points
    pub fn enumerate(&self, params: &[i64]) -> Vec<Vec<i64>> {
        // ...
    }
    
    /// Get number of dimensions
    pub fn dim(&self) -> usize {
        // ...
    }
}
```

### Example Usage

```rust
let ctx = IslContext::new();

// Create sets
let a = IslSet::from_str(&ctx, "{ [i] : 0 <= i < 10 }")?;
let b = IslSet::from_str(&ctx, "{ [i] : 5 <= i < 15 }")?;

// Operations
let intersection = a.intersect(&b);  // { [i] : 5 <= i < 10 }
let union = a.union(&b);             // { [i] : 0 <= i < 15 }
let diff = a.subtract(&b);           // { [i] : 0 <= i < 5 }

// Check properties
assert!(!intersection.is_empty());
assert_eq!(intersection.enumerate(&[]), vec![
    vec![5], vec![6], vec![7], vec![8], vec![9]
]);
```

---

## 5. Affine Maps (`map.rs`)

### Map Operations

```rust
pub struct IslMap {
    context: Arc<IslContext>,
    // Native or simulated storage
}

impl IslMap {
    /// Parse from ISL notation
    pub fn from_str(ctx: &IslContext, s: &str) -> Result<Self, Error> {
        // ...
    }
    
    /// Apply map to a set (image)
    pub fn apply_range(&self, set: &IslSet) -> IslSet {
        // ...
    }
    
    /// Compose maps: (self ∘ other)(x) = self(other(x))
    pub fn compose(&self, other: &IslMap) -> IslMap {
        // ...
    }
    
    /// Inverse map
    pub fn inverse(&self) -> IslMap {
        // ...
    }
    
    /// Domain of the map
    pub fn domain(&self) -> IslSet {
        // ...
    }
    
    /// Range of the map
    pub fn range(&self) -> IslSet {
        // ...
    }
    
    /// Check if map is bijective
    pub fn is_bijective(&self) -> bool {
        // ...
    }
}
```

### Example: Access Functions

```rust
let ctx = IslContext::new();

// Access A[i][j+1]
let access = IslMap::from_str(&ctx, "{ S[i,j] -> A[i, j+1] }")?;

// Domain: iteration space
let domain = access.domain();  // { S[i,j] }

// Range: accessed elements
let range = access.range();    // { A[i,j+1] }

// Inverse: which iterations access element (x,y)?
let inv = access.inverse();    // { A[x,y] -> S[x, y-1] }
```

---

## 6. Scheduling (`schedule.rs`)

### Computing Schedules

```rust
pub struct IslSchedule {
    context: Arc<IslContext>,
    // Schedule tree representation
}

/// Schedule constraints
pub struct ScheduleConstraints {
    /// Iteration domain
    pub domain: IslUnionSet,
    /// Validity constraints (dependences)
    pub validity: IslUnionMap,
    /// Proximity constraints (locality hints)
    pub proximity: IslUnionMap,
    /// Coincidence constraints (parallelism hints)
    pub coincidence: IslUnionMap,
}

impl IslSchedule {
    /// Compute schedule from constraints
    pub fn from_constraints(constraints: &ScheduleConstraints) -> Result<Self, Error> {
        // Uses ISL's scheduler (Feautrier/Pluto hybrid)
        // ...
    }
    
    /// Get schedule as a map
    pub fn get_map(&self) -> IslUnionMap {
        // ...
    }
    
    /// Check if a loop is parallel
    pub fn is_parallel(&self, band: usize) -> bool {
        // ...
    }
}
```

### Example: Automatic Scheduling

```rust
let ctx = IslContext::new();

// Domain: two statements
let domain = IslUnionSet::from_str(&ctx, 
    "{ S1[i] : 0 <= i < N; S2[i] : 0 <= i < N }")?;

// Dependence: S1[i] -> S2[i] (must execute S1 before S2 at same i)
let validity = IslUnionMap::from_str(&ctx,
    "{ S1[i] -> S2[i] }")?;

// Proximity: keep S1[i] and S2[i] close (fusion hint)
let proximity = validity.clone();

let constraints = ScheduleConstraints {
    domain,
    validity,
    proximity,
    coincidence: IslUnionMap::empty(&ctx),
};

let schedule = IslSchedule::from_constraints(&constraints)?;
// Result: { S1[i] -> [i, 0]; S2[i] -> [i, 1] }
// This fuses the loops and orders S1 before S2
```

---

## 7. Code Generation (`codegen.rs`)

### Generating Loops from Schedules

```rust
pub struct IslCodeGen {
    context: Arc<IslContext>,
}

impl IslCodeGen {
    /// Generate AST from schedule
    pub fn generate(&self, schedule: &IslSchedule) -> LoopAst {
        // ...
    }
    
    /// Generate with specific options
    pub fn generate_with_options(
        &self,
        schedule: &IslSchedule,
        options: &CodeGenOptions,
    ) -> LoopAst {
        // ...
    }
}

pub struct CodeGenOptions {
    /// Separate loops for different statements
    pub separate: bool,
    /// Unroll small loops
    pub unroll: Option<usize>,
    /// Generate OpenMP pragmas
    pub openmp: bool,
}
```

---

## 8. Pure Rust Simulation (`simulation.rs`)

### Why a Simulation?

Not everyone has ISL installed. The simulation provides:
- **No external dependencies**
- **Easier building** (just `cargo build`)
- **Sufficient for learning** and simple cases

### PolySet Implementation

```rust
/// A polyhedral set implemented in pure Rust
#[derive(Clone, Debug)]
pub struct PolySet {
    /// Dimension names
    pub dimensions: Vec<String>,
    /// Parameter names
    pub parameters: Vec<String>,
    /// Linear constraints (conjunction)
    pub constraints: Vec<Constraint>,
}

impl PolySet {
    /// Parse ISL notation
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let mut parser = IslParser::new(input);
        parser.parse_set()
    }
    
    /// Check if a point is in the set
    pub fn contains(&self, point: &[i64], params: &[i64]) -> bool {
        self.constraints.iter()
            .all(|c| c.is_satisfied(point, params))
    }
    
    /// Enumerate all integer points (for small sets)
    pub fn enumerate(&self, params: &[i64]) -> Vec<Vec<i64>> {
        // Find bounds
        let bounds = self.compute_bounds(params);
        let mut points = Vec::new();
        
        // Enumerate recursively
        self.enumerate_impl(&bounds, params, &mut vec![], &mut points);
        points
    }
    
    fn enumerate_impl(
        &self,
        bounds: &[(i64, i64)],
        params: &[i64],
        current: &mut Vec<i64>,
        results: &mut Vec<Vec<i64>>,
    ) {
        if current.len() == bounds.len() {
            if self.contains(current, params) {
                results.push(current.clone());
            }
            return;
        }
        
        let dim = current.len();
        let (lower, upper) = bounds[dim];
        
        for val in lower..=upper {
            current.push(val);
            self.enumerate_impl(bounds, params, current, results);
            current.pop();
        }
    }
}
```

### Constraint Implementation

```rust
#[derive(Clone, Debug)]
pub struct Constraint {
    /// Coefficients for dimensions: a₁x₁ + a₂x₂ + ...
    pub dim_coeffs: Vec<i64>,
    /// Coefficients for parameters: b₁p₁ + b₂p₂ + ...
    pub param_coeffs: Vec<i64>,
    /// Constant term: c
    pub constant: i64,
    /// Constraint type: >= 0 or == 0
    pub is_equality: bool,
}

impl Constraint {
    /// Evaluate: returns a₁x₁ + a₂x₂ + ... + b₁p₁ + ... + c
    pub fn evaluate(&self, point: &[i64], params: &[i64]) -> i64 {
        let dim_sum: i64 = self.dim_coeffs.iter()
            .zip(point)
            .map(|(c, v)| c * v)
            .sum();
        
        let param_sum: i64 = self.param_coeffs.iter()
            .zip(params)
            .map(|(c, v)| c * v)
            .sum();
        
        dim_sum + param_sum + self.constant
    }
    
    /// Check if satisfied
    pub fn is_satisfied(&self, point: &[i64], params: &[i64]) -> bool {
        let value = self.evaluate(point, params);
        if self.is_equality {
            value == 0
        } else {
            value >= 0
        }
    }
}
```

### Tiled Code Generation

```rust
impl PolySet {
    /// Generate tiled loop structure
    pub fn generate_tiled_loops(&self, tile_size: usize) -> String {
        let mut code = String::new();
        let n = self.dimensions.len();
        
        // Tile loops
        for (i, dim) in self.dimensions.iter().enumerate() {
            let tile_var = format!("{}t", dim);
            let bound = self.get_upper_bound(i);
            code.push_str(&format!(
                "for ({tile_var} = 0; {tile_var} < CEIL_DIV({bound}, {tile_size}); {tile_var}++) {{\n"
            ));
        }
        
        // Point loops
        for (i, dim) in self.dimensions.iter().enumerate() {
            let tile_var = format!("{}t", dim);
            let bound = self.get_upper_bound(i);
            code.push_str(&format!(
                "  for ({dim} = {tile_var}*{tile_size}; {dim} < MIN(({tile_var}+1)*{tile_size}, {bound}); {dim}++) {{\n"
            ));
        }
        
        // Body placeholder
        code.push_str("    // Body\n");
        
        // Close loops
        for _ in 0..(2 * n) {
            code.push_str("  }\n");
        }
        
        code
    }
}
```

---

## 9. CLI Usage

### ISL Command

```bash
# Parse and display set
polyopt isl "{ [i,j] : 0 <= i < 4 and 0 <= j < 4 }" --expr

# Enumerate points
polyopt isl "{ [i,j] : 0 <= i < 4 and 0 <= j < 4 }" --expr --enumerate

# With parameters
polyopt isl "{ [i,j] : 0 <= i < N and 0 <= j < M }" --expr --params "N=3,M=4" --enumerate

# Generate tiled code
polyopt isl "{ [i,j] : 0 <= i < N and 0 <= j < N }" --expr --tile 32
```

### Example Output

```
$ polyopt isl "{ [i,j] : 0 <= i < 3 and 0 <= j <= i }" --expr --enumerate

ISL Expression Analysis
=======================

Input: { [i,j] : 0 <= i < 3 and 0 <= j <= i }

Dimensions: [i, j]
Constraints:
  i >= 0
  i <= 2
  j >= 0
  j <= i

Points (6 total):
  (0, 0)
  (1, 0)
  (1, 1)
  (2, 0)
  (2, 1)
  (2, 2)
```

---

## 10. Installing Native ISL

For advanced features, install ISL:

### macOS
```bash
brew install isl
```

### Ubuntu/Debian
```bash
apt install libisl-dev
```

### Fedora
```bash
dnf install isl-devel
```

### Build with ISL support
```bash
cargo build --release --features native-isl
```

---

## Key Takeaways

1. **ISL** is the standard library for polyhedral mathematics
2. **ISL notation** provides a concise way to express sets and maps
3. **Pure Rust simulation** works without external dependencies
4. **Scheduling** algorithms find optimal execution orders
5. **Code generation** produces loops from polyhedral representations

## Limitations of Simulation

The pure Rust simulation handles basic cases but doesn't support:
- Full scheduling algorithms (Feautrier, Pluto)
- Complex set operations (coalescing, simplification)
- Exact integer linear programming

For production use, native ISL is recommended.

## Further Reading

- [ISL Manual](https://libisl.sourceforge.io/manual.pdf) - Complete documentation
- [Polly](https://polly.llvm.org/) - LLVM's polyhedral optimizer (uses ISL)
- [PPCG](https://github.com/Meinersbur/ppcg) - GPU code generator (uses ISL)
- [Barvinok](https://barvinok.sourceforge.net/) - Counting integer points