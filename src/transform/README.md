# Transform Module

## Overview

The **transform module** implements loop transformations that improve performance. These transformations change *how* code executes without changing *what* it computes.

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFORMATIONS                           │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │  Tiling  │ │Interchange│ │  Fusion  │ │ Skewing  │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│       │            │            │            │              │
│       └────────────┴────────────┴────────────┘              │
│                          │                                  │
│                    ┌──────────┐                             │
│                    │ Pipeline │  ← Combines transforms      │
│                    └──────────┘                             │
│                          │                                  │
│                    ┌──────────┐                             │
│                    │Scheduler │  ← Finds optimal order      │
│                    └──────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Files in This Module

| File | Purpose |
|------|---------|
| `tiling.rs` | Loop tiling (blocking) for cache locality |
| `interchange.rs` | Loop interchange (reordering) |
| `fusion.rs` | Loop fusion (combining loops) |
| `skewing.rs` | Loop skewing for parallelization |
| `unrolling.rs` | Loop unrolling |
| `scheduler.rs` | Automatic schedule optimization |
| `pipeline.rs` | Combines multiple transformations |
| `mod.rs` | Module exports |

---

## 1. Loop Tiling (`tiling.rs`)

### What is Tiling?

**Tiling** (also called blocking) divides a loop's iteration space into smaller "tiles" that fit in cache. This dramatically improves **data locality**.

### The Cache Problem

```c
// Naive matrix multiply - terrible cache behavior
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        for (k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
```

Problem: When N is large:
- `A[i][k]` accesses row `i` sequentially ✓
- `B[k][j]` jumps across rows (column access) ✗ Cache misses!

### The Solution: Tiling

```c
// Tiled matrix multiply - excellent cache behavior
for (ii = 0; ii < N; ii += TILE)
    for (jj = 0; jj < N; jj += TILE)
        for (kk = 0; kk < N; kk += TILE)
            // Process one tile
            for (i = ii; i < min(ii+TILE, N); i++)
                for (j = jj; j < min(jj+TILE, N); j++)
                    for (k = kk; k < min(kk+TILE, N); k++)
                        C[i][j] += A[i][k] * B[k][j];
```

Now:
- Each tile of A (TILE × TILE) fits in cache
- Each tile of B (TILE × TILE) fits in cache
- Massive reduction in cache misses!

### Visualization

```
Original iteration order (i-major):
┌─────────────────┐
│ 1  2  3  4  5...│  ← Process entire row
│ N+1 ...         │  ← Then next row
│                 │
└─────────────────┘

Tiled iteration order:
┌────┬────┬────┬──┐
│ 1  │ 5  │ 9  │..│  ← Process tile
│ 2  │ 6  │10  │  │
│ 3  │ 7  │11  │  │
│ 4  │ 8  │12  │  │
├────┼────┼────┼──┤
│13  │17  │    │  │  ← Next tile
│... │    │    │  │
└────┴────┴────┴──┘
```

### Implementation

```rust
/// Apply tiling to a loop nest
pub fn tile_loops(
    program: &mut PolyProgram,
    loops_to_tile: &[String],  // Which loops: ["i", "j"]
    tile_sizes: &[usize],       // Tile sizes: [32, 32]
) -> Result<(), TransformError> {
    for stmt in &mut program.statements {
        // 1. Identify tileable dimensions
        let tile_dims: Vec<usize> = loops_to_tile.iter()
            .filter_map(|name| stmt.domain.iterators.iter().position(|i| i == name))
            .collect();
        
        // 2. Create new iterators: i → (ii, i) where ii = tile iterator
        let mut new_iterators = Vec::new();
        let mut new_constraints = Vec::new();
        
        for (idx, &dim) in tile_dims.iter().enumerate() {
            let orig_iter = &stmt.domain.iterators[dim];
            let tile_iter = format!("{}t", orig_iter);  // "it", "jt"
            let point_iter = orig_iter.clone();         // "i", "j"
            let tile_size = tile_sizes[idx] as i64;
            
            // Tile iterator: 0 <= it < ceil(N / TILE)
            // Point iterator: it*TILE <= i < min((it+1)*TILE, N)
            
            new_constraints.push(/* it >= 0 */);
            new_constraints.push(/* it*TILE <= i */);
            new_constraints.push(/* i < (it+1)*TILE */);
            new_constraints.push(/* i < N (original bound) */);
            
            new_iterators.push(tile_iter);
            new_iterators.push(point_iter);
        }
        
        // 3. Update schedule to iterate tiles first, then points
        // Original: (i, j) → (i, j)
        // Tiled:    (it, jt, i, j) → (it, jt, i, j)
        stmt.schedule = build_tiled_schedule(&tile_dims, tile_sizes);
    }
    
    Ok(())
}
```

### Choosing Tile Size

The optimal tile size depends on cache size:

```
L1 cache = 32KB, line = 64 bytes, double = 8 bytes
Working set for one tile: 3 * TILE² * 8 bytes (for A, B, C tiles)
3 * TILE² * 8 ≤ 32KB
TILE ≤ sqrt(32KB / 24) ≈ 37

Practical choice: TILE = 32 (power of 2)
```

---

## 2. Loop Interchange (`interchange.rs`)

### What is Interchange?

**Loop interchange** swaps the nesting order of loops. This can improve:
- Cache access patterns (row-major vs column-major)
- Parallelization opportunities

### Example

```c
// Original: Column-major access (bad for row-major arrays)
for (j = 0; j < N; j++)
    for (i = 0; i < N; i++)
        A[i][j] = B[i][j] * 2;

// Interchanged: Row-major access (good!)
for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
        A[i][j] = B[i][j] * 2;
```

### When is Interchange Legal?

Interchange is legal when it doesn't violate dependences:

```rust
pub fn can_interchange(
    program: &PolyProgram,
    loop1: &str,
    loop2: &str,
    deps: &DependenceGraph,
) -> bool {
    // Get positions of loops to interchange
    let pos1 = get_loop_position(program, loop1);
    let pos2 = get_loop_position(program, loop2);
    
    // Check all dependences
    for dep in &deps.dependences {
        let dir = &dep.direction;
        
        // After interchange, direction at pos1 and pos2 swap
        let new_dir1 = dir[pos2];
        let new_dir2 = dir[pos1];
        
        // Dependence must remain forward (< or =, not >)
        // Check lexicographic ordering is preserved
        if would_violate_dependence(dir, pos1, pos2) {
            return false;
        }
    }
    
    true
}
```

### Interchange and Dependence

```
Dependence direction: (<, =)  means: outer loop carries dependence

If we interchange:
- (<, =) becomes (=, <)  ← Still valid (inner carries)

Dependence direction: (=, <)  means: inner loop carries dependence

If we interchange:
- (=, <) becomes (<, =)  ← Still valid (outer carries)

Dependence direction: (<, >)  means: CANNOT interchange!
- (<, >) would become (>, <)  ← Backward dependence = ILLEGAL
```

---

## 3. Loop Fusion (`fusion.rs`)

### What is Fusion?

**Loop fusion** combines multiple loops with the same bounds into a single loop. Benefits:
- Reduces loop overhead
- Improves data locality (use data while it's in cache)

### Example

```c
// Before fusion: A computed, evicted from cache, then reloaded
for (i = 0; i < N; i++)
    A[i] = B[i] + 1;

for (i = 0; i < N; i++)
    C[i] = A[i] * 2;  // A[i] might be cache miss!

// After fusion: A[i] used immediately while in cache
for (i = 0; i < N; i++) {
    A[i] = B[i] + 1;
    C[i] = A[i] * 2;  // A[i] still in register/cache!
}
```

### When is Fusion Legal?

Fusion is legal when:
1. Loops have compatible iteration spaces
2. No dependence is violated (reordering statements)

```rust
pub fn can_fuse(
    loop1: &Statement,
    loop2: &Statement,
    deps: &DependenceGraph,
) -> bool {
    // 1. Check same iteration space
    if loop1.domain != loop2.domain {
        return false;
    }
    
    // 2. Check no backward dependence between loops
    for dep in deps.dependences_between(loop1, loop2) {
        // After fusion, iterations interleave:
        // L1[0], L2[0], L1[1], L2[1], ...
        // 
        // If L2[i] depends on L1[j] where j > i, we have a problem
        if dep.sink == loop1.id && dep.source == loop2.id {
            // Dependence from loop2 to loop1 at same iteration
            if dep.direction.iter().all(|d| *d == Direction::Equal) {
                continue;  // Same iteration, will still be satisfied
            }
            return false;  // Would violate dependence
        }
    }
    
    true
}
```

### Fusion Preventing Dependence

```c
// These CANNOT be fused:
for (i = 0; i < N; i++)
    A[i] = A[i+1] + 1;   // Reads A[i+1]

for (i = 0; i < N; i++)
    B[i] = A[i];         // Reads A[i]

// If fused:
// i=0: A[0] = A[1] + 1; B[0] = A[0];  -- OK so far
// i=1: A[1] = A[2] + 1; B[1] = A[1];  -- B[1] gets NEW A[1], wrong!

// Original: B[1] should get ORIGINAL A[1], before it's overwritten
```

---

## 4. Loop Skewing (`skewing.rs`)

### What is Skewing?

**Skewing** transforms the iteration space to enable parallelization when there are loop-carried dependences.

### The Problem

```c
// Wavefront pattern - cannot directly parallelize
for (i = 1; i < N; i++)
    for (j = 1; j < N; j++)
        A[i][j] = A[i-1][j] + A[i][j-1];
```

Dependences: `(<, =)` and `(=, <)` — every iteration depends on left AND above.

### The Solution: Skewing

Transform `(i, j)` to `(i, j+i)` (skew inner loop by outer):

```c
// Skewed: now we can parallelize the inner loop!
for (t = 2; t < 2*N; t++)           // t = i + j (wavefront)
    for (j = max(1,t-N); j < min(t,N); j++)  // parallel!
        i = t - j;
        A[i][j] = A[i-1][j] + A[i][j-1];
```

### Visualization

```
Original (i,j):              Skewed (t=i+j, j):
j                            t
│                            │
│ 4  5  6  7                 │    7
│ 3  4  5  6                 │   6 6
│ 2  3  4  5      →          │  5 5 5
│ 1  2  3  4                 │ 4 4 4 4
└────────────── i            │3 3 3
                             │2 2
                             │1
                             └────────────── j

Numbers show execution order. In skewed version,
each row (same t) can run in parallel!
```

### Implementation

```rust
pub fn skew_loop(
    schedule: &mut Schedule,
    outer_loop: usize,
    inner_loop: usize,
    skew_factor: i64,  // Usually 1
) {
    // Original: (i, j, ...)
    // Skewed:   (i, j + skew_factor * i, ...)
    
    let inner_dim = &mut schedule.dimensions[inner_loop];
    
    // Add outer iterator contribution to inner
    if let ScheduleDim::Iterator(ref iter) = &schedule.dimensions[outer_loop] {
        *inner_dim = ScheduleDim::Affine {
            original: inner_dim.clone(),
            added: (iter.clone(), skew_factor),
        };
    }
}
```

---

## 5. Loop Unrolling (`unrolling.rs`)

### What is Unrolling?

**Unrolling** replicates the loop body multiple times, reducing loop overhead and enabling more optimizations.

### Example

```c
// Original
for (i = 0; i < N; i++)
    A[i] = B[i] + 1;

// Unrolled by 4
for (i = 0; i < N; i += 4) {
    A[i]   = B[i]   + 1;
    A[i+1] = B[i+1] + 1;
    A[i+2] = B[i+2] + 1;
    A[i+3] = B[i+3] + 1;
}
// + cleanup loop for remainder
```

### Benefits

1. **Less overhead**: Loop counter updated N/4 times instead of N
2. **ILP**: Multiple independent operations for CPU pipeline
3. **Vectorization**: Compiler can generate SIMD instructions

### Implementation

```rust
pub fn unroll_loop(
    stmt: &Statement,
    loop_name: &str,
    unroll_factor: usize,
) -> Vec<Statement> {
    let mut unrolled = Vec::new();
    
    for offset in 0..unroll_factor {
        let mut new_stmt = stmt.clone();
        
        // Replace iterator i with i + offset
        new_stmt.substitute_iterator(loop_name, |expr| {
            expr.add_constant(offset as i64)
        });
        
        unrolled.push(new_stmt);
    }
    
    // Adjust loop bounds: i < N becomes i < N, step = unroll_factor
    
    unrolled
}
```

---

## 6. Scheduler (`scheduler.rs`)

### What is the Scheduler?

The **scheduler** automatically finds an optimal execution order (schedule) that:
1. Respects all dependences
2. Maximizes parallelism
3. Optimizes locality

### The Feautrier Algorithm

One approach to automatic scheduling:

```rust
pub fn compute_schedule(
    program: &PolyProgram,
    deps: &DependenceGraph,
) -> Schedule {
    let mut schedule_dims = Vec::new();
    let mut remaining_deps = deps.clone();
    
    // Iteratively find schedule dimensions
    while !remaining_deps.is_empty() {
        // Find an affine function that:
        // 1. Is non-negative for all dependences
        // 2. Is strictly positive for at least one dependence
        
        let (coeffs, satisfied) = solve_scheduling_ilp(&remaining_deps);
        
        schedule_dims.push(coeffs);
        
        // Remove satisfied dependences
        remaining_deps.remove_all(&satisfied);
    }
    
    Schedule { dimensions: schedule_dims }
}
```

### Pluto Algorithm (Simplified)

The Pluto algorithm optimizes for both parallelism and locality:

```rust
pub fn pluto_schedule(
    program: &PolyProgram,
    deps: &DependenceGraph,
) -> Schedule {
    // Objective: minimize dependence distances (maximize locality)
    // Subject to: all dependence distances >= 0 (legality)
    
    let mut schedule = Schedule::identity();
    
    for dim in 0..max_loop_depth(program) {
        // Solve ILP:
        // minimize: sum of dependence distances at this dimension
        // subject to: all distances >= 0
        
        let transform = solve_pluto_ilp(program, deps, dim);
        schedule.add_dimension(transform);
    }
    
    schedule
}
```

---

## 7. Transformation Pipeline (`pipeline.rs`)

### What is a Pipeline?

A **pipeline** chains multiple transformations together:

```rust
pub struct TransformPipeline {
    transforms: Vec<Box<dyn Transform>>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self { transforms: vec![] }
    }
    
    /// Add tiling
    pub fn tile(mut self, sizes: &[usize]) -> Self {
        self.transforms.push(Box::new(Tiling::new(sizes)));
        self
    }
    
    /// Add interchange
    pub fn interchange(mut self, loop1: &str, loop2: &str) -> Self {
        self.transforms.push(Box::new(Interchange::new(loop1, loop2)));
        self
    }
    
    /// Add parallelization
    pub fn parallelize(mut self, loop_name: &str) -> Self {
        self.transforms.push(Box::new(Parallelize::new(loop_name)));
        self
    }
    
    /// Execute all transformations
    pub fn apply(&self, program: &mut PolyProgram) -> Result<()> {
        for transform in &self.transforms {
            transform.apply(program)?;
        }
        Ok(())
    }
}

// Usage:
let pipeline = TransformPipeline::new()
    .interchange("i", "j")    // Fix access pattern
    .tile(&[32, 32])          // Cache blocking
    .parallelize("i");         // OpenMP outer loop

pipeline.apply(&mut program)?;
```

### Common Optimization Sequences

**For matrix multiply:**
```rust
pipeline
    .interchange("j", "k")    // Better access pattern
    .tile(&[32, 32, 32])      // Cache blocking
    .parallelize("i")          // Parallelize outer
```

**For stencils:**
```rust
pipeline
    .skew("t", "i", 1)        // Enable parallelism
    .tile(&[64, 256])         // Time-space tiling
    .parallelize("i")          // Parallelize space
```

**For reductions:**
```rust
pipeline
    .privatize("sum")         // Thread-private accumulator
    .parallelize("i")          // Parallel accumulation
    .reduce("sum", Add)       // Final reduction
```

---

## Transformation Composition

Transformations can be represented as **matrix operations** on the schedule:

```
Original schedule: S(i, j) = (i, j)

Interchange i↔j:   S'(i, j) = (j, i) = [0 1; 1 0] × (i, j)

Skew j by i:       S'(i, j) = (i, i+j) = [1 0; 1 1] × (i, j)

Tiling (32×32):    S'(i, j) = (i/32, j/32, i%32, j%32)
```

Composing transformations = multiplying matrices:

```
Interchange then skew: [1 0; 1 1] × [0 1; 1 0] = [0 1; 1 1]

This transforms (i, j) → (j, i+j)
```

---

## Key Takeaways

1. **Tiling** improves cache locality by processing data in blocks
2. **Interchange** reorders loops for better memory access
3. **Fusion** combines loops to keep data in cache
4. **Skewing** enables parallelization of wavefront computations
5. **Unrolling** reduces overhead and enables vectorization
6. **Scheduler** automatically finds good transformations
7. **Pipeline** combines transformations for compound optimization

## Performance Impact

| Transformation | Typical Speedup | When to Use |
|----------------|-----------------|-------------|
| Tiling | 2-10x | Large working sets |
| Interchange | 1.5-5x | Column-major access |
| Fusion | 1.2-3x | Sequential loops on same data |
| Parallelization | Nx (N cores) | Independent iterations |
| Unrolling | 1.2-2x | Small loop bodies |

## Further Reading

- "Loop Transformations for Restructuring Compilers" by Banerjee
- Pluto paper: "A Practical Automatic Polyhedral Parallelizer"
- "Optimizing Compilers for Modern Architectures" (Allen & Kennedy)
- Polly documentation: https://polly.llvm.org/