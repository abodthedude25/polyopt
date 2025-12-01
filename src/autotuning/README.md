# Auto-Tuning Module

## Overview

The **auto-tuning module** automatically discovers optimal transformation parameters by compiling and benchmarking multiple variants. Instead of guessing tile sizes or parallelization strategies, auto-tuning finds the best configuration empirically.

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTO-TUNING FLOW                          │
│                                                             │
│  Input Code                                                 │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────┐                                          │
│  │   Generate   │  ← Create many variants                  │
│  │   Variants   │    (tile sizes, parallel, etc.)          │
│  └──────────────┘                                          │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────┐                                          │
│  │   Compile    │  ← gcc/clang with optimization flags     │
│  │   Each       │                                          │
│  └──────────────┘                                          │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────┐                                          │
│  │  Benchmark   │  ← Time each variant                     │
│  │   Each       │                                          │
│  └──────────────┘                                          │
│      │                                                      │
│      ▼                                                      │
│  Best Configuration                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Files in This Module

| File | Purpose |
|------|---------|
| `config.rs` | Configuration structures |
| `search.rs` | Search strategies |
| `runner.rs` | Compilation and benchmarking |
| `results.rs` | Result storage and analysis |
| `mod.rs` | Module exports |

Also: `autotune.rs` in src root provides CLI integration.

---

## 1. Why Auto-Tuning?

### The Problem

Optimal parameters depend on:
- **Hardware**: Cache sizes, core count, SIMD width
- **Problem size**: Small vs large matrices
- **Data patterns**: Dense vs sparse, regular vs irregular
- **Compiler**: GCC vs Clang, version, flags

### Example: Tile Size

```
Tile size 8:   Fast for small arrays (fits L1 cache)
Tile size 32:  Best for medium arrays (fits L2 cache)
Tile size 128: Optimal for large arrays (minimizes TLB misses)

But which is best for YOUR machine with YOUR data?
```

### The Solution

Instead of guessing, **try many configurations and measure**.

---

## 2. Configuration (`config.rs`)

### What Can Be Tuned?

```rust
/// A single tuning configuration
#[derive(Clone, Debug)]
pub struct TuningConfig {
    /// Tile sizes for each loop dimension
    /// Example: [32, 32] for 2D tiling
    pub tile_sizes: Vec<usize>,
    
    /// Enable OpenMP parallelization
    pub parallel: bool,
    
    /// Enable compiler vectorization hints
    pub vectorize: bool,
    
    /// Loop interchange specification
    /// Example: Some((0, 1)) to swap first two loops
    pub interchange: Option<(usize, usize)>,
    
    /// Unroll factor (1 = no unrolling)
    pub unroll_factor: usize,
}

impl TuningConfig {
    /// Human-readable description
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        
        // Tile sizes
        let tiles: String = self.tile_sizes.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x");
        parts.push(format!("tile={}", tiles));
        
        // Flags
        if self.parallel {
            parts.push("omp".to_string());
        }
        if self.vectorize {
            parts.push("vec".to_string());
        }
        if let Some((i, j)) = self.interchange {
            parts.push(format!("swap({},{})", i, j));
        }
        if self.unroll_factor > 1 {
            parts.push(format!("unroll={}", self.unroll_factor));
        }
        
        parts.join(",")
    }
}
```

### Configuration Space

The **configuration space** is all possible combinations:

```rust
pub struct ConfigSpace {
    /// Possible tile sizes to try
    pub tile_sizes: Vec<usize>,      // e.g., [8, 16, 32, 64, 128]
    
    /// Try with/without OpenMP
    pub try_parallel: bool,
    
    /// Try with/without vectorization
    pub try_vectorize: bool,
    
    /// Try loop interchanges
    pub try_interchange: bool,
    
    /// Unroll factors to try
    pub unroll_factors: Vec<usize>,  // e.g., [1, 2, 4, 8]
}

impl ConfigSpace {
    /// Generate all configurations
    pub fn enumerate(&self, num_loops: usize) -> Vec<TuningConfig> {
        let mut configs = Vec::new();
        
        // Baseline (no optimization)
        configs.push(TuningConfig::baseline());
        
        // All tile size combinations
        for tile in &self.tile_sizes {
            let tile_vec = vec![*tile; num_loops];
            
            // Sequential
            configs.push(TuningConfig {
                tile_sizes: tile_vec.clone(),
                parallel: false,
                vectorize: false,
                interchange: None,
                unroll_factor: 1,
            });
            
            // With OpenMP
            if self.try_parallel {
                configs.push(TuningConfig {
                    tile_sizes: tile_vec.clone(),
                    parallel: true,
                    vectorize: false,
                    ..Default::default()
                });
            }
            
            // With vectorization
            if self.try_vectorize {
                configs.push(TuningConfig {
                    tile_sizes: tile_vec.clone(),
                    parallel: true,
                    vectorize: true,
                    ..Default::default()
                });
            }
        }
        
        configs
    }
    
    /// Number of configurations
    pub fn size(&self, num_loops: usize) -> usize {
        // Rough estimate
        self.tile_sizes.len() * 
        (1 + self.try_parallel as usize) *
        (1 + self.try_vectorize as usize)
    }
}
```

---

## 3. Search Strategies (`search.rs`)

### Exhaustive Search

Try **all** configurations. Best quality, but expensive.

```rust
pub struct ExhaustiveSearch {
    space: ConfigSpace,
}

impl SearchStrategy for ExhaustiveSearch {
    fn search(&self, evaluator: &dyn Evaluator) -> TuningResult {
        let configs = self.space.enumerate(evaluator.num_loops());
        let mut best = None;
        let mut best_time = f64::MAX;
        
        for config in configs {
            let time = evaluator.evaluate(&config);
            if time < best_time {
                best_time = time;
                best = Some(config);
            }
        }
        
        TuningResult { config: best.unwrap(), time: best_time }
    }
}
```

**Complexity**: O(n) where n = number of configurations.

### Random Search

Sample **random** configurations. Surprisingly effective!

```rust
pub struct RandomSearch {
    space: ConfigSpace,
    num_samples: usize,
}

impl SearchStrategy for RandomSearch {
    fn search(&self, evaluator: &dyn Evaluator) -> TuningResult {
        let mut rng = rand::thread_rng();
        let mut best = None;
        let mut best_time = f64::MAX;
        
        for _ in 0..self.num_samples {
            let config = self.space.random_sample(&mut rng);
            let time = evaluator.evaluate(&config);
            
            if time < best_time {
                best_time = time;
                best = Some(config);
            }
        }
        
        TuningResult { config: best.unwrap(), time: best_time }
    }
}
```

**Advantage**: Finds good solutions with far fewer evaluations than exhaustive.

### Genetic Algorithm

Use **evolution** to find good configurations:

```rust
pub struct GeneticSearch {
    population_size: usize,
    generations: usize,
    mutation_rate: f64,
    crossover_rate: f64,
}

impl SearchStrategy for GeneticSearch {
    fn search(&self, evaluator: &dyn Evaluator) -> TuningResult {
        let mut rng = rand::thread_rng();
        
        // Initialize population
        let mut population: Vec<TuningConfig> = (0..self.population_size)
            .map(|_| self.space.random_sample(&mut rng))
            .collect();
        
        for _gen in 0..self.generations {
            // Evaluate fitness (lower time = higher fitness)
            let mut fitness: Vec<(TuningConfig, f64)> = population.iter()
                .map(|c| (c.clone(), 1.0 / evaluator.evaluate(c)))
                .collect();
            
            // Sort by fitness
            fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Selection: keep top 50%
            let survivors: Vec<_> = fitness.iter()
                .take(self.population_size / 2)
                .map(|(c, _)| c.clone())
                .collect();
            
            // Reproduction
            population.clear();
            while population.len() < self.population_size {
                let parent1 = &survivors[rng.gen_range(0..survivors.len())];
                let parent2 = &survivors[rng.gen_range(0..survivors.len())];
                
                // Crossover
                let mut child = if rng.gen::<f64>() < self.crossover_rate {
                    self.crossover(parent1, parent2, &mut rng)
                } else {
                    parent1.clone()
                };
                
                // Mutation
                if rng.gen::<f64>() < self.mutation_rate {
                    self.mutate(&mut child, &mut rng);
                }
                
                population.push(child);
            }
        }
        
        // Return best from final population
        population.into_iter()
            .map(|c| {
                let time = evaluator.evaluate(&c);
                TuningResult { config: c, time }
            })
            .min_by(|a, b| a.time.partial_cmp(&b.time).unwrap())
            .unwrap()
    }
    
    fn crossover(&self, p1: &TuningConfig, p2: &TuningConfig, rng: &mut impl Rng) -> TuningConfig {
        TuningConfig {
            tile_sizes: if rng.gen() { p1.tile_sizes.clone() } else { p2.tile_sizes.clone() },
            parallel: if rng.gen() { p1.parallel } else { p2.parallel },
            vectorize: if rng.gen() { p1.vectorize } else { p2.vectorize },
            interchange: if rng.gen() { p1.interchange } else { p2.interchange },
            unroll_factor: if rng.gen() { p1.unroll_factor } else { p2.unroll_factor },
        }
    }
    
    fn mutate(&self, config: &mut TuningConfig, rng: &mut impl Rng) {
        match rng.gen_range(0..4) {
            0 => {
                // Mutate tile size
                let idx = rng.gen_range(0..config.tile_sizes.len());
                config.tile_sizes[idx] = *self.space.tile_sizes.choose(rng).unwrap();
            }
            1 => config.parallel = !config.parallel,
            2 => config.vectorize = !config.vectorize,
            3 => config.unroll_factor = *self.space.unroll_factors.choose(rng).unwrap_or(&1),
            _ => {}
        }
    }
}
```

**Advantage**: Good for large search spaces; converges to good solutions.

### Simulated Annealing

Explore with decreasing "temperature":

```rust
pub struct SimulatedAnnealing {
    initial_temp: f64,
    cooling_rate: f64,
    iterations: usize,
}

impl SearchStrategy for SimulatedAnnealing {
    fn search(&self, evaluator: &dyn Evaluator) -> TuningResult {
        let mut rng = rand::thread_rng();
        let mut current = self.space.random_sample(&mut rng);
        let mut current_time = evaluator.evaluate(&current);
        let mut best = current.clone();
        let mut best_time = current_time;
        let mut temp = self.initial_temp;
        
        for _ in 0..self.iterations {
            // Generate neighbor
            let neighbor = self.neighbor(&current, &mut rng);
            let neighbor_time = evaluator.evaluate(&neighbor);
            
            // Accept or reject
            let delta = neighbor_time - current_time;
            if delta < 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
                current = neighbor;
                current_time = neighbor_time;
                
                if current_time < best_time {
                    best = current.clone();
                    best_time = current_time;
                }
            }
            
            // Cool down
            temp *= self.cooling_rate;
        }
        
        TuningResult { config: best, time: best_time }
    }
    
    fn neighbor(&self, config: &TuningConfig, rng: &mut impl Rng) -> TuningConfig {
        let mut neighbor = config.clone();
        // Small random modification
        // ...
        neighbor
    }
}
```

**Advantage**: Escapes local minima; good for noisy landscapes.

---

## 4. Benchmarking (`runner.rs`)

### Compilation

```rust
pub struct CompileRunner {
    compiler: String,         // "gcc-13", "clang"
    base_flags: Vec<String>,  // ["-O3"]
    omp_flags: String,        // "-fopenmp"
    work_dir: PathBuf,
}

impl CompileRunner {
    pub fn compile(&self, code: &str, config: &TuningConfig) -> Result<PathBuf, Error> {
        // Write source file
        let src_path = self.work_dir.join("variant.c");
        fs::write(&src_path, code)?;
        
        // Build compile command
        let exe_path = self.work_dir.join("variant");
        let mut args = self.base_flags.clone();
        args.push("-o".to_string());
        args.push(exe_path.to_string_lossy().to_string());
        args.push(src_path.to_string_lossy().to_string());
        
        // Add OpenMP if parallel
        if config.parallel {
            for flag in self.omp_flags.split_whitespace() {
                args.push(flag.to_string());
            }
        }
        
        // Add vectorization flags
        if config.vectorize {
            args.push("-march=native".to_string());
            args.push("-ffast-math".to_string());
        }
        
        // Compile
        let output = Command::new(&self.compiler)
            .args(&args)
            .output()?;
        
        if !output.status.success() {
            return Err(Error::CompileFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }
        
        Ok(exe_path)
    }
}
```

### Benchmarking

```rust
pub struct BenchmarkRunner {
    iterations: usize,
    warmup_iterations: usize,
}

impl BenchmarkRunner {
    pub fn benchmark(&self, exe_path: &Path) -> Result<f64, Error> {
        // Warmup runs (not measured)
        for _ in 0..self.warmup_iterations {
            Command::new(exe_path).output()?;
        }
        
        // Measured runs
        let mut times = Vec::new();
        for _ in 0..self.iterations {
            let output = Command::new(exe_path).output()?;
            
            if !output.status.success() {
                return Err(Error::RuntimeFailed);
            }
            
            // Parse time from output
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(time) = self.parse_time(&stdout) {
                times.push(time);
            }
        }
        
        // Return median (robust to outliers)
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(times[times.len() / 2])
    }
    
    fn parse_time(&self, output: &str) -> Option<f64> {
        // Look for "Time: X.XXXXXX" pattern
        for line in output.lines() {
            if line.starts_with("Time:") {
                return line.split_whitespace().nth(1)?.parse().ok();
            }
        }
        None
    }
}
```

### Benchmark Harness

The code is wrapped with timing:

```rust
fn wrap_with_harness(code: &str, kernel_name: &str, problem_size: usize) -> String {
    format!(r#"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_time() {{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}}

{code}

int main() {{
    int N = {size};
    
    // Allocate and initialize arrays
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)calloc(N * N, sizeof(double));
    
    for (int i = 0; i < N * N; i++) {{
        A[i] = (i % 100) * 0.01;
        B[i] = (i % 100) * 0.01;
    }}
    
    // Warmup
    {kernel}(N, A, B, C);
    
    // Benchmark
    double start = get_time();
    for (int rep = 0; rep < 3; rep++) {{
        {kernel}(N, A, B, C);
    }}
    double end = get_time();
    
    printf("Time: %.6f\n", (end - start) / 3.0);
    printf("Check: %.4f\n", C[0]);  // Prevent dead code elimination
    
    free(A); free(B); free(C);
    return 0;
}}
"#, code = code, kernel = kernel_name, size = problem_size)
}
```

---

## 5. Results (`results.rs`)

### Result Structure

```rust
#[derive(Clone, Debug)]
pub struct TuningResult {
    pub config: TuningConfig,
    pub compile_success: bool,
    pub run_success: bool,
    pub time_seconds: f64,
    pub speedup: f64,  // vs baseline
}

pub struct TuningReport {
    pub kernel_name: String,
    pub problem_size: usize,
    pub baseline_time: f64,
    pub results: Vec<TuningResult>,
    pub best: TuningResult,
}

impl TuningReport {
    /// Sort results by time
    pub fn sorted_by_time(&self) -> Vec<&TuningResult> {
        let mut sorted: Vec<_> = self.results.iter().collect();
        sorted.sort_by(|a, b| a.time_seconds.partial_cmp(&b.time_seconds).unwrap());
        sorted
    }
    
    /// Export to CSV
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("config,time_seconds,speedup\n");
        
        for result in &self.results {
            csv.push_str(&format!(
                "{},{:.6},{:.2}\n",
                result.config.describe(),
                result.time_seconds,
                result.speedup
            ));
        }
        
        csv
    }
    
    /// Pretty print
    pub fn display(&self) {
        println!("════════════════════════════════════════════════════════════");
        println!("                    TUNING RESULTS");
        println!("════════════════════════════════════════════════════════════");
        println!();
        println!("{:<40} {:>12} {:>10}", "Configuration", "Time (s)", "Speedup");
        println!("────────────────────────────────────────────────────────────");
        
        for (i, result) in self.sorted_by_time().iter().enumerate() {
            let marker = if i == 0 { "★" } else { " " };
            println!(
                "{} {:<38} {:>12.6} {:>9.2}x",
                marker,
                result.config.describe(),
                result.time_seconds,
                result.speedup
            );
        }
        
        println!();
        println!("★ Best configuration: {}", self.best.config.describe());
        println!("  Speedup: {:.2}x over baseline", self.best.speedup);
    }
}
```

---

## 6. CLI Usage

### Basic Tuning

```bash
# Auto-tune with defaults
polyopt autotune kernel.poly

# Specify problem size
polyopt autotune kernel.poly -N 1000

# Enable OpenMP variants
polyopt autotune kernel.poly --openmp

# Custom tile sizes
polyopt autotune kernel.poly --tiles "16,32,64,128"

# Save results
polyopt autotune kernel.poly -o results.csv
```

### Search Strategies

```bash
# Exhaustive (try all)
polyopt autotune kernel.poly --strategy exhaustive

# Random sampling (faster)
polyopt autotune kernel.poly --strategy random

# Genetic algorithm
polyopt autotune kernel.poly --strategy genetic

# Simulated annealing
polyopt autotune kernel.poly --strategy annealing
```

### Example Session

```
$ polyopt autotune matmul.poly --openmp -N 500

╔════════════════════════════════════════════════════════════╗
║              PolyOpt Auto-Tuning Framework                 ║
╚════════════════════════════════════════════════════════════╝

Input: matmul.poly
Configuration:
  Tile sizes to try: [8, 16, 32, 64, 128]
  Problem size: N=500
  Iterations per variant: 3
  OpenMP: true
  Strategy: Exhaustive

Compiler: gcc-13 
OpenMP flags: -fopenmp

Baseline time: 0.892341s
Testing 16 configurations...

[1/16] tile=0 ... 0.892341s (1.00x)
[2/16] tile=8 ... 0.423156s (2.11x)
[3/16] tile=8,omp ... 0.098234s (9.08x)
[4/16] tile=16 ... 0.312456s (2.86x)
[5/16] tile=16,omp ... 0.067891s (13.14x)
[6/16] tile=32 ... 0.287654s (3.10x)
[7/16] tile=32,omp ... 0.054321s (16.43x) ★
...

════════════════════════════════════════════════════════════
                    TUNING RESULTS
════════════════════════════════════════════════════════════

Configuration                         Time (s)    Speedup
────────────────────────────────────────────────────────────
★ tile=32x32x32,omp                   0.054321     16.43x
  tile=64x64x64,omp                   0.058234     15.32x
  tile=16x16x16,omp                   0.067891     13.14x
  tile=32x32x32,omp,vec               0.071234     12.53x
  ...

★ Best configuration: tile=32x32x32,omp
  Speedup: 16.43x over baseline
```

---

## 7. Comparison of Strategies

| Strategy | Configs Tried | Quality | Best For |
|----------|---------------|---------|----------|
| **Exhaustive** | All | Optimal | Small spaces (<100 configs) |
| **Random** | N samples | Good | Medium spaces, quick results |
| **Genetic** | Pop × Gen | Very Good | Large spaces, complex interactions |
| **Annealing** | Iterations | Good | Noisy measurements, local minima |

### When to Use What

- **Quick exploration**: Random with 20-50 samples
- **Production tuning**: Genetic or exhaustive
- **Very large spaces**: Genetic algorithm
- **Noisy hardware**: Simulated annealing

---

## Key Takeaways

1. **Auto-tuning** finds optimal parameters empirically
2. **Multiple strategies** trade off thoroughness vs speed
3. **Benchmarking** must be accurate (warmup, multiple runs, median)
4. **Results** show clear speedups from optimization
5. **Hardware-specific** tuning matters more than universal heuristics

## Further Reading

- OpenTuner: An extensible auto-tuning framework
- ATLAS: Automatically Tuned Linear Algebra Software
- SPIRAL: Automatic algorithm generation and tuning
- "Auto-tuning" on Wikipedia