//! Auto-tuning Framework
//!
//! Automatically searches for optimal transformation parameters
//! by compiling and benchmarking multiple variants.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

/// Configuration for a single tuning variant
#[derive(Debug, Clone)]
pub struct TuningConfig {
    pub tile_sizes: Vec<usize>,
    pub parallel: bool,
    pub vectorize: bool,
    pub interchange: Option<(usize, usize)>,
}

impl TuningConfig {
    pub fn describe(&self) -> String {
        let tiles = self.tile_sizes.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x");
        
        let mut desc = format!("tile={}", tiles);
        if self.parallel {
            desc.push_str(",omp");
        }
        if self.vectorize {
            desc.push_str(",vec");
        }
        if let Some((i, j)) = self.interchange {
            desc.push_str(&format!(",interchange({},{})", i, j));
        }
        desc
    }
}

/// Result of benchmarking a variant
#[derive(Debug, Clone)]
pub struct TuningResult {
    pub config: TuningConfig,
    pub compile_success: bool,
    pub run_success: bool,
    pub time_seconds: f64,
    pub speedup: f64,  // vs baseline
}

/// Auto-tuner that searches the transformation space
pub struct AutoTuner {
    /// Tile sizes to try
    pub tile_sizes: Vec<usize>,
    /// Whether to try OpenMP variants
    pub try_openmp: bool,
    /// Whether to try vectorization
    pub try_vectorize: bool,
    /// Whether to try loop interchange
    pub try_interchange: bool,
    /// Number of benchmark iterations
    pub iterations: usize,
    /// Problem size for benchmarking
    pub problem_size: usize,
    /// Working directory for generated files
    pub work_dir: String,
    /// Compiler to use
    pub compiler: String,
    /// OpenMP flags
    pub omp_flags: String,
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self {
            tile_sizes: vec![8, 16, 32, 64, 128],
            try_openmp: true,
            try_vectorize: true,
            try_interchange: true,
            iterations: 3,
            problem_size: 1000,
            work_dir: "/tmp/polyopt_autotune".to_string(),
            compiler: detect_compiler(),
            omp_flags: detect_omp_flags(),
        }
    }
}

impl AutoTuner {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Generate all configurations to try
    pub fn generate_configs(&self, num_loops: usize) -> Vec<TuningConfig> {
        let mut configs = Vec::new();
        
        // Baseline: no tiling
        configs.push(TuningConfig {
            tile_sizes: vec![0; num_loops],
            parallel: false,
            vectorize: false,
            interchange: None,
        });
        
        // Try different tile sizes
        for &tile_size in &self.tile_sizes {
            let tile_sizes = vec![tile_size; num_loops];
            
            // Sequential tiled
            configs.push(TuningConfig {
                tile_sizes: tile_sizes.clone(),
                parallel: false,
                vectorize: false,
                interchange: None,
            });
            
            // Parallel tiled
            if self.try_openmp {
                configs.push(TuningConfig {
                    tile_sizes: tile_sizes.clone(),
                    parallel: true,
                    vectorize: false,
                    interchange: None,
                });
            }
            
            // Parallel + vectorized
            if self.try_openmp && self.try_vectorize {
                configs.push(TuningConfig {
                    tile_sizes: tile_sizes.clone(),
                    parallel: true,
                    vectorize: true,
                    interchange: None,
                });
            }
        }
        
        // Try interchange for 2+ loop nests
        if self.try_interchange && num_loops >= 2 {
            for &tile_size in &[32, 64] {
                configs.push(TuningConfig {
                    tile_sizes: vec![tile_size; num_loops],
                    parallel: self.try_openmp,
                    vectorize: false,
                    interchange: Some((0, 1)),
                });
            }
        }
        
        configs
    }
    
    /// Run auto-tuning on generated C code
    pub fn tune(&self, c_code: &str, kernel_name: &str) -> Result<Vec<TuningResult>, String> {
        // Create work directory
        fs::create_dir_all(&self.work_dir)
            .map_err(|e| format!("Failed to create work dir: {}", e))?;
        
        // Show compiler info
        println!("Compiler: {} ", self.compiler);
        if self.try_openmp {
            if self.omp_flags.is_empty() {
                println!("WARNING: OpenMP requested but no OpenMP support detected!");
                println!("         Install GCC: brew install gcc");
                println!("         Or libomp:   brew install libomp");
                println!();
            } else {
                println!("OpenMP flags: {}", self.omp_flags);
            }
        }
        println!();
        
        // Detect number of loops (simple heuristic: count "for" statements)
        let num_loops = c_code.matches("for (").count().max(1).min(3);
        
        let configs = self.generate_configs(num_loops);
        let mut results = Vec::new();
        
        // Get baseline time first
        let baseline_time = self.benchmark_variant(c_code, kernel_name, &configs[0])?;
        
        println!("Baseline time: {:.6}s", baseline_time);
        println!("Testing {} configurations...\n", configs.len());
        
        for (i, config) in configs.iter().enumerate() {
            print!("[{}/{}] {} ... ", i + 1, configs.len(), config.describe());
            std::io::stdout().flush().ok();
            
            match self.benchmark_variant(c_code, kernel_name, config) {
                Ok(time) => {
                    let speedup = if time > 0.0 { baseline_time / time } else { 0.0 };
                    println!("{:.6}s ({:.2}x)", time, speedup);
                    
                    results.push(TuningResult {
                        config: config.clone(),
                        compile_success: true,
                        run_success: true,
                        time_seconds: time,
                        speedup,
                    });
                }
                Err(e) => {
                    println!("FAILED: {}", e);
                    results.push(TuningResult {
                        config: config.clone(),
                        compile_success: false,
                        run_success: false,
                        time_seconds: f64::INFINITY,
                        speedup: 0.0,
                    });
                }
            }
        }
        
        // Sort by time
        results.sort_by(|a, b| a.time_seconds.partial_cmp(&b.time_seconds).unwrap());
        
        Ok(results)
    }
    
    /// Benchmark a single variant
    fn benchmark_variant(&self, c_code: &str, kernel_name: &str, config: &TuningConfig) -> Result<f64, String> {
        let variant_name = format!("{}_{}", kernel_name, config.describe().replace(",", "_").replace("(", "").replace(")", ""));
        let c_file = format!("{}/{}.c", self.work_dir, variant_name);
        let exe_file = format!("{}/{}", self.work_dir, variant_name);
        
        // Generate variant code with benchmark harness
        let modified_code = self.apply_config_to_code(c_code, config);
        let bench_code = self.wrap_with_harness(&modified_code, kernel_name);
        
        // Write C file
        fs::write(&c_file, &bench_code)
            .map_err(|e| format!("Failed to write C file: {}", e))?;
        
        // Compile
        let mut compile_args = vec![
            "-O3".to_string(),
            "-o".to_string(),
            exe_file.clone(),
            c_file.clone(),
            "-lm".to_string(),
        ];
        
        if config.parallel && !self.omp_flags.is_empty() {
            for flag in self.omp_flags.split_whitespace() {
                compile_args.push(flag.to_string());
            }
        }
        
        if config.vectorize {
            compile_args.push("-march=native".to_string());
            compile_args.push("-ffast-math".to_string());
        }
        
        let compile_output = Command::new(&self.compiler)
            .args(&compile_args)
            .output()
            .map_err(|e| format!("Compile failed: {}", e))?;
        
        if !compile_output.status.success() {
            return Err(format!("Compilation failed: {}", String::from_utf8_lossy(&compile_output.stderr)));
        }
        
        // Run benchmark
        let mut times = Vec::new();
        for _ in 0..self.iterations {
            let output = Command::new(&exe_file)
                .output()
                .map_err(|e| format!("Run failed: {}", e))?;
            
            if !output.status.success() {
                return Err("Execution failed".to_string());
            }
            
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(time) = parse_time(&stdout) {
                times.push(time);
            }
        }
        
        if times.is_empty() {
            return Err("No timing data".to_string());
        }
        
        // Return median
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(times[times.len() / 2])
    }
    
    /// Apply configuration transformations to C code
    fn apply_config_to_code(&self, code: &str, config: &TuningConfig) -> String {
        let mut result = code.to_string();
        
        if config.parallel {
            // Add OpenMP pragma if parallel and not already present
            if !result.contains("#pragma omp") {
                if let Some(pos) = result.find("for (") {
                    let pragma = "#pragma omp parallel for\n    ";
                    result.insert_str(pos, pragma);
                }
            }
        } else {
            // Remove OpenMP includes and pragmas for sequential variants
            result = result
                .lines()
                .filter(|line| !line.contains("#include <omp.h>"))
                .filter(|line| !line.contains("#pragma omp"))
                .collect::<Vec<_>>()
                .join("\n");
        }
        
        // Note: Real tiling would require AST transformation
        // For now, we rely on compiler auto-vectorization
        
        result
    }
    
    /// Wrap code with timing harness
    fn wrap_with_harness(&self, code: &str, kernel_name: &str) -> String {
        // Strip the original headers since we'll add our own
        let code_stripped = code.lines()
            .filter(|line| !line.starts_with("#include"))
            .filter(|line| !line.starts_with("// Generated"))
            .collect::<Vec<_>>()
            .join("\n");
        
        let harness = format!(r#"#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

double get_time() {{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}}

{code}

int main() {{
    int N = {size};
    
    // Allocate arrays (large enough for 1D or 2D use)
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)calloc(N * N, sizeof(double));
    
    if (!A || !B || !C) {{
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }}
    
    // Initialize
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
    printf("Check: %.4f\n", C[0]);
    
    free(A); free(B); free(C);
    return 0;
}}
"#, code = code_stripped, size = self.problem_size, kernel = kernel_name);
        
        harness
    }
}

/// Detect available C compiler with OpenMP support
fn detect_compiler() -> String {
    // First try GCC versions which have built-in OpenMP
    for compiler in &["gcc-14", "gcc-13", "gcc-12", "gcc-11", "gcc"] {
        if let Ok(output) = Command::new(compiler).arg("--version").output() {
            if output.status.success() {
                // Verify it actually supports -fopenmp
                let test = Command::new(compiler)
                    .args(&["-fopenmp", "-x", "c", "-c", "-", "-o", "/dev/null"])
                    .stdin(std::process::Stdio::null())
                    .output();
                
                if test.map(|o| o.status.success()).unwrap_or(false) {
                    return compiler.to_string();
                }
            }
        }
    }
    
    // Fall back to clang (may need libomp on macOS)
    "clang".to_string()
}

/// Detect OpenMP flags for the compiler
fn detect_omp_flags() -> String {
    let compiler = detect_compiler();
    
    if compiler.starts_with("gcc") {
        return "-fopenmp".to_string();
    }
    
    // Try clang with libomp (macOS with Homebrew)
    if Path::new("/opt/homebrew/opt/libomp").exists() {
        return "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp".to_string();
    }
    if Path::new("/usr/local/opt/libomp").exists() {
        return "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include -L/usr/local/opt/libomp/lib -lomp".to_string();
    }
    
    // Linux clang typically has OpenMP support
    "-fopenmp".to_string()
}

/// Parse time from benchmark output
fn parse_time(output: &str) -> Option<f64> {
    for line in output.lines() {
        if line.starts_with("Time:") {
            return line.split_whitespace().nth(1)?.parse().ok();
        }
    }
    None
}

/// Search strategy for auto-tuning
#[derive(Debug, Clone, Copy)]
pub enum SearchStrategy {
    /// Try all configurations
    Exhaustive,
    /// Random sampling
    Random { samples: usize },
    /// Genetic algorithm
    Genetic { population: usize, generations: usize },
    /// Simulated annealing
    Annealing { iterations: usize },
}

/// Advanced auto-tuner with multiple search strategies
pub struct AdvancedAutoTuner {
    pub base: AutoTuner,
    pub strategy: SearchStrategy,
}

impl AdvancedAutoTuner {
    pub fn new(strategy: SearchStrategy) -> Self {
        Self {
            base: AutoTuner::default(),
            strategy,
        }
    }
    
    /// Run tuning with the selected strategy
    pub fn tune(&self, c_code: &str, kernel_name: &str) -> Result<Vec<TuningResult>, String> {
        match self.strategy {
            SearchStrategy::Exhaustive => self.base.tune(c_code, kernel_name),
            SearchStrategy::Random { samples } => self.tune_random(c_code, kernel_name, samples),
            SearchStrategy::Genetic { population, generations } => {
                self.tune_genetic(c_code, kernel_name, population, generations)
            }
            SearchStrategy::Annealing { iterations } => {
                self.tune_annealing(c_code, kernel_name, iterations)
            }
        }
    }
    
    /// Random sampling
    fn tune_random(&self, c_code: &str, kernel_name: &str, samples: usize) -> Result<Vec<TuningResult>, String> {
        use std::collections::HashSet;
        
        let num_loops = c_code.matches("for (").count().max(1).min(3);
        let all_configs = self.base.generate_configs(num_loops);
        
        // Randomly select configurations
        let mut selected = HashSet::new();
        let mut rng_state = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as usize;
        
        while selected.len() < samples.min(all_configs.len()) {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let idx = rng_state % all_configs.len();
            selected.insert(idx);
        }
        
        let configs: Vec<_> = selected.iter()
            .map(|&i| all_configs[i].clone())
            .collect();
        
        // Benchmark selected configs
        let mut results = Vec::new();
        let baseline_time = self.base.benchmark_variant(c_code, kernel_name, &all_configs[0])?;
        
        println!("Random sampling {} configurations...\n", configs.len());
        
        for config in &configs {
            print!("{} ... ", config.describe());
            std::io::stdout().flush().ok();
            
            match self.base.benchmark_variant(c_code, kernel_name, config) {
                Ok(time) => {
                    let speedup = baseline_time / time;
                    println!("{:.6}s ({:.2}x)", time, speedup);
                    results.push(TuningResult {
                        config: config.clone(),
                        compile_success: true,
                        run_success: true,
                        time_seconds: time,
                        speedup,
                    });
                }
                Err(e) => {
                    println!("FAILED: {}", e);
                }
            }
        }
        
        results.sort_by(|a, b| a.time_seconds.partial_cmp(&b.time_seconds).unwrap());
        Ok(results)
    }
    
    /// Genetic algorithm tuning
    fn tune_genetic(&self, c_code: &str, kernel_name: &str, pop_size: usize, generations: usize) -> Result<Vec<TuningResult>, String> {
        let num_loops = c_code.matches("for (").count().max(1).min(3);
        let tile_options = &self.base.tile_sizes;
        
        // Initialize population
        let mut population: Vec<TuningConfig> = Vec::new();
        let mut rng_state = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as usize;
        
        for _ in 0..pop_size {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let tile_idx = rng_state % tile_options.len();
            let parallel = (rng_state / tile_options.len()) % 2 == 1;
            
            population.push(TuningConfig {
                tile_sizes: vec![tile_options[tile_idx]; num_loops],
                parallel,
                vectorize: parallel && (rng_state % 4 == 0),
                interchange: None,
            });
        }
        
        let baseline_config = TuningConfig {
            tile_sizes: vec![0; num_loops],
            parallel: false,
            vectorize: false,
            interchange: None,
        };
        let baseline_time = self.base.benchmark_variant(c_code, kernel_name, &baseline_config)?;
        
        println!("Genetic algorithm: {} population, {} generations\n", pop_size, generations);
        
        let mut best_results: Vec<TuningResult> = Vec::new();
        
        for gen in 0..generations {
            println!("Generation {}/{}", gen + 1, generations);
            
            // Evaluate fitness
            let mut fitness: Vec<(TuningConfig, f64)> = Vec::new();
            for config in &population {
                match self.base.benchmark_variant(c_code, kernel_name, config) {
                    Ok(time) => fitness.push((config.clone(), 1.0 / time)),
                    Err(_) => fitness.push((config.clone(), 0.0)),
                }
            }
            
            // Sort by fitness
            fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Record best
            if let Some((best_config, best_fitness)) = fitness.first() {
                if best_fitness > &0.0 {
                    let time = 1.0 / best_fitness;
                    println!("  Best: {} -> {:.6}s ({:.2}x)", 
                             best_config.describe(), time, baseline_time / time);
                    
                    best_results.push(TuningResult {
                        config: best_config.clone(),
                        compile_success: true,
                        run_success: true,
                        time_seconds: time,
                        speedup: baseline_time / time,
                    });
                }
            }
            
            // Selection: top 50%
            let survivors: Vec<_> = fitness.iter()
                .take(pop_size / 2)
                .map(|(c, _)| c.clone())
                .collect();
            
            // Crossover and mutation
            population.clear();
            population.extend(survivors.clone());
            
            while population.len() < pop_size {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let parent1 = &survivors[rng_state % survivors.len()];
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let parent2 = &survivors[rng_state % survivors.len()];
                
                // Crossover
                let mut child = TuningConfig {
                    tile_sizes: if rng_state % 2 == 0 { parent1.tile_sizes.clone() } else { parent2.tile_sizes.clone() },
                    parallel: if (rng_state / 2) % 2 == 0 { parent1.parallel } else { parent2.parallel },
                    vectorize: parent1.vectorize || parent2.vectorize,
                    interchange: None,
                };
                
                // Mutation (10% chance)
                if rng_state % 10 == 0 {
                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    let new_tile = tile_options[rng_state % tile_options.len()];
                    child.tile_sizes = vec![new_tile; num_loops];
                }
                
                population.push(child);
            }
        }
        
        best_results.sort_by(|a, b| a.time_seconds.partial_cmp(&b.time_seconds).unwrap());
        best_results.dedup_by(|a, b| a.config.describe() == b.config.describe());
        Ok(best_results)
    }
    
    /// Simulated annealing
    fn tune_annealing(&self, c_code: &str, kernel_name: &str, iterations: usize) -> Result<Vec<TuningResult>, String> {
        let num_loops = c_code.matches("for (").count().max(1).min(3);
        let tile_options = &self.base.tile_sizes;
        
        let mut rng_state = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as usize;
        
        // Start with a random config
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let mut current = TuningConfig {
            tile_sizes: vec![tile_options[rng_state % tile_options.len()]; num_loops],
            parallel: true,
            vectorize: false,
            interchange: None,
        };
        
        let baseline_config = TuningConfig {
            tile_sizes: vec![0; num_loops],
            parallel: false,
            vectorize: false,
            interchange: None,
        };
        let baseline_time = self.base.benchmark_variant(c_code, kernel_name, &baseline_config)?;
        
        let mut current_time = self.base.benchmark_variant(c_code, kernel_name, &current)
            .unwrap_or(f64::INFINITY);
        
        let mut best = current.clone();
        let mut best_time = current_time;
        
        let mut results = Vec::new();
        
        println!("Simulated annealing: {} iterations\n", iterations);
        
        for i in 0..iterations {
            let temperature = 1.0 - (i as f64 / iterations as f64);
            
            // Generate neighbor
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let mut neighbor = current.clone();
            
            match rng_state % 3 {
                0 => {
                    // Change tile size
                    rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                    neighbor.tile_sizes = vec![tile_options[rng_state % tile_options.len()]; num_loops];
                }
                1 => {
                    // Toggle parallel
                    neighbor.parallel = !neighbor.parallel;
                }
                _ => {
                    // Toggle vectorize
                    neighbor.vectorize = !neighbor.vectorize;
                }
            }
            
            if let Ok(neighbor_time) = self.base.benchmark_variant(c_code, kernel_name, &neighbor) {
                let delta = neighbor_time - current_time;
                
                // Accept if better, or probabilistically if worse
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let random_val = (rng_state % 1000) as f64 / 1000.0;
                
                if delta < 0.0 || random_val < (-delta / temperature).exp() {
                    current = neighbor;
                    current_time = neighbor_time;
                    
                    if current_time < best_time {
                        best = current.clone();
                        best_time = current_time;
                        println!("[{}] New best: {} -> {:.6}s ({:.2}x)", 
                                 i, best.describe(), best_time, baseline_time / best_time);
                        
                        results.push(TuningResult {
                            config: best.clone(),
                            compile_success: true,
                            run_success: true,
                            time_seconds: best_time,
                            speedup: baseline_time / best_time,
                        });
                    }
                }
            }
        }
        
        results.sort_by(|a, b| a.time_seconds.partial_cmp(&b.time_seconds).unwrap());
        Ok(results)
    }
}

/// Print tuning results summary
pub fn print_results(results: &[TuningResult]) {
    println!("\n{}", "═".repeat(60));
    println!("                    TUNING RESULTS");
    println!("{}\n", "═".repeat(60));
    
    println!("{:<35} {:>10} {:>10}", "Configuration", "Time (s)", "Speedup");
    println!("{}", "─".repeat(60));
    
    for (i, result) in results.iter().take(10).enumerate() {
        let marker = if i == 0 { "★" } else { " " };
        println!("{} {:<33} {:>10.6} {:>9.2}x", 
                 marker,
                 result.config.describe(),
                 result.time_seconds,
                 result.speedup);
    }
    
    if !results.is_empty() {
        println!("\n★ Best configuration: {}", results[0].config.describe());
        println!("  Speedup: {:.2}x over baseline", results[0].speedup);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_generation() {
        let tuner = AutoTuner::default();
        let configs = tuner.generate_configs(2);
        assert!(configs.len() > 10);
    }
    
    #[test]
    fn test_config_describe() {
        let config = TuningConfig {
            tile_sizes: vec![32, 32],
            parallel: true,
            vectorize: false,
            interchange: None,
        };
        assert!(config.describe().contains("32x32"));
        assert!(config.describe().contains("omp"));
    }
}