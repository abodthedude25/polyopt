//! Auto-Tuning Framework
//!
//! Automatically finds the best optimization parameters for a given program
//! by exploring different configurations and benchmarking them.
//!
//! # Features
//!
//! - **Tile size tuning**: Find optimal tile sizes for tiled loops
//! - **Loop order exploration**: Try different permutations
//! - **Unroll factor tuning**: Find best unroll factors
//! - **Parallelization strategies**: Compare different OpenMP schedules
//!
//! # Example
//!
//! ```ignore
//! use polyopt::autotuning::{AutoTuner, TuningConfig};
//!
//! let config = TuningConfig::default()
//!     .tile_sizes(vec![16, 32, 64, 128])
//!     .max_time_seconds(60);
//!
//! let tuner = AutoTuner::new(config);
//! let best = tuner.tune("examples/matmul.poly", &["N=1000"])?;
//! println!("Best config: {:?}", best);
//! ```

mod search;
mod config;
mod runner;
mod results;

pub use config::{TuningConfig, TuningSpace, TuningParameter};
pub use search::{SearchStrategy, RandomSearch, GridSearch, GeneticSearch};
pub use runner::{BenchmarkRunner, BenchmarkResult};
pub use results::{TuningResults, ConfigResult};

use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TuningError {
    #[error("Compilation failed: {0}")]
    CompilationFailed(String),
    
    #[error("Benchmark failed: {0}")]
    BenchmarkFailed(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Timeout exceeded")]
    Timeout,
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type TuningResult<T> = Result<T, TuningError>;

/// A configuration to test
#[derive(Clone, Debug, PartialEq)]
pub struct Configuration {
    /// Tile sizes for each tiled loop (empty = no tiling)
    pub tile_sizes: Vec<usize>,
    
    /// Loop permutation (indices into original loop order)
    pub loop_order: Vec<usize>,
    
    /// Unroll factors for each loop
    pub unroll_factors: Vec<usize>,
    
    /// Number of OpenMP threads (0 = sequential)
    pub num_threads: usize,
    
    /// OpenMP schedule type
    pub omp_schedule: OmpSchedule,
    
    /// Enable vectorization
    pub vectorize: bool,
}

/// OpenMP schedule types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OmpSchedule {
    #[default]
    Static,
    Dynamic,
    Guided,
    Auto,
}

impl std::fmt::Display for OmpSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OmpSchedule::Static => write!(f, "static"),
            OmpSchedule::Dynamic => write!(f, "dynamic"),
            OmpSchedule::Guided => write!(f, "guided"),
            OmpSchedule::Auto => write!(f, "auto"),
        }
    }
}

impl Default for Configuration {
    fn default() -> Self {
        Self {
            tile_sizes: vec![],
            loop_order: vec![],
            unroll_factors: vec![],
            num_threads: 0,
            omp_schedule: OmpSchedule::Static,
            vectorize: false,
        }
    }
}

impl Configuration {
    /// Create a configuration with tiling
    pub fn with_tiling(tile_sizes: Vec<usize>) -> Self {
        Self {
            tile_sizes,
            ..Default::default()
        }
    }
    
    /// Create a parallel configuration
    pub fn parallel(num_threads: usize) -> Self {
        Self {
            num_threads,
            ..Default::default()
        }
    }
    
    /// Create a fully optimized configuration
    pub fn optimized(tile_sizes: Vec<usize>, num_threads: usize) -> Self {
        Self {
            tile_sizes,
            num_threads,
            omp_schedule: OmpSchedule::Static,
            vectorize: true,
            ..Default::default()
        }
    }
    
    /// Get a string representation for file naming
    pub fn to_string_compact(&self) -> String {
        let mut parts = vec![];
        
        if !self.tile_sizes.is_empty() {
            parts.push(format!("t{}", self.tile_sizes.iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x")));
        }
        
        if self.num_threads > 0 {
            parts.push(format!("p{}", self.num_threads));
        }
        
        if self.vectorize {
            parts.push("v".to_string());
        }
        
        if parts.is_empty() {
            "baseline".to_string()
        } else {
            parts.join("_")
        }
    }
}

/// The main auto-tuner
pub struct AutoTuner {
    /// Tuning configuration
    config: TuningConfig,
    /// Search strategy
    strategy: Box<dyn SearchStrategy>,
    /// Benchmark runner
    runner: BenchmarkRunner,
    /// Results collected so far
    results: TuningResults,
}

impl AutoTuner {
    /// Create a new auto-tuner with default settings
    pub fn new(config: TuningConfig) -> Self {
        let strategy: Box<dyn SearchStrategy> = match config.search_strategy {
            SearchType::Grid => Box::new(GridSearch::new()),
            SearchType::Random => Box::new(RandomSearch::new(config.random_seed)),
            SearchType::Genetic => Box::new(GeneticSearch::new(
                config.population_size,
                config.mutation_rate,
            )),
        };
        
        Self {
            config: config.clone(),
            strategy,
            runner: BenchmarkRunner::new(config),
            results: TuningResults::new(),
        }
    }
    
    /// Run auto-tuning on a program
    pub fn tune<P: AsRef<Path>>(
        &mut self,
        source: P,
        params: &[&str],
    ) -> TuningResult<Configuration> {
        let start = Instant::now();
        let deadline = start + Duration::from_secs(self.config.max_time_seconds);
        
        // Generate search space
        let space = self.generate_search_space()?;
        
        println!("=== Auto-Tuning ===");
        println!("Search space: {} configurations", space.size());
        println!("Time budget: {}s", self.config.max_time_seconds);
        println!();
        
        // Run baseline first
        let baseline_config = Configuration::default();
        let baseline_result = self.runner.benchmark(source.as_ref(), &baseline_config, params)?;
        self.results.add(baseline_config.clone(), baseline_result.clone());
        println!("Baseline: {:.4}s", baseline_result.median_time);
        
        let mut best_config = baseline_config;
        let mut best_time = baseline_result.median_time;
        let mut configs_tested = 1;
        
        // Main tuning loop
        while Instant::now() < deadline {
            // Get next configuration to try
            let config = match self.strategy.next(&space, &self.results) {
                Some(c) => c,
                None => break, // Search exhausted
            };
            
            // Skip if already tested
            if self.results.contains(&config) {
                continue;
            }
            
            // Benchmark this configuration
            match self.runner.benchmark(source.as_ref(), &config, params) {
                Ok(result) => {
                    configs_tested += 1;
                    let speedup = baseline_result.median_time / result.median_time;
                    
                    if self.config.verbose {
                        println!(
                            "[{:3}] {} -> {:.4}s ({:.2}x)",
                            configs_tested,
                            config.to_string_compact(),
                            result.median_time,
                            speedup
                        );
                    }
                    
                    if result.median_time < best_time {
                        best_time = result.median_time;
                        best_config = config.clone();
                        println!(
                            "  *** New best: {} ({:.2}x speedup)",
                            config.to_string_compact(),
                            speedup
                        );
                    }
                    
                    self.results.add(config, result);
                }
                Err(e) => {
                    if self.config.verbose {
                        println!("[{:3}] {} -> FAILED: {}", configs_tested, config.to_string_compact(), e);
                    }
                }
            }
        }
        
        // Print summary
        println!();
        println!("=== Tuning Complete ===");
        println!("Configurations tested: {}", configs_tested);
        println!("Time elapsed: {:.1}s", start.elapsed().as_secs_f64());
        println!("Best configuration: {}", best_config.to_string_compact());
        println!("Best time: {:.4}s", best_time);
        println!("Speedup over baseline: {:.2}x", baseline_result.median_time / best_time);
        
        Ok(best_config)
    }
    
    /// Generate the search space based on config
    fn generate_search_space(&self) -> TuningResult<TuningSpace> {
        let mut space = TuningSpace::new();
        
        // Tile sizes
        if !self.config.tile_sizes.is_empty() {
            for dim in 0..self.config.max_loop_depth {
                space.add_parameter(TuningParameter::TileSize {
                    dim,
                    values: self.config.tile_sizes.clone(),
                });
            }
        }
        
        // Thread counts
        if !self.config.thread_counts.is_empty() {
            space.add_parameter(TuningParameter::Threads(self.config.thread_counts.clone()));
        }
        
        // OpenMP schedules
        space.add_parameter(TuningParameter::OmpSchedule(vec![
            OmpSchedule::Static,
            OmpSchedule::Dynamic,
            OmpSchedule::Guided,
        ]));
        
        // Vectorization
        space.add_parameter(TuningParameter::Vectorize(vec![false, true]));
        
        Ok(space)
    }
    
    /// Get all results
    pub fn results(&self) -> &TuningResults {
        &self.results
    }
}

/// Search type
#[derive(Clone, Copy, Debug, Default)]
pub enum SearchType {
    /// Exhaustive grid search
    #[default]
    Grid,
    /// Random sampling
    Random,
    /// Genetic algorithm
    Genetic,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_configuration() {
        let config = Configuration::with_tiling(vec![32, 32]);
        assert_eq!(config.to_string_compact(), "t32x32");
        
        let config = Configuration::parallel(4);
        assert_eq!(config.to_string_compact(), "p4");
        
        let config = Configuration::optimized(vec![64], 8);
        assert_eq!(config.to_string_compact(), "t64_p8_v");
    }
}
