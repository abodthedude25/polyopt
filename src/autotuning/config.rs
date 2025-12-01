//! Auto-tuning configuration

use super::{Configuration, OmpSchedule, SearchType};

/// Configuration for the auto-tuner
#[derive(Clone, Debug)]
pub struct TuningConfig {
    /// Maximum tuning time in seconds
    pub max_time_seconds: u64,
    
    /// Tile sizes to explore
    pub tile_sizes: Vec<usize>,
    
    /// Thread counts to try
    pub thread_counts: Vec<usize>,
    
    /// Maximum loop depth to tune
    pub max_loop_depth: usize,
    
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
    
    /// Search strategy
    pub search_strategy: SearchType,
    
    /// Random seed for reproducibility
    pub random_seed: u64,
    
    /// Population size for genetic search
    pub population_size: usize,
    
    /// Mutation rate for genetic search
    pub mutation_rate: f64,
    
    /// Verbose output
    pub verbose: bool,
    
    /// Compiler to use
    pub compiler: String,
    
    /// Extra compiler flags
    pub compiler_flags: Vec<String>,
    
    /// Working directory for temporary files
    pub work_dir: Option<String>,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            max_time_seconds: 60,
            tile_sizes: vec![16, 32, 64, 128],
            thread_counts: vec![1, 2, 4, 8],
            max_loop_depth: 3,
            benchmark_iterations: 3,
            search_strategy: SearchType::Grid,
            random_seed: 42,
            population_size: 20,
            mutation_rate: 0.1,
            verbose: true,
            compiler: "gcc".to_string(),
            compiler_flags: vec!["-O3".to_string(), "-march=native".to_string()],
            work_dir: None,
        }
    }
}

impl TuningConfig {
    /// Create a new config with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set maximum tuning time
    pub fn max_time(mut self, seconds: u64) -> Self {
        self.max_time_seconds = seconds;
        self
    }
    
    /// Set tile sizes to explore
    pub fn tile_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.tile_sizes = sizes;
        self
    }
    
    /// Set thread counts to try
    pub fn thread_counts(mut self, counts: Vec<usize>) -> Self {
        self.thread_counts = counts;
        self
    }
    
    /// Set search strategy
    pub fn search(mut self, strategy: SearchType) -> Self {
        self.search_strategy = strategy;
        self
    }
    
    /// Enable/disable verbose output
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }
    
    /// Set compiler
    pub fn compiler(mut self, compiler: &str) -> Self {
        self.compiler = compiler.to_string();
        self
    }
    
    /// Quick tuning preset (fast but less thorough)
    pub fn quick() -> Self {
        Self {
            max_time_seconds: 30,
            tile_sizes: vec![32, 64],
            thread_counts: vec![1, 4],
            benchmark_iterations: 2,
            ..Default::default()
        }
    }
    
    /// Thorough tuning preset (slow but comprehensive)
    pub fn thorough() -> Self {
        Self {
            max_time_seconds: 300,
            tile_sizes: vec![8, 16, 32, 64, 128, 256],
            thread_counts: vec![1, 2, 4, 8, 16],
            benchmark_iterations: 5,
            search_strategy: SearchType::Genetic,
            ..Default::default()
        }
    }
}

/// A tuning parameter
#[derive(Clone, Debug)]
pub enum TuningParameter {
    /// Tile size for a loop dimension
    TileSize { dim: usize, values: Vec<usize> },
    /// Number of threads
    Threads(Vec<usize>),
    /// OpenMP schedule type
    OmpSchedule(Vec<OmpSchedule>),
    /// Vectorization enabled
    Vectorize(Vec<bool>),
    /// Unroll factor
    Unroll { dim: usize, values: Vec<usize> },
    /// Loop permutation
    Permutation(Vec<Vec<usize>>),
}

impl TuningParameter {
    /// Get the number of possible values
    pub fn cardinality(&self) -> usize {
        match self {
            TuningParameter::TileSize { values, .. } => values.len(),
            TuningParameter::Threads(v) => v.len(),
            TuningParameter::OmpSchedule(v) => v.len(),
            TuningParameter::Vectorize(v) => v.len(),
            TuningParameter::Unroll { values, .. } => values.len(),
            TuningParameter::Permutation(v) => v.len(),
        }
    }
}

/// The search space for auto-tuning
#[derive(Clone, Debug, Default)]
pub struct TuningSpace {
    /// Parameters to tune
    parameters: Vec<TuningParameter>,
}

impl TuningSpace {
    /// Create a new empty search space
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a parameter to tune
    pub fn add_parameter(&mut self, param: TuningParameter) {
        self.parameters.push(param);
    }
    
    /// Get total number of configurations
    pub fn size(&self) -> usize {
        self.parameters.iter()
            .map(|p| p.cardinality())
            .product()
    }
    
    /// Get all parameters
    pub fn parameters(&self) -> &[TuningParameter] {
        &self.parameters
    }
    
    /// Generate all configurations (for grid search)
    pub fn all_configurations(&self) -> Vec<Configuration> {
        let mut configs = vec![Configuration::default()];
        
        for param in &self.parameters {
            let mut new_configs = Vec::new();
            
            for config in &configs {
                match param {
                    TuningParameter::TileSize { dim, values } => {
                        for &size in values {
                            let mut new = config.clone();
                            while new.tile_sizes.len() <= *dim {
                                new.tile_sizes.push(0);
                            }
                            new.tile_sizes[*dim] = size;
                            new_configs.push(new);
                        }
                    }
                    TuningParameter::Threads(counts) => {
                        for &count in counts {
                            let mut new = config.clone();
                            new.num_threads = count;
                            new_configs.push(new);
                        }
                    }
                    TuningParameter::OmpSchedule(schedules) => {
                        for &sched in schedules {
                            let mut new = config.clone();
                            new.omp_schedule = sched;
                            new_configs.push(new);
                        }
                    }
                    TuningParameter::Vectorize(options) => {
                        for &vec in options {
                            let mut new = config.clone();
                            new.vectorize = vec;
                            new_configs.push(new);
                        }
                    }
                    _ => {
                        new_configs.push(config.clone());
                    }
                }
            }
            
            configs = new_configs;
        }
        
        configs
    }
    
    /// Sample a random configuration
    #[cfg(feature = "autotuning")]
    pub fn random_configuration(&self, rng: &mut impl rand::Rng) -> Configuration {
        use rand::seq::SliceRandom;
        
        let mut config = Configuration::default();
        
        for param in &self.parameters {
            match param {
                TuningParameter::TileSize { dim, values } => {
                    if let Some(&size) = values.choose(rng) {
                        while config.tile_sizes.len() <= *dim {
                            config.tile_sizes.push(32); // default
                        }
                        config.tile_sizes[*dim] = size;
                    }
                }
                TuningParameter::Threads(counts) => {
                    if let Some(&count) = counts.choose(rng) {
                        config.num_threads = count;
                    }
                }
                TuningParameter::OmpSchedule(schedules) => {
                    if let Some(&sched) = schedules.choose(rng) {
                        config.omp_schedule = sched;
                    }
                }
                TuningParameter::Vectorize(options) => {
                    if let Some(&vec) = options.choose(rng) {
                        config.vectorize = vec;
                    }
                }
                _ => {}
            }
        }
        
        config
    }
}
