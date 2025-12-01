//! Benchmark runner for auto-tuning

use super::{Configuration, TuningConfig, TuningResult, TuningError};
use std::path::Path;
use std::process::Command;
use std::time::Instant;
use std::fs;

/// Runs benchmarks for configurations
pub struct BenchmarkRunner {
    /// Configuration
    config: TuningConfig,
    /// Working directory
    work_dir: String,
    /// Counter for unique file names
    counter: usize,
}

/// Result of a benchmark run
#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    /// Individual run times
    pub times: Vec<f64>,
    /// Median time
    pub median_time: f64,
    /// Minimum time
    pub min_time: f64,
    /// Maximum time
    pub max_time: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Whether compilation succeeded
    pub compiled: bool,
    /// Whether execution succeeded
    pub executed: bool,
}

impl BenchmarkResult {
    /// Create a new result from times
    pub fn from_times(times: Vec<f64>) -> Self {
        if times.is_empty() {
            return Self {
                times: vec![],
                median_time: f64::INFINITY,
                min_time: f64::INFINITY,
                max_time: f64::INFINITY,
                std_dev: 0.0,
                compiled: true,
                executed: false,
            };
        }
        
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median_time = sorted[sorted.len() / 2];
        let min_time = sorted[0];
        let max_time = sorted[sorted.len() - 1];
        
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();
        
        Self {
            times,
            median_time,
            min_time,
            max_time,
            std_dev,
            compiled: true,
            executed: true,
        }
    }
    
    /// Create a failed result
    pub fn failed() -> Self {
        Self {
            times: vec![],
            median_time: f64::INFINITY,
            min_time: f64::INFINITY,
            max_time: f64::INFINITY,
            std_dev: 0.0,
            compiled: false,
            executed: false,
        }
    }
}

impl BenchmarkRunner {
    /// Create a new runner
    pub fn new(config: TuningConfig) -> Self {
        let work_dir = config.work_dir.clone()
            .unwrap_or_else(|| "/tmp/polyopt_tune".to_string());
        
        // Create work directory
        let _ = fs::create_dir_all(&work_dir);
        
        Self {
            config,
            work_dir,
            counter: 0,
        }
    }
    
    /// Benchmark a configuration
    pub fn benchmark(
        &mut self,
        source: &Path,
        config: &Configuration,
        params: &[&str],
    ) -> TuningResult<BenchmarkResult> {
        self.counter += 1;
        
        let base_name = format!("tune_{}", self.counter);
        let c_file = format!("{}/{}.c", self.work_dir, base_name);
        let exe_file = format!("{}/{}", self.work_dir, base_name);
        
        // Step 1: Generate C code with this configuration
        self.generate_code(source, config, &c_file)?;
        
        // Step 2: Compile
        self.compile(&c_file, &exe_file, config)?;
        
        // Step 3: Run benchmarks
        let mut times = Vec::new();
        for _ in 0..self.config.benchmark_iterations {
            match self.run_benchmark(&exe_file, params) {
                Ok(time) => times.push(time),
                Err(_) => {}
            }
        }
        
        if times.is_empty() {
            return Err(TuningError::BenchmarkFailed("No successful runs".to_string()));
        }
        
        // Cleanup
        let _ = fs::remove_file(&c_file);
        let _ = fs::remove_file(&exe_file);
        
        Ok(BenchmarkResult::from_times(times))
    }
    
    /// Generate C code with the given configuration
    fn generate_code(
        &self,
        source: &Path,
        config: &Configuration,
        output: &str,
    ) -> TuningResult<()> {
        let mut args = vec![
            "compile".to_string(),
            source.to_string_lossy().to_string(),
            "-o".to_string(),
            output.to_string(),
            "--benchmark".to_string(),
        ];
        
        // Add OpenMP if threads > 0
        if config.num_threads > 0 {
            args.push("--openmp".to_string());
        }
        
        // Add tiling
        if !config.tile_sizes.is_empty() {
            args.push("--tile".to_string());
            args.push(config.tile_sizes[0].to_string());
        }
        
        // Add vectorization
        if config.vectorize {
            args.push("--vectorize".to_string());
        }
        
        // Run polyopt
        let polyopt = std::env::current_exe()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| "polyopt".to_string());
        
        let output = Command::new(&polyopt)
            .args(&args)
            .output()
            .map_err(|e| TuningError::CompilationFailed(e.to_string()))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TuningError::CompilationFailed(stderr.to_string()));
        }
        
        Ok(())
    }
    
    /// Compile the C code
    fn compile(
        &self,
        c_file: &str,
        exe_file: &str,
        config: &Configuration,
    ) -> TuningResult<()> {
        let mut args = vec![
            "-o".to_string(),
            exe_file.to_string(),
            c_file.to_string(),
        ];
        
        // Add optimization flags
        args.extend(self.config.compiler_flags.iter().cloned());
        
        // Add OpenMP
        if config.num_threads > 0 {
            args.push("-fopenmp".to_string());
        }
        
        // Add math library
        args.push("-lm".to_string());
        
        let output = Command::new(&self.config.compiler)
            .args(&args)
            .output()
            .map_err(|e| TuningError::CompilationFailed(e.to_string()))?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(TuningError::CompilationFailed(stderr.to_string()));
        }
        
        Ok(())
    }
    
    /// Run the benchmark and extract timing
    fn run_benchmark(
        &self,
        exe_file: &str,
        params: &[&str],
    ) -> TuningResult<f64> {
        let start = Instant::now();
        
        let output = Command::new(exe_file)
            .args(params)
            .output()
            .map_err(|e| TuningError::BenchmarkFailed(e.to_string()))?;
        
        if !output.status.success() {
            return Err(TuningError::BenchmarkFailed("Execution failed".to_string()));
        }
        
        // Try to extract time from output
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // Look for "Time: X.XXX" pattern
        for line in stdout.lines() {
            if line.contains("Time:") {
                if let Some(time_str) = line.split(':').nth(1) {
                    if let Ok(time) = time_str.trim().trim_end_matches('s').parse::<f64>() {
                        return Ok(time);
                    }
                }
            }
        }
        
        // Fall back to wall clock time
        Ok(start.elapsed().as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult::from_times(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.median_time, 3.0);
        assert_eq!(result.min_time, 1.0);
        assert_eq!(result.max_time, 5.0);
    }
}
