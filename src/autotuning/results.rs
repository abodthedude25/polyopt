//! Auto-tuning results tracking

use super::{Configuration, BenchmarkResult};
use std::collections::HashMap;

/// Results from auto-tuning
#[derive(Clone, Debug, Default)]
pub struct TuningResults {
    /// All results indexed by configuration
    results: Vec<ConfigResult>,
}

/// Result for a single configuration
#[derive(Clone, Debug)]
pub struct ConfigResult {
    /// The configuration
    pub config: Configuration,
    /// Benchmark result
    pub result: BenchmarkResult,
}

impl TuningResults {
    /// Create empty results
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a result
    pub fn add(&mut self, config: Configuration, result: BenchmarkResult) {
        self.results.push(ConfigResult { config, result });
    }
    
    /// Check if a configuration has been tested
    pub fn contains(&self, config: &Configuration) -> bool {
        self.results.iter().any(|r| &r.config == config)
    }
    
    /// Get result for a configuration
    pub fn get(&self, config: &Configuration) -> Option<&BenchmarkResult> {
        self.results.iter()
            .find(|r| &r.config == config)
            .map(|r| &r.result)
    }
    
    /// Get the best configuration
    pub fn best(&self) -> Option<&ConfigResult> {
        self.results.iter()
            .filter(|r| r.result.executed)
            .min_by(|a, b| {
                a.result.median_time.partial_cmp(&b.result.median_time).unwrap()
            })
    }
    
    /// Get all results sorted by performance
    pub fn sorted(&self) -> Vec<&ConfigResult> {
        let mut sorted: Vec<_> = self.results.iter()
            .filter(|r| r.result.executed)
            .collect();
        sorted.sort_by(|a, b| {
            a.result.median_time.partial_cmp(&b.result.median_time).unwrap()
        });
        sorted
    }
    
    /// Get number of configurations tested
    pub fn len(&self) -> usize {
        self.results.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
    
    /// Get all results
    pub fn all(&self) -> &[ConfigResult] {
        &self.results
    }
    
    /// Export results to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("config,median_time,min_time,max_time,std_dev\n");
        
        for r in &self.results {
            csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6}\n",
                r.config.to_string_compact(),
                r.result.median_time,
                r.result.min_time,
                r.result.max_time,
                r.result.std_dev
            ));
        }
        
        csv
    }
    
    /// Print summary report
    pub fn print_summary(&self) {
        println!("=== Tuning Results Summary ===");
        println!("Configurations tested: {}", self.len());
        println!();
        
        let sorted = self.sorted();
        if sorted.is_empty() {
            println!("No successful configurations");
            return;
        }
        
        println!("Top 5 configurations:");
        println!("{:<20} {:>12} {:>12}", "Config", "Median", "Speedup");
        println!("{}", "-".repeat(46));
        
        let baseline = sorted.last().map(|r| r.result.median_time).unwrap_or(1.0);
        
        for (i, r) in sorted.iter().take(5).enumerate() {
            let speedup = baseline / r.result.median_time;
            println!(
                "{:<20} {:>10.4}s {:>10.2}x",
                r.config.to_string_compact(),
                r.result.median_time,
                speedup
            );
        }
        
        println!();
        if let Some(best) = sorted.first() {
            let worst = sorted.last().unwrap();
            println!(
                "Range: {:.4}s - {:.4}s ({:.1}x difference)",
                best.result.median_time,
                worst.result.median_time,
                worst.result.median_time / best.result.median_time
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_results() {
        let mut results = TuningResults::new();
        
        let c1 = Configuration::with_tiling(vec![32]);
        let r1 = BenchmarkResult::from_times(vec![1.0, 1.1, 1.2]);
        results.add(c1.clone(), r1);
        
        let c2 = Configuration::with_tiling(vec![64]);
        let r2 = BenchmarkResult::from_times(vec![0.8, 0.9, 1.0]);
        results.add(c2.clone(), r2);
        
        assert_eq!(results.len(), 2);
        assert!(results.contains(&c1));
        
        let best = results.best().unwrap();
        assert_eq!(best.config.tile_sizes, vec![64]);
    }
}
