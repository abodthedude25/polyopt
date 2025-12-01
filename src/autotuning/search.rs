//! Search strategies for auto-tuning

use super::{Configuration, TuningSpace, TuningResults};

/// Trait for search strategies
pub trait SearchStrategy: Send {
    /// Get the next configuration to try
    fn next(&mut self, space: &TuningSpace, results: &TuningResults) -> Option<Configuration>;
    
    /// Reset the search
    fn reset(&mut self);
    
    /// Get name of the strategy
    fn name(&self) -> &str;
}

/// Grid search - exhaustively try all configurations
pub struct GridSearch {
    /// Current index in the configuration space
    index: usize,
    /// Cached configurations
    configs: Vec<Configuration>,
}

impl GridSearch {
    pub fn new() -> Self {
        Self {
            index: 0,
            configs: Vec::new(),
        }
    }
}

impl SearchStrategy for GridSearch {
    fn next(&mut self, space: &TuningSpace, _results: &TuningResults) -> Option<Configuration> {
        // Generate all configs on first call
        if self.configs.is_empty() {
            self.configs = space.all_configurations();
        }
        
        if self.index < self.configs.len() {
            let config = self.configs[self.index].clone();
            self.index += 1;
            Some(config)
        } else {
            None
        }
    }
    
    fn reset(&mut self) {
        self.index = 0;
        self.configs.clear();
    }
    
    fn name(&self) -> &str {
        "grid"
    }
}

impl Default for GridSearch {
    fn default() -> Self {
        Self::new()
    }
}

/// Random search - sample configurations randomly
pub struct RandomSearch {
    /// Random seed
    seed: u64,
    /// Number of configurations tried
    count: usize,
    /// Maximum configurations to try
    max_configs: usize,
}

impl RandomSearch {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            count: 0,
            max_configs: 100,
        }
    }
    
    pub fn with_max(seed: u64, max: usize) -> Self {
        Self {
            seed,
            count: 0,
            max_configs: max,
        }
    }
}

impl SearchStrategy for RandomSearch {
    fn next(&mut self, space: &TuningSpace, _results: &TuningResults) -> Option<Configuration> {
        if self.count >= self.max_configs {
            return None;
        }
        
        self.count += 1;
        
        // Simple pseudo-random configuration
        // In real implementation, would use rand crate
        let configs = space.all_configurations();
        if configs.is_empty() {
            return None;
        }
        
        let idx = ((self.seed.wrapping_mul(self.count as u64) % 1000003) as usize) % configs.len();
        Some(configs[idx].clone())
    }
    
    fn reset(&mut self) {
        self.count = 0;
    }
    
    fn name(&self) -> &str {
        "random"
    }
}

/// Genetic algorithm search
pub struct GeneticSearch {
    /// Population size
    population_size: usize,
    /// Mutation rate
    mutation_rate: f64,
    /// Current generation
    generation: usize,
    /// Current population
    population: Vec<Configuration>,
    /// Current individual index
    individual_idx: usize,
}

impl GeneticSearch {
    pub fn new(population_size: usize, mutation_rate: f64) -> Self {
        Self {
            population_size,
            mutation_rate,
            generation: 0,
            population: Vec::new(),
            individual_idx: 0,
        }
    }
    
    /// Initialize population
    fn init_population(&mut self, space: &TuningSpace) {
        let all = space.all_configurations();
        
        // Take first N configurations as initial population
        self.population = all.into_iter()
            .take(self.population_size)
            .collect();
        
        self.generation = 0;
        self.individual_idx = 0;
    }
    
    /// Evolve to next generation based on results
    fn evolve(&mut self, results: &TuningResults) {
        if self.population.is_empty() {
            return;
        }
        
        // Sort by performance (best first)
        let mut scored: Vec<_> = self.population.iter()
            .filter_map(|c| {
                results.get(c).map(|r| (c.clone(), r.median_time))
            })
            .collect();
        
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Keep top 50% as parents
        let parents: Vec<_> = scored.iter()
            .take(self.population_size / 2)
            .map(|(c, _)| c.clone())
            .collect();
        
        if parents.is_empty() {
            return;
        }
        
        // Create new population through crossover and mutation
        let mut new_pop = parents.clone();
        
        while new_pop.len() < self.population_size {
            // Simple crossover: take random attributes from parents
            let p1 = &parents[self.generation % parents.len()];
            let p2 = &parents[(self.generation + 1) % parents.len()];
            
            let mut child = Configuration {
                tile_sizes: if self.generation % 2 == 0 { p1.tile_sizes.clone() } else { p2.tile_sizes.clone() },
                num_threads: if self.generation % 3 == 0 { p1.num_threads } else { p2.num_threads },
                omp_schedule: if self.generation % 5 == 0 { p1.omp_schedule } else { p2.omp_schedule },
                vectorize: if self.generation % 7 == 0 { p1.vectorize } else { p2.vectorize },
                ..Default::default()
            };
            
            // Simple mutation
            if (self.generation as f64 * self.mutation_rate) as usize % 3 == 0 {
                if !child.tile_sizes.is_empty() {
                    child.tile_sizes[0] = match child.tile_sizes[0] {
                        16 => 32,
                        32 => 64,
                        64 => 128,
                        128 => 16,
                        x => x,
                    };
                }
            }
            
            new_pop.push(child);
        }
        
        self.population = new_pop;
        self.generation += 1;
        self.individual_idx = 0;
    }
}

impl SearchStrategy for GeneticSearch {
    fn next(&mut self, space: &TuningSpace, results: &TuningResults) -> Option<Configuration> {
        // Initialize on first call
        if self.population.is_empty() {
            self.init_population(space);
        }
        
        // Check if we've evaluated everyone in this generation
        if self.individual_idx >= self.population.len() {
            // Evolve to next generation
            self.evolve(results);
            
            // Stop after 10 generations
            if self.generation > 10 {
                return None;
            }
        }
        
        if self.individual_idx < self.population.len() {
            let config = self.population[self.individual_idx].clone();
            self.individual_idx += 1;
            Some(config)
        } else {
            None
        }
    }
    
    fn reset(&mut self) {
        self.population.clear();
        self.generation = 0;
        self.individual_idx = 0;
    }
    
    fn name(&self) -> &str {
        "genetic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autotuning::config::TuningParameter;
    
    #[test]
    fn test_grid_search() {
        let mut space = TuningSpace::new();
        space.add_parameter(TuningParameter::Threads(vec![1, 2, 4]));
        
        let mut search = GridSearch::new();
        let results = TuningResults::new();
        
        let mut count = 0;
        while let Some(_) = search.next(&space, &results) {
            count += 1;
        }
        
        assert_eq!(count, 3);
    }
}
