//! ISL Schedule - represents optimized loop schedules

use super::{IslResult, IslError, run_isl};
use super::map::IslMap;
use super::set::IslSet;

/// Options for schedule computation
#[derive(Clone, Debug, Default)]
pub struct ScheduleOptions {
    /// Try to maximize parallelism (find parallel dimensions)
    pub maximize_parallelism: bool,
    
    /// Minimize dependence distances (improve locality)
    pub minimize_dependence_distance: bool,
    
    /// Fuse statements when possible
    pub enable_fusion: bool,
    
    /// Maximum fusion level (0 = no fusion, -1 = unlimited)
    pub max_fusion_level: i32,
    
    /// Target tile sizes for tiling (empty = no tiling)
    pub tile_sizes: Vec<usize>,
    
    /// Outer parallelism only (don't parallelize inner loops)
    pub outer_parallelism: bool,
}

impl ScheduleOptions {
    /// Create options optimized for parallelism
    pub fn parallel() -> Self {
        Self {
            maximize_parallelism: true,
            minimize_dependence_distance: false,
            enable_fusion: true,
            max_fusion_level: -1,
            tile_sizes: vec![],
            outer_parallelism: true,
        }
    }
    
    /// Create options optimized for locality
    pub fn locality() -> Self {
        Self {
            maximize_parallelism: false,
            minimize_dependence_distance: true,
            enable_fusion: true,
            max_fusion_level: -1,
            tile_sizes: vec![32, 32], // Default tile size
            outer_parallelism: false,
        }
    }
    
    /// Create balanced options (good for most cases)
    pub fn balanced() -> Self {
        Self {
            maximize_parallelism: true,
            minimize_dependence_distance: true,
            enable_fusion: true,
            max_fusion_level: 2,
            tile_sizes: vec![],
            outer_parallelism: true,
        }
    }
}

/// An ISL schedule representing an optimized loop transformation
/// 
/// Schedules map statement instances to logical execution times.
/// The schedule determines the order of execution and which iterations
/// can be executed in parallel.
#[derive(Clone, Debug)]
pub struct IslSchedule {
    /// Raw schedule from ISL
    raw: String,
    /// Parsed schedule tree (if available)
    tree: Option<ScheduleTree>,
}

/// Schedule tree node types
#[derive(Clone, Debug)]
pub enum ScheduleTree {
    /// Sequential execution
    Sequence(Vec<ScheduleTree>),
    /// Parallel execution (independent)
    Parallel(Vec<ScheduleTree>),
    /// Loop band (one or more loops)
    Band {
        /// Schedule dimensions (loop variables)
        dims: Vec<String>,
        /// Whether each dimension is parallel
        parallel: Vec<bool>,
        /// Tile sizes (if tiled)
        tile_sizes: Vec<Option<usize>>,
        /// Child
        child: Option<Box<ScheduleTree>>,
    },
    /// Leaf (statement execution)
    Leaf(String),
    /// Filter (subset of domain)
    Filter {
        domain: String,
        child: Box<ScheduleTree>,
    },
}

impl IslSchedule {
    /// Create a new schedule from ISL output
    pub fn new(raw: String) -> Self {
        Self {
            raw,
            tree: None, // Parsing TODO
        }
    }
    
    /// Get the raw schedule string
    pub fn raw(&self) -> &str {
        &self.raw
    }
    
    /// Check if schedule is valid (not empty)
    pub fn is_valid(&self) -> bool {
        !self.raw.is_empty() && !self.raw.contains("error")
    }
    
    /// Get schedule as a map
    pub fn as_map(&self) -> IslResult<IslMap> {
        // Try to extract map from schedule
        let result = run_isl(&format!("map_from_schedule({});", self.raw))?;
        Ok(IslMap::new(self.raw.clone(), result))
    }
    
    /// Get the schedule tree
    pub fn tree(&self) -> Option<&ScheduleTree> {
        self.tree.as_ref()
    }
    
    /// Detect parallel dimensions
    /// 
    /// Returns indices of dimensions that can be executed in parallel
    pub fn parallel_dims(&self) -> IslResult<Vec<usize>> {
        // This would require more sophisticated parsing
        // For now, return empty
        Ok(vec![])
    }
    
    /// Apply tiling to the schedule
    pub fn tile(&self, tile_sizes: &[usize]) -> IslResult<Self> {
        if tile_sizes.is_empty() {
            return Ok(self.clone());
        }
        
        let sizes_str = tile_sizes.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        
        let script = format!(
            r#"
            S := {};
            T := tile S with sizes [{}];
            T;
            "#,
            self.raw,
            sizes_str
        );
        
        let result = run_isl(&script)?;
        Ok(Self::new(result))
    }
}

/// Compute an optimized schedule for a set of statements
/// 
/// # Arguments
/// * `domains` - Map from statement name to iteration domain
/// * `dependences` - Dependence relations
/// * `options` - Scheduling options
/// 
/// # Returns
/// An optimized schedule
pub fn compute_schedule(
    domains: &[(&str, &IslSet)],
    dependences: &IslMap,
    options: &ScheduleOptions,
) -> IslResult<IslSchedule> {
    let mut script = String::new();
    
    // Define domains
    for (name, domain) in domains {
        script.push_str(&format!("{} := {};\n", name, domain.expr()));
    }
    
    // Union of all domains
    let union_expr = domains.iter()
        .map(|(name, _)| format!("({})", name))
        .collect::<Vec<_>>()
        .join(" + ");
    script.push_str(&format!("D := {};\n", union_expr));
    
    // Define dependences
    script.push_str(&format!("DEP := {};\n", dependences.expr()));
    
    // Compute schedule
    script.push_str("S := schedule D respecting DEP");
    
    if options.maximize_parallelism {
        script.push_str(" maximizing parallelism");
    }
    
    script.push_str(";\n");
    script.push_str("S;\n");
    
    let result = run_isl(&script)?;
    
    // Apply tiling if requested
    let schedule = IslSchedule::new(result);
    if !options.tile_sizes.is_empty() {
        schedule.tile(&options.tile_sizes)
    } else {
        Ok(schedule)
    }
}

/// Pluto-style scheduling algorithm
/// 
/// Computes a schedule that:
/// 1. Maximizes outermost parallel loops
/// 2. Minimizes reuse distances for locality
/// 3. Enables tiling
pub fn pluto_schedule(
    domain: &IslSet,
    dependences: &IslMap,
) -> IslResult<IslSchedule> {
    let script = format!(
        r#"
        D := {};
        DEP := {};
        # Pluto-style: maximize outer parallelism
        S := schedule D respecting DEP 
             minimizing sum of dependence distances;
        S;
        "#,
        domain.expr(),
        dependences.expr()
    );
    
    let result = run_isl(&script)?;
    Ok(IslSchedule::new(result))
}

/// Feautrier's scheduling algorithm
/// 
/// Computes multi-dimensional schedules for maximal parallelism.
pub fn feautrier_schedule(
    domain: &IslSet,
    dependences: &IslMap,
) -> IslResult<IslSchedule> {
    let script = format!(
        r#"
        D := {};
        DEP := {};
        # Feautrier: maximize parallelism at all levels
        S := schedule D respecting DEP maximizing parallelism;
        S;
        "#,
        domain.expr(),
        dependences.expr()
    );
    
    let result = run_isl(&script)?;
    Ok(IslSchedule::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isl::is_isl_available;
    
    #[test]
    fn test_schedule_options() {
        let opt = ScheduleOptions::parallel();
        assert!(opt.maximize_parallelism);
        
        let opt = ScheduleOptions::locality();
        assert!(opt.minimize_dependence_distance);
    }
}
