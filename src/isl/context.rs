//! ISL Context - manages ISL state and provides high-level operations

use super::{IslResult, IslError, run_isl, is_isl_available};
use super::set::IslSet;
use super::map::IslMap;
use super::schedule::{IslSchedule, ScheduleOptions};

/// ISL computation context
/// 
/// Provides high-level interface to ISL operations.
/// All ISL computations go through this context.
pub struct IslContext {
    /// Parameters (symbolic constants like N, M)
    parameters: Vec<String>,
}

impl IslContext {
    /// Create a new ISL context
    pub fn new() -> IslResult<Self> {
        if !is_isl_available() {
            return Err(IslError::IslNotFound);
        }
        Ok(Self {
            parameters: Vec::new(),
        })
    }
    
    /// Add a symbolic parameter
    pub fn add_parameter(&mut self, name: &str) {
        if !self.parameters.contains(&name.to_string()) {
            self.parameters.push(name.to_string());
        }
    }
    
    /// Get parameter context string for ISL
    pub fn param_context(&self) -> String {
        if self.parameters.is_empty() {
            String::new()
        } else {
            format!("[{}]", self.parameters.join(","))
        }
    }
    
    /// Parse an ISL set from string
    /// 
    /// # Example
    /// ```ignore
    /// let set = ctx.parse_set("{ [i,j] : 0 <= i < N and 0 <= j < M }")?;
    /// ```
    pub fn parse_set(&self, expr: &str) -> IslResult<IslSet> {
        // Validate by running through ISL
        let result = run_isl(&format!("{};", expr))?;
        Ok(IslSet::new(expr.to_string(), result))
    }
    
    /// Parse an ISL map (relation) from string
    /// 
    /// # Example
    /// ```ignore
    /// let map = ctx.parse_map("{ [i,j] -> [i+1,j] }")?;
    /// ```
    pub fn parse_map(&self, expr: &str) -> IslResult<IslMap> {
        let result = run_isl(&format!("{};", expr))?;
        Ok(IslMap::new(expr.to_string(), result))
    }
    
    /// Compute the intersection of two sets
    pub fn intersect(&self, s1: &IslSet, s2: &IslSet) -> IslResult<IslSet> {
        let expr = format!("({}) * ({});", s1.expr(), s2.expr());
        let result = run_isl(&expr)?;
        Ok(IslSet::new(format!("({}) * ({})", s1.expr(), s2.expr()), result))
    }
    
    /// Compute the union of two sets
    pub fn union(&self, s1: &IslSet, s2: &IslSet) -> IslResult<IslSet> {
        let expr = format!("({}) + ({});", s1.expr(), s2.expr());
        let result = run_isl(&expr)?;
        Ok(IslSet::new(format!("({}) + ({})", s1.expr(), s2.expr()), result))
    }
    
    /// Compute set difference
    pub fn subtract(&self, s1: &IslSet, s2: &IslSet) -> IslResult<IslSet> {
        let expr = format!("({}) - ({});", s1.expr(), s2.expr());
        let result = run_isl(&expr)?;
        Ok(IslSet::new(format!("({}) - ({})", s1.expr(), s2.expr()), result))
    }
    
    /// Apply a map to a set (image)
    pub fn apply(&self, set: &IslSet, map: &IslMap) -> IslResult<IslSet> {
        let expr = format!("({}) . ({});", set.expr(), map.expr());
        let result = run_isl(&expr)?;
        Ok(IslSet::new(expr, result))
    }
    
    /// Compute dependence polyhedra between write and read accesses
    /// 
    /// Given:
    /// - write_domain: iteration domain of write statement
    /// - read_domain: iteration domain of read statement  
    /// - write_access: map from iteration to array element written
    /// - read_access: map from iteration to array element read
    /// 
    /// Returns the dependence relation
    pub fn compute_dependences(
        &self,
        write_domain: &IslSet,
        read_domain: &IslSet,
        write_access: &IslMap,
        read_access: &IslMap,
    ) -> IslResult<IslMap> {
        // Dependence exists when:
        // 1. Same array element accessed (write_access(i) = read_access(j))
        // 2. Write happens before read (lexicographically i < j)
        // 3. Both iterations are in their domains
        
        let expr = format!(
            r#"
            W := {domain};
            R := {read_domain};
            WA := {write_access};
            RA := {read_access};
            # Compute raw dependences (flow dependences)
            # i -> j where WA(i) = RA(j) and i << j
            DEP := (WA . (RA^-1)) * (W -> R);
            DEP;
            "#,
            domain = write_domain.expr(),
            read_domain = read_domain.expr(),
            write_access = write_access.expr(),
            read_access = read_access.expr(),
        );
        
        let result = run_isl(&expr)?;
        Ok(IslMap::new("computed_dependences".to_string(), result))
    }
    
    /// Compute an optimizing schedule using ISL's scheduler
    /// 
    /// This uses ISL's Pluto-like algorithm to find a schedule that:
    /// - Respects all dependences
    /// - Maximizes parallelism
    /// - Optimizes data locality
    pub fn compute_schedule(
        &self,
        domain: &IslSet,
        dependences: &IslMap,
        options: &ScheduleOptions,
    ) -> IslResult<IslSchedule> {
        let mut script = String::new();
        
        script.push_str(&format!("D := {};\n", domain.expr()));
        script.push_str(&format!("DEP := {};\n", dependences.expr()));
        
        // Use ISL's schedule computation
        script.push_str("S := schedule D respecting DEP");
        
        if options.maximize_parallelism {
            script.push_str(" maximizing parallelism");
        }
        if options.minimize_dependence_distance {
            script.push_str(" minimizing dependence distance");  
        }
        
        script.push_str(";\n");
        script.push_str("S;\n");
        
        let result = run_isl(&script)?;
        Ok(IslSchedule::new(result))
    }
    
    /// Check if a set is empty
    pub fn is_empty(&self, set: &IslSet) -> IslResult<bool> {
        let expr = format!("({}) = {{}};", set.expr());
        let result = run_isl(&expr)?;
        Ok(result.trim() == "True")
    }
    
    /// Compute lexicographic minimum of a set
    pub fn lexmin(&self, set: &IslSet) -> IslResult<IslSet> {
        let expr = format!("lexmin({});", set.expr());
        let result = run_isl(&expr)?;
        Ok(IslSet::new(expr, result))
    }
    
    /// Compute lexicographic maximum of a set
    pub fn lexmax(&self, set: &IslSet) -> IslResult<IslSet> {
        let expr = format!("lexmax({});", set.expr());
        let result = run_isl(&expr)?;
        Ok(IslSet::new(expr, result))
    }
    
    /// Project out dimensions from a set
    pub fn project_out(&self, set: &IslSet, dims: &[usize]) -> IslResult<IslSet> {
        // ISL syntax: project out dimension by existential quantification
        // For now, simple approach
        let expr = format!("domain({})", set.expr());
        let result = run_isl(&format!("{};", expr))?;
        Ok(IslSet::new(expr, result))
    }
    
    /// Count the number of integer points in a set (if bounded)
    pub fn card(&self, set: &IslSet) -> IslResult<String> {
        let expr = format!("card({});", set.expr());
        let result = run_isl(&expr)?;
        Ok(result)
    }
}

impl Default for IslContext {
    fn default() -> Self {
        Self::new().expect("ISL should be available")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_context_creation() {
        if is_isl_available() {
            let ctx = IslContext::new();
            assert!(ctx.is_ok());
        }
    }
}
