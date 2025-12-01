//! ISL Set - represents a polyhedral set

use super::{IslResult, run_isl};

/// An ISL set representing a polyhedral domain
/// 
/// Sets are collections of integer tuples satisfying affine constraints.
/// 
/// # Examples
/// 
/// ```text
/// { [i] : 0 <= i < N }           // 1D loop domain
/// { [i,j] : 0 <= i < N and 0 <= j < M }  // 2D loop nest
/// { [i,j] : 0 <= i < N and i <= j < N }  // Triangular domain
/// ```
#[derive(Clone, Debug)]
pub struct IslSet {
    /// Original expression
    expression: String,
    /// Simplified/canonical form from ISL
    canonical: String,
}

impl IslSet {
    /// Create a new ISL set
    pub fn new(expression: String, canonical: String) -> Self {
        Self { expression, canonical }
    }
    
    /// Create a set from a loop nest
    /// 
    /// # Arguments
    /// * `vars` - Loop variables, e.g., ["i", "j"]
    /// * `constraints` - Affine constraints, e.g., ["0 <= i", "i < N", "0 <= j", "j < M"]
    pub fn from_loop(vars: &[&str], constraints: &[&str]) -> IslResult<Self> {
        let vars_str = vars.join(",");
        let constraints_str = constraints.join(" and ");
        let expr = format!("{{ [{}] : {} }}", vars_str, constraints_str);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Create an empty set
    pub fn empty() -> Self {
        Self::new("{ }".to_string(), "{ }".to_string())
    }
    
    /// Create a universe set (all integer points)
    pub fn universe(dims: usize) -> Self {
        let vars: Vec<String> = (0..dims).map(|i| format!("i{}", i)).collect();
        let expr = format!("{{ [{}] }}", vars.join(","));
        Self::new(expr.clone(), expr)
    }
    
    /// Get the original expression
    pub fn expr(&self) -> &str {
        &self.expression
    }
    
    /// Get the canonical form
    pub fn canonical(&self) -> &str {
        &self.canonical
    }
    
    /// Check if this set is empty
    pub fn is_empty(&self) -> IslResult<bool> {
        let result = run_isl(&format!("({}) = {{}};", self.expression))?;
        Ok(result.trim() == "True")
    }
    
    /// Get the number of dimensions
    pub fn dim(&self) -> IslResult<usize> {
        // Parse from canonical form - count commas + 1 inside brackets
        if let Some(start) = self.canonical.find('[') {
            if let Some(end) = self.canonical.find(']') {
                let inner = &self.canonical[start+1..end];
                if inner.is_empty() {
                    return Ok(0);
                }
                return Ok(inner.matches(',').count() + 1);
            }
        }
        Ok(0)
    }
    
    /// Compute cardinality (number of points) - returns parametric expression
    pub fn cardinality(&self) -> IslResult<String> {
        run_isl(&format!("card({});", self.expression))
    }
    
    /// Get sample point from the set (if non-empty)
    pub fn sample(&self) -> IslResult<Option<Vec<i64>>> {
        let result = run_isl(&format!("sample({});", self.expression))?;
        if result.contains("{ }") {
            return Ok(None);
        }
        // Parse result like "{ [1, 2, 3] }"
        if let Some(start) = result.find('[') {
            if let Some(end) = result.find(']') {
                let inner = &result[start+1..end];
                let values: Result<Vec<i64>, _> = inner
                    .split(',')
                    .map(|s| s.trim().parse())
                    .collect();
                if let Ok(v) = values {
                    return Ok(Some(v));
                }
            }
        }
        Ok(None)
    }
    
    /// Get lexicographic minimum
    pub fn lexmin(&self) -> IslResult<Self> {
        let expr = format!("lexmin({})", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Get lexicographic maximum
    pub fn lexmax(&self) -> IslResult<Self> {
        let expr = format!("lexmax({})", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Coalesce - simplify representation by merging constraints
    pub fn coalesce(&self) -> IslResult<Self> {
        let result = run_isl(&format!("coalesce({});", self.expression))?;
        Ok(Self::new(self.expression.clone(), result))
    }
    
    /// Compute the convex hull
    pub fn convex_hull(&self) -> IslResult<Self> {
        let expr = format!("convex_hull({})", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Compute affine hull (smallest affine space containing the set)
    pub fn affine_hull(&self) -> IslResult<Self> {
        let expr = format!("affine_hull({})", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
}

impl std::fmt::Display for IslSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.canonical)
    }
}

/// Helper to create a rectangular iteration domain
pub fn rectangular_domain(
    vars: &[&str],
    lower_bounds: &[&str],
    upper_bounds: &[&str],
) -> IslResult<IslSet> {
    assert_eq!(vars.len(), lower_bounds.len());
    assert_eq!(vars.len(), upper_bounds.len());
    
    let mut constraints = Vec::new();
    for i in 0..vars.len() {
        constraints.push(format!("{} <= {}", lower_bounds[i], vars[i]));
        constraints.push(format!("{} < {}", vars[i], upper_bounds[i]));
    }
    
    let vars_str = vars.join(",");
    let constraints_str = constraints.join(" and ");
    let expr = format!("{{ [{}] : {} }}", vars_str, constraints_str);
    let result = run_isl(&format!("{};", expr))?;
    Ok(IslSet::new(expr, result))
}

/// Helper to create a triangular domain
pub fn triangular_domain(
    var1: &str, var2: &str,
    n: &str,
) -> IslResult<IslSet> {
    let expr = format!(
        "{{ [{},{}] : 0 <= {} < {} and {} <= {} < {} }}",
        var1, var2, var1, n, var1, var2, n
    );
    let result = run_isl(&format!("{};", expr))?;
    Ok(IslSet::new(expr, result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isl::is_isl_available;
    
    #[test]
    fn test_from_loop() {
        if !is_isl_available() {
            return;
        }
        
        let set = IslSet::from_loop(
            &["i", "j"],
            &["0 <= i", "i < 10", "0 <= j", "j < 10"]
        );
        assert!(set.is_ok());
    }
}
