//! ISL Map - represents relations and access functions

use super::{IslResult, run_isl};
use super::set::IslSet;

/// An ISL map representing a relation between integer tuples
/// 
/// Maps represent access functions, dependence relations, and schedule functions.
/// 
/// # Examples
/// 
/// ```text
/// { [i] -> [i+1] }           // Successor relation
/// { [i,j] -> A[i,j] }        // Array access
/// { S[i,j] -> [i,j,0] }      // Schedule function
/// ```
#[derive(Clone, Debug)]
pub struct IslMap {
    /// Original expression
    expression: String,
    /// Simplified/canonical form from ISL
    canonical: String,
}

impl IslMap {
    /// Create a new ISL map
    pub fn new(expression: String, canonical: String) -> Self {
        Self { expression, canonical }
    }
    
    /// Create an identity map
    pub fn identity(dims: usize) -> IslResult<Self> {
        let vars: Vec<String> = (0..dims).map(|i| format!("i{}", i)).collect();
        let vars_str = vars.join(",");
        let expr = format!("{{ [{}] -> [{}] }}", vars_str, vars_str);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Create an empty map
    pub fn empty() -> Self {
        Self::new("{ }".to_string(), "{ }".to_string())
    }
    
    /// Create an array access map
    /// 
    /// # Arguments
    /// * `stmt_name` - Statement identifier (e.g., "S0")
    /// * `loop_vars` - Loop iteration variables
    /// * `array_name` - Array being accessed
    /// * `subscripts` - Array subscript expressions
    /// 
    /// # Example
    /// ```ignore
    /// // A[i+1][j] access in statement S0
    /// let map = IslMap::array_access("S0", &["i","j"], "A", &["i+1", "j"])?;
    /// // Creates: { S0[i,j] -> A[i+1,j] }
    /// ```
    pub fn array_access(
        stmt_name: &str,
        loop_vars: &[&str],
        array_name: &str,
        subscripts: &[&str],
    ) -> IslResult<Self> {
        let domain = format!("{}[{}]", stmt_name, loop_vars.join(","));
        let range = format!("{}[{}]", array_name, subscripts.join(","));
        let expr = format!("{{ {} -> {} }}", domain, range);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Create a schedule map
    /// 
    /// # Arguments
    /// * `stmt_name` - Statement identifier
    /// * `loop_vars` - Original loop variables
    /// * `schedule` - Schedule expression (time dimensions)
    pub fn schedule(
        stmt_name: &str,
        loop_vars: &[&str],
        schedule: &[&str],
    ) -> IslResult<Self> {
        let domain = format!("{}[{}]", stmt_name, loop_vars.join(","));
        let range = format!("[{}]", schedule.join(","));
        let expr = format!("{{ {} -> {} }}", domain, range);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Get the expression
    pub fn expr(&self) -> &str {
        &self.expression
    }
    
    /// Get the canonical form
    pub fn canonical(&self) -> &str {
        &self.canonical
    }
    
    /// Compute the domain of the map (source set)
    pub fn domain(&self) -> IslResult<IslSet> {
        let expr = format!("domain({})", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(IslSet::new(expr, result))
    }
    
    /// Compute the range of the map (target set)
    pub fn range(&self) -> IslResult<IslSet> {
        let expr = format!("range({})", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(IslSet::new(expr, result))
    }
    
    /// Compute the inverse of the map
    pub fn inverse(&self) -> IslResult<Self> {
        let expr = format!("({})^-1", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Compose two maps: (self . other)
    /// Result maps x to z where self maps x to y and other maps y to z
    pub fn compose(&self, other: &IslMap) -> IslResult<Self> {
        let expr = format!("({}) . ({})", self.expression, other.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Apply map to a set (compute image)
    pub fn apply(&self, set: &IslSet) -> IslResult<IslSet> {
        let expr = format!("({}) . ({})", set.expr(), self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(IslSet::new(expr, result))
    }
    
    /// Compute the intersection with another map
    pub fn intersect(&self, other: &IslMap) -> IslResult<Self> {
        let expr = format!("({}) * ({})", self.expression, other.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Compute the union with another map
    pub fn union(&self, other: &IslMap) -> IslResult<Self> {
        let expr = format!("({}) + ({})", self.expression, other.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Restrict the domain
    pub fn restrict_domain(&self, set: &IslSet) -> IslResult<Self> {
        let expr = format!("({}) * (({})->{{}})", self.expression, set.expr());
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Restrict the range
    pub fn restrict_range(&self, set: &IslSet) -> IslResult<Self> {
        let expr = format!("({}) * ({{}}->({})))", self.expression, set.expr());
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Check if map is empty
    pub fn is_empty(&self) -> IslResult<bool> {
        let result = run_isl(&format!("({}) = {{}};", self.expression))?;
        Ok(result.trim() == "True")
    }
    
    /// Check if map is single-valued (a function)
    pub fn is_single_valued(&self) -> IslResult<bool> {
        let result = run_isl(&format!("is_single_valued({});", self.expression))?;
        Ok(result.trim() == "True")
    }
    
    /// Compute transitive closure
    pub fn transitive_closure(&self) -> IslResult<Self> {
        let expr = format!("({})^+", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Compute reflexive transitive closure
    pub fn reflexive_transitive_closure(&self) -> IslResult<Self> {
        let expr = format!("({})^*", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Lexicographically positive relations (source < target)
    pub fn lex_lt(&self) -> IslResult<Self> {
        let expr = format!("lex_lt({})", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Lexicographically non-negative relations (source <= target)  
    pub fn lex_le(&self) -> IslResult<Self> {
        let expr = format!("lex_le({})", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(Self::new(expr, result))
    }
    
    /// Coalesce - simplify representation
    pub fn coalesce(&self) -> IslResult<Self> {
        let result = run_isl(&format!("coalesce({});", self.expression))?;
        Ok(Self::new(self.expression.clone(), result))
    }
    
    /// Deltas - compute difference between source and target
    /// For { [i] -> [j] }, returns { [j-i] }
    pub fn deltas(&self) -> IslResult<IslSet> {
        let expr = format!("deltas({})", self.expression);
        let result = run_isl(&format!("{};", expr))?;
        Ok(IslSet::new(expr, result))
    }
}

impl std::fmt::Display for IslMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.canonical)
    }
}

/// Compute dependence polyhedron between two accesses
/// 
/// # Arguments
/// * `write_domain` - Domain of write statement
/// * `read_domain` - Domain of read statement
/// * `write_access` - Write access function (iteration -> array element)
/// * `read_access` - Read access function (iteration -> array element)
/// 
/// Returns the dependence relation: iterations where dependence exists
pub fn compute_dependence(
    write_domain: &IslSet,
    read_domain: &IslSet,
    write_access: &IslMap,
    read_access: &IslMap,
) -> IslResult<IslMap> {
    // Flow dependence: write -> read where same element, write before read
    let script = format!(
        r#"
        W := {};
        R := {};
        WA := {};
        RA := {};
        # Same element accessed
        SAME := (WA . (RA^-1));
        # Write iteration in domain, read iteration in domain
        VALID := (W -> R);
        # Combine
        DEP := SAME * VALID;
        DEP;
        "#,
        write_domain.expr(),
        read_domain.expr(),
        write_access.expr(),
        read_access.expr(),
    );
    
    let result = run_isl(&script)?;
    Ok(IslMap::new("flow_dependence".to_string(), result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isl::is_isl_available;
    
    #[test]
    fn test_array_access() {
        if !is_isl_available() {
            return;
        }
        
        let map = IslMap::array_access(
            "S0",
            &["i", "j"],
            "A",
            &["i", "j+1"]
        );
        assert!(map.is_ok());
    }
}
