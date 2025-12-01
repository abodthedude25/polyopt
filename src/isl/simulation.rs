//! Pure Rust Polyhedral Simulation
//!
//! This module provides a pure Rust simulation of basic polyhedral operations
//! for use when the ISL library is not installed.

use std::collections::HashSet;

/// Represents a polyhedral set as a collection of constraints
#[derive(Debug, Clone)]
pub struct PolySet {
    /// Variable names
    pub vars: Vec<String>,
    /// Parameter names
    pub params: Vec<String>,
    /// Constraints in the form: coeffs[0]*v0 + coeffs[1]*v1 + ... + const >= 0
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    /// Coefficients for variables
    pub var_coeffs: Vec<i64>,
    /// Coefficients for parameters  
    pub param_coeffs: Vec<i64>,
    /// Constant term
    pub constant: i64,
    /// True if equality (=), false if inequality (>=)
    pub is_equality: bool,
}

impl PolySet {
    /// Parse ISL set notation like "{ [i, j] : 0 <= i < N and 0 <= j < M }"
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        
        // Extract variables from brackets
        let vars_start = s.find('[').ok_or("Missing [")?;
        let vars_end = s.find(']').ok_or("Missing ]")?;
        let vars_str = &s[vars_start + 1..vars_end];
        let vars: Vec<String> = vars_str.split(',')
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .collect();
        
        // Extract constraints after ':'
        let mut constraints = Vec::new();
        let mut params = HashSet::new();
        
        if let Some(colon_pos) = s.find(':') {
            let constraint_str = &s[colon_pos + 1..s.rfind('}').unwrap_or(s.len())];
            
            // Split by "and"
            for part in constraint_str.split(" and ") {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                
                // Parse constraint
                if let Some(constraint) = parse_constraint(part, &vars, &mut params) {
                    constraints.push(constraint);
                }
            }
        }
        
        Ok(PolySet {
            vars,
            params: params.into_iter().collect(),
            constraints,
        })
    }
    
    /// Check if a point satisfies all constraints
    pub fn contains(&self, point: &[i64], param_values: &[i64]) -> bool {
        for c in &self.constraints {
            let mut sum = c.constant;
            
            for (i, &coeff) in c.var_coeffs.iter().enumerate() {
                if i < point.len() {
                    sum += coeff * point[i];
                }
            }
            
            for (i, &coeff) in c.param_coeffs.iter().enumerate() {
                if i < param_values.len() {
                    sum += coeff * param_values[i];
                }
            }
            
            if c.is_equality {
                if sum != 0 {
                    return false;
                }
            } else {
                if sum < 0 {
                    return false;
                }
            }
        }
        true
    }
    
    /// Enumerate all integer points (for small sets)
    pub fn enumerate(&self, param_values: &[i64], max_per_dim: i64) -> Vec<Vec<i64>> {
        let mut points = Vec::new();
        
        fn enumerate_rec(
            set: &PolySet, 
            param_values: &[i64],
            current: &mut Vec<i64>, 
            dim: usize, 
            max_val: i64,
            points: &mut Vec<Vec<i64>>
        ) {
            if dim == set.vars.len() {
                if set.contains(current, param_values) {
                    points.push(current.clone());
                }
                return;
            }
            
            for val in 0..=max_val {
                current.push(val);
                enumerate_rec(set, param_values, current, dim + 1, max_val, points);
                current.pop();
            }
        }
        
        let mut current = Vec::new();
        enumerate_rec(self, param_values, &mut current, 0, max_per_dim, &mut points);
        points
    }
    
    /// Compute intersection with another set
    pub fn intersect(&self, other: &PolySet) -> PolySet {
        let mut result = self.clone();
        result.constraints.extend(other.constraints.clone());
        result
    }
    
    /// Check if set is empty
    pub fn is_empty(&self, param_values: &[i64]) -> bool {
        self.enumerate(param_values, 10).is_empty()
    }
    
    /// Convert to ISL string notation
    pub fn to_isl_string(&self) -> String {
        let vars_str = self.vars.join(", ");
        
        if self.constraints.is_empty() {
            return format!("{{ [{}] }}", vars_str);
        }
        
        let constraints_str: Vec<String> = self.constraints.iter()
            .map(|c| constraint_to_string(c, &self.vars, &self.params))
            .collect();
        
        format!("{{ [{}] : {} }}", vars_str, constraints_str.join(" and "))
    }
}

fn parse_constraint(s: &str, vars: &[String], params: &mut HashSet<String>) -> Option<Constraint> {
    let s = s.trim();
    
    // Handle <= or <
    if let Some(pos) = s.find("<=") {
        let left = s[..pos].trim();
        let right = s[pos + 2..].trim();
        return Some(make_constraint(right, left, vars, params, false));
    }
    
    if let Some(pos) = s.find('<') {
        if !s[pos..].starts_with("<=") {
            let left = s[..pos].trim();
            let right = s[pos + 1..].trim();
            let mut c = make_constraint(right, left, vars, params, false);
            c.constant -= 1;
            return Some(c);
        }
    }
    
    if let Some(pos) = s.find(">=") {
        let left = s[..pos].trim();
        let right = s[pos + 2..].trim();
        return Some(make_constraint(left, right, vars, params, false));
    }
    
    if let Some(pos) = s.find('>') {
        if !s[pos..].starts_with(">=") {
            let left = s[..pos].trim();
            let right = s[pos + 1..].trim();
            let mut c = make_constraint(left, right, vars, params, false);
            c.constant -= 1;
            return Some(c);
        }
    }
    
    if let Some(pos) = s.find('=') {
        if !s[..pos].ends_with('<') && !s[..pos].ends_with('>') && !s[pos + 1..].starts_with('=') {
            let left = s[..pos].trim();
            let right = s[pos + 1..].trim();
            return Some(make_constraint(left, right, vars, params, true));
        }
    }
    
    None
}

fn make_constraint(pos_term: &str, neg_term: &str, vars: &[String], params: &mut HashSet<String>, is_eq: bool) -> Constraint {
    let mut var_coeffs = vec![0i64; vars.len()];
    let mut param_coeffs = Vec::new();
    let mut constant = 0i64;
    
    parse_term(pos_term, vars, params, &mut var_coeffs, &mut param_coeffs, &mut constant, 1);
    parse_term(neg_term, vars, params, &mut var_coeffs, &mut param_coeffs, &mut constant, -1);
    
    Constraint {
        var_coeffs,
        param_coeffs,
        constant,
        is_equality: is_eq,
    }
}

fn parse_term(term: &str, vars: &[String], params: &mut HashSet<String>, 
              var_coeffs: &mut [i64], param_coeffs: &mut Vec<i64>, 
              constant: &mut i64, sign: i64) {
    let term = term.trim();
    
    if let Ok(num) = term.parse::<i64>() {
        *constant += sign * num;
        return;
    }
    
    for (i, var) in vars.iter().enumerate() {
        if term == var {
            var_coeffs[i] += sign;
            return;
        }
    }
    
    params.insert(term.to_string());
}

fn constraint_to_string(c: &Constraint, vars: &[String], _params: &[String]) -> String {
    let mut parts = Vec::new();
    
    for (i, &coeff) in c.var_coeffs.iter().enumerate() {
        if coeff != 0 && i < vars.len() {
            if coeff == 1 {
                parts.push(vars[i].clone());
            } else if coeff == -1 {
                parts.push(format!("-{}", vars[i]));
            } else {
                parts.push(format!("{}*{}", coeff, vars[i]));
            }
        }
    }
    
    if c.constant != 0 {
        parts.push(c.constant.to_string());
    }
    
    if parts.is_empty() {
        parts.push("0".to_string());
    }
    
    let lhs = parts.join(" + ").replace("+ -", "- ");
    let op = if c.is_equality { "=" } else { ">=" };
    format!("{} {} 0", lhs, op)
}

/// Represents an affine map (access function or schedule)
#[derive(Debug, Clone)]
pub struct PolyMap {
    pub in_vars: Vec<String>,
    pub out_vars: Vec<String>,
    pub params: Vec<String>,
    /// For each output dimension: coefficients for input vars + constant
    pub mappings: Vec<Vec<i64>>,
}

impl PolyMap {
    /// Apply map to a point
    pub fn apply(&self, input: &[i64]) -> Vec<i64> {
        self.mappings.iter().map(|m| {
            let mut sum = *m.last().unwrap_or(&0);
            for (i, &coeff) in m.iter().take(input.len()).enumerate() {
                sum += coeff * input[i];
            }
            sum
        }).collect()
    }
}

/// Code generation from polyhedral representation
pub mod codegen {
    use super::*;
    
    /// Generate nested loops from a polyhedral set
    pub fn generate_loops(domain: &PolySet, body: &str) -> String {
        let mut code = String::new();
        let indent = "    ";
        
        for (i, var) in domain.vars.iter().enumerate() {
            let ind = indent.repeat(i);
            let (lower, upper) = find_bounds(domain, i);
            
            code.push_str(&format!("{}for (int {} = {}; {} < {}; {}++) {{\n",
                ind, var, lower, var, upper, var));
        }
        
        let body_indent = indent.repeat(domain.vars.len());
        code.push_str(&format!("{}{}\n", body_indent, body));
        
        for i in (0..domain.vars.len()).rev() {
            let ind = indent.repeat(i);
            code.push_str(&format!("{}}}\n", ind));
        }
        
        code
    }
    
    fn find_bounds(domain: &PolySet, dim: usize) -> (String, String) {
        let var = &domain.vars[dim];
        let mut lower = "0".to_string();
        let mut upper = "N".to_string();
        
        for c in &domain.constraints {
            if dim < c.var_coeffs.len() {
                let coeff = c.var_coeffs[dim];
                if coeff == 1 && c.constant <= 0 && !c.is_equality {
                    lower = format!("{}", -c.constant);
                } else if coeff == -1 && !c.is_equality {
                    upper = format!("{}", c.constant);
                }
            }
        }
        
        (lower, upper)
    }
    
    /// Generate tiled loops
    pub fn generate_tiled_loops(domain: &PolySet, body: &str, tile_sizes: &[usize]) -> String {
        let mut code = String::new();
        let indent = "    ";
        
        // Generate tile loops
        for (i, var) in domain.vars.iter().enumerate() {
            let tile_size = tile_sizes.get(i).copied().unwrap_or(32);
            let tile_var = format!("{}_tile", var);
            
            code.push_str(&format!("{}for (int {} = 0; {} < N; {} += {}) {{\n",
                indent.repeat(i), tile_var, tile_var, tile_var, tile_size));
        }
        
        // Generate point loops within tiles
        let base_indent = domain.vars.len();
        for (i, var) in domain.vars.iter().enumerate() {
            let tile_size = tile_sizes.get(i).copied().unwrap_or(32);
            let tile_var = format!("{}_tile", var);
            
            code.push_str(&format!("{}for (int {} = {}; {} < min({} + {}, N); {}++) {{\n",
                indent.repeat(base_indent + i), var, tile_var, var, tile_var, tile_size, var));
        }
        
        // Generate body
        let body_indent = indent.repeat(base_indent * 2);
        code.push_str(&format!("{}{}\n", body_indent, body));
        
        // Close all loops
        for i in (0..domain.vars.len() * 2).rev() {
            code.push_str(&format!("{}}}\n", indent.repeat(i)));
        }
        
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_set() {
        let set = PolySet::parse("{ [i, j] : 0 <= i and i < N and 0 <= j and j < M }").unwrap();
        assert_eq!(set.vars, vec!["i", "j"]);
    }
    
    #[test]
    fn test_enumerate() {
        let set = PolySet::parse("{ [i, j] : 0 <= i and i < 3 and 0 <= j and j < 3 }").unwrap();
        let points = set.enumerate(&[], 5);
        assert!(!points.is_empty());
    }
}
