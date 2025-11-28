//! Polyhedral spaces define the dimensions of iteration domains.
//!
//! A space describes the structure of an iteration domain or map:
//! - Input dimensions (for maps)
//! - Output dimensions
//! - Parameter dimensions (symbolic constants)

use serde::{Serialize, Deserialize};
use std::fmt;

/// A polyhedral space describes the dimensionality and structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Space {
    /// Number of set/output dimensions
    pub n_dim: usize,
    /// Number of parameter dimensions
    pub n_param: usize,
    /// Number of input dimensions (for maps only)
    pub n_in: usize,
    /// Names of dimensions (optional, for debugging)
    pub dim_names: Vec<String>,
    /// Names of parameters (optional)
    pub param_names: Vec<String>,
}

impl Space {
    /// Create a new set space with the given dimensions.
    pub fn set(n_dim: usize) -> Self {
        Self {
            n_dim,
            n_param: 0,
            n_in: 0,
            dim_names: Vec::new(),
            param_names: Vec::new(),
        }
    }

    /// Create a new set space with parameters.
    pub fn set_with_params(n_dim: usize, n_param: usize) -> Self {
        Self {
            n_dim,
            n_param,
            n_in: 0,
            dim_names: Vec::new(),
            param_names: Vec::new(),
        }
    }

    /// Create a new map space.
    pub fn map(n_in: usize, n_out: usize) -> Self {
        Self {
            n_dim: n_out,
            n_param: 0,
            n_in,
            dim_names: Vec::new(),
            param_names: Vec::new(),
        }
    }

    /// Create a new map space with parameters.
    pub fn map_with_params(n_in: usize, n_out: usize, n_param: usize) -> Self {
        Self {
            n_dim: n_out,
            n_param,
            n_in,
            dim_names: Vec::new(),
            param_names: Vec::new(),
        }
    }

    /// Check if this is a set space (no input dimensions).
    pub fn is_set(&self) -> bool {
        self.n_in == 0
    }

    /// Check if this is a map space (has input dimensions).
    pub fn is_map(&self) -> bool {
        self.n_in > 0
    }

    /// Get the number of output/set dimensions.
    pub fn dim(&self) -> usize {
        self.n_dim
    }

    /// Get the total number of dimensions.
    pub fn total_dim(&self) -> usize {
        self.n_param + self.n_in + self.n_dim
    }

    /// Set dimension names.
    pub fn with_dim_names(mut self, names: Vec<String>) -> Self {
        self.dim_names = names;
        self
    }

    /// Set parameter names.
    pub fn with_param_names(mut self, names: Vec<String>) -> Self {
        self.param_names = names;
        self
    }

    /// Get the name of a dimension.
    pub fn dim_name(&self, idx: usize) -> Option<&str> {
        self.dim_names.get(idx).map(|s| s.as_str())
    }

    /// Get the name of a parameter.
    pub fn param_name(&self, idx: usize) -> Option<&str> {
        self.param_names.get(idx).map(|s| s.as_str())
    }

    /// Get all dimension names with defaults.
    pub fn all_dim_names(&self) -> Vec<String> {
        (0..self.n_dim)
            .map(|i| {
                self.dim_names.get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("d{}", i))
            })
            .collect()
    }

    /// Get all parameter names with defaults.
    pub fn all_param_names(&self) -> Vec<String> {
        (0..self.n_param)
            .map(|i| {
                self.param_names.get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("p{}", i))
            })
            .collect()
    }

    /// Create the domain space from a map space.
    pub fn domain(&self) -> Space {
        Space::set_with_params(self.n_in, self.n_param)
            .with_param_names(self.param_names.clone())
    }

    /// Create the range space from a map space.
    pub fn range(&self) -> Space {
        Space::set_with_params(self.n_dim, self.n_param)
            .with_param_names(self.param_names.clone())
    }

    /// Wrap this set space as a map from itself.
    pub fn identity_map(&self) -> Space {
        Space::map_with_params(self.n_dim, self.n_dim, self.n_param)
            .with_param_names(self.param_names.clone())
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_map() {
            write!(f, "[{}] -> [{}]", self.n_in, self.n_dim)?;
        } else {
            write!(f, "[{}]", self.n_dim)?;
        }
        if self.n_param > 0 {
            write!(f, " : {} params", self.n_param)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_space() {
        let space = Space::set(3);
        assert!(space.is_set());
        assert!(!space.is_map());
        assert_eq!(space.dim(), 3);
    }

    #[test]
    fn test_map_space() {
        let space = Space::map(2, 3);
        assert!(!space.is_set());
        assert!(space.is_map());
        assert_eq!(space.n_in, 2);
        assert_eq!(space.dim(), 3);
    }

    #[test]
    fn test_with_names() {
        let space = Space::set(2)
            .with_dim_names(vec!["i".to_string(), "j".to_string()]);
        assert_eq!(space.dim_name(0), Some("i"));
        assert_eq!(space.dim_name(1), Some("j"));
    }
}
