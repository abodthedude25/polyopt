//! Affine expressions for polyhedral representation.
//!
//! An affine expression is a linear combination of variables plus a constant:
//! `aff(x) = c0 + c1*x1 + c2*x2 + ... + cn*xn`

use num_rational::Rational64;
use serde::{Serialize, Deserialize};
use std::fmt;
use std::ops::{Add, Sub, Mul, Neg};

/// Integer floor division (rounds toward negative infinity).
fn floor_div_i64(a: i64, b: i64) -> i64 {
    if b == 0 { return 0; }
    let d = a / b;
    let r = a % b;
    if (r != 0) && ((r < 0) != (b < 0)) {
        d - 1
    } else {
        d
    }
}

/// An affine expression: constant + sum(coeff[i] * var[i])
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineExpr {
    /// Constant term
    pub constant: i64,
    /// Coefficients for each dimension (index = dimension index)
    pub coeffs: Vec<i64>,
    /// Coefficients for parameters (index = parameter index)
    pub param_coeffs: Vec<i64>,
}

impl AffineExpr {
    /// Create a zero expression.
    pub fn zero(n_dim: usize, n_param: usize) -> Self {
        Self {
            constant: 0,
            coeffs: vec![0; n_dim],
            param_coeffs: vec![0; n_param],
        }
    }

    /// Create a constant expression.
    pub fn constant(value: i64, n_dim: usize, n_param: usize) -> Self {
        Self {
            constant: value,
            coeffs: vec![0; n_dim],
            param_coeffs: vec![0; n_param],
        }
    }

    /// Create an expression for a single dimension variable.
    pub fn var(dim: usize, n_dim: usize, n_param: usize) -> Self {
        let mut coeffs = vec![0; n_dim];
        if dim < n_dim {
            coeffs[dim] = 1;
        }
        Self {
            constant: 0,
            coeffs,
            param_coeffs: vec![0; n_param],
        }
    }

    /// Create an expression for a parameter.
    pub fn param(param_idx: usize, n_dim: usize, n_param: usize) -> Self {
        let mut param_coeffs = vec![0; n_param];
        if param_idx < n_param {
            param_coeffs[param_idx] = 1;
        }
        Self {
            constant: 0,
            coeffs: vec![0; n_dim],
            param_coeffs,
        }
    }

    /// Check if this is a constant expression.
    pub fn is_constant(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0) &&
        self.param_coeffs.iter().all(|&c| c == 0)
    }

    /// Check if this expression is zero.
    pub fn is_zero(&self) -> bool {
        self.constant == 0 && self.is_constant()
    }

    /// Get the constant value if this is a constant expression.
    pub fn as_constant(&self) -> Option<i64> {
        if self.is_constant() {
            Some(self.constant)
        } else {
            None
        }
    }

    /// Get the number of dimensions.
    pub fn n_dim(&self) -> usize {
        self.coeffs.len()
    }

    /// Get the number of parameters.
    pub fn n_param(&self) -> usize {
        self.param_coeffs.len()
    }

    /// Get coefficient for a dimension.
    pub fn coeff(&self, dim: usize) -> i64 {
        self.coeffs.get(dim).copied().unwrap_or(0)
    }

    /// Get coefficient for a parameter.
    pub fn param_coeff(&self, idx: usize) -> i64 {
        self.param_coeffs.get(idx).copied().unwrap_or(0)
    }

    /// Set coefficient for a dimension.
    pub fn set_coeff(&mut self, dim: usize, value: i64) {
        if dim < self.coeffs.len() {
            self.coeffs[dim] = value;
        }
    }

    /// Set coefficient for a parameter.
    pub fn set_param_coeff(&mut self, idx: usize, value: i64) {
        if idx < self.param_coeffs.len() {
            self.param_coeffs[idx] = value;
        }
    }

    /// Evaluate the expression given concrete values.
    pub fn evaluate(&self, dim_values: &[i64], param_values: &[i64]) -> i64 {
        let mut result = self.constant;
        for (i, &c) in self.coeffs.iter().enumerate() {
            if let Some(&v) = dim_values.get(i) {
                result += c * v;
            }
        }
        for (i, &c) in self.param_coeffs.iter().enumerate() {
            if let Some(&v) = param_values.get(i) {
                result += c * v;
            }
        }
        result
    }

    /// Scale the expression by a constant.
    pub fn scale(&self, factor: i64) -> Self {
        Self {
            constant: self.constant * factor,
            coeffs: self.coeffs.iter().map(|&c| c * factor).collect(),
            param_coeffs: self.param_coeffs.iter().map(|&c| c * factor).collect(),
        }
    }

    /// Create a floor division expression (for tiling).
    /// This is approximate - for exact floor division when divisible.
    pub fn floordiv(&self, divisor: i64) -> Self {
        if divisor == 0 {
            return self.clone();
        }
        // Create an approximation: divide all coefficients
        // This is exact when the original is divisible by the divisor
        Self {
            constant: floor_div_i64(self.constant, divisor),
            coeffs: self.coeffs.iter().map(|&c| floor_div_i64(c, divisor)).collect(),
            param_coeffs: self.param_coeffs.iter().map(|&c| floor_div_i64(c, divisor)).collect(),
        }
    }

    /// Floor division of the expression by a constant.
    pub fn floor_div(&self, divisor: i64) -> Option<Self> {
        if divisor == 0 {
            return None;
        }
        // For floor division to be exact, all coefficients must be divisible
        if self.constant % divisor != 0 {
            return None;
        }
        for &c in &self.coeffs {
            if c % divisor != 0 {
                return None;
            }
        }
        for &c in &self.param_coeffs {
            if c % divisor != 0 {
                return None;
            }
        }
        Some(Self {
            constant: self.constant / divisor,
            coeffs: self.coeffs.iter().map(|&c| c / divisor).collect(),
            param_coeffs: self.param_coeffs.iter().map(|&c| c / divisor).collect(),
        })
    }

    /// Get GCD of all coefficients.
    pub fn gcd(&self) -> i64 {
        use num_integer::Integer;
        let mut g = self.constant.abs();
        for &c in &self.coeffs {
            g = g.gcd(&c.abs());
        }
        for &c in &self.param_coeffs {
            g = g.gcd(&c.abs());
        }
        if g == 0 { 1 } else { g }
    }

    /// Normalize by dividing by GCD.
    pub fn normalize(&self) -> Self {
        let g = self.gcd();
        if g <= 1 {
            self.clone()
        } else {
            self.floor_div(g).unwrap_or_else(|| self.clone())
        }
    }

    /// Convert to string with given dimension and parameter names.
    pub fn to_string_with_names(&self, dim_names: &[String], param_names: &[String]) -> String {
        let mut parts = Vec::new();
        
        // Add constant
        if self.constant != 0 || (self.is_constant()) {
            parts.push(format!("{}", self.constant));
        }

        // Add dimension terms
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c != 0 {
                let default_name = format!("d{}", i);
                let name = dim_names.get(i)
                    .map(|s| s.as_str())
                    .unwrap_or(&default_name);
                if c == 1 {
                    parts.push(name.to_string());
                } else if c == -1 {
                    parts.push(format!("-{}", name));
                } else {
                    parts.push(format!("{}*{}", c, name));
                }
            }
        }

        // Add parameter terms
        for (i, &c) in self.param_coeffs.iter().enumerate() {
            if c != 0 {
                let default_name = format!("p{}", i);
                let name = param_names.get(i)
                    .map(|s| s.as_str())
                    .unwrap_or(&default_name);
                if c == 1 {
                    parts.push(name.to_string());
                } else if c == -1 {
                    parts.push(format!("-{}", name));
                } else {
                    parts.push(format!("{}*{}", c, name));
                }
            }
        }

        if parts.is_empty() {
            "0".to_string()
        } else {
            parts.join(" + ").replace("+ -", "- ")
        }
    }
}

impl Add for AffineExpr {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.coeffs.len(), other.coeffs.len());
        assert_eq!(self.param_coeffs.len(), other.param_coeffs.len());
        Self {
            constant: self.constant + other.constant,
            coeffs: self.coeffs.iter().zip(&other.coeffs)
                .map(|(&a, &b)| a + b).collect(),
            param_coeffs: self.param_coeffs.iter().zip(&other.param_coeffs)
                .map(|(&a, &b)| a + b).collect(),
        }
    }
}

impl Sub for AffineExpr {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.coeffs.len(), other.coeffs.len());
        assert_eq!(self.param_coeffs.len(), other.param_coeffs.len());
        Self {
            constant: self.constant - other.constant,
            coeffs: self.coeffs.iter().zip(&other.coeffs)
                .map(|(&a, &b)| a - b).collect(),
            param_coeffs: self.param_coeffs.iter().zip(&other.param_coeffs)
                .map(|(&a, &b)| a - b).collect(),
        }
    }
}

impl Neg for AffineExpr {
    type Output = Self;

    fn neg(self) -> Self {
        self.scale(-1)
    }
}

impl fmt::Display for AffineExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dim_names: Vec<String> = (0..self.n_dim()).map(|i| format!("d{}", i)).collect();
        let param_names: Vec<String> = (0..self.n_param()).map(|i| format!("p{}", i)).collect();
        write!(f, "{}", self.to_string_with_names(&dim_names, &param_names))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant() {
        let expr = AffineExpr::constant(5, 2, 1);
        assert!(expr.is_constant());
        assert_eq!(expr.evaluate(&[1, 2], &[3]), 5);
    }

    #[test]
    fn test_var() {
        let expr = AffineExpr::var(0, 2, 0);
        assert!(!expr.is_constant());
        assert_eq!(expr.evaluate(&[7, 3], &[]), 7);
    }

    #[test]
    fn test_add() {
        let e1 = AffineExpr::var(0, 2, 0);
        let e2 = AffineExpr::var(1, 2, 0);
        let sum = e1 + e2;
        assert_eq!(sum.evaluate(&[3, 4], &[]), 7);
    }

    #[test]
    fn test_scale() {
        let expr = AffineExpr::var(0, 2, 0);
        let scaled = expr.scale(3);
        assert_eq!(scaled.evaluate(&[5, 0], &[]), 15);
    }

    #[test]
    fn test_display() {
        let mut expr = AffineExpr::zero(2, 1);
        expr.constant = 5;
        expr.coeffs[0] = 2;
        expr.coeffs[1] = -1;
        expr.param_coeffs[0] = 1;
        
        let s = expr.to_string_with_names(
            &["i".to_string(), "j".to_string()],
            &["N".to_string()],
        );
        assert!(s.contains("2*i"));
    }
}
