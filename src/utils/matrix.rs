//! Matrix operations for polyhedral transformations.
//!
//! This module provides matrix utilities used throughout the framework,
//! particularly for representing and manipulating affine transformations.

use nalgebra::{DMatrix, DVector};
use num_rational::Rational64;
use num_integer::Integer;
use num_traits::Signed;
use std::fmt;

/// A matrix with rational entries, used for exact arithmetic in polyhedral operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RationalMatrix {
    data: Vec<Vec<Rational64>>,
    rows: usize,
    cols: usize,
}

impl RationalMatrix {
    /// Create a new matrix with the given dimensions, initialized to zero.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![Rational64::from_integer(0); cols]; rows],
            rows,
            cols,
        }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut mat = Self::zeros(n, n);
        for i in 0..n {
            mat.data[i][i] = Rational64::from_integer(1);
        }
        mat
    }

    /// Create a matrix from a 2D vector.
    pub fn from_vec(data: Vec<Vec<i64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        let rational_data: Vec<Vec<Rational64>> = data
            .into_iter()
            .map(|row| row.into_iter().map(Rational64::from_integer).collect())
            .collect();
        Self {
            data: rational_data,
            rows,
            cols,
        }
    }

    /// Get the number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Get an element.
    pub fn get(&self, row: usize, col: usize) -> Option<&Rational64> {
        self.data.get(row)?.get(col)
    }

    /// Get a mutable reference to an element.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Rational64> {
        self.data.get_mut(row)?.get_mut(col)
    }

    /// Set an element.
    pub fn set(&mut self, row: usize, col: usize, value: Rational64) {
        if row < self.rows && col < self.cols {
            self.data[row][col] = value;
        }
    }

    /// Set an element from an integer.
    pub fn set_int(&mut self, row: usize, col: usize, value: i64) {
        self.set(row, col, Rational64::from_integer(value));
    }

    /// Get a row as a vector.
    pub fn row(&self, row: usize) -> Option<&Vec<Rational64>> {
        self.data.get(row)
    }

    /// Get a column as a vector.
    pub fn column(&self, col: usize) -> Vec<Rational64> {
        self.data.iter().map(|row| row[col]).collect()
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &Self) -> Option<Self> {
        if self.cols != other.rows {
            return None;
        }
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = Rational64::from_integer(0);
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        Some(result)
    }

    /// Matrix-vector multiplication.
    pub fn mul_vec(&self, vec: &[Rational64]) -> Option<Vec<Rational64>> {
        if self.cols != vec.len() {
            return None;
        }
        let mut result = vec![Rational64::from_integer(0); self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.data[i][j] * vec[j];
            }
        }
        Some(result)
    }

    /// Compute the determinant (for square matrices).
    pub fn determinant(&self) -> Option<Rational64> {
        if self.rows != self.cols {
            return None;
        }
        if self.rows == 0 {
            return Some(Rational64::from_integer(1));
        }
        if self.rows == 1 {
            return Some(self.data[0][0]);
        }
        if self.rows == 2 {
            return Some(
                self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
            );
        }
        
        // LU decomposition for larger matrices
        let mut det = Rational64::from_integer(1);
        let mut lu = self.clone();
        
        for k in 0..self.rows {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..self.rows {
                if lu.data[i][k].abs() > lu.data[max_row][k].abs() {
                    max_row = i;
                }
            }
            
            if max_row != k {
                lu.data.swap(k, max_row);
                det = -det;
            }
            
            if lu.data[k][k] == Rational64::from_integer(0) {
                return Some(Rational64::from_integer(0));
            }
            
            det *= lu.data[k][k];
            
            for i in (k + 1)..self.rows {
                let factor = lu.data[i][k] / lu.data[k][k];
                let lu_row_k: Vec<_> = lu.data[k].clone();
                for j in k..self.rows {
                    lu.data[i][j] -= factor * lu_row_k[j];
                }
            }
        }
        
        Some(det)
    }

    /// Check if the matrix is unimodular (determinant is ±1).
    pub fn is_unimodular(&self) -> bool {
        if let Some(det) = self.determinant() {
            det.abs() == Rational64::from_integer(1)
        } else {
            false
        }
    }

    /// Compute the inverse (for square matrices with non-zero determinant).
    pub fn inverse(&self) -> Option<Self> {
        if self.rows != self.cols {
            return None;
        }
        let n = self.rows;
        
        // Augmented matrix [A | I]
        let mut aug = Self::zeros(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.data[i][j] = self.data[i][j];
            }
            aug.data[i][n + i] = Rational64::from_integer(1);
        }
        
        // Gauss-Jordan elimination
        for k in 0..n {
            // Find pivot
            let mut max_row = k;
            for i in (k + 1)..n {
                if aug.data[i][k].abs() > aug.data[max_row][k].abs() {
                    max_row = i;
                }
            }
            aug.data.swap(k, max_row);
            
            if aug.data[k][k] == Rational64::from_integer(0) {
                return None; // Singular matrix
            }
            
            // Scale pivot row
            let pivot = aug.data[k][k];
            for j in 0..(2 * n) {
                aug.data[k][j] /= pivot;
            }
            
            // Eliminate column
            for i in 0..n {
                if i != k {
                    let factor = aug.data[i][k];
                    let aug_row_k: Vec<_> = aug.data[k].clone();
                    for j in 0..(2 * n) {
                        aug.data[i][j] -= factor * aug_row_k[j];
                    }
                }
            }
        }
        
        // Extract inverse
        let mut inv = Self::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.data[i][j] = aug.data[i][n + j];
            }
        }
        
        Some(inv)
    }

    /// Compute the Hermite Normal Form (HNF).
    /// Returns (H, U) where H = U * A and H is in HNF.
    pub fn hermite_normal_form(&self) -> (Self, Self) {
        let m = self.rows;
        let n = self.cols;
        let mut h = self.clone();
        let mut u = Self::identity(m);
        
        let mut pivot_col = 0;
        for row in 0..m {
            if pivot_col >= n {
                break;
            }
            
            // Find non-zero entry in this column
            let mut found = false;
            for k in row..m {
                if h.data[k][pivot_col] != Rational64::from_integer(0) {
                    // Swap rows
                    h.data.swap(row, k);
                    u.data.swap(row, k);
                    found = true;
                    break;
                }
            }
            
            if !found {
                pivot_col += 1;
                continue;
            }
            
            // Make pivot positive
            if h.data[row][pivot_col] < Rational64::from_integer(0) {
                for j in 0..n {
                    h.data[row][j] = -h.data[row][j];
                }
                for j in 0..m {
                    u.data[row][j] = -u.data[row][j];
                }
            }
            
            // Eliminate below
            for k in (row + 1)..m {
                if h.data[k][pivot_col] != Rational64::from_integer(0) {
                    let q = (h.data[k][pivot_col] / h.data[row][pivot_col]).floor();
                    let h_row: Vec<_> = h.data[row].clone();
                    for j in 0..n {
                        h.data[k][j] -= q * h_row[j];
                    }
                    let u_row: Vec<_> = u.data[row].clone();
                    for j in 0..m {
                        u.data[k][j] -= q * u_row[j];
                    }
                }
            }
            
            pivot_col += 1;
        }
        
        (h, u)
    }

    /// Convert to nalgebra DMatrix<f64> for numerical operations.
    pub fn to_nalgebra(&self) -> DMatrix<f64> {
        let data: Vec<f64> = self.data
            .iter()
            .flat_map(|row| row.iter().map(|r| (*r.numer() as f64) / (*r.denom() as f64)))
            .collect();
        DMatrix::from_row_slice(self.rows, self.cols, &data)
    }

    /// Create from nalgebra DMatrix, rounding to nearest integer.
    pub fn from_nalgebra_round(mat: &DMatrix<f64>) -> Self {
        let mut result = Self::zeros(mat.nrows(), mat.ncols());
        for i in 0..mat.nrows() {
            for j in 0..mat.ncols() {
                result.data[i][j] = Rational64::from_integer(mat[(i, j)].round() as i64);
            }
        }
        result
    }

    /// Check if all entries are integers.
    pub fn is_integer(&self) -> bool {
        self.data.iter().all(|row| {
            row.iter().all(|r| r.is_integer())
        })
    }

    /// Convert to integer matrix if possible.
    pub fn to_integer_matrix(&self) -> Option<Vec<Vec<i64>>> {
        if !self.is_integer() {
            return None;
        }
        Some(
            self.data
                .iter()
                .map(|row| row.iter().map(|r| *r.numer()).collect())
                .collect()
        )
    }
}

impl fmt::Display for RationalMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for row in &self.data {
            write!(f, "  [")?;
            for (j, val) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                if val.is_integer() {
                    write!(f, "{}", val.numer())?;
                } else {
                    write!(f, "{}/{}", val.numer(), val.denom())?;
                }
            }
            writeln!(f, "]")?;
        }
        write!(f, "]")
    }
}

/// Compute the GCD of a vector of integers.
pub fn vector_gcd(v: &[i64]) -> i64 {
    v.iter().fold(0, |acc, &x| acc.gcd(&x))
}

/// Compute the LCM of a vector of integers.
pub fn vector_lcm(v: &[i64]) -> i64 {
    v.iter().fold(1, |acc, &x| acc.lcm(&x))
}

/// Extended Euclidean algorithm: returns (gcd, x, y) such that ax + by = gcd.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a.abs(), a.signum(), 0)
    } else {
        let (g, x, y) = extended_gcd(b, a % b);
        (g, y, x - (a / b) * y)
    }
}

/// Solve a system of linear Diophantine equations Ax = b.
/// Returns None if no integer solution exists.
pub fn solve_diophantine(a: &RationalMatrix, b: &[i64]) -> Option<Vec<i64>> {
    // Convert to integer matrix
    let a_int = a.to_integer_matrix()?;
    
    // Use Hermite Normal Form for solving
    let (h, u) = a.hermite_normal_form();
    
    // Transform b: b' = U * b
    let b_rational: Vec<Rational64> = b.iter().map(|&x| Rational64::from_integer(x)).collect();
    let b_prime = u.mul_vec(&b_rational)?;
    
    // Back-substitute to find solution
    let n = a.ncols();
    let mut x = vec![0i64; n];
    
    for i in (0..a.nrows()).rev() {
        let mut sum = b_prime[i];
        for j in (i + 1)..n.min(a.nrows()) {
            if let Some(&hij) = h.get(i, j) {
                sum -= hij * Rational64::from_integer(x[j]);
            }
        }
        
        if let Some(&hii) = h.get(i, i) {
            if hii == Rational64::from_integer(0) {
                if sum != Rational64::from_integer(0) {
                    return None; // No solution
                }
                // Free variable, set to 0
            } else {
                let xi = sum / hii;
                if !xi.is_integer() {
                    return None; // No integer solution
                }
                x[i] = *xi.numer();
            }
        }
    }
    
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = RationalMatrix::from_vec(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = RationalMatrix::from_vec(vec![
            vec![1, 2],
            vec![3, 4],
        ]);
        let b = RationalMatrix::from_vec(vec![
            vec![5, 6],
            vec![7, 8],
        ]);
        let c = a.mul(&b).unwrap();
        assert_eq!(c.get(0, 0).map(|r| *r.numer()), Some(19));
        assert_eq!(c.get(0, 1).map(|r| *r.numer()), Some(22));
        assert_eq!(c.get(1, 0).map(|r| *r.numer()), Some(43));
        assert_eq!(c.get(1, 1).map(|r| *r.numer()), Some(50));
    }

    #[test]
    fn test_determinant() {
        let m = RationalMatrix::from_vec(vec![
            vec![1, 2],
            vec![3, 4],
        ]);
        let det = m.determinant().unwrap();
        assert_eq!(*det.numer(), -2);
    }

    #[test]
    fn test_inverse() {
        let m = RationalMatrix::from_vec(vec![
            vec![1, 2],
            vec![3, 4],
        ]);
        let inv = m.inverse().unwrap();
        let identity = m.mul(&inv).unwrap();
        
        // Should be close to identity
        assert_eq!(identity.get(0, 0).map(|r| *r.numer()), Some(1));
        assert_eq!(identity.get(1, 1).map(|r| *r.numer()), Some(1));
    }

    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(12, 8);
        assert_eq!(g, 4);
        assert_eq!(12 * x + 8 * y, 4);
    }

    #[test]
    fn test_unimodular() {
        let m = RationalMatrix::from_vec(vec![
            vec![1, 1],
            vec![0, 1],
        ]);
        assert!(m.is_unimodular());
    }
}
