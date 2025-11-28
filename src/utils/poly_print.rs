//! Domain printing and visualization utilities.
//!
//! This module provides tools to visualize polyhedral domains
//! and other polyhedral objects in a human-readable format.

use crate::polyhedral::set::IntegerSet;
use crate::polyhedral::map::AffineMap;
use crate::polyhedral::expr::AffineExpr;
use crate::ir::pir::{PolyProgram, PolyStmt};
use std::fmt::Write;

/// Pretty printer for polyhedral objects.
pub struct PolyPrinter {
    /// Indentation level
    indent: usize,
    /// Output buffer
    buffer: String,
}

impl PolyPrinter {
    /// Create a new printer.
    pub fn new() -> Self {
        Self {
            indent: 0,
            buffer: String::new(),
        }
    }

    /// Get the output.
    pub fn output(&self) -> &str {
        &self.buffer
    }

    /// Take the output.
    pub fn take_output(self) -> String {
        self.buffer
    }

    /// Add indentation.
    fn write_indent(&mut self) {
        for _ in 0..self.indent {
            self.buffer.push_str("  ");
        }
    }

    /// Print an integer set (domain).
    pub fn print_set(&mut self, set: &IntegerSet) {
        let dim_names = set.dim_names();
        let param_names = set.param_names();
        
        // Print in ISL-like notation: { [i, j] : constraints }
        self.buffer.push_str("{ [");
        self.buffer.push_str(&dim_names.join(", "));
        self.buffer.push_str("]");
        
        if !param_names.is_empty() {
            self.buffer.push_str(" : ");
            
            let mut constraints = Vec::new();
            for c in &set.constraints.constraints {
                constraints.push(c.to_string_with_names(&dim_names, &param_names));
            }
            
            if constraints.is_empty() {
                self.buffer.push_str("true");
            } else {
                self.buffer.push_str(&constraints.join(" and "));
            }
        }
        
        self.buffer.push_str(" }");
    }

    /// Print an affine map (schedule or access).
    pub fn print_map(&mut self, map: &AffineMap) {
        let in_dim = map.n_in();
        let out_dim = map.n_out();
        
        let in_names: Vec<String> = (0..in_dim).map(|i| format!("i{}", i)).collect();
        let out_names: Vec<String> = (0..out_dim).map(|i| format!("o{}", i)).collect();
        let param_names = map.space.all_param_names();
        
        // Print in ISL-like notation: { [i0, i1] -> [o0, o1] }
        self.buffer.push_str("{ [");
        self.buffer.push_str(&in_names.join(", "));
        self.buffer.push_str("] -> [");
        
        let mut out_exprs = Vec::new();
        for output in &map.outputs {
            out_exprs.push(output.to_string_with_names(&in_names, &param_names));
        }
        self.buffer.push_str(&out_exprs.join(", "));
        
        self.buffer.push_str("] }");
    }

    /// Print a polyhedral statement.
    pub fn print_stmt(&mut self, stmt: &PolyStmt) {
        self.write_indent();
        writeln!(self.buffer, "Statement {}:", stmt.name).unwrap();
        self.indent += 1;
        
        self.write_indent();
        write!(self.buffer, "Domain: ").unwrap();
        self.print_set(&stmt.domain);
        self.buffer.push('\n');
        
        self.write_indent();
        write!(self.buffer, "Schedule: ").unwrap();
        self.print_map(&stmt.schedule);
        self.buffer.push('\n');
        
        if !stmt.reads.is_empty() {
            self.write_indent();
            self.buffer.push_str("Reads:\n");
            self.indent += 1;
            for read in &stmt.reads {
                self.write_indent();
                write!(self.buffer, "{}:", read.array).unwrap();
                self.print_map(&read.relation);
                self.buffer.push('\n');
            }
            self.indent -= 1;
        }
        
        if !stmt.writes.is_empty() {
            self.write_indent();
            self.buffer.push_str("Writes:\n");
            self.indent += 1;
            for write in &stmt.writes {
                self.write_indent();
                write!(self.buffer, "{}:", write.array).unwrap();
                self.print_map(&write.relation);
                self.buffer.push('\n');
            }
            self.indent -= 1;
        }
        
        self.indent -= 1;
    }

    /// Print a polyhedral program.
    pub fn print_program(&mut self, program: &PolyProgram) {
        writeln!(self.buffer, "Polyhedral Program: {}", program.name).unwrap();
        writeln!(self.buffer, "Parameters: [{}]", program.parameters.join(", ")).unwrap();
        writeln!(self.buffer, "Arrays: {}", program.arrays.len()).unwrap();
        
        for arr in &program.arrays {
            writeln!(self.buffer, "  {} ({}D): [{}]", arr.name, arr.ndims, arr.sizes.join(", ")).unwrap();
        }
        
        self.buffer.push('\n');
        writeln!(self.buffer, "Context:").unwrap();
        self.indent += 1;
        self.write_indent();
        self.print_set(&program.context);
        self.buffer.push('\n');
        self.indent -= 1;
        
        self.buffer.push('\n');
        writeln!(self.buffer, "Statements ({}):", program.statements.len()).unwrap();
        for stmt in &program.statements {
            self.print_stmt(stmt);
            self.buffer.push('\n');
        }
    }
}

impl Default for PolyPrinter {
    fn default() -> Self {
        Self::new()
    }
}

/// Print a domain to a string.
pub fn print_domain(set: &IntegerSet) -> String {
    let mut printer = PolyPrinter::new();
    printer.print_set(set);
    printer.take_output()
}

/// Print a map to a string.
pub fn print_map(map: &AffineMap) -> String {
    let mut printer = PolyPrinter::new();
    printer.print_map(map);
    printer.take_output()
}

/// Print a polyhedral program to a string.
pub fn print_program(program: &PolyProgram) -> String {
    let mut printer = PolyPrinter::new();
    printer.print_program(program);
    printer.take_output()
}

/// Enumerate points in a domain (for small domains only).
pub fn enumerate_points(set: &IntegerSet, params: &[i64], max_points: usize) -> Vec<Vec<i64>> {
    let n_dim = set.dim();
    if n_dim == 0 {
        return vec![vec![]];
    }
    
    let mut points = Vec::new();
    
    // Simple brute-force enumeration for small domains
    // In practice, you'd use a smarter algorithm
    let bounds: Vec<(i64, i64)> = (0..n_dim)
        .map(|_| (-100, 100)) // Default bounds
        .collect();
    
    enumerate_recursive(set, params, &bounds, &mut vec![0; n_dim], 0, &mut points, max_points);
    
    points
}

fn enumerate_recursive(
    set: &IntegerSet,
    params: &[i64],
    bounds: &[(i64, i64)],
    current: &mut Vec<i64>,
    dim: usize,
    points: &mut Vec<Vec<i64>>,
    max_points: usize,
) {
    if points.len() >= max_points {
        return;
    }
    
    if dim == current.len() {
        if set.contains(current, params) {
            points.push(current.clone());
        }
        return;
    }
    
    let (lo, hi) = bounds[dim];
    for val in lo..=hi {
        current[dim] = val;
        enumerate_recursive(set, params, bounds, current, dim + 1, points, max_points);
        if points.len() >= max_points {
            return;
        }
    }
}

/// Generate a 2D ASCII visualization of a domain (first 2 dimensions).
pub fn visualize_2d(set: &IntegerSet, params: &[i64], x_range: (i64, i64), y_range: (i64, i64)) -> String {
    if set.dim() < 2 {
        return "Domain has less than 2 dimensions".to_string();
    }
    
    let mut output = String::new();
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;
    
    // Header
    output.push_str("    ");
    for x in x_min..=x_max {
        write!(output, "{:3}", x).unwrap();
    }
    output.push('\n');
    
    // Grid
    for y in (y_min..=y_max).rev() {
        write!(output, "{:3} ", y).unwrap();
        for x in x_min..=x_max {
            let mut point = vec![x, y];
            // Pad with zeros for higher dimensions
            while point.len() < set.dim() {
                point.push(0);
            }
            
            if set.contains(&point, params) {
                output.push_str(" * ");
            } else {
                output.push_str(" . ");
            }
        }
        output.push('\n');
    }
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedral::Space;
    use crate::polyhedral::constraint::{Constraint, ConstraintSystem};

    #[test]
    fn test_print_domain() {
        // Create domain: { [i] : 0 <= i < N }
        let space = Space::set_with_params(1, 1)
            .with_dim_names(vec!["i".to_string()])
            .with_param_names(vec!["N".to_string()]);
        
        let mut constraints = ConstraintSystem::new(1, 1);
        // i >= 0
        constraints.add(Constraint::ge_zero(AffineExpr::var(0, 1, 1)));
        // N - 1 - i >= 0  (i < N)
        let mut upper = AffineExpr::param(0, 1, 1);
        upper.constant = -1;
        upper = upper - AffineExpr::var(0, 1, 1);
        constraints.add(Constraint::ge_zero(upper));
        
        let set = IntegerSet { space, constraints };
        
        let output = print_domain(&set);
        assert!(output.contains("[i]"));
    }

    #[test]
    fn test_enumerate_points() {
        // Create domain: { [i] : 0 <= i < 5 }
        let space = Space::set_with_params(1, 1)
            .with_dim_names(vec!["i".to_string()])
            .with_param_names(vec!["N".to_string()]);
        
        let mut constraints = ConstraintSystem::new(1, 1);
        constraints.add(Constraint::ge_zero(AffineExpr::var(0, 1, 1)));
        let mut upper = AffineExpr::param(0, 1, 1);
        upper.constant = -1;
        upper = upper - AffineExpr::var(0, 1, 1);
        constraints.add(Constraint::ge_zero(upper));
        
        let set = IntegerSet { space, constraints };
        
        let points = enumerate_points(&set, &[5], 100);
        assert_eq!(points.len(), 5);
        assert!(points.contains(&vec![0]));
        assert!(points.contains(&vec![4]));
        assert!(!points.contains(&vec![5]));
    }

    #[test]
    fn test_visualize_2d() {
        // Create domain: { [i, j] : 0 <= i < 5 and 0 <= j < 5 }
        let set = IntegerSet::rectangular(&[5, 5]);
        
        let output = visualize_2d(&set, &[], (0, 5), (0, 5));
        assert!(output.contains("*"));
        assert!(output.contains("."));
    }
}
