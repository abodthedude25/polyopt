//! AST builder for generating loop nests from schedules.

use crate::ir::pir::{PolyProgram, PolyStmt, StmtId};
use crate::polyhedral::set::IntegerSet;
use crate::polyhedral::map::AffineMap;

/// A node in the generated AST.
#[derive(Debug, Clone)]
pub enum AstNode {
    /// A for loop
    For {
        iterator: String,
        lower: AstExpr,
        upper: AstExpr,
        step: i64,
        body: Vec<AstNode>,
        is_parallel: bool,
    },
    /// An if statement
    If {
        condition: AstExpr,
        then_body: Vec<AstNode>,
        else_body: Option<Vec<AstNode>>,
    },
    /// A statement instance
    Stmt {
        id: StmtId,
        iterators: Vec<AstExpr>,
    },
    /// A block of statements
    Block {
        statements: Vec<AstNode>,
    },
}

/// An expression in the generated AST.
#[derive(Debug, Clone)]
pub enum AstExpr {
    /// Integer constant
    Int(i64),
    /// Variable
    Var(String),
    /// Binary operation
    Binary {
        op: AstBinOp,
        left: Box<AstExpr>,
        right: Box<AstExpr>,
    },
    /// Minimum
    Min(Box<AstExpr>, Box<AstExpr>),
    /// Maximum
    Max(Box<AstExpr>, Box<AstExpr>),
    /// Floor division
    FloorDiv(Box<AstExpr>, Box<AstExpr>),
    /// Ceiling division
    CeilDiv(Box<AstExpr>, Box<AstExpr>),
}

impl AstExpr {
    pub fn int(v: i64) -> Self { Self::Int(v) }
    pub fn var(name: &str) -> Self { Self::Var(name.to_string()) }
    
    pub fn add(self, other: Self) -> Self {
        Self::Binary { op: AstBinOp::Add, left: Box::new(self), right: Box::new(other) }
    }
    
    pub fn sub(self, other: Self) -> Self {
        Self::Binary { op: AstBinOp::Sub, left: Box::new(self), right: Box::new(other) }
    }
    
    pub fn mul(self, other: Self) -> Self {
        Self::Binary { op: AstBinOp::Mul, left: Box::new(self), right: Box::new(other) }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AstBinOp {
    Add, Sub, Mul, Div, Mod,
    Lt, Le, Gt, Ge, Eq, Ne,
    And, Or,
}

/// AST builder using polyhedral scanning.
pub struct AstBuilder {
    /// Generated AST nodes
    nodes: Vec<AstNode>,
    /// Current loop iterators
    iterators: Vec<String>,
    /// Iterator counter for generating names
    iter_counter: usize,
}

impl AstBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            iterators: Vec::new(),
            iter_counter: 0,
        }
    }

    /// Build AST from a polyhedral program.
    pub fn build(&mut self, program: &PolyProgram) -> Vec<AstNode> {
        // Simplified implementation: generate one loop nest per statement
        // Real implementation needs proper schedule tree construction
        
        for stmt in &program.statements {
            let node = self.build_stmt_loop_nest(stmt, program);
            self.nodes.push(node);
        }

        std::mem::take(&mut self.nodes)
    }

    /// Build loop nest for a single statement.
    fn build_stmt_loop_nest(&mut self, stmt: &PolyStmt, program: &PolyProgram) -> AstNode {
        let depth = stmt.depth();
        let dim_names = stmt.domain.dim_names();

        // Build nested loops from outside in
        self.build_loops_recursive(stmt, program, 0, depth, &dim_names)
    }

    fn build_loops_recursive(
        &mut self,
        stmt: &PolyStmt,
        program: &PolyProgram,
        current_depth: usize,
        total_depth: usize,
        dim_names: &[String],
    ) -> AstNode {
        if current_depth >= total_depth {
            // Base case: generate statement
            let iterators = dim_names.iter()
                .map(|n| AstExpr::var(n))
                .collect();
            return AstNode::Stmt {
                id: stmt.id,
                iterators,
            };
        }

        let var_name = dim_names.get(current_depth)
            .cloned()
            .unwrap_or_else(|| format!("c{}", current_depth));

        // Get bounds from parameters (simplified)
        let lower = AstExpr::int(0);
        let upper = if current_depth < program.parameters.len() {
            AstExpr::var(&program.parameters[current_depth])
        } else {
            AstExpr::var("N")
        };

        let body = self.build_loops_recursive(stmt, program, current_depth + 1, total_depth, dim_names);

        AstNode::For {
            iterator: var_name,
            lower,
            upper,
            step: 1,
            body: vec![body],
            is_parallel: current_depth == 0, // Mark outermost as parallel candidate
        }
    }

    /// Generate a fresh iterator name.
    fn fresh_iter(&mut self) -> String {
        let name = format!("c{}", self.iter_counter);
        self.iter_counter += 1;
        name
    }
}

impl Default for AstBuilder {
    fn default() -> Self { Self::new() }
}

/// Convert AST to string (for debugging).
pub fn ast_to_string(nodes: &[AstNode], indent: usize) -> String {
    let mut result = String::new();
    let prefix = "  ".repeat(indent);

    for node in nodes {
        match node {
            AstNode::For { iterator, lower, upper, step, body, is_parallel } => {
                if *is_parallel {
                    result.push_str(&format!("{}#parallel\n", prefix));
                }
                result.push_str(&format!(
                    "{}for {} = {:?} to {:?} step {} {{\n",
                    prefix, iterator, lower, upper, step
                ));
                result.push_str(&ast_to_string(body, indent + 1));
                result.push_str(&format!("{}}}\n", prefix));
            }
            AstNode::If { condition, then_body, else_body } => {
                result.push_str(&format!("{}if {:?} {{\n", prefix, condition));
                result.push_str(&ast_to_string(then_body, indent + 1));
                result.push_str(&format!("{}}}", prefix));
                if let Some(else_b) = else_body {
                    result.push_str(" else {\n");
                    result.push_str(&ast_to_string(else_b, indent + 1));
                    result.push_str(&format!("{}}}", prefix));
                }
                result.push('\n');
            }
            AstNode::Stmt { id, iterators } => {
                result.push_str(&format!("{}S{}({:?});\n", prefix, id.0, iterators));
            }
            AstNode::Block { statements } => {
                result.push_str(&ast_to_string(statements, indent));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_expr() {
        let e = AstExpr::var("i").add(AstExpr::int(1));
        assert!(matches!(e, AstExpr::Binary { op: AstBinOp::Add, .. }));
    }
}
