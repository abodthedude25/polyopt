//! C code generation.

use crate::ir::pir::{PolyProgram, PolyStmt, StmtBody, ComputeExpr, AccessExpr, BinaryComputeOp};
use crate::utils::pretty::CodeFormatter;
use anyhow::Result;

/// C code generator.
pub struct CCodeGen {
    /// Enable OpenMP pragmas
    openmp: bool,
    /// Formatter for output
    formatter: CodeFormatter,
}

impl CCodeGen {
    pub fn new(openmp: bool) -> Self {
        Self {
            openmp,
            formatter: CodeFormatter::new("    "),
        }
    }

    /// Generate C code for a program.
    pub fn generate(&self, program: &PolyProgram) -> Result<String> {
        let mut f = CodeFormatter::new("    ");

        // Header
        f.writeln("#include <stdio.h>");
        f.writeln("#include <stdlib.h>");
        if self.openmp {
            f.writeln("#include <omp.h>");
        }
        f.writeln("");

        // Generate function
        self.generate_function(&mut f, program)?;

        Ok(f.finish())
    }

    fn generate_function(&self, f: &mut CodeFormatter, program: &PolyProgram) -> Result<()> {
        // Function signature
        f.write(&format!("void {}(", program.name));
        
        // Parameters
        for (i, param) in program.parameters.iter().enumerate() {
            if i > 0 { f.write(", "); }
            f.write(&format!("int {}", param));
        }
        
        // Arrays
        for (i, array) in program.arrays.iter().enumerate() {
            if !program.parameters.is_empty() || i > 0 { f.write(", "); }
            let ty = match array.element_type {
                crate::ir::pir::ElementType::Int => "int",
                crate::ir::pir::ElementType::Float => "float",
                crate::ir::pir::ElementType::Double => "double",
            };
            f.write(&format!("{}* {}", ty, array.name));
        }
        
        f.writeln(") {");
        f.indent();

        // Generate code for each statement using AST scanning
        // This is a simplified version - real implementation needs proper AST generation
        for stmt in &program.statements {
            self.generate_stmt(f, stmt, program)?;
        }

        f.dedent();
        f.writeln("}");

        Ok(())
    }

    fn generate_stmt(&self, f: &mut CodeFormatter, stmt: &PolyStmt, program: &PolyProgram) -> Result<()> {
        let depth = stmt.depth();
        let dim_names = stmt.domain.dim_names();

        // Generate nested loops based on domain
        for d in 0..depth {
            let var = dim_names.get(d).map(|s| s.as_str()).unwrap_or("i");
            
            if self.openmp && d == 0 {
                f.writeln("#pragma omp parallel for");
            }
            
            // Simplified: assume 0 to N bounds
            let bound = program.parameters.get(d).map(|s| s.as_str()).unwrap_or("N");
            f.writeln(&format!("for (int {} = 0; {} < {}; {}++) {{", var, var, bound, var));
            f.indent();
        }

        // Generate statement body
        self.generate_body(f, &stmt.body)?;

        // Close loops
        for _ in 0..depth {
            f.dedent();
            f.writeln("}");
        }

        Ok(())
    }

    fn generate_body(&self, f: &mut CodeFormatter, body: &StmtBody) -> Result<()> {
        match body {
            StmtBody::Assignment { target, expr } => {
                let target_str = self.generate_access(target);
                let expr_str = self.generate_expr(expr);
                f.writeln(&format!("{} = {};", target_str, expr_str));
            }
            StmtBody::CompoundAssign { target, op, expr } => {
                let target_str = self.generate_access(target);
                let expr_str = self.generate_expr(expr);
                let op_str = match op {
                    crate::ir::pir::CompoundOp::Add => "+=",
                    crate::ir::pir::CompoundOp::Sub => "-=",
                    crate::ir::pir::CompoundOp::Mul => "*=",
                    crate::ir::pir::CompoundOp::Div => "/=",
                };
                f.writeln(&format!("{} {} {};", target_str, op_str, expr_str));
            }
        }
        Ok(())
    }

    fn generate_access(&self, access: &AccessExpr) -> String {
        if access.indices.is_empty() {
            access.array.clone()
        } else {
            let indices: Vec<String> = access.indices.iter()
                .map(|i| format!("[{}]", i.0))
                .collect();
            format!("{}{}", access.array, indices.join(""))
        }
    }

    fn generate_expr(&self, expr: &ComputeExpr) -> String {
        match expr {
            ComputeExpr::Int(v) => v.to_string(),
            ComputeExpr::Float(v) => format!("{:.6}", v),
            ComputeExpr::Var(name) => name.clone(),
            ComputeExpr::Access(acc) => self.generate_access(acc),
            ComputeExpr::Binary { op, left, right } => {
                let l = self.generate_expr(left);
                let r = self.generate_expr(right);
                let op_str = match op {
                    BinaryComputeOp::Add => "+",
                    BinaryComputeOp::Sub => "-",
                    BinaryComputeOp::Mul => "*",
                    BinaryComputeOp::Div => "/",
                    BinaryComputeOp::Mod => "%",
                };
                format!("({} {} {})", l, op_str, r)
            }
            ComputeExpr::Unary { op, operand } => {
                let o = self.generate_expr(operand);
                match op {
                    crate::ir::pir::UnaryComputeOp::Neg => format!("(-{})", o),
                }
            }
            ComputeExpr::Call { func, args } => {
                let args_str: Vec<String> = args.iter().map(|a| self.generate_expr(a)).collect();
                format!("{}({})", func, args_str.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_new() {
        let cg = CCodeGen::new(false);
        assert!(!cg.openmp);
    }
}
