//! Code generation from polyhedral representation.

pub mod c;
pub mod ast_builder;

pub use c::CCodeGen;
pub use ast_builder::AstBuilder;

use crate::ir::pir::PolyProgram;
use anyhow::Result;

/// Target for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// Standard C
    C,
    /// C with OpenMP pragmas
    OpenMP,
    /// CUDA
    Cuda,
    /// OpenCL
    OpenCL,
    /// LLVM IR
    LLVM,
}

/// Generate code for a target.
pub fn generate(program: &PolyProgram, target: Target) -> Result<String> {
    match target {
        Target::C | Target::OpenMP => {
            let codegen = CCodeGen::new(target == Target::OpenMP);
            codegen.generate(program)
        }
        Target::Cuda => {
            // Placeholder
            Ok("// CUDA code generation not yet implemented\n".to_string())
        }
        Target::OpenCL => {
            Ok("// OpenCL code generation not yet implemented\n".to_string())
        }
        Target::LLVM => {
            Ok("; LLVM IR code generation not yet implemented\n".to_string())
        }
    }
}
