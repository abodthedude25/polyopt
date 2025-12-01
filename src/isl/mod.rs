//! ISL (Integer Set Library) Integration
//!
//! This module provides integration with the ISL library for exact polyhedral
//! computations. ISL is the industry-standard library used by LLVM's Polly,
//! GCC's Graphite, and other production polyhedral compilers.
//!
//! # Requirements
//!
//! This module requires `iscc` (ISL calculator) to be installed:
//! - macOS: `brew install isl`
//! - Ubuntu/Debian: `apt install libisl-dev isl-utils`
//! - Fedora: `dnf install isl-devel`
//!
//! # Features
//!
//! - Exact dependence analysis using ISL's dependence computation
//! - Optimal schedule computation using ISL's Pluto-like scheduler
//! - Polyhedral set operations (union, intersection, projection)
//! - Code generation bounds computation
//!
//! # Example
//!
//! ```ignore
//! use polyopt::isl::{IslContext, IslSet, IslMap};
//!
//! let ctx = IslContext::new()?;
//! let domain = ctx.parse_set("{ [i,j] : 0 <= i < N and 0 <= j < M }")?;
//! let schedule = ctx.compute_schedule(&domain, &deps)?;
//! ```

mod context;
mod set;
mod map;
mod schedule;
mod codegen;
pub mod simulation;

pub use context::IslContext;
pub use set::IslSet;
pub use map::IslMap;
pub use schedule::{IslSchedule, ScheduleOptions};
pub use codegen::IslCodegen;

use std::process::Command;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum IslError {
    #[error("ISL not found. Install with: brew install isl (macOS) or apt install isl-utils (Linux)")]
    IslNotFound,
    
    #[error("ISL command failed: {0}")]
    CommandFailed(String),
    
    #[error("Failed to parse ISL output: {0}")]
    ParseError(String),
    
    #[error("Invalid ISL expression: {0}")]
    InvalidExpression(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type IslResult<T> = Result<T, IslError>;

/// Check if ISL (iscc) is available on the system
pub fn is_isl_available() -> bool {
    Command::new("iscc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Get ISL version string
pub fn isl_version() -> Option<String> {
    Command::new("iscc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

/// Run an ISL expression through iscc and return the result
pub fn run_isl(expr: &str) -> IslResult<String> {
    if !is_isl_available() {
        return Err(IslError::IslNotFound);
    }
    
    let output = Command::new("iscc")
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?
        .wait_with_output()?;
    
    // Actually pipe the input
    let mut child = Command::new("iscc")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    
    use std::io::Write;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(expr.as_bytes())?;
    }
    
    let output = child.wait_with_output()?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(IslError::CommandFailed(stderr.to_string()));
    }
    
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_isl_availability() {
        // Just check - don't fail if not installed
        let available = is_isl_available();
        println!("ISL available: {}", available);
        if available {
            println!("ISL version: {:?}", isl_version());
        }
    }
}
