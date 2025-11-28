//! Utility modules for the polyhedral optimizer.
//!
//! This module contains common utilities used throughout the codebase:
//! - Error types
//! - Matrix operations
//! - Source location tracking
//! - Symbol interning
//! - Visualization helpers
//! - Polyhedral printing

pub mod errors;
pub mod matrix;
pub mod location;
pub mod intern;
pub mod pretty;
pub mod poly_print;

#[cfg(feature = "visualization")]
pub mod visualizer;

// Re-exports
pub use errors::*;
pub use location::{SourceLocation, Span};
pub use intern::{Symbol, SymbolInterner};
pub use poly_print::{print_domain, print_map, print_program, PolyPrinter};
