//! Utility modules for the polyhedral optimizer.
//!
//! This module contains common utilities used throughout the codebase:
//! - Error types
//! - Matrix operations
//! - Source location tracking
//! - Symbol interning
//! - Visualization helpers

pub mod errors;
pub mod matrix;
pub mod location;
pub mod intern;
pub mod pretty;

#[cfg(feature = "visualization")]
pub mod visualizer;

// Re-exports
pub use errors::*;
pub use location::{SourceLocation, Span};
pub use intern::{Symbol, SymbolInterner};
