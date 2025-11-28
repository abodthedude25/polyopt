//! Source location tracking for error reporting.
//!
//! This module provides types for tracking source locations and spans,
//! which are used for error reporting and source mapping.

use std::fmt;
use serde::{Serialize, Deserialize};

/// A position in source code (line and column).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct SourceLocation {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
    /// Byte offset from start of file
    pub offset: usize,
}

impl SourceLocation {
    /// Create a new source location.
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self { line, column, offset }
    }

    /// Create a location at the start of a file.
    pub fn start() -> Self {
        Self { line: 1, column: 1, offset: 0 }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// A span in source code (start and end positions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Span {
    /// Start line (1-indexed)
    pub start_line: usize,
    /// Start column (1-indexed)
    pub start_column: usize,
    /// End line (1-indexed)
    pub end_line: usize,
    /// End column (1-indexed)
    pub end_column: usize,
    /// Byte offset of start
    pub start_offset: usize,
    /// Byte offset of end
    pub end_offset: usize,
}

impl Span {
    /// Create a new span.
    pub fn new(start_line: usize, start_column: usize, end_line: usize, end_column: usize) -> Self {
        Self {
            start_line,
            start_column,
            end_line,
            end_column,
            start_offset: 0,
            end_offset: 0,
        }
    }

    /// Create a span from start and end locations.
    pub fn from_locations(start: SourceLocation, end: SourceLocation) -> Self {
        Self {
            start_line: start.line,
            start_column: start.column,
            end_line: end.line,
            end_column: end.column,
            start_offset: start.offset,
            end_offset: end.offset,
        }
    }

    /// Create a span with byte offsets.
    pub fn with_offsets(mut self, start: usize, end: usize) -> Self {
        self.start_offset = start;
        self.end_offset = end;
        self
    }

    /// Create a dummy span (for generated code).
    pub fn dummy() -> Self {
        Self::default()
    }

    /// Check if this span is a dummy span.
    pub fn is_dummy(&self) -> bool {
        self.start_line == 0 && self.end_line == 0
    }

    /// Get the start location.
    pub fn start(&self) -> SourceLocation {
        SourceLocation {
            line: self.start_line,
            column: self.start_column,
            offset: self.start_offset,
        }
    }

    /// Get the end location.
    pub fn end(&self) -> SourceLocation {
        SourceLocation {
            line: self.end_line,
            column: self.end_column,
            offset: self.end_offset,
        }
    }

    /// Merge two spans to create a span covering both.
    pub fn merge(&self, other: &Span) -> Span {
        let start = if (self.start_line, self.start_column) <= (other.start_line, other.start_column) {
            self.start()
        } else {
            other.start()
        };
        let end = if (self.end_line, self.end_column) >= (other.end_line, other.end_column) {
            self.end()
        } else {
            other.end()
        };
        Span::from_locations(start, end)
    }

    /// Check if this span contains a location.
    pub fn contains(&self, loc: &SourceLocation) -> bool {
        if loc.line < self.start_line || loc.line > self.end_line {
            return false;
        }
        if loc.line == self.start_line && loc.column < self.start_column {
            return false;
        }
        if loc.line == self.end_line && loc.column > self.end_column {
            return false;
        }
        true
    }

    /// Get the length of this span in bytes.
    pub fn len(&self) -> usize {
        self.end_offset.saturating_sub(self.start_offset)
    }

    /// Check if span is empty.
    pub fn is_empty(&self) -> bool {
        self.start_offset == self.end_offset
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start_line == self.end_line {
            write!(f, "{}:{}-{}", self.start_line, self.start_column, self.end_column)
        } else {
            write!(
                f,
                "{}:{}-{}:{}",
                self.start_line, self.start_column, self.end_line, self.end_column
            )
        }
    }
}

/// A value with an associated source span.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Spanned<T> {
    /// The value
    pub value: T,
    /// The source span
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Create a new spanned value.
    pub fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }

    /// Map the inner value.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Spanned<U> {
        Spanned {
            value: f(self.value),
            span: self.span,
        }
    }

    /// Get a reference to the inner value.
    pub fn as_ref(&self) -> Spanned<&T> {
        Spanned {
            value: &self.value,
            span: self.span,
        }
    }
}

impl<T> std::ops::Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// Helper to create spans from a source string.
#[derive(Debug, Clone)]
pub struct SourceMap {
    /// The source text
    source: String,
    /// Line start offsets
    line_starts: Vec<usize>,
}

impl SourceMap {
    /// Create a new source map.
    pub fn new(source: String) -> Self {
        let mut line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self { source, line_starts }
    }

    /// Get the source text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Convert a byte offset to a source location.
    pub fn offset_to_location(&self, offset: usize) -> SourceLocation {
        let line = match self.line_starts.binary_search(&offset) {
            Ok(line) => line,
            Err(line) => line.saturating_sub(1),
        };
        let column = offset - self.line_starts[line] + 1;
        SourceLocation::new(line + 1, column, offset)
    }

    /// Get the text for a span.
    pub fn span_text(&self, span: &Span) -> &str {
        &self.source[span.start_offset..span.end_offset]
    }

    /// Get a line of source code.
    pub fn line(&self, line_number: usize) -> Option<&str> {
        if line_number == 0 || line_number > self.line_starts.len() {
            return None;
        }
        let start = self.line_starts[line_number - 1];
        let end = self.line_starts
            .get(line_number)
            .copied()
            .unwrap_or(self.source.len());
        Some(self.source[start..end].trim_end_matches('\n'))
    }

    /// Get the number of lines.
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_display() {
        let span = Span::new(1, 5, 1, 10);
        assert_eq!(format!("{}", span), "1:5-10");

        let span = Span::new(1, 5, 3, 10);
        assert_eq!(format!("{}", span), "1:5-3:10");
    }

    #[test]
    fn test_span_merge() {
        let span1 = Span::new(1, 1, 1, 5);
        let span2 = Span::new(1, 10, 1, 15);
        let merged = span1.merge(&span2);
        assert_eq!(merged.start_column, 1);
        assert_eq!(merged.end_column, 15);
    }

    #[test]
    fn test_source_map() {
        let source = "line1\nline2\nline3".to_string();
        let map = SourceMap::new(source);
        
        assert_eq!(map.line_count(), 3);
        assert_eq!(map.line(1), Some("line1"));
        assert_eq!(map.line(2), Some("line2"));
        assert_eq!(map.line(3), Some("line3"));

        let loc = map.offset_to_location(7); // 'i' in line2
        assert_eq!(loc.line, 2);
        assert_eq!(loc.column, 2);
    }
}
