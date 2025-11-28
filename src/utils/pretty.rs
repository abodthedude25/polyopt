//! Pretty printing utilities for IR and AST nodes.
//!
//! This module provides utilities for pretty-printing various
//! intermediate representations in a human-readable format.

use pretty::{DocAllocator, DocBuilder, BoxAllocator};
use std::fmt;

/// Default line width for pretty printing.
pub const DEFAULT_WIDTH: usize = 80;

/// A pretty-printable value.
pub trait PrettyPrint {
    /// Convert to a pretty document.
    fn to_doc<'a, D: DocAllocator<'a>>(&self, allocator: &'a D) -> DocBuilder<'a, D>;

    /// Pretty print to a string with the given width.
    fn pretty_print(&self, width: usize) -> String {
        let allocator = BoxAllocator;
        let doc = self.to_doc(&allocator);
        let mut output = String::new();
        doc.render_fmt(width, &mut output).unwrap();
        output
    }

    /// Pretty print with default width.
    fn pretty(&self) -> String {
        self.pretty_print(DEFAULT_WIDTH)
    }
}

/// Indent a block of text.
pub fn indent(s: &str, spaces: usize) -> String {
    let indent_str = " ".repeat(spaces);
    s.lines()
        .map(|line| {
            if line.is_empty() {
                line.to_string()
            } else {
                format!("{}{}", indent_str, line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// A simple code formatter for generated code.
#[derive(Debug)]
pub struct CodeFormatter {
    output: String,
    indent_level: usize,
    indent_str: String,
    at_line_start: bool,
}

impl CodeFormatter {
    /// Create a new formatter with the given indent string.
    pub fn new(indent_str: &str) -> Self {
        Self {
            output: String::new(),
            indent_level: 0,
            indent_str: indent_str.to_string(),
            at_line_start: true,
        }
    }

    /// Create a formatter with default settings (2 spaces).
    pub fn default_indent() -> Self {
        Self::new("  ")
    }

    /// Increase indentation level.
    pub fn indent(&mut self) {
        self.indent_level += 1;
    }

    /// Decrease indentation level.
    pub fn dedent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    /// Write text.
    pub fn write(&mut self, s: &str) {
        for c in s.chars() {
            if c == '\n' {
                self.output.push('\n');
                self.at_line_start = true;
            } else {
                if self.at_line_start {
                    for _ in 0..self.indent_level {
                        self.output.push_str(&self.indent_str);
                    }
                    self.at_line_start = false;
                }
                self.output.push(c);
            }
        }
    }

    /// Write a line.
    pub fn writeln(&mut self, s: &str) {
        self.write(s);
        self.write("\n");
    }

    /// Write an empty line.
    pub fn newline(&mut self) {
        self.write("\n");
    }

    /// Write a block with braces.
    pub fn block<F: FnOnce(&mut Self)>(&mut self, header: &str, f: F) {
        self.write(header);
        self.writeln(" {");
        self.indent();
        f(self);
        self.dedent();
        self.writeln("}");
    }

    /// Get the formatted output.
    pub fn finish(self) -> String {
        self.output
    }

    /// Get a reference to the current output.
    pub fn output(&self) -> &str {
        &self.output
    }
}

impl fmt::Write for CodeFormatter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write(s);
        Ok(())
    }
}

/// Format a list with separators.
pub fn format_list<T: fmt::Display>(items: &[T], sep: &str) -> String {
    items
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(sep)
}

/// Format a list with separators using a custom formatter.
pub fn format_list_with<T, F: Fn(&T) -> String>(items: &[T], sep: &str, f: F) -> String {
    items
        .iter()
        .map(f)
        .collect::<Vec<_>>()
        .join(sep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_formatter() {
        let mut fmt = CodeFormatter::default_indent();
        fmt.writeln("int main() {");
        fmt.indent();
        fmt.writeln("printf(\"Hello\");");
        fmt.writeln("return 0;");
        fmt.dedent();
        fmt.writeln("}");
        
        let output = fmt.finish();
        assert!(output.contains("  printf"));
        assert!(output.contains("  return"));
    }

    #[test]
    fn test_block() {
        let mut fmt = CodeFormatter::default_indent();
        fmt.block("for (i = 0; i < N; i++)", |f| {
            f.writeln("sum += a[i];");
        });
        
        let output = fmt.finish();
        assert!(output.contains("for (i = 0; i < N; i++) {"));
        assert!(output.contains("  sum += a[i];"));
        assert!(output.contains("}"));
    }

    #[test]
    fn test_indent_helper() {
        let text = "line1\nline2\nline3";
        let indented = indent(text, 4);
        assert!(indented.starts_with("    line1"));
    }
}
