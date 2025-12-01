//! ISL Code Generation - generate loop bounds from polyhedral domains

use super::{IslResult, run_isl};
use super::set::IslSet;
use super::schedule::IslSchedule;

/// ISL code generator
/// 
/// Generates scanning code (loop nests) from polyhedral domains and schedules.
pub struct IslCodegen {
    /// Target language
    language: CodegenLanguage,
}

/// Target language for code generation
#[derive(Clone, Copy, Debug, Default)]
pub enum CodegenLanguage {
    #[default]
    C,
    OpenMP,
    CUDA,
}

/// Generated code structure
#[derive(Clone, Debug)]
pub struct GeneratedCode {
    /// Loop structure
    pub loops: Vec<GeneratedLoop>,
    /// Statements in execution order
    pub statements: Vec<String>,
}

/// A generated loop
#[derive(Clone, Debug)]
pub struct GeneratedLoop {
    /// Loop variable name
    pub var: String,
    /// Lower bound expression
    pub lower: String,
    /// Upper bound expression  
    pub upper: String,
    /// Stride (usually 1)
    pub stride: i64,
    /// Whether this loop is parallel
    pub parallel: bool,
    /// Nested loops
    pub body: Vec<GeneratedLoop>,
}

impl IslCodegen {
    /// Create a new code generator
    pub fn new(language: CodegenLanguage) -> Self {
        Self { language }
    }
    
    /// Generate loop bounds for a domain
    /// 
    /// Given a polyhedral domain, generates the loop structure
    /// needed to iterate over all points.
    pub fn generate_loops(&self, domain: &IslSet) -> IslResult<String> {
        // Use ISL's code generation
        let script = format!(
            r#"
            D := {};
            codegen D;
            "#,
            domain.expr()
        );
        
        run_isl(&script)
    }
    
    /// Generate code with a schedule
    /// 
    /// Applies the schedule transformation and generates code.
    pub fn generate_scheduled(
        &self,
        domain: &IslSet,
        schedule: &IslSchedule,
    ) -> IslResult<String> {
        let script = format!(
            r#"
            D := {};
            S := {};
            codegen S on D;
            "#,
            domain.expr(),
            schedule.raw()
        );
        
        run_isl(&script)
    }
    
    /// Generate tiled loop code
    pub fn generate_tiled(
        &self,
        domain: &IslSet,
        tile_sizes: &[usize],
    ) -> IslResult<String> {
        if tile_sizes.is_empty() {
            return self.generate_loops(domain);
        }
        
        let sizes_str = tile_sizes.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        
        let script = format!(
            r#"
            D := {};
            T := tile D with sizes [{}];
            codegen T;
            "#,
            domain.expr(),
            sizes_str
        );
        
        run_isl(&script)
    }
    
    /// Generate parallel loop annotations
    fn parallel_annotation(&self) -> &str {
        match self.language {
            CodegenLanguage::C => "",
            CodegenLanguage::OpenMP => "#pragma omp parallel for",
            CodegenLanguage::CUDA => "// CUDA kernel launch",
        }
    }
}

/// Compute loop bounds for scanning a domain
/// 
/// Given constraints, computes the tightest bounds for each variable.
pub fn compute_bounds(domain: &IslSet) -> IslResult<Vec<(String, String, String)>> {
    // For each dimension, compute min and max
    let dim = domain.dim()?;
    let mut bounds = Vec::new();
    
    for i in 0..dim {
        let var = format!("i{}", i);
        
        // Compute lexmin/lexmax projected to this dimension
        let min_expr = format!("lexmin({})", domain.expr());
        let max_expr = format!("lexmax({})", domain.expr());
        
        let min_result = run_isl(&format!("{};", min_expr))?;
        let max_result = run_isl(&format!("{};", max_expr))?;
        
        bounds.push((var, min_result, max_result));
    }
    
    Ok(bounds)
}

/// Generate C-style loop code for a rectangular domain
pub fn generate_rectangular_loops(
    vars: &[&str],
    lower_bounds: &[&str],
    upper_bounds: &[&str],
    body: &str,
    parallel_dims: &[usize],
) -> String {
    let mut code = String::new();
    let indent = "    ";
    
    for (i, var) in vars.iter().enumerate() {
        let prefix = indent.repeat(i);
        
        if parallel_dims.contains(&i) {
            code.push_str(&format!("{}#pragma omp parallel for\n", prefix));
        }
        
        code.push_str(&format!(
            "{}for (int {} = {}; {} < {}; {}++) {{\n",
            prefix, var, lower_bounds[i], var, upper_bounds[i], var
        ));
    }
    
    // Body
    let body_indent = indent.repeat(vars.len());
    code.push_str(&format!("{}{}\n", body_indent, body));
    
    // Close loops
    for i in (0..vars.len()).rev() {
        let prefix = indent.repeat(i);
        code.push_str(&format!("{}}}\n", prefix));
    }
    
    code
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rectangular_codegen() {
        let code = generate_rectangular_loops(
            &["i", "j"],
            &["0", "0"],
            &["N", "M"],
            "A[i][j] = B[i][j] + C[i][j];",
            &[0], // Outer loop parallel
        );
        
        assert!(code.contains("#pragma omp parallel for"));
        assert!(code.contains("for (int i = 0; i < N; i++)"));
        assert!(code.contains("for (int j = 0; j < M; j++)"));
    }
}
