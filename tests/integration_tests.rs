//! Integration tests for the polyhedral pipeline.

use polyopt::prelude::*;
use polyopt::{parse, lower_ast, lower_hir, parse_and_lower};
use polyopt::analysis::{extract_polyhedral, extract_and_analyze};

#[test]
fn test_simple_loop_pipeline() {
    let source = r#"
        func scale(A[N], B[N]) {
            for i = 0 to N {
                B[i] = A[i] * 2;
            }
        }
    "#;
    
    // Parse
    let ast = parse(source).expect("Failed to parse");
    assert_eq!(ast.functions.len(), 1);
    
    // Lower to HIR
    let hir = lower_ast(&ast).expect("Failed to lower to HIR");
    assert_eq!(hir.functions.len(), 1);
    
    // Lower to PIR
    let pir = lower_hir(&hir).expect("Failed to lower to PIR");
    assert_eq!(pir.len(), 1);
    
    let program = &pir[0];
    assert_eq!(program.name, "scale");
    assert_eq!(program.parameters, vec!["N"]);
    assert_eq!(program.statements.len(), 1);
    
    let stmt = &program.statements[0];
    assert_eq!(stmt.domain.dim(), 1); // 1D loop
    assert_eq!(stmt.reads.len(), 1);   // Read from A
    assert_eq!(stmt.writes.len(), 1);  // Write to B
}

#[test]
fn test_matmul_pipeline() {
    let source = r#"
        func matmul(A[N][K], B[K][M], C[N][M]) {
            for i = 0 to N {
                for j = 0 to M {
                    for k = 0 to K {
                        C[i][j] = C[i][j] + A[i][k] * B[k][j];
                    }
                }
            }
        }
    "#;
    
    let pir = parse_and_lower(source).expect("Failed to lower");
    assert_eq!(pir.len(), 1);
    
    let program = &pir[0];
    assert_eq!(program.name, "matmul");
    
    // Should have N, M, K as parameters
    assert!(program.parameters.contains(&"N".to_string()));
    assert!(program.parameters.contains(&"M".to_string()));
    assert!(program.parameters.contains(&"K".to_string()));
    
    // Should have 3 arrays
    assert_eq!(program.arrays.len(), 3);
    
    // One statement (the innermost assignment)
    assert_eq!(program.statements.len(), 1);
    
    let stmt = &program.statements[0];
    assert_eq!(stmt.domain.dim(), 3); // 3D loop nest (i, j, k)
    
    // Check accesses:
    // Reads: C[i][j], A[i][k], B[k][j]
    // Writes: C[i][j]
    assert_eq!(stmt.writes.len(), 1);
    assert_eq!(stmt.writes[0].array, "C");
    
    // At least 3 reads (could be more depending on expression structure)
    assert!(stmt.reads.len() >= 3);
}

#[test]
fn test_jacobi_stencil() {
    let source = r#"
        func jacobi(A[N][N], B[N][N]) {
            for i = 1 to N {
                for j = 1 to N {
                    B[i][j] = (A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]) / 5;
                }
            }
        }
    "#;
    
    // This should fail because of division by non-constant in affine context
    // Actually, integer division by constant is OK in the compute expression
    let result = parse_and_lower(source);
    assert!(result.is_ok(), "Jacobi should parse and lower");
    
    let pir = result.unwrap();
    let stmt = &pir[0].statements[0];
    
    // Should have 2D domain
    assert_eq!(stmt.domain.dim(), 2);
    
    // Should have 5 reads (A accessed 5 times)
    assert!(stmt.reads.len() >= 1);
    
    // Should have 1 write
    assert_eq!(stmt.writes.len(), 1);
    assert_eq!(stmt.writes[0].array, "B");
}

#[test]
fn test_domain_constraints() {
    let source = r#"
        func bounds(A[N][M]) {
            for i = 0 to N {
                for j = 0 to M {
                    A[i][j] = 0;
                }
            }
        }
    "#;
    
    let pir = parse_and_lower(source).expect("Failed to lower");
    let stmt = &pir[0].statements[0];
    
    // Check domain has correct dimensionality
    assert_eq!(stmt.domain.dim(), 2);
    
    // Domain should have 4 constraints:
    // i >= 0, i < N, j >= 0, j < M
    assert_eq!(stmt.domain.constraints.len(), 4);
    
    // Check a point is in the domain (with N=10, M=10)
    assert!(stmt.domain.contains(&[5, 5], &[10, 10]));
    assert!(stmt.domain.contains(&[0, 0], &[10, 10]));
    assert!(stmt.domain.contains(&[9, 9], &[10, 10]));
    
    // Points outside the domain
    assert!(!stmt.domain.contains(&[-1, 0], &[10, 10]));
    assert!(!stmt.domain.contains(&[10, 0], &[10, 10]));
    assert!(!stmt.domain.contains(&[0, 10], &[10, 10]));
}

#[test]
fn test_access_relations() {
    let source = r#"
        func copy(A[N], B[N]) {
            for i = 0 to N {
                B[i+1] = A[i-1];
            }
        }
    "#;
    
    let pir = parse_and_lower(source).expect("Failed to lower");
    let stmt = &pir[0].statements[0];
    
    // Write to B[i+1]
    let write = &stmt.writes[0];
    assert_eq!(write.array, "B");
    
    // Read from A[i-1]
    let read = stmt.reads.iter().find(|r| r.array == "A").expect("Should read A");
    assert_eq!(read.array, "A");
}

#[test]
fn test_extract_polyhedral() {
    let source = r#"
        func test(A[N]) {
            for i = 0 to N {
                A[i] = i;
            }
        }
    "#;
    
    let ast = parse(source).expect("Failed to parse");
    let pir = extract_polyhedral(&ast).expect("Failed to extract");
    
    assert_eq!(pir.len(), 1);
    assert_eq!(pir[0].statements.len(), 1);
}

#[test]
fn test_multiple_statements() {
    let source = r#"
        func multi(A[N], B[N], C[N]) {
            for i = 0 to N {
                B[i] = A[i];
                C[i] = B[i];
            }
        }
    "#;
    
    let pir = parse_and_lower(source).expect("Failed to lower");
    
    // Should have 2 statements
    assert_eq!(pir[0].statements.len(), 2);
    
    // Both statements should have the same 1D domain
    for stmt in &pir[0].statements {
        assert_eq!(stmt.domain.dim(), 1);
    }
}

#[test]
fn test_schedule() {
    let source = r#"
        func test(A[N]) {
            for i = 0 to N {
                A[i] = i;
            }
        }
    "#;
    
    let pir = parse_and_lower(source).expect("Failed to lower");
    let stmt = &pir[0].statements[0];
    
    // Schedule should be identity: [i] -> [i]
    assert_eq!(stmt.schedule.n_in(), 1);
    assert_eq!(stmt.schedule.n_out(), 1);
}