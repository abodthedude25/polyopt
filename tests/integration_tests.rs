//! Integration tests for the polyhedral pipeline.

use polyopt::prelude::*;
use polyopt::{parse, lower_ast, lower_hir, parse_and_lower};
use polyopt::analysis::{
    extract_polyhedral, extract_and_analyze,
    DependenceAnalysis, DependenceKind, Direction,
    gcd_test, banerjee_test,
};

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
    assert_eq!(stmt.reads.len(), 1);  // Read from A
    assert_eq!(stmt.writes.len(), 1); // Write to B
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
    assert!(program.parameters.contains(&"N".to_string()));
    assert!(program.parameters.contains(&"M".to_string()));
    assert!(program.parameters.contains(&"K".to_string()));
    assert_eq!(program.arrays.len(), 3);
    assert_eq!(program.statements.len(), 1);

    let stmt = &program.statements[0];
    assert_eq!(stmt.domain.dim(), 3);
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
    assert_eq!(stmt.domain.dim(), 2);
    assert!(stmt.domain.contains(&[5, 5], &[10, 10]));
}

#[test]
fn test_dependence_analysis_flow() {
    let source = r#"
        func recurrence(A[N]) {
            for i = 1 to N {
                A[i] = A[i-1] + 1;
            }
        }
    "#;

    let pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(&pir[0]).expect("Failed to analyze");

    let flow_deps: Vec<_> = deps.iter()
        .filter(|d| d.kind == DependenceKind::Flow)
        .collect();
    assert!(!flow_deps.is_empty());
}

#[test]
fn test_dependence_graph() {
    let source = r#"
        func two_stmts(A[N], B[N]) {
            for i = 0 to N {
                A[i] = i;
                B[i] = A[i];
            }
        }
    "#;

    let pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let graph = analyzer.build_graph(&pir[0]).expect("Failed to build graph");

    assert_eq!(graph.statements.len(), 2);
}

#[test]
fn test_gcd_test() {
    assert!(!gcd_test(&[2, -2], 1));
    assert!(gcd_test(&[2, -2], 0));
    assert!(gcd_test(&[3, -6], 9));
}

#[test]
fn test_banerjee_test() {
    assert!(banerjee_test(&[1, -1], 0, &[0, 0], &[9, 9]));
    assert!(!banerjee_test(&[1, -1], 20, &[0, 0], &[9, 9]));
}

#[test]
fn test_parallelizable_loop() {
    let source = r#"
        func parallel_test(A[N], B[N]) {
            for i = 0 to N {
                A[i] = B[i] * 2;
            }
        }
    "#;

    let pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let graph = analyzer.build_graph(&pir[0]).expect("Failed to build graph");
    assert!(graph.is_parallel_at(0));
}

#[test]
fn test_non_parallelizable_loop() {
    let source = r#"
        func sequential(A[N]) {
            for i = 1 to N {
                A[i] = A[i-1] + 1;
            }
        }
    "#;

    let pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let graph = analyzer.build_graph(&pir[0]).expect("Failed to build graph");
    assert!(!graph.is_parallel_at(0));
}

// ============================================================
// Phase 4: Transformation Tests
// ============================================================

#[test]
fn test_tiling_transformation() {
    use polyopt::transform::{Tiling, Transform};
    
    let source = r#"
        func simple_nest(A[N][M]) {
            for i = 0 to N {
                for j = 0 to M {
                    A[i][j] = i + j;
                }
            }
        }
    "#;
    
    let mut pir = parse_and_lower(source).expect("Failed to lower");
    let tiling = Tiling::new(vec![32, 32]);
    let changed = tiling.apply(&mut pir[0]).expect("Failed to apply tiling");
    
    assert!(changed);
    let stmt = &pir[0].statements[0];
    assert!(stmt.schedule.n_out() > 2);
}

#[test]
fn test_interchange_transformation() {
    use polyopt::transform::{Interchange, Transform};
    
    let source = r#"
        func transpose_access(A[N][M], B[M][N]) {
            for i = 0 to N {
                for j = 0 to M {
                    B[j][i] = A[i][j];
                }
            }
        }
    "#;
    
    let mut pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(&pir[0]).expect("Failed to analyze");
    
    let interchange = Interchange::new(0, 1);
    assert!(interchange.is_legal(&pir[0], &deps));
    
    let original_sched = pir[0].statements[0].schedule.clone();
    let changed = interchange.apply(&mut pir[0]).expect("Failed to apply interchange");
    
    assert!(changed);
    let new_sched = &pir[0].statements[0].schedule;
    assert_eq!(original_sched.apply(&[1, 2], &[]), vec![1, 2]);
    assert_eq!(new_sched.apply(&[1, 2], &[]), vec![2, 1]);
}

#[test]
fn test_fusion_transformation() {
    use polyopt::transform::{Fusion, Transform};
    use polyopt::ir::pir::StmtId;
    
    let source = r#"
        func two_loops(A[N], B[N], C[N]) {
            for i = 0 to N {
                A[i] = B[i] + 1;
            }
            for j = 0 to N {
                C[j] = A[j] * 2;
            }
        }
    "#;
    
    let mut pir = parse_and_lower(source).expect("Failed to lower");
    let stmt_ids: Vec<StmtId> = pir[0].statements.iter().map(|s| s.id).collect();
    
    let fusion = Fusion::new(stmt_ids);
    let changed = fusion.apply(&mut pir[0]).expect("Failed to apply fusion");
    assert!(changed);
}

#[test]
fn test_scheduler_pluto() {
    use polyopt::transform::{Scheduler, ScheduleAlgorithm};
    
    let source = r#"
        func stencil(A[N], B[N]) {
            for i = 1 to N - 1 {
                B[i] = A[i-1] + A[i] + A[i+1];
            }
        }
    "#;
    
    let mut pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(&pir[0]).expect("Failed to analyze");
    
    let scheduler = Scheduler::new()
        .with_algorithm(ScheduleAlgorithm::Pluto)
        .with_parallelism(true);
    
    scheduler.schedule(&mut pir[0], &deps).expect("Failed to schedule");
    
    let stmt = &pir[0].statements[0];
    assert!(stmt.schedule.n_out() > 1);
}

#[test]
fn test_scheduler_feautrier() {
    use polyopt::transform::{Scheduler, ScheduleAlgorithm};
    
    let source = r#"
        func simple(A[N]) {
            for i = 0 to N {
                A[i] = i * 2;
            }
        }
    "#;
    
    let mut pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(&pir[0]).expect("Failed to analyze");
    
    let scheduler = Scheduler::new()
        .with_algorithm(ScheduleAlgorithm::Feautrier);
    
    scheduler.schedule(&mut pir[0], &deps).expect("Failed to schedule");
    
    let stmt = &pir[0].statements[0];
    assert!(stmt.schedule.n_out() >= 1);
}

#[test]
fn test_auto_schedule() {
    use polyopt::transform::auto_schedule;
    
    let source = r#"
        func jacobi(A[N][N], B[N][N]) {
            for i = 1 to N - 1 {
                for j = 1 to N - 1 {
                    B[i][j] = A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1];
                }
            }
        }
    "#;
    
    let mut pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(&pir[0]).expect("Failed to analyze");
    
    auto_schedule(&mut pir[0], &deps).expect("Auto-scheduling failed");
    assert!(!pir[0].statements.is_empty());
}

#[test]
fn test_interchange_illegal() {
    use polyopt::transform::Interchange;
    use polyopt::analysis::{Dependence, DependenceKind, Direction, DependenceRelation};
    use polyopt::ir::pir::StmtId;
    
    let deps = vec![Dependence {
        source: StmtId::new(0),
        target: StmtId::new(0),
        kind: DependenceKind::Flow,
        relation: DependenceRelation::universe(2, 2, 0),
        distance: Some(vec![1, -1]),
        direction: vec![Direction::Lt, Direction::Gt],
        array: "A".to_string(),
        level: Some(0),
        is_loop_independent: false,
    }];
    
    let interchange = Interchange::new(0, 1);
    assert!(!interchange.is_interchange_legal_for(&deps));
}

#[test]
fn test_permutation_transformation() {
    use polyopt::transform::{Permutation, Transform};
    
    let source = r#"
        func three_loops(A[N][M][K]) {
            for i = 0 to N {
                for j = 0 to M {
                    for k = 0 to K {
                        A[i][j][k] = i + j + k;
                    }
                }
            }
        }
    "#;
    
    let mut pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(&pir[0]).expect("Failed to analyze");
    
    let perm = Permutation::reverse(3);
    assert!(perm.is_legal(&pir[0], &deps));
    
    let original = pir[0].statements[0].schedule.apply(&[1, 2, 3], &[]);
    perm.apply(&mut pir[0]).expect("Failed to apply permutation");
    let permuted = pir[0].statements[0].schedule.apply(&[1, 2, 3], &[]);
    
    assert_eq!(original, vec![1, 2, 3]);
    assert_eq!(permuted, vec![3, 2, 1]);
}

#[test]
fn test_optimize_for_parallel_tiled() {
    use polyopt::transform::optimize_for_parallel_tiled;
    
    let source = r#"
        func gemm(A[N][K], B[K][M], C[N][M]) {
            for i = 0 to N {
                for j = 0 to M {
                    for k = 0 to K {
                        C[i][j] = C[i][j] + A[i][k] * B[k][j];
                    }
                }
            }
        }
    "#;
    
    let mut pir = parse_and_lower(source).expect("Failed to lower");
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(&pir[0]).expect("Failed to analyze");
    
    optimize_for_parallel_tiled(&mut pir[0], &deps, vec![32, 32, 32])
        .expect("Optimization failed");
    
    let stmt = &pir[0].statements[0];
    assert!(stmt.schedule.n_out() > 3);
}
