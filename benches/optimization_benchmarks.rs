//! Benchmarks for the polyhedral optimizer.
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark parsing speed.
fn bench_parsing(c: &mut Criterion) {
    let source = r#"
        func matmul(A[N][K], B[K][M], C[N][M]) {
            for i = 0 to N {
                for j = 0 to M {
                    C[i][j] = 0;
                    for k = 0 to K {
                        C[i][j] = C[i][j] + A[i][k] * B[k][j];
                    }
                }
            }
        }
    "#;

    c.bench_function("parse_matmul", |b| {
        b.iter(|| {
            let lexer = polyopt::frontend::Lexer::new(black_box(source));
            let mut parser = polyopt::frontend::Parser::new(lexer).unwrap();
            parser.parse_program().unwrap()
        })
    });
}

/// Benchmark lexer speed.
fn bench_lexing(c: &mut Criterion) {
    let source = r#"
        func test(A[N][M], B[M][K]) {
            for i = 0 to N {
                for j = 0 to M {
                    for k = 0 to K {
                        A[i][j] = A[i][j] + B[j][k];
                    }
                }
            }
        }
    "#;

    c.bench_function("lex_nested_loops", |b| {
        b.iter(|| {
            let lexer = polyopt::frontend::Lexer::new(black_box(source));
            lexer.tokenize().unwrap()
        })
    });
}

/// Benchmark polyhedral set operations.
fn bench_polyhedral_ops(c: &mut Criterion) {
    use polyopt::polyhedral::IntegerSet;
    
    c.bench_function("create_rectangular_set", |b| {
        b.iter(|| {
            IntegerSet::rectangular(black_box(&[100, 100, 100]))
        })
    });
}

criterion_group!(benches, bench_parsing, bench_lexing, bench_polyhedral_ops);
criterion_main!(benches);
