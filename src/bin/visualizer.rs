//! Visualizer binary for polyhedral iteration spaces.
//!
//! Usage:
//!   polyvis input.poly                    # Text visualization
//!   polyvis input.poly --params N=10,M=10 # With parameter values

use polyopt::prelude::*;
use polyopt::parse_and_lower;
use polyopt::analysis::DependenceAnalysis;
use clap::Parser;
use anyhow::{Result, Context};
use std::fs;

#[derive(Parser)]
#[command(name = "polyvis")]
#[command(about = "Visualize polyhedral iteration spaces and dependences")]
struct Args {
    /// Input .poly file
    input: String,
    
    /// Parameter values (e.g., N=10,M=20)
    #[arg(short, long)]
    params: Option<String>,
    
    /// Maximum iterations to display per dimension
    #[arg(long, default_value = "15")]
    max_iters: i64,
    
    /// Show detailed dependence info
    #[arg(long)]
    verbose: bool,
}

fn parse_params(params_str: &str) -> Vec<(&str, i64)> {
    params_str.split(',')
        .filter_map(|p| {
            let parts: Vec<&str> = p.split('=').collect();
            if parts.len() == 2 {
                parts[1].parse::<i64>().ok().map(|v| (parts[0].trim(), v))
            } else {
                None
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Read input file
    let source = fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read input file: {}", args.input))?;
    
    // Parse and lower to PIR
    let pir = parse_and_lower(&source)
        .with_context(|| "Failed to parse and lower input")?;
    
    if pir.is_empty() {
        eprintln!("No functions found in input");
        return Ok(());
    }
    
    let program = &pir[0];
    
    // Parse parameter values
    let param_values: Vec<i64> = if let Some(ref params_str) = args.params {
        let parsed = parse_params(params_str);
        program.parameters.iter()
            .map(|p| {
                parsed.iter()
                    .find(|(name, _)| *name == p.as_str())
                    .map(|(_, v)| *v)
                    .unwrap_or(args.max_iters)
            })
            .collect()
    } else {
        vec![args.max_iters; program.parameters.len()]
    };
    
    // Analyze dependences
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(program).unwrap_or_default();
    
    print_visualization(program, &param_values, &deps, args.max_iters, args.verbose);
    
    Ok(())
}

fn print_visualization(
    program: &polyopt::ir::pir::PolyProgram,
    param_values: &[i64],
    deps: &[polyopt::analysis::Dependence],
    max_iters: i64,
    verbose: bool,
) {
    let width = 66;
    println!("╔{}╗", "═".repeat(width));
    println!("║  Polyhedral Visualization: {:<37}║", program.name);
    println!("╠{}╣", "═".repeat(width));
    
    // Print parameters
    let params_str = program.parameters.iter()
        .zip(param_values.iter())
        .map(|(p, v)| format!("{}={}", p, v))
        .collect::<Vec<_>>()
        .join(", ");
    println!("║  Parameters: {:<51}║", params_str);
    println!("║  Arrays: {:<55}║", 
        program.arrays.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", "));
    println!("╠{}╣", "═".repeat(width));
    
    // Print statements
    for (idx, stmt) in program.statements.iter().enumerate() {
        println!("║  Statement S{}: {:<50}║", idx, stmt.name);
        println!("║    Domain: {} dimensions{:<41}║", stmt.domain.dim(), "");
        
        let reads: Vec<_> = stmt.reads.iter().map(|r| r.array.as_str()).collect();
        let writes: Vec<_> = stmt.writes.iter().map(|w| w.array.as_str()).collect();
        println!("║    Reads: {:<54}║", reads.join(", "));
        println!("║    Writes: {:<53}║", writes.join(", "));
        
        // Print iteration space visualization
        if stmt.domain.dim() == 1 {
            print!("║    Iterations: ");
            let bound = param_values.get(0).copied().unwrap_or(max_iters).min(max_iters);
            let mut line = String::new();
            for i in 0..bound.min(45) {
                if stmt.domain.contains(&[i], param_values) {
                    line.push('●');
                } else {
                    line.push('○');
                }
            }
            if bound > 45 {
                line.push_str("...");
            }
            println!("{:<48}║", line);
        } else if stmt.domain.dim() == 2 {
            println!("║    Iteration Space (i→, j↓):{:<36}║", "");
            let bound_i = param_values.get(0).copied().unwrap_or(max_iters).min(max_iters);
            let bound_j = param_values.get(1).copied().unwrap_or(max_iters).min(max_iters);
            
            // Print j axis label
            print!("║      j\\i");
            for i in 0..bound_i.min(25) {
                print!("{}", i % 10);
            }
            println!("{:>width$}║", "", width = 66 - 9 - bound_i.min(25) as usize);
            
            for j in 0..bound_j.min(12) {
                print!("║      {:2} ", j);
                for i in 0..bound_i.min(25) {
                    if stmt.domain.contains(&[i, j], param_values) {
                        print!("●");
                    } else {
                        print!("·");
                    }
                }
                println!("{:>width$}║", "", width = 66 - 9 - bound_i.min(25) as usize);
            }
            if bound_i > 25 || bound_j > 12 {
                println!("║      ... (showing {}x{} of {}x{}){:<24}║", 
                    bound_i.min(25), bound_j.min(12), bound_i, bound_j, "");
            }
        } else if stmt.domain.dim() == 3 {
            println!("║    3D iteration space: {} × {} × {}{:<27}║",
                param_values.get(0).copied().unwrap_or(max_iters),
                param_values.get(1).copied().unwrap_or(max_iters),
                param_values.get(2).copied().unwrap_or(max_iters), "");
        }
        println!("║{:width$}║", "", width = width);
    }
    
    // Print dependences
    println!("╠{}╣", "═".repeat(width));
    println!("║  Dependences: {:<50}║", deps.len());
    
    let flow_count = deps.iter().filter(|d| d.kind == polyopt::analysis::DependenceKind::Flow).count();
    let anti_count = deps.iter().filter(|d| d.kind == polyopt::analysis::DependenceKind::Anti).count();
    let output_count = deps.iter().filter(|d| d.kind == polyopt::analysis::DependenceKind::Output).count();
    
    println!("║    Flow (RAW): {}, Anti (WAR): {}, Output (WAW): {}{:<17}║", 
        flow_count, anti_count, output_count, "");
    
    // Show parallelism info
    let analyzer = DependenceAnalysis::new();
    if let Some(parallel_level) = analyzer.find_parallel_level(deps, program.statements.first().map(|s| s.depth()).unwrap_or(0)) {
        println!("║    ✓ Parallel at level {}{:<41}║", parallel_level, "");
    } else {
        println!("║    ✗ No parallel loops detected{:<32}║", "");
    }
    
    if verbose {
        println!("║{:width$}║", "", width = width);
        for dep in deps.iter().take(8) {
            let kind = match dep.kind {
                polyopt::analysis::DependenceKind::Flow => "Flow",
                polyopt::analysis::DependenceKind::Anti => "Anti",
                polyopt::analysis::DependenceKind::Output => "Out ",
                polyopt::analysis::DependenceKind::Input => "Inp ",
            };
            let dist = dep.distance.as_ref()
                .map(|d| format!("{:?}", d))
                .unwrap_or_else(|| "?".to_string());
            println!("║    {} on {}: S{}→S{} dist={:<28}║", 
                kind, dep.array, dep.source.0, dep.target.0, dist);
        }
        if deps.len() > 8 {
            println!("║    ... and {} more{:<45}║", deps.len() - 8, "");
        }
    }
    
    println!("╚{}╝", "═".repeat(width));
}
