//! PolyOpt - A Polyhedral Loop Optimizer
//!
//! Main command-line interface for the polyhedral optimizer.

use polyopt::{parse, parse_and_lower};
use polyopt::analysis::{DependenceAnalysis, DependenceGraph};
use polyopt::codegen::{generate, Target, generate_benchmark, CodeGenOptions, CCodeGen};
use polyopt::transform::{
    Tiling, Interchange, Fusion,
    Scheduler, ScheduleAlgorithm,
};

use clap::{Parser, Subcommand, ValueEnum};
use anyhow::{Result, Context};
use std::fs;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "polyopt")]
#[command(author = "PolyOpt Team")]
#[command(version = "0.1.0")]
#[command(about = "Polyhedral loop optimizer for high-performance computing")]
#[command(long_about = r#"
PolyOpt is a polyhedral compiler that optimizes loop nests for parallelism
and data locality. It performs:

  - Dependence analysis to identify loop-carried dependencies
  - Transformations like tiling, interchange, and fusion
  - Automatic scheduling using Pluto-like algorithms
  - Code generation to C/OpenMP

Example usage:
  polyopt compile input.poly -o output.c --openmp
  polyopt analyze input.poly --deps --parallel
  polyopt optimize input.poly --tile 32 --schedule pluto
"#)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a .poly file to C code
    Compile {
        /// Input .poly file
        input: PathBuf,
        
        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Target format
        #[arg(short, long, value_enum, default_value = "c")]
        target: TargetArg,
        
        /// Enable OpenMP parallelization
        #[arg(long)]
        openmp: bool,
        
        /// Enable vectorization hints
        #[arg(long)]
        vectorize: bool,
        
        /// Generate timing/benchmark code
        #[arg(long)]
        benchmark: bool,
    },
    
    /// Analyze a .poly file
    Analyze {
        /// Input .poly file
        input: PathBuf,
        
        /// Show dependence information
        #[arg(long)]
        deps: bool,
        
        /// Show parallelism opportunities
        #[arg(long)]
        parallel: bool,
        
        /// Show detailed statistics
        #[arg(long)]
        stats: bool,
        
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
    
    /// Optimize and transform a .poly file
    Optimize {
        /// Input .poly file
        input: PathBuf,
        
        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Apply tiling with given size
        #[arg(long)]
        tile: Option<i64>,
        
        /// Scheduling algorithm
        #[arg(long, value_enum, default_value = "pluto")]
        schedule: ScheduleArg,
        
        /// Enable OpenMP in output
        #[arg(long)]
        openmp: bool,
    },
    
    /// Print parsed AST (for debugging)
    Parse {
        /// Input .poly file
        input: PathBuf,
        
        /// Show HIR instead of AST
        #[arg(long)]
        hir: bool,
        
        /// Show PIR (polyhedral IR)
        #[arg(long)]
        pir: bool,
    },
    
    /// Run benchmarks
    Bench {
        /// Input .poly file
        input: PathBuf,
        
        /// Problem size
        #[arg(short = 'N', long, default_value = "1000")]
        size: i64,
        
        /// Number of iterations
        #[arg(short, long, default_value = "5")]
        iterations: usize,
        
        /// Enable OpenMP
        #[arg(long)]
        openmp: bool,
    },
    
    /// Analyze with ISL (exact polyhedral math)
    Isl {
        /// Input .poly file or ISL expression
        input: String,
        
        /// Parse as raw ISL set expression (not file)
        #[arg(long)]
        expr: bool,
        
        /// Show domain information
        #[arg(long)]
        domain: bool,
        
        /// Compute schedule
        #[arg(long)]
        schedule: bool,
        
        /// Show dependences
        #[arg(long)]
        deps: bool,
        
        /// Enumerate points (for small sets)
        #[arg(long)]
        enumerate: bool,
        
        /// Parameter values (e.g., N=10,M=20)
        #[arg(long)]
        params: Option<String>,
        
        /// Generate tiled code
        #[arg(long)]
        tile: Option<usize>,
    },
    
    /// Auto-tune optimization parameters
    Autotune {
        /// Input .poly file
        input: PathBuf,
        
        /// Output file for best configuration
        #[arg(short, long)]
        output: Option<PathBuf>,
        
        /// Tile sizes to try (comma-separated)
        #[arg(long, default_value = "8,16,32,64,128")]
        tiles: String,
        
        /// Problem size for benchmarking
        #[arg(short = 'N', long, default_value = "500")]
        size: usize,
        
        /// Number of benchmark iterations
        #[arg(long, default_value = "3")]
        iterations: usize,
        
        /// Search strategy (exhaustive, random, genetic, annealing)
        #[arg(long, default_value = "exhaustive")]
        strategy: String,
        
        /// Enable OpenMP in variants
        #[arg(long)]
        openmp: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TargetArg {
    C,
    Openmp,
    Cuda,
    Opencl,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ScheduleArg {
    Pluto,
    Feautrier,
    Greedy,
}

impl From<ScheduleArg> for ScheduleAlgorithm {
    fn from(arg: ScheduleArg) -> Self {
        match arg {
            ScheduleArg::Pluto => ScheduleAlgorithm::Pluto,
            ScheduleArg::Feautrier => ScheduleAlgorithm::Feautrier,
            ScheduleArg::Greedy => ScheduleAlgorithm::Greedy,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compile { input, output, target, openmp, vectorize, benchmark } => {
            cmd_compile(input, output, target, openmp, vectorize, benchmark)
        }
        Commands::Analyze { input, deps, parallel, stats, json } => {
            cmd_analyze(input, deps, parallel, stats, json)
        }
        Commands::Optimize { input, output, tile, schedule, openmp } => {
            cmd_optimize(input, output, tile, schedule, openmp)
        }
        Commands::Parse { input, hir, pir } => {
            cmd_parse(input, hir, pir)
        }
        Commands::Bench { input, size, iterations, openmp } => {
            cmd_bench(input, size, iterations, openmp)
        }
        Commands::Isl { input, expr, domain, schedule, deps, enumerate, params, tile } => {
            cmd_isl(input, expr, domain, schedule, deps, enumerate, params, tile)
        }
        Commands::Autotune { input, output, tiles, size, iterations, strategy, openmp } => {
            cmd_autotune(input, output, tiles, size, iterations, strategy, openmp)
        }
    }
}

fn cmd_compile(
    input: PathBuf,
    output: Option<PathBuf>,
    target: TargetArg,
    openmp: bool,
    vectorize: bool,
    benchmark: bool,
) -> Result<()> {
    let source = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read {}", input.display()))?;
    
    let programs = parse_and_lower(&source)
        .with_context(|| "Failed to parse and lower")?;
    
    if programs.is_empty() {
        anyhow::bail!("No functions found in input");
    }
    
    let program = &programs[0];
    
    // Generate code
    let target_enum = match target {
        TargetArg::C => if openmp { Target::OpenMP } else { Target::C },
        TargetArg::Openmp => Target::OpenMP,
        TargetArg::Cuda => Target::Cuda,
        TargetArg::Opencl => Target::OpenCL,
    };
    
    let code = if benchmark {
        generate_benchmark(program, openmp || matches!(target, TargetArg::Openmp))?
    } else {
        let options = CodeGenOptions {
            openmp: openmp || matches!(target, TargetArg::Openmp),
            vectorize,
            ..Default::default()
        };
        let codegen = CCodeGen::with_options(options);
        codegen.generate(program)?
    };
    
    // Output
    if let Some(out_path) = output {
        fs::write(&out_path, &code)
            .with_context(|| format!("Failed to write {}", out_path.display()))?;
        eprintln!("Wrote {} bytes to {}", code.len(), out_path.display());
    } else {
        print!("{}", code);
    }
    
    Ok(())
}

fn cmd_analyze(
    input: PathBuf,
    deps: bool,
    parallel: bool,
    stats: bool,
    json: bool,
) -> Result<()> {
    let source = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read {}", input.display()))?;
    
    let programs = parse_and_lower(&source)
        .with_context(|| "Failed to parse and lower")?;
    
    if programs.is_empty() {
        anyhow::bail!("No functions found in input");
    }
    
    let program = &programs[0];
    let analyzer = DependenceAnalysis::new();
    let dep_list = analyzer.analyze(program).unwrap_or_default();
    let dep_graph = DependenceGraph::from_dependences(dep_list.clone(), program);
    
    // Count dependence types
    let num_flow = dep_list.iter().filter(|d| d.kind == polyopt::analysis::DependenceKind::Flow).count();
    let num_anti = dep_list.iter().filter(|d| d.kind == polyopt::analysis::DependenceKind::Anti).count();
    let num_output = dep_list.iter().filter(|d| d.kind == polyopt::analysis::DependenceKind::Output).count();
    let num_loop_carried = dep_list.iter().filter(|d| d.is_loop_carried()).count();
    
    if json {
        // JSON output
        println!("{{");
        println!("  \"name\": \"{}\",", program.name);
        println!("  \"statements\": {},", program.statements.len());
        println!("  \"parameters\": {:?},", program.parameters);
        println!("  \"dependences\": {},", dep_list.len());
        println!("  \"flow_deps\": {},", num_flow);
        println!("  \"anti_deps\": {},", num_anti);
        println!("  \"output_deps\": {},", num_output);
        
        // Find parallel loops
        let max_depth = program.statements.iter().map(|s| s.depth()).max().unwrap_or(0);
        let parallel_levels: Vec<usize> = (0..max_depth)
            .filter(|&level| dep_graph.is_parallel_at(level))
            .collect();
        println!("  \"parallel_levels\": {:?}", parallel_levels);
        println!("}}");
    } else {
        // Human-readable output
        println!("=== Analysis: {} ===\n", program.name);
        println!("Statements: {}", program.statements.len());
        println!("Parameters: {:?}", program.parameters);
        println!("Arrays: {:?}", program.arrays.iter().map(|a| &a.name).collect::<Vec<_>>());
        
        if deps || stats {
            println!("\n--- Dependences ---");
            println!("Total: {}", dep_list.len());
            println!("  Flow (RAW): {}", num_flow);
            println!("  Anti (WAR): {}", num_anti);
            println!("  Output (WAW): {}", num_output);
            println!("  Loop-carried: {}", num_loop_carried);
            println!("  Loop-independent: {}", dep_list.len() - num_loop_carried);
            
            if deps {
                println!("\nDependence details:");
                for (i, dep) in dep_list.iter().enumerate().take(20) {
                    let kind = match dep.kind {
                        polyopt::analysis::DependenceKind::Flow => "Flow",
                        polyopt::analysis::DependenceKind::Anti => "Anti",
                        polyopt::analysis::DependenceKind::Output => "Output",
                        polyopt::analysis::DependenceKind::Input => "Input",
                    };
                    println!("  [{}] {} on {}: S{} -> S{}", 
                        i, kind, dep.array, dep.source.0, dep.target.0);
                    if let Some(ref dist) = dep.distance {
                        println!("       distance: {:?}", dist);
                    }
                }
                if dep_list.len() > 20 {
                    println!("  ... and {} more", dep_list.len() - 20);
                }
            }
        }
        
        if parallel {
            println!("\n--- Parallelism ---");
            let max_depth = program.statements.iter().map(|s| s.depth()).max().unwrap_or(0);
            for level in 0..max_depth {
                let is_parallel = dep_graph.is_parallel_at(level);
                let marker = if is_parallel { "✓" } else { "✗" };
                println!("  Level {}: {} {}", level, marker, 
                    if is_parallel { "PARALLEL" } else { "sequential" });
            }
            
            if let Some(level) = analyzer.find_parallel_level(&dep_list, max_depth) {
                println!("\nRecommendation: Parallelize at level {}", level);
            } else {
                println!("\nNo directly parallelizable loops found");
                println!("Consider applying transformations (tiling, skewing)");
            }
        }
    }
    
    Ok(())
}

fn cmd_optimize(
    input: PathBuf,
    output: Option<PathBuf>,
    tile: Option<i64>,
    schedule: ScheduleArg,
    openmp: bool,
) -> Result<()> {
    let source = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read {}", input.display()))?;
    
    let mut programs = parse_and_lower(&source)
        .with_context(|| "Failed to parse and lower")?;
    
    if programs.is_empty() {
        anyhow::bail!("No functions found in input");
    }
    
    let mut program = programs.remove(0);
    
    // Analyze dependencies
    let analyzer = DependenceAnalysis::new();
    let deps = analyzer.analyze(&program).unwrap_or_default();
    
    eprintln!("Optimizing {} with {} statements...", program.name, program.statements.len());
    eprintln!("Found {} dependences", deps.len());
    
    // Apply scheduling
    let scheduler = Scheduler::new()
        .with_algorithm(schedule.into());
    scheduler.schedule(&mut program, &deps).ok();
    eprintln!("Applied {:?} scheduling", schedule);
    
    // Apply tiling if requested
    if let Some(tile_size) = tile {
        let depth = program.statements.first().map(|s| s.depth()).unwrap_or(0);
        if depth > 0 {
            let tiling = Tiling::with_default_size(depth, tile_size);
            for stmt in &mut program.statements {
                stmt.schedule = tiling.apply_to_schedule(&stmt.schedule);
            }
            eprintln!("Applied tiling with size {}", tile_size);
        }
    }
    
    // Generate code
    let codegen = CCodeGen::new(openmp);
    let code = codegen.generate(&program)?;
    
    // Output
    if let Some(out_path) = output {
        fs::write(&out_path, &code)
            .with_context(|| format!("Failed to write {}", out_path.display()))?;
        eprintln!("Wrote optimized code to {}", out_path.display());
    } else {
        print!("{}", code);
    }
    
    Ok(())
}

fn cmd_parse(input: PathBuf, hir: bool, pir: bool) -> Result<()> {
    let source = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read {}", input.display()))?;
    
    if pir {
        let programs = parse_and_lower(&source)
            .with_context(|| "Failed to parse and lower")?;
        
        for program in &programs {
            println!("=== Polyhedral IR: {} ===", program.name);
            println!("Parameters: {:?}", program.parameters);
            println!("Arrays: {:?}", program.arrays.iter().map(|a| &a.name).collect::<Vec<_>>());
            println!("\nStatements:");
            for stmt in &program.statements {
                println!("  {} (depth {})", stmt.name, stmt.depth());
                println!("    Domain: {} dims, {} constraints", 
                    stmt.domain.dim(), stmt.domain.constraints.constraints.len());
                println!("    Schedule: {} in, {} out",
                    stmt.schedule.n_in(), stmt.schedule.n_out());
                println!("    Reads: {:?}", stmt.reads.iter().map(|r| &r.array).collect::<Vec<_>>());
                println!("    Writes: {:?}", stmt.writes.iter().map(|w| &w.array).collect::<Vec<_>>());
            }
        }
    } else if hir {
        let hir_programs = polyopt::lower_to_hir(&source)
            .with_context(|| "Failed to parse and lower to HIR")?;
        
        for func in &hir_programs {
            println!("=== HIR: {} ===", func.name);
            println!("Parameters: {}", func.params.len());
            println!("Body statements: {}", func.body.statements.len());
        }
    } else {
        let ast = parse(&source)
            .with_context(|| "Failed to parse")?;
        
        println!("=== AST ===");
        println!("Functions: {}", ast.functions.len());
        for func in &ast.functions {
            println!("  func {} ({} params)", func.name, func.params.len());
        }
    }
    
    Ok(())
}

fn cmd_bench(input: PathBuf, size: i64, iterations: usize, openmp: bool) -> Result<()> {
    let source = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read {}", input.display()))?;
    
    let programs = parse_and_lower(&source)
        .with_context(|| "Failed to parse and lower")?;
    
    if programs.is_empty() {
        anyhow::bail!("No functions found in input");
    }
    
    let program = &programs[0];
    
    println!("=== Benchmark: {} ===", program.name);
    println!("Size: N = {}", size);
    println!("Iterations: {}", iterations);
    println!("OpenMP: {}", if openmp { "enabled" } else { "disabled" });
    println!();
    
    // Generate benchmark code
    let code = generate_benchmark(program, openmp)?;
    
    // Write to temp file
    let temp_dir = std::env::temp_dir();
    let c_file = temp_dir.join("polyopt_bench.c");
    let exe_file = temp_dir.join("polyopt_bench");
    
    fs::write(&c_file, &code)?;
    
    // Compile
    let compiler = "gcc";
    let mut compile_args = vec![
        "-O3", "-march=native",
        c_file.to_str().unwrap(),
        "-o", exe_file.to_str().unwrap(),
        "-lm"
    ];
    if openmp {
        compile_args.push("-fopenmp");
    }
    
    println!("Compiling with: {} {}", compiler, compile_args.join(" "));
    
    let compile_status = std::process::Command::new(compiler)
        .args(&compile_args)
        .status();
    
    match compile_status {
        Ok(status) if status.success() => {
            println!("Compilation successful\n");
            
            // Run benchmark
            println!("Running {} iterations...\n", iterations);
            for i in 0..iterations {
                let output = std::process::Command::new(&exe_file)
                    .arg(size.to_string())
                    .output()?;
                
                let stdout = String::from_utf8_lossy(&output.stdout);
                print!("  Run {}: {}", i + 1, stdout);
            }
        }
        Ok(_) => {
            eprintln!("Compilation failed");
            eprintln!("Generated code saved to: {}", c_file.display());
        }
        Err(e) => {
            eprintln!("Could not run compiler: {}", e);
            eprintln!("Generated code saved to: {}", c_file.display());
        }
    }
    
    Ok(())
}

fn cmd_isl(
    input: String,
    expr: bool,
    domain: bool,
    schedule: bool,
    deps: bool,
    enumerate: bool,
    params: Option<String>,
    tile: Option<usize>,
) -> Result<()> {
    use polyopt::isl::simulation::{PolySet, codegen};
    use polyopt::isl::is_isl_available;
    
    println!("=== ISL Polyhedral Analysis ===\n");
    
    // Check if native ISL is available
    if is_isl_available() {
        println!("Using: Native ISL library (iscc)");
    } else {
        println!("Using: Pure Rust polyhedral simulation");
        println!("       (Install ISL for exact operations: brew install isl)");
    }
    println!();
    
    // Parse parameter values
    let param_values: Vec<i64> = if let Some(ref p) = params {
        p.split(',')
            .filter_map(|s| {
                let parts: Vec<&str> = s.split('=').collect();
                if parts.len() == 2 {
                    parts[1].trim().parse().ok()
                } else {
                    s.trim().parse().ok()
                }
            })
            .collect()
    } else {
        vec![10] // Default N=10
    };
    
    // Parse input as ISL expression or file
    let isl_set = if expr {
        // Direct ISL expression
        PolySet::parse(&input).map_err(|e| anyhow::anyhow!("{}", e))?
    } else {
        // Read from .poly file and extract domain
        let path = std::path::PathBuf::from(&input);
        let source = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        
        let programs = parse_and_lower(&source)?;
        if programs.is_empty() {
            anyhow::bail!("No functions found in input");
        }
        
        // Extract domain from first statement
        let program = &programs[0];
        if program.statements.is_empty() {
            anyhow::bail!("No statements found");
        }
        
        let stmt = &program.statements[0];
        let dim = stmt.domain.dim();
        let vars: Vec<String> = (0..dim).map(|i| format!("i{}", i)).collect();
        
        // Build ISL-format string from constraints
        let constraints: Vec<String> = stmt.domain.constraints.constraints.iter()
            .map(|c| {
                let terms: Vec<String> = c.expr.coeffs.iter().enumerate()
                    .filter(|(_, &coeff)| coeff != 0)
                    .map(|(i, &coeff)| {
                        if i < vars.len() {
                            if coeff == 1 { vars[i].clone() }
                            else { format!("{}*{}", coeff, vars[i]) }
                        } else {
                            coeff.to_string()
                        }
                    })
                    .collect();
                
                let lhs = if terms.is_empty() { "0".to_string() } else { terms.join(" + ") };
                format!("{} >= {}", lhs, -c.expr.constant)
            })
            .collect();
        
        let isl_str = if constraints.is_empty() {
            format!("{{ [{}] : 0 <= {} and {} < N }}", vars.join(", "), vars[0], vars[0])
        } else {
            format!("{{ [{}] : {} }}", vars.join(", "), constraints.join(" and "))
        };
        
        println!("Extracted domain from {}: {}", program.name, stmt.name);
        PolySet::parse(&isl_str).map_err(|e| anyhow::anyhow!("{}", e))?
    };
    
    println!("Set: {}", isl_set.to_isl_string());
    println!("Variables: {:?}", isl_set.vars);
    println!("Parameters: {:?}", isl_set.params);
    println!("Constraints: {}", isl_set.constraints.len());
    println!();
    
    if domain || (!enumerate && !schedule && tile.is_none()) {
        println!("--- Domain Analysis ---");
        println!("ISL notation: {}", isl_set.to_isl_string());
        println!("Dimensions: {}", isl_set.vars.len());
        
        // Check emptiness
        let empty = isl_set.is_empty(&param_values);
        println!("Empty (with params {:?}): {}", param_values, empty);
    }
    
    if enumerate {
        println!("\n--- Point Enumeration ---");
        let max = if param_values.is_empty() { 10 } else { param_values[0] };
        let points = isl_set.enumerate(&param_values, max);
        
        println!("Points (up to {} per dimension):", max);
        for (i, p) in points.iter().enumerate().take(50) {
            println!("  {:3}: ({:?})", i, p);
        }
        
        if points.len() > 50 {
            println!("  ... and {} more", points.len() - 50);
        }
        
        println!("\nTotal points: {}", points.len());
    }
    
    if schedule {
        println!("\n--- Scheduling ---");
        println!("Available algorithms:");
        println!("  - Feautrier: Lexicographic minimum latency");
        println!("  - Pluto: Maximize parallelism + locality");
        
        // Show identity schedule
        println!("\nIdentity schedule (no transformation):");
        let code = codegen::generate_loops(&isl_set, "// body");
        println!("{}", code);
    }
    
    if let Some(ts) = tile {
        println!("\n--- Tiled Code Generation ---");
        println!("Tile size: {}", ts);
        
        let tile_sizes = vec![ts; isl_set.vars.len()];
        let code = codegen::generate_tiled_loops(&isl_set, "C[i][j] = A[i][j] + B[i][j];", &tile_sizes);
        println!("{}", code);
    }
    
    Ok(())
}

fn cmd_autotune(
    input: PathBuf,
    output: Option<PathBuf>,
    tiles: String,
    size: usize,
    iterations: usize,
    strategy: String,
    openmp: bool,
) -> Result<()> {
    use polyopt::autotune::{AutoTuner, AdvancedAutoTuner, SearchStrategy, print_results};
    
    // Parse tile sizes
    let tile_sizes: Vec<usize> = tiles.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    // Parse search strategy
    let search_strategy = match strategy.to_lowercase().as_str() {
        "random" => SearchStrategy::Random { samples: 20 },
        "genetic" => SearchStrategy::Genetic { population: 10, generations: 5 },
        "annealing" => SearchStrategy::Annealing { iterations: 50 },
        _ => SearchStrategy::Exhaustive,
    };
    
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              PolyOpt Auto-Tuning Framework                 ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
    
    println!("Input: {}", input.display());
    println!("Configuration:");
    println!("  Tile sizes to try: {:?}", tile_sizes);
    println!("  Problem size: N={}", size);
    println!("  Iterations per variant: {}", iterations);
    println!("  OpenMP: {}", openmp);
    println!("  Strategy: {:?}", search_strategy);
    println!();
    
    // Read and compile the input file
    let source = fs::read_to_string(&input)
        .with_context(|| format!("Failed to read {}", input.display()))?;
    
    let programs = parse_and_lower(&source)?;
    if programs.is_empty() {
        anyhow::bail!("No functions found in input");
    }
    
    let program = &programs[0];
    
    // Generate baseline C code
    let options = CodeGenOptions {
        openmp,
        ..Default::default()
    };
    
    let codegen = CCodeGen::with_options(options);
    let c_code = codegen.generate(program)?;
    
    println!("Generated baseline code for: {}", program.name);
    println!("Statements: {}", program.statements.len());
    println!();
    
    // Create tuner
    let mut tuner = AutoTuner::new();
    tuner.tile_sizes = tile_sizes;
    tuner.try_openmp = openmp;
    tuner.iterations = iterations;
    tuner.problem_size = size;
    
    // Run tuning
    println!("Starting auto-tuning...\n");
    
    let results = if matches!(search_strategy, SearchStrategy::Exhaustive) {
        tuner.tune(&c_code, &program.name)
            .map_err(|e| anyhow::anyhow!("{}", e))?
    } else {
        let advanced = AdvancedAutoTuner::new(search_strategy);
        advanced.tune(&c_code, &program.name)
            .map_err(|e| anyhow::anyhow!("{}", e))?
    };
    
    // Print results
    print_results(&results);
    
    // Save results if output specified
    if let Some(out_path) = output {
        let mut csv = String::from("config,time_seconds,speedup\n");
        for r in &results {
            csv.push_str(&format!("{},{:.6},{:.2}\n", 
                r.config.describe(), r.time_seconds, r.speedup));
        }
        fs::write(&out_path, &csv)?;
        println!("\nResults saved to: {}", out_path.display());
    }
    
    // Show command for best config
    if let Some(best) = results.first() {
        if best.speedup > 1.0 {
            println!("\n To generate optimized code:");
            let tile_str = if !best.config.tile_sizes.is_empty() && best.config.tile_sizes[0] > 0 {
                format!(" --tile {}", best.config.tile_sizes[0])
            } else {
                String::new()
            };
            let omp_str = if best.config.parallel { " --openmp" } else { "" };
            println!("  polyopt compile {}{}{}", input.display(), tile_str, omp_str);
        }
    }
    
    Ok(())
}