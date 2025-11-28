//! PolyOpt Command Line Interface
//!
//! Usage:
//!   polyopt [OPTIONS] <input-file>
//!   polyopt --help
//!
//! Examples:
//!   polyopt matmul.poly                    # Optimize with defaults
//!   polyopt -O3 --target=openmp matmul.poly  # Max optimization, OpenMP target
//!   polyopt --tile-sizes=64,64,32 gemm.poly  # Custom tile sizes
//!   polyopt --emit=ast matmul.poly          # Just parse and dump AST

use clap::{Parser, ValueEnum};
use polyopt::{OptimizationConfig, codegen::Target};
use std::path::PathBuf;
use std::fs;
use anyhow::{Result, Context};
use log::{info, debug, error};

/// PolyOpt - Polyhedral Compiler Optimization Framework
#[derive(Parser, Debug)]
#[command(name = "polyopt")]
#[command(author = "PolyOpt Contributors")]
#[command(version)]
#[command(about = "A polyhedral compiler optimization framework", long_about = None)]
struct Cli {
    /// Input file to optimize (.poly format)
    #[arg(value_name = "FILE")]
    input: PathBuf,

    /// Output file (defaults to stdout)
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Optimization level (0-3)
    #[arg(short = 'O', long, default_value = "2", value_parser = clap::value_parser!(u8).range(0..=3))]
    opt_level: u8,

    /// Code generation target
    #[arg(short, long, default_value = "c")]
    target: TargetArg,

    /// Tile sizes (comma-separated)
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    tile_sizes: Option<Vec<i64>>,

    /// Disable loop tiling
    #[arg(long)]
    no_tiling: bool,

    /// Disable loop fusion
    #[arg(long)]
    no_fusion: bool,

    /// Disable automatic scheduling
    #[arg(long)]
    no_auto_schedule: bool,

    /// Disable parallelization
    #[arg(long)]
    no_parallel: bool,

    /// Disable vectorization hints
    #[arg(long)]
    no_vectorize: bool,

    /// What to emit
    #[arg(long, default_value = "code")]
    emit: EmitKind,

    /// Verbose output (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Quiet mode (suppress warnings)
    #[arg(short, long)]
    quiet: bool,

    /// Dump intermediate representations
    #[arg(long)]
    dump_ir: bool,

    /// Enable auto-tuning (experimental)
    #[cfg(feature = "autotuning")]
    #[arg(long)]
    autotune: bool,

    /// Enable ML-based scheduling (experimental)
    #[cfg(feature = "ml")]
    #[arg(long)]
    ml_schedule: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TargetArg {
    /// Plain C code
    C,
    /// C with OpenMP pragmas
    Openmp,
    /// CUDA kernels
    Cuda,
    /// OpenCL kernels
    Opencl,
    /// LLVM IR
    Llvm,
}

impl From<TargetArg> for Target {
    fn from(arg: TargetArg) -> Self {
        match arg {
            TargetArg::C => Target::C,
            TargetArg::Openmp => Target::OpenMP,
            TargetArg::Cuda => Target::Cuda,
            TargetArg::Opencl => Target::OpenCL,
            TargetArg::Llvm => Target::LLVM,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EmitKind {
    /// Generated source code
    Code,
    /// Abstract Syntax Tree
    Ast,
    /// High-level IR
    Hir,
    /// Polyhedral IR
    Pir,
    /// Dependence graph (DOT format)
    Deps,
    /// Schedule (isl format)
    Schedule,
    /// All stages (for debugging)
    All,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.quiet {
        log::LevelFilter::Error
    } else {
        match cli.verbose {
            0 => log::LevelFilter::Warn,
            1 => log::LevelFilter::Info,
            2 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        }
    };
    
    env_logger::Builder::from_default_env()
        .filter_level(log_level)
        .format_timestamp(None)
        .init();

    info!("PolyOpt v{}", polyopt::VERSION);
    debug!("Input file: {:?}", cli.input);

    // Read input file
    let source = fs::read_to_string(&cli.input)
        .with_context(|| format!("Failed to read input file: {:?}", cli.input))?;

    // Parse the source
    info!("Parsing...");
    let program = polyopt::parse(&source)
        .with_context(|| "Failed to parse input")?;

    // Handle --emit=ast early exit
    if matches!(cli.emit, EmitKind::Ast) {
        let output = format!("{:#?}", program);
        write_output(&cli.output, &output)?;
        return Ok(());
    }

    // Build optimization config
    let config = build_config(&cli);
    debug!("Optimization config: {:?}", config);

    // Run optimization pipeline
    info!("Optimizing...");
    match polyopt::optimize(program, config) {
        Ok(_optimized) => {
            // TODO: Code generation based on emit kind
            info!("Optimization complete!");
            println!("// Optimization successful - code generation not yet implemented");
        }
        Err(e) => {
            error!("Optimization failed: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

fn build_config(cli: &Cli) -> OptimizationConfig {
    let mut config = match cli.opt_level {
        0 => OptimizationConfig {
            enable_tiling: false,
            enable_fusion: false,
            enable_interchange: false,
            enable_auto_schedule: false,
            enable_vectorization: false,
            enable_parallelization: false,
            ..Default::default()
        },
        1 => OptimizationConfig {
            enable_tiling: true,
            enable_fusion: false,
            enable_interchange: true,
            enable_auto_schedule: false,
            ..Default::default()
        },
        2 => OptimizationConfig::default(),
        _ => OptimizationConfig {
            enable_tiling: true,
            enable_fusion: true,
            enable_interchange: true,
            enable_auto_schedule: true,
            enable_vectorization: true,
            enable_parallelization: true,
            ..Default::default()
        },
    };

    // Override with CLI flags
    if cli.no_tiling {
        config.enable_tiling = false;
    }
    if cli.no_fusion {
        config.enable_fusion = false;
    }
    if cli.no_auto_schedule {
        config.enable_auto_schedule = false;
    }
    if cli.no_parallel {
        config.enable_parallelization = false;
    }
    if cli.no_vectorize {
        config.enable_vectorization = false;
    }
    if let Some(ref sizes) = cli.tile_sizes {
        config.tile_sizes = sizes.clone();
    }

    config.target = cli.target.into();
    config.verbosity = cli.verbose;

    #[cfg(feature = "autotuning")]
    {
        config.enable_autotuning = cli.autotune;
    }

    #[cfg(feature = "ml")]
    {
        config.enable_ml_scheduling = cli.ml_schedule;
    }

    config
}

fn write_output(path: &Option<PathBuf>, content: &str) -> Result<()> {
    match path {
        Some(p) => {
            fs::write(p, content)
                .with_context(|| format!("Failed to write output file: {:?}", p))?;
        }
        None => {
            println!("{}", content);
        }
    }
    Ok(())
}
