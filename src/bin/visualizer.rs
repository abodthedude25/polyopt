//! Visualizer binary for polyhedral iteration spaces.

use polyopt::prelude::*;
use clap::Parser;
use anyhow::Result;

#[derive(Parser)]
#[command(name = "polyopt-viz")]
#[command(about = "Visualize polyhedral iteration spaces")]
struct Args {
    /// Input file
    input: String,
    
    /// Output file (SVG)
    #[arg(short, long)]
    output: Option<String>,
    
    /// Width in pixels
    #[arg(long, default_value = "800")]
    width: u32,
    
    /// Height in pixels  
    #[arg(long, default_value = "600")]
    height: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("Visualizer not yet fully implemented");
    println!("Input: {}", args.input);
    
    Ok(())
}
