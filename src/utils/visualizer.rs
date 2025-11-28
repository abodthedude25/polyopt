//! Visualization utilities for polyhedra and iteration spaces.
//!
//! This module provides utilities for visualizing:
//! - 2D and 3D iteration spaces
//! - Dependence graphs
//! - Schedules
//! - Transformation effects
//!
//! Requires the "visualization" feature to be enabled.

#[cfg(feature = "visualization")]
use plotters::prelude::*;

#[cfg(feature = "visualization")]
use svg::Document;

/// Configuration for visualization output.
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Output width in pixels
    pub width: u32,
    /// Output height in pixels
    pub height: u32,
    /// Background color
    pub background: (u8, u8, u8),
    /// Show grid lines
    pub show_grid: bool,
    /// Show axis labels
    pub show_labels: bool,
    /// Point size
    pub point_size: u32,
    /// Line width
    pub line_width: u32,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            background: (255, 255, 255),
            show_grid: true,
            show_labels: true,
            point_size: 5,
            line_width: 2,
        }
    }
}

/// A point in 2D iteration space.
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: i64,
    pub y: i64,
}

/// A point in 3D iteration space.
#[derive(Debug, Clone, Copy)]
pub struct Point3D {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

/// An edge in a dependence graph.
#[derive(Debug, Clone)]
pub struct DependenceEdge {
    pub from: Point2D,
    pub to: Point2D,
    pub kind: DependenceEdgeKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependenceEdgeKind {
    Flow,   // RAW
    Anti,   // WAR
    Output, // WAW
}

/// Visualize a 2D iteration space.
#[cfg(feature = "visualization")]
pub fn visualize_2d_iteration_space(
    points: &[Point2D],
    config: &VisualizationConfig,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (config.width, config.height))
        .into_drawing_area();
    
    root.fill(&RGBColor(
        config.background.0,
        config.background.1,
        config.background.2,
    ))?;

    if points.is_empty() {
        return Ok(());
    }

    let (min_x, max_x, min_y, max_y) = points.iter().fold(
        (i64::MAX, i64::MIN, i64::MAX, i64::MIN),
        |(min_x, max_x, min_y, max_y), p| {
            (
                min_x.min(p.x),
                max_x.max(p.x),
                min_y.min(p.y),
                max_y.max(p.y),
            )
        },
    );

    let margin = 2;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (min_x - margin)..(max_x + margin),
            (min_y - margin)..(max_y + margin),
        )?;

    if config.show_grid {
        chart.configure_mesh().draw()?;
    }

    chart.draw_series(
        points.iter().map(|p| {
            Circle::new((p.x, p.y), config.point_size, BLUE.filled())
        }),
    )?;

    root.present()?;
    Ok(())
}

/// Visualize dependencies as arrows.
#[cfg(feature = "visualization")]
pub fn visualize_dependencies(
    points: &[Point2D],
    edges: &[DependenceEdge],
    config: &VisualizationConfig,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (config.width, config.height))
        .into_drawing_area();
    
    root.fill(&WHITE)?;

    if points.is_empty() {
        return Ok(());
    }

    let (min_x, max_x, min_y, max_y) = points.iter().fold(
        (i64::MAX, i64::MIN, i64::MAX, i64::MIN),
        |(min_x, max_x, min_y, max_y), p| {
            (
                min_x.min(p.x),
                max_x.max(p.x),
                min_y.min(p.y),
                max_y.max(p.y),
            )
        },
    );

    let margin = 2;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(
            (min_x - margin)..(max_x + margin),
            (min_y - margin)..(max_y + margin),
        )?;

    chart.configure_mesh().draw()?;

    // Draw edges
    for edge in edges {
        let color = match edge.kind {
            DependenceEdgeKind::Flow => &RED,
            DependenceEdgeKind::Anti => &GREEN,
            DependenceEdgeKind::Output => &BLUE,
        };
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(edge.from.x, edge.from.y), (edge.to.x, edge.to.y)],
            color.stroke_width(config.line_width),
        )))?;
    }

    // Draw points
    chart.draw_series(
        points.iter().map(|p| {
            Circle::new((p.x, p.y), config.point_size, BLACK.filled())
        }),
    )?;

    root.present()?;
    Ok(())
}

/// Generate SVG visualization of iteration space.
#[cfg(feature = "visualization")]
pub fn visualize_to_svg(
    points: &[Point2D],
    edges: &[DependenceEdge],
    config: &VisualizationConfig,
) -> String {
    use svg::node::element::{Circle, Line, Rectangle};
    
    let mut document = Document::new()
        .set("width", config.width)
        .set("height", config.height)
        .set("viewBox", (0, 0, config.width, config.height));

    // Background
    document = document.add(
        Rectangle::new()
            .set("width", "100%")
            .set("height", "100%")
            .set("fill", format!(
                "rgb({},{},{})",
                config.background.0,
                config.background.1,
                config.background.2
            )),
    );

    if points.is_empty() {
        return document.to_string();
    }

    // Calculate bounds and scale
    let (min_x, max_x, min_y, max_y) = points.iter().fold(
        (i64::MAX, i64::MIN, i64::MAX, i64::MIN),
        |(min_x, max_x, min_y, max_y), p| {
            (
                min_x.min(p.x),
                max_x.max(p.x),
                min_y.min(p.y),
                max_y.max(p.y),
            )
        },
    );

    let range_x = (max_x - min_x + 4) as f64;
    let range_y = (max_y - min_y + 4) as f64;
    let scale_x = (config.width as f64 - 40.0) / range_x;
    let scale_y = (config.height as f64 - 40.0) / range_y;
    let scale = scale_x.min(scale_y);

    let transform = |p: &Point2D| -> (f64, f64) {
        let x = 20.0 + ((p.x - min_x + 2) as f64) * scale;
        let y = config.height as f64 - 20.0 - ((p.y - min_y + 2) as f64) * scale;
        (x, y)
    };

    // Draw edges
    for edge in edges {
        let (x1, y1) = transform(&edge.from);
        let (x2, y2) = transform(&edge.to);
        let color = match edge.kind {
            DependenceEdgeKind::Flow => "red",
            DependenceEdgeKind::Anti => "green",
            DependenceEdgeKind::Output => "blue",
        };
        document = document.add(
            Line::new()
                .set("x1", x1)
                .set("y1", y1)
                .set("x2", x2)
                .set("y2", y2)
                .set("stroke", color)
                .set("stroke-width", config.line_width),
        );
    }

    // Draw points
    for point in points {
        let (cx, cy) = transform(point);
        document = document.add(
            Circle::new()
                .set("cx", cx)
                .set("cy", cy)
                .set("r", config.point_size)
                .set("fill", "black"),
        );
    }

    document.to_string()
}

// Stub implementations when visualization is disabled
#[cfg(not(feature = "visualization"))]
pub fn visualize_2d_iteration_space(
    _points: &[Point2D],
    _config: &VisualizationConfig,
    _output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("Visualization feature not enabled. Rebuild with --features visualization".into())
}

#[cfg(not(feature = "visualization"))]
pub fn visualize_dependencies(
    _points: &[Point2D],
    _edges: &[DependenceEdge],
    _config: &VisualizationConfig,
    _output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("Visualization feature not enabled. Rebuild with --features visualization".into())
}

#[cfg(not(feature = "visualization"))]
pub fn visualize_to_svg(
    _points: &[Point2D],
    _edges: &[DependenceEdge],
    _config: &VisualizationConfig,
) -> String {
    "<!-- Visualization feature not enabled -->".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }
}
