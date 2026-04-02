// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! # Figure Visualization Backend
//!
//! This module renders circuits with a Rust-native SVG-first pipeline and optional PNG
//! rasterization (via `resvg`).
//!
//! ## Core Features
//!
//! - **SVG-first rendering**: directly generates scalable vector output.
//! - **Optional PNG export**: rasterizes SVG through `resvg` when bitmap output is needed.
//! - **Shared IR pipeline**: consumes [`VisualCircuit`](crate::visualization::VisualCircuit)
//!   built by the common visualization builder.
//! - **Style-map driven drawing**: gate colors/fonts/line styles come from `styles/default.json`
//!   with optional overrides.
//!
//! ## Typical Workflow
//!
//! 1. Build visualization IR from a circuit.
//! 2. Convert IR into SVG output.
//! 3. Save SVG directly or rasterize to PNG.
//!
//! ## Example
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Qubit};
//! use cqlib_core::visualization::{FigureDrawerOptions, circuit_to_figure};
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! let script = circuit_to_figure(&circuit, &FigureDrawerOptions::default()).unwrap();
//! assert!(script.contains("<svg"));
//! ```

use crate::circuit::Circuit;
use crate::visualization::circuit::builder::{VisualBuildOptions, build_visual_circuit};
use crate::visualization::circuit::error::VisualizationError;
use crate::visualization::circuit::ir_utils::{flatten_control_flow_visual, reverse_visual_lanes};
use crate::visualization::circuit::model::{VisualCircuit, VisualControlFlowKind, VisualOpStyle};
use crate::visualization::circuit::parameter_formatter::ParameterFormatOptions;
use crate::visualization::circuit::style::{GateStyle, StyleBook};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[path = "figure_helpers.rs"]
mod helpers;
use helpers::*;

/// Figure rendering theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FigureDrawStyle {
    /// Cqlib default style loaded from `styles/default.json`.
    Cqlib,
}

/// Options for figure drawing.
///
/// This configuration controls geometry, fold behavior, and text rendering.
/// The defaults are tuned for medium-sized circuits and readable SVG output.
///
/// # Layout Notes
///
/// - `gate_width`/`gate_height` define the base gate box size in logical data units.
/// - `moment_spacing` controls horizontal spacing between adjacent columns.
/// - `fold` controls row splitting (`-1` means no folding).
/// - `width_per_column`/`height_per_qubit` are final canvas scale factors.
#[derive(Debug, Clone)]
pub struct FigureDrawerOptions {
    /// Whether to append parameter text to gate labels.
    pub show_params: bool,
    /// Whether to decompose circuit-gates before drawing.
    pub decompose_circuit_gates: bool,
    /// Parameter display format used by visualization IR builder.
    pub parameter_format: ParameterFormatOptions,
    /// Figure width scale per logical column.
    pub width_per_column: f64,
    /// Figure height scale per qubit.
    pub height_per_qubit: f64,
    /// Rasterization DPI when exporting PNG.
    pub dpi: u32,
    /// Base gate width (data units), also used as minimum column width.
    pub gate_width: f64,
    /// Base gate height (data units).
    pub gate_height: f64,
    /// Horizontal spacing between adjacent columns.
    pub moment_spacing: f64,
    /// Vertical spacing between folded rows.
    pub connect_height: f64,
    /// Maximum columns per row (`-1` disables folding).
    pub fold: i32,
    /// Plot style preset.
    pub style: FigureDrawStyle,
    /// Optional per-gate style overrides (merged over base style map).
    pub gate_styles: HashMap<String, GateStyle>,
    /// Whether to show `|0>` in qubit labels.
    pub initial_state: bool,
    /// Whether to reverse display order of qubits.
    pub reverse_bits: bool,
}

impl Default for FigureDrawerOptions {
    fn default() -> Self {
        Self {
            show_params: true,
            decompose_circuit_gates: false,
            parameter_format: ParameterFormatOptions::default(),
            width_per_column: 1.2,
            height_per_qubit: 0.9,
            dpi: 160,
            gate_width: 1.1,
            gate_height: 1.5,
            moment_spacing: 0.3,
            connect_height: 2.0,
            fold: 18,
            style: FigureDrawStyle::Cqlib,
            gate_styles: HashMap::new(),
            initial_state: false,
            reverse_bits: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FigurePalette {
    wire_color: &'static str,
    connector_color: &'static str,
    text_color: &'static str,
    gate_edge_color: &'static str,
    gate_fill_color: Option<&'static str>,
    barrier_color: &'static str,
    wire_linewidth: f64,
    connector_linewidth: f64,
    gate_linewidth: f64,
    gate_fontsize: u8,
}

/// Minimum logical width reserved for an empty folded row.
const MIN_ROW_LOGICAL_WIDTH: f64 = 2.0;
/// Logical Y pitch between adjacent qubit wires.
const WIRE_PITCH: f64 = 2.0;
/// Left X bound of the drawing area (keeps qubit labels visible).
const CANVAS_MIN_X: f64 = -1.4;
/// Extra padding on the right side of the drawing area.
const CANVAS_RIGHT_PADDING: f64 = 0.3;
/// Symmetric Y padding around the drawing area.
const CANVAS_Y_PADDING: f64 = 1.0;
/// Base pixel scale per logical unit (further scaled by options).
const LOGICAL_UNIT_TO_PX: f64 = 80.0;
/// Inner text padding for gate labels (in pixels).
const LABEL_INNER_PADDING_PX: f64 = 10.0;
/// Minimum usable text area per direction (in pixels).
const LABEL_MIN_INNER_PX: f64 = 4.0;
/// Relative parameter font size against the gate label font.
const PARAM_FONT_SCALE: f64 = 0.78;
/// Approximate width factor for gate-name fitting.
const NAME_WIDTH_FACTOR: f64 = 0.60;
/// Approximate width factor for parameter-line fitting.
const PARAM_WIDTH_FACTOR: f64 = 0.56;
/// Relative vertical gap between gate name and parameter line.
const LABEL_LINE_GAP_SCALE: f64 = 0.22;
/// Maximum fitting iterations for label down-scaling.
const LABEL_FIT_MAX_ITERS: usize = 24;
/// Upper/lower clamp for each fitting step scale factor.
const LABEL_FIT_MAX_STEP: f64 = 0.95;
const LABEL_FIT_MIN_STEP: f64 = 0.10;
/// Module/generic span-gate label width estimator tuning.
const MODULE_LABEL_WIDTH_DIVISOR: f64 = 4.0;
const MODULE_LABEL_PADDING_THRESHOLD: f64 = 6.0;
const MODULE_LABEL_PADDING_CHARS: f64 = 1.0;
/// Extra headroom when packing columns into folded rows.
const FOLD_TARGET_SLACK: f64 = 1.12;

fn figure_palette(style: FigureDrawStyle) -> FigurePalette {
    match style {
        FigureDrawStyle::Cqlib => FigurePalette {
            wire_color: "black",
            connector_color: "black",
            text_color: "black",
            gate_edge_color: "black",
            gate_fill_color: Some("white"),
            barrier_color: "gray",
            wire_linewidth: 1.1,
            connector_linewidth: 1.0,
            gate_linewidth: 1.1,
            gate_fontsize: 9,
        },
    }
}

/// Generate SVG markup for a circuit.
///
/// # Arguments
///
/// * `circuit` - Input circuit to render.
/// * `options` - Figure rendering options.
///
/// # Errors
///
/// Returns [`VisualizationError`] when IR build fails (for example, unknown qubit references).
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::visualization::{FigureDrawerOptions, circuit_to_figure};
///
/// let mut circuit = Circuit::new(1);
/// circuit.h(Qubit::new(0)).unwrap();
///
/// let script = circuit_to_figure(&circuit, &FigureDrawerOptions::default()).unwrap();
/// assert!(script.contains("<svg"));
/// ```
pub fn circuit_to_figure(
    circuit: &Circuit,
    options: &FigureDrawerOptions,
) -> Result<String, VisualizationError> {
    let visual_options = VisualBuildOptions {
        decompose_circuit_gates: options.decompose_circuit_gates,
        parameter_format: options.parameter_format,
        ..VisualBuildOptions::default()
    };
    let visual = build_visual_circuit(circuit, &visual_options)?;
    Ok(draw_figure_svg_from_visual(&visual, options))
}

/// Render a circuit directly to an output file (`.svg` or `.png`).
///
/// # Arguments
///
/// * `circuit` - Input circuit to render.
/// * `output_path` - Target file path. `.svg` writes vector output, `.png` writes raster output.
/// * `options` - Figure rendering options.
///
/// # Errors
///
/// Returns [`VisualizationError`] when IR build, file writing, or PNG rasterization fails.
///
/// # Example
///
/// ```no_run
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::visualization::{FigureDrawerOptions, render_figure_to_file};
///
/// let mut circuit = Circuit::new(1);
/// circuit.h(Qubit::new(0)).unwrap();
///
/// render_figure_to_file(&circuit, "circuit.png", &FigureDrawerOptions::default()).unwrap();
/// ```
pub fn render_figure_to_file(
    circuit: &Circuit,
    output_path: &str,
    options: &FigureDrawerOptions,
) -> Result<(), VisualizationError> {
    let visual_options = VisualBuildOptions {
        decompose_circuit_gates: options.decompose_circuit_gates,
        parameter_format: options.parameter_format,
        ..VisualBuildOptions::default()
    };
    let visual = build_visual_circuit(circuit, &visual_options)?;
    let svg = draw_figure_svg_from_visual(&visual, options);
    let out_path = Path::new(output_path);
    match out_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => rasterize_svg_to_png_data(svg.as_bytes(), out_path),
        _ => fs::write(out_path, svg).map_err(VisualizationError::Io),
    }
}

fn rasterize_svg_to_png_data(svg_data: &[u8], png_path: &Path) -> Result<(), VisualizationError> {
    let mut options = resvg::usvg::Options::default();
    options.fontdb_mut().load_system_fonts();
    let tree = resvg::usvg::Tree::from_data(svg_data, &options)
        .map_err(|e| VisualizationError::SvgRenderFailed(e.to_string()))?;
    let size = tree.size().to_int_size();
    let mut pixmap =
        resvg::tiny_skia::Pixmap::new(size.width(), size.height()).ok_or_else(|| {
            VisualizationError::SvgRenderFailed("failed to allocate pixmap".to_string())
        })?;
    let mut pixmap_mut = pixmap.as_mut();
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::default(),
        &mut pixmap_mut,
    );
    pixmap
        .save_png(png_path)
        .map_err(|e| VisualizationError::SvgRenderFailed(e.to_string()))?;
    Ok(())
}

fn draw_figure_svg_from_visual(visual: &VisualCircuit, options: &FigureDrawerOptions) -> String {
    let mut visual_data = flatten_control_flow_visual(visual);
    if options.reverse_bits {
        visual_data = reverse_visual_lanes(visual_data);
    }

    let num_qubits = visual_data.num_qubits();
    let num_columns = visual_data.num_columns.max(1);
    let palette = figure_palette(options.style);
    let style_book = StyleBook::new("default", &options.gate_styles);

    let mut cols_ops: Vec<Vec<&_>> = vec![Vec::new(); num_columns];
    for op in &visual_data.operations {
        if op.column < num_columns {
            cols_ops[op.column].push(op);
        }
    }
    // Keep a uniform base slot width. Selected operations reserve extra width:
    // - module/unitary span gates with long labels,
    // - control-flow markers whose labels should not shrink.
    let mut col_widths = vec![options.gate_width; num_columns];
    for col in 0..num_columns {
        for op in &cols_ops[col] {
            if is_module_span_gate(op) {
                col_widths[col] = col_widths[col].max(module_span_column_width(
                    op,
                    options.show_params,
                    options.gate_width,
                ));
            } else if is_control_flow_box(op) {
                col_widths[col] =
                    col_widths[col].max(control_flow_column_width(op, options.gate_width));
            }
        }
    }

    let row_columns = split_columns_by_fold(
        &col_widths,
        options.fold,
        options.moment_spacing,
        options.gate_width,
    );
    let mut row_widths = Vec::with_capacity(row_columns.len());
    for row in &row_columns {
        if row.is_empty() {
            row_widths.push(MIN_ROW_LOGICAL_WIDTH);
            continue;
        }
        let mut width = options.moment_spacing;
        for (i, col) in row.iter().enumerate() {
            width += col_widths[*col];
            if i + 1 < row.len() {
                width += options.moment_spacing;
            }
        }
        width += options.moment_spacing;
        row_widths.push(width.max(MIN_ROW_LOGICAL_WIDTH));
    }
    let x_max = row_widths
        .iter()
        .fold(0.0f64, |acc, w| acc.max(*w))
        .max(MIN_ROW_LOGICAL_WIDTH);
    let qubits_height = if num_qubits == 0 {
        WIRE_PITCH
    } else {
        (num_qubits as f64 - 1.0) * WIRE_PITCH
    };
    let total_height = if row_columns.is_empty() {
        WIRE_PITCH
    } else {
        (row_columns.len() as f64 - 1.0) * (qubits_height + options.connect_height) + qubits_height
    };

    let min_x = CANVAS_MIN_X;
    let max_x = x_max + CANVAS_RIGHT_PADDING;
    let min_y = -CANVAS_Y_PADDING;
    let max_y = total_height + CANVAS_Y_PADDING;
    let sx = LOGICAL_UNIT_TO_PX * options.width_per_column;
    let sy = LOGICAL_UNIT_TO_PX * options.height_per_qubit;
    let canvas_w = ((max_x - min_x) * sx).max(1.0);
    let canvas_h = ((max_y - min_y) * sy).max(1.0);
    let px = |x: f64| (x - min_x) * sx;
    let py = |y: f64| (y - min_y) * sy;

    let mut elements = Vec::new();
    elements.push(format!(
        "<rect x=\"0\" y=\"0\" width=\"{:.3}\" height=\"{:.3}\" fill=\"#dcdcdc\"/>",
        canvas_w, canvas_h
    ));

    let wire_color = style_book
        .get("default")
        .line_color
        .as_deref()
        .unwrap_or(palette.wire_color);
    let default_text_color = style_book
        .get("default")
        .text_color
        .as_deref()
        .unwrap_or(palette.text_color);
    let global_text_fs = style_book
        .get("default")
        .font_size
        .unwrap_or(palette.gate_fontsize as f64)
        .clamp(8.0, 48.0);

    for (row_idx, row_cols) in row_columns.iter().enumerate() {
        // Keep all folded rows at a consistent visual width.
        let row_x_max = x_max;
        let y_base = row_idx as f64 * (qubits_height + options.connect_height);

        for (lane, qubit) in visual_data.qubits.iter().enumerate() {
            let y = lane_to_y(lane, y_base);
            let q_label = if options.initial_state {
                format!("q{} |0>", qubit.id())
            } else {
                format!("q{}", qubit.id())
            };
            elements.push(svg_line(
                px(0.0),
                py(y),
                px(row_x_max),
                py(y),
                wire_color,
                palette.wire_linewidth,
                None,
            ));
            elements.push(svg_text(
                px(-0.08),
                py(y),
                &q_label,
                global_text_fs,
                default_text_color,
                "end",
            ));
        }
        if row_idx > 0 && num_qubits > 0 {
            elements.push(svg_line(
                px(0.0),
                py(y_base),
                px(0.0),
                py(y_base + qubits_height),
                wire_color,
                palette.wire_linewidth * 1.3,
                None,
            ));
        }
        if row_idx + 1 < row_columns.len() && num_qubits > 0 {
            elements.push(svg_line(
                px(row_x_max),
                py(y_base),
                px(row_x_max),
                py(y_base + qubits_height),
                wire_color,
                palette.wire_linewidth * 1.3,
                None,
            ));
        }

        let mut x = options.moment_spacing;
        for (i, col) in row_cols.iter().enumerate() {
            let col_w = col_widths[*col];
            x += col_w / 2.0;
            let x_center = x;

            for op in &cols_ops[*col] {
                // Keep regular gate boxes fixed-size. Module and control-flow boxes can expand
                // according to column width to preserve readable labels.
                let op_w = if is_module_span_gate(op) || is_control_flow_box(op) {
                    col_w
                } else {
                    options.gate_width
                };
                let label = compose_label(&op.label, &op.params, options.show_params);
                let gate_style = style_book.get(op_style_key(op));
                let min_lane = op.covered_lanes.iter().copied().min();
                let max_lane = op.covered_lanes.iter().copied().max();
                let connector_color = gate_style
                    .line_color
                    .as_deref()
                    .unwrap_or(palette.connector_color);
                let connector_lw = gate_style.line_width.unwrap_or(palette.connector_linewidth);

                match op.style {
                    VisualOpStyle::Gate => {
                        if op.label == "FSIM" && op.lanes.len() >= 2 {
                            if let (Some(min_l), Some(max_l)) = (
                                op.lanes.iter().copied().min(),
                                op.lanes.iter().copied().max(),
                            ) {
                                if max_l > min_l {
                                    let y0 = lane_to_y(min_l, y_base);
                                    let y1 = lane_to_y(max_l, y_base);
                                    elements.push(svg_line(
                                        px(x_center),
                                        py(y0.min(y1)),
                                        px(x_center),
                                        py(y0.max(y1)),
                                        connector_color,
                                        connector_lw,
                                        None,
                                    ));
                                }
                            }
                            let r = (options.gate_width * 0.35 * sx.min(sy)).clamp(14.0, 28.0);
                            let circle_face = normalized_fill_color(gate_style, &palette)
                                .unwrap_or_else(|| "white".to_string());
                            let circle_edge = normalized_edge_color(gate_style, &palette)
                                .unwrap_or_else(|| connector_color.to_string());
                            let text_color = gate_style
                                .text_color
                                .as_deref()
                                .unwrap_or(palette.text_color);
                            // Keep FSIM text inside the circular marker.
                            let fsim_font = (r * 0.62).clamp(7.0, (global_text_fs * 0.9).max(7.0));
                            for lane in &op.lanes {
                                let y = lane_to_y(*lane, y_base);
                                elements.push(svg_circle(
                                    px(x_center),
                                    py(y),
                                    r,
                                    Some(&circle_face),
                                    Some(&circle_edge),
                                    connector_lw,
                                ));
                                elements.push(svg_text(
                                    px(x_center),
                                    py(y),
                                    "FSIM",
                                    fsim_font,
                                    text_color,
                                    "middle",
                                ));
                            }
                            continue;
                        }

                        if op.lanes.len() > 1 {
                            let gate_box_w = op_w;
                            let show_markers = show_span_lane_markers(op);
                            let marker_gutter = if show_markers {
                                (gate_box_w * 0.28).clamp(0.24, 0.42)
                            } else {
                                0.0
                            };
                            let start_lane = op.lanes.iter().copied().min().unwrap_or(0);
                            let end_lane = op.lanes.iter().copied().max().unwrap_or(start_lane);
                            let y0 = lane_to_y(start_lane, y_base);
                            let y1 = lane_to_y(end_lane, y_base);
                            let y_min = y0.min(y1) - options.gate_height / 2.0;
                            let box_h = (y0.max(y1) - y0.min(y1) + options.gate_height)
                                .max(options.gate_height);
                            elements.extend(draw_span_box_svg(
                                // Keep span-gate center strictly aligned to the moment center.
                                x_center,
                                y_min,
                                box_h,
                                &label,
                                &palette,
                                gate_style,
                                global_text_fs,
                                gate_box_w,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                            if show_markers {
                                // Keep lane markers inside the box with a small left inset.
                                let marker_x = x_center - gate_box_w / 2.0 + marker_gutter * 0.2;
                                let marker_font_size = global_text_fs;
                                for (idx, lane) in op.lanes.iter().enumerate() {
                                    let y = lane_to_y(*lane, y_base);
                                    elements.push(svg_text(
                                        px(marker_x),
                                        py(y),
                                        &idx.to_string(),
                                        marker_font_size,
                                        gate_style
                                            .text_color
                                            .as_deref()
                                            .unwrap_or(palette.text_color),
                                        "start",
                                    ));
                                }
                            }
                        } else {
                            let anchor = op.lanes.iter().copied().min().unwrap_or(0);
                            let y = lane_to_y(anchor, y_base);
                            elements.extend(draw_box_svg(
                                x_center,
                                y,
                                &label,
                                &palette,
                                gate_style,
                                global_text_fs,
                                op_w,
                                options.gate_height,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                        }
                    }
                    VisualOpStyle::Controlled { num_controls } => {
                        if let (Some(min_l), Some(max_l)) = (
                            op.lanes.iter().copied().min(),
                            op.lanes.iter().copied().max(),
                        ) {
                            if max_l > min_l {
                                let y0 = lane_to_y(min_l, y_base);
                                let y1 = lane_to_y(max_l, y_base);
                                elements.push(svg_line(
                                    px(x_center),
                                    py(y0.min(y1)),
                                    px(x_center),
                                    py(y0.max(y1)),
                                    connector_color,
                                    connector_lw,
                                    None,
                                ));
                            }
                        }
                        for lane in op.lanes.iter().take(num_controls) {
                            let y = lane_to_y(*lane, y_base);
                            elements.push(svg_circle(
                                px(x_center),
                                py(y),
                                (0.07 * sx.min(sy)).max(4.0),
                                Some(connector_color),
                                Some(connector_color),
                                connector_lw,
                            ));
                        }
                        for lane in op.lanes.iter().skip(num_controls) {
                            let y = lane_to_y(*lane, y_base);
                            elements.extend(draw_box_svg(
                                x_center,
                                y,
                                &label,
                                &palette,
                                gate_style,
                                global_text_fs,
                                op_w,
                                options.gate_height,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                        }
                    }
                    VisualOpStyle::Cz => {
                        if let (Some(min_l), Some(max_l)) = (
                            op.lanes.iter().copied().min(),
                            op.lanes.iter().copied().max(),
                        ) {
                            if max_l > min_l {
                                let y0 = lane_to_y(min_l, y_base);
                                let y1 = lane_to_y(max_l, y_base);
                                elements.push(svg_line(
                                    px(x_center),
                                    py(y0.min(y1)),
                                    px(x_center),
                                    py(y0.max(y1)),
                                    connector_color,
                                    connector_lw,
                                    None,
                                ));
                            }
                        }
                        for lane in &op.lanes {
                            let y = lane_to_y(*lane, y_base);
                            elements.push(svg_circle(
                                px(x_center),
                                py(y),
                                (0.07 * sx.min(sy)).max(4.0),
                                Some(connector_color),
                                Some(connector_color),
                                connector_lw,
                            ));
                        }
                    }
                    VisualOpStyle::Swap => {
                        let swap_lw = connector_lw * 1.8;
                        if let (Some(min_l), Some(max_l)) = (
                            op.lanes.iter().copied().min(),
                            op.lanes.iter().copied().max(),
                        ) {
                            if max_l > min_l {
                                let y0 = lane_to_y(min_l, y_base);
                                let y1 = lane_to_y(max_l, y_base);
                                elements.push(svg_line(
                                    px(x_center),
                                    py(y0.min(y1)),
                                    px(x_center),
                                    py(y0.max(y1)),
                                    connector_color,
                                    swap_lw,
                                    None,
                                ));
                            }
                        }
                        for lane in &op.lanes {
                            let y = lane_to_y(*lane, y_base);
                            elements.push(svg_line(
                                px(x_center - 0.15),
                                py(y - 0.15),
                                px(x_center + 0.15),
                                py(y + 0.15),
                                connector_color,
                                swap_lw,
                                None,
                            ));
                            elements.push(svg_line(
                                px(x_center - 0.15),
                                py(y + 0.15),
                                px(x_center + 0.15),
                                py(y - 0.15),
                                connector_color,
                                swap_lw,
                                None,
                            ));
                        }
                    }
                    VisualOpStyle::Barrier => {
                        let barrier_lw =
                            gate_style.line_width.unwrap_or(palette.gate_linewidth) * 1.8;
                        let (start_lane, end_lane) =
                            if let (Some(min_l), Some(max_l)) = (min_lane, max_lane) {
                                (min_l, max_l)
                            } else if num_qubits > 0 {
                                (0, num_qubits - 1)
                            } else {
                                (0, 0)
                            };
                        let y0 = lane_to_y(start_lane, y_base);
                        let y1 = lane_to_y(end_lane, y_base);
                        elements.push(svg_line(
                            px(x_center),
                            py(y0.min(y1) - options.gate_height / 2.0),
                            px(x_center),
                            py(y0.max(y1) + options.gate_height / 2.0),
                            gate_style
                                .line_color
                                .as_deref()
                                .unwrap_or(palette.barrier_color),
                            barrier_lw,
                            Some("6,4"),
                        ));
                    }
                    VisualOpStyle::Measure => {
                        if op.lanes.is_empty() {
                            elements.extend(draw_measure_svg(
                                x_center,
                                lane_to_y(0, y_base),
                                &palette,
                                gate_style,
                                global_text_fs,
                                op_w,
                                options.gate_height,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                        } else {
                            for lane in &op.lanes {
                                elements.extend(draw_measure_svg(
                                    x_center,
                                    lane_to_y(*lane, y_base),
                                    &palette,
                                    gate_style,
                                    global_text_fs,
                                    op_w,
                                    options.gate_height,
                                    sx,
                                    sy,
                                    &px,
                                    &py,
                                ));
                            }
                        }
                    }
                    VisualOpStyle::Reset | VisualOpStyle::Delay => {
                        if op.lanes.is_empty() {
                            elements.extend(draw_box_svg(
                                x_center,
                                lane_to_y(0, y_base),
                                &label,
                                &palette,
                                gate_style,
                                global_text_fs,
                                op_w,
                                options.gate_height,
                                sx,
                                sy,
                                &px,
                                &py,
                            ));
                        } else {
                            for lane in &op.lanes {
                                elements.extend(draw_box_svg(
                                    x_center,
                                    lane_to_y(*lane, y_base),
                                    &label,
                                    &palette,
                                    gate_style,
                                    global_text_fs,
                                    op_w,
                                    options.gate_height,
                                    sx,
                                    sy,
                                    &px,
                                    &py,
                                ));
                            }
                        }
                    }
                    VisualOpStyle::ControlFlow { kind } => {
                        let start_lane = min_lane
                            .or_else(|| op.lanes.iter().copied().min())
                            .unwrap_or(0);
                        let end_lane = max_lane
                            .or_else(|| op.lanes.iter().copied().max())
                            .unwrap_or(start_lane);
                        let y0 = lane_to_y(start_lane, y_base);
                        let y1 = lane_to_y(end_lane, y_base);
                        let y_min = y0.min(y1) - options.gate_height / 2.0;
                        let box_h = (y0.max(y1) - y0.min(y1) + options.gate_height)
                            .max(options.gate_height);
                        elements.extend(draw_flow_box_svg(
                            x_center,
                            y_min,
                            box_h,
                            &label,
                            &palette,
                            gate_style,
                            global_text_fs,
                            op_w,
                            sx,
                            sy,
                            &px,
                            &py,
                        ));
                        if matches!(
                            kind,
                            VisualControlFlowKind::IfElseBlock {
                                has_false_branch: true,
                                ..
                            }
                        ) {
                            elements.push(svg_text(
                                px(x_center + op_w / 2.0 + 0.08),
                                py(y_min + box_h - 0.12),
                                "else",
                                global_text_fs,
                                gate_style
                                    .text_color
                                    .as_deref()
                                    .unwrap_or(palette.text_color),
                                "start",
                            ));
                        }
                    }
                }
            }

            x += col_w / 2.0;
            if i + 1 < row_cols.len() {
                x += options.moment_spacing;
            }
        }
    }

    let mut out = String::new();
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" viewBox=\"0 0 {:.3} {:.3}\">",
        canvas_w, canvas_h, canvas_w, canvas_h
    ));
    for e in elements {
        out.push_str(&e);
    }
    out.push_str("</svg>");
    out
}

/// Generate SVG markup from pre-built visual IR.
///
/// This method is useful when you want to cache or transform visualization IR once and render
/// it multiple times with different backends/options.
pub fn draw_figure_from_visual(
    visual: &VisualCircuit,
    options: &FigureDrawerOptions,
    _output_path: Option<&str>,
) -> String {
    draw_figure_svg_from_visual(visual, options)
}

#[cfg(test)]
#[path = "figure_tests.rs"]
mod tests;
