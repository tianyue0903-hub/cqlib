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

//! # Unicode Text Visualization Backend
//!
//! This module renders circuits as UTF-8 text diagrams using box-drawing characters.
//! It supports:
//! - layered operation layout,
//! - multi-qubit connectors and span boxes,
//! - control-flow markers (`If/Else/While/End`),
//! - optional line wrapping with continuation arrows.
//!
//! ## Example
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Qubit};
//! use cqlib_core::visualization::{TextDrawerOptions, circuit_to_text};
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! let text = circuit_to_text(&circuit, &TextDrawerOptions::default()).unwrap();
//! assert!(text.contains("H"));
//! assert!(text.contains("■"));
//! ```

use crate::circuit::Circuit;
use crate::visualization::circuit::builder::{VisualBuildOptions, build_visual_circuit};
use crate::visualization::circuit::error::VisualizationError;
use crate::visualization::circuit::ir_utils::{flatten_control_flow_visual, reverse_visual_lanes};
use crate::visualization::circuit::model::{
    VisualCircuit, VisualControlFlowKind, VisualOpStyle, VisualOperation,
};
#[path = "text_helpers.rs"]
mod helpers;
use helpers::*;

const MIN_LINE_WIDTH: isize = 10;
const DEFAULT_LINE_WIDTH: isize = 80;

struct BoxChar;

#[allow(dead_code)]
impl BoxChar {
    const TOP: &'static str = "╵";
    const BOTTOM: &'static str = "╷";
    const LEFT: &'static str = "╴";
    const RIGHT: &'static str = "╶";
    const TOP_BOTTOM: &'static str = "│";
    const LEFT_RIGHT: &'static str = "─";

    const TOP_LEFT: &'static str = "┘";
    const TOP_RIGHT: &'static str = "└";
    const BOTTOM_LEFT: &'static str = "┐";
    const BOTTOM_RIGHT: &'static str = "┌";

    const TOP_BOTTOM_LEFT: &'static str = "┤";
    const TOP_BOTTOM_RIGHT: &'static str = "├";
    const TOP_LEFT_RIGHT: &'static str = "┴";
    const BOTTOM_LEFT_RIGHT: &'static str = "┬";
    const TOP_BOTTOM_LEFT_RIGHT: &'static str = "┼";

    const DOT: &'static str = "■";
    const CONNECT: &'static str = "X";
    const LEFT_ARROW: &'static str = "«";
    const RIGHT_ARROW: &'static str = "»";
}

/// Options for Unicode text drawing.
///
/// # Example
///
/// ```rust
/// use cqlib_core::visualization::TextDrawerOptions;
///
/// let options = TextDrawerOptions {
///     show_params: true,
///     line_width: 80,
///     initial_state: true,
///     reverse_bits: false,
///     ..TextDrawerOptions::default()
/// };
/// assert!(options.initial_state);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TextDrawerOptions {
    /// Kept for API compatibility. Rendering width is dynamic by content.
    pub cell_width: usize,
    /// Whether to append parameter text to gate labels.
    pub show_params: bool,
    /// Whether to decompose circuit-gates before drawing.
    pub decompose_circuit_gates: bool,
    /// Max width per wrapped segment. Set `-1` to disable wrapping.
    pub line_width: isize,
    /// Whether to show `|0>` at the beginning of each qubit wire.
    pub initial_state: bool,
    /// Whether to reverse the displayed qubit order.
    pub reverse_bits: bool,
}

impl Default for TextDrawerOptions {
    fn default() -> Self {
        Self {
            cell_width: 9,
            show_params: true,
            decompose_circuit_gates: false,
            line_width: DEFAULT_LINE_WIDTH,
            initial_state: false,
            reverse_bits: false,
        }
    }
}

/// Draw a circuit as Unicode text.
///
/// # Example
///
/// ```rust
/// use cqlib_core::circuit::{Circuit, Qubit};
/// use cqlib_core::visualization::{TextDrawerOptions, circuit_to_text};
///
/// let mut circuit = Circuit::new(1);
/// circuit.h(Qubit::new(0)).unwrap();
///
/// let text = circuit_to_text(&circuit, &TextDrawerOptions::default()).unwrap();
/// assert!(text.contains("Q0:"));
/// ```
pub fn circuit_to_text(
    circuit: &Circuit,
    options: &TextDrawerOptions,
) -> Result<String, VisualizationError> {
    let visual_options = VisualBuildOptions {
        decompose_circuit_gates: options.decompose_circuit_gates,
        ..VisualBuildOptions::default()
    };
    let visual = build_visual_circuit(circuit, &visual_options)?;
    draw_text_from_visual(&visual, options)
}

/// Draw from pre-built visual IR.
///
/// Use this API when you already have cached [`VisualCircuit`] IR.
pub fn draw_text_from_visual(
    visual: &VisualCircuit,
    options: &TextDrawerOptions,
) -> Result<String, VisualizationError> {
    if visual.num_qubits() == 0 {
        return Ok("empty circuit".to_string());
    }

    let mut flattened = flatten_control_flow_visual(visual);
    if options.reverse_bits {
        flattened = reverse_visual_lanes(flattened);
    }
    let lines = make_lines(&flattened, options.show_params, options.initial_state);
    let lines_count = total_rows(flattened.num_qubits());
    let max_line_width = effective_line_width(options);

    let start_qubits = lines[0].clone();
    let mut current_data = start_qubits.clone();
    let mut current_width = 0usize;
    let mut data: Vec<Vec<Vec<String>>> = Vec::new();

    for moment in lines.iter().skip(1) {
        let moment_len = str_len(&moment[0].concat());
        if max_line_width != usize::MAX && moment_len.saturating_add(current_width) > max_line_width
        {
            for row in &mut current_data {
                row.push(BoxChar::RIGHT_ARROW.to_string());
            }
            data.push(current_data);

            let mut next_data = Vec::with_capacity(lines_count);
            for (i, sq) in start_qubits.iter().enumerate().take(lines_count) {
                let mut row = vec![BoxChar::LEFT_ARROW.to_string()];
                row.extend(sq.clone());
                next_data.push(row);
                let _ = i;
            }
            current_data = next_data;
            current_width = 0;
        }

        for (i, mrow) in moment.iter().enumerate().take(lines_count) {
            current_data[i].extend(mrow.clone());
        }
        current_width = current_width.saturating_add(moment_len);
    }
    data.push(current_data);

    let block_strings: Vec<String> = data
        .into_iter()
        .map(|block| {
            block
                .into_iter()
                .map(|row| row.concat())
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect();
    Ok(block_strings.join("\n\n"))
}