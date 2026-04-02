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

use crate::circuit::PyCircuit;
use cqlib_core::visualization::{
    FigureDrawerOptions, TextDrawerOptions, circuit_to_figure, circuit_to_text,
    render_figure_to_file,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Draw circuit as unicode text diagram.
#[pyfunction(
    name = "draw_text",
    signature = (
        circuit,
        *,
        line_width = None,
        initial_state = false,
        reverse_bits = false,
        show_params = true,
        decompose_circuit_gates = false
    )
)]
pub fn py_draw_text(
    circuit: &PyCircuit,
    line_width: Option<isize>,
    initial_state: bool,
    reverse_bits: bool,
    show_params: bool,
    decompose_circuit_gates: bool,
) -> PyResult<String> {
    let mut options = TextDrawerOptions {
        initial_state,
        reverse_bits,
        show_params,
        decompose_circuit_gates,
        ..TextDrawerOptions::default()
    };
    if let Some(width) = line_width {
        options.line_width = width;
    }

    circuit_to_text(&circuit.inner, &options)
        .map_err(|e| PyValueError::new_err(format!("text visualization error: {e}")))
}

/// Draw circuit as SVG string.
#[pyfunction(
    name = "draw_figure",
    signature = (
        circuit,
        *,
        fold = None,
        initial_state = false,
        reverse_bits = false,
        show_params = true,
        decompose_circuit_gates = false,
        output_path = None
    )
)]
pub fn py_draw_figure(
    circuit: &PyCircuit,
    fold: Option<i32>,
    initial_state: bool,
    reverse_bits: bool,
    show_params: bool,
    decompose_circuit_gates: bool,
    output_path: Option<&str>,
) -> PyResult<String> {
    let mut options = FigureDrawerOptions {
        initial_state,
        reverse_bits,
        show_params,
        decompose_circuit_gates,
        ..FigureDrawerOptions::default()
    };
    if let Some(fold_value) = fold {
        options.fold = fold_value;
    }

    let svg = circuit_to_figure(&circuit.inner, &options)
        .map_err(|e| PyValueError::new_err(format!("figure visualization error: {e}")))?;

    if let Some(path) = output_path {
        render_figure_to_file(&circuit.inner, path, &options)
            .map_err(|e| PyValueError::new_err(format!("figure render error: {e}")))?;
    }

    Ok(svg)
}
