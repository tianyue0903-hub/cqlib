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

//! Trivial initial layout.

use super::{
    CircuitLayoutAnalysis, LayoutDiagnostics, LayoutObjective, LayoutResult, PhysicalLayoutGraph,
    build_physical_layout_graph,
};
use crate::compiler::CompilerError;
use crate::device::{Device, Layout};

/// Maps logical qubits to usable physical qubits in their existing order.
///
/// This layout method deliberately does not optimize topology, direction, or
/// calibration data. It is useful as a deterministic baseline and for circuits
/// whose logical order already matches the target device.
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] if the circuit has more logical
/// qubits than the device has usable physical qubits, or if scoring rejects the
/// resulting layout.
pub fn trivial_layout(
    analysis: &CircuitLayoutAnalysis,
    device: &Device,
    objective: &LayoutObjective,
) -> Result<LayoutResult, CompilerError> {
    let physical = build_physical_layout_graph(device)?;
    trivial_layout_with_physical(analysis, &physical, objective)
}

/// Maps logical qubits to an already-built physical graph in their existing order.
///
/// This advanced entry point is useful when a workflow evaluates multiple
/// layout algorithms against the same device and wants to reuse the derived
/// physical graph instead of rebuilding the distance table each time.
pub fn trivial_layout_with_physical(
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
) -> Result<LayoutResult, CompilerError> {
    if analysis.logical_qubits.len() > physical.physical_qubits().len() {
        return Err(CompilerError::InvalidInput(format!(
            "trivial layout requires at least as many usable physical qubits as logical qubits; got {} logical qubits and {} usable physical qubits",
            analysis.logical_qubits.len(),
            physical.physical_qubits().len()
        )));
    }

    let layout = Layout::new(
        analysis.logical_qubits.clone(),
        physical.physical_qubits().to_vec(),
        None,
    )
    .map_err(|error| {
        CompilerError::InvariantViolation(format!(
            "trivial layout failed to construct a valid layout: {error}"
        ))
    })?;

    let score = objective.score_layout(analysis, physical, &layout)?;
    let is_perfect = analysis
        .interactions
        .interactions()
        .iter()
        .all(|interaction| {
            let Some(left) = layout.get_physical(interaction.left) else {
                return false;
            };
            let Some(right) = layout.get_physical(interaction.right) else {
                return false;
            };
            physical.is_adjacent_undirected(left, right)
        });

    let diagnostics = LayoutDiagnostics {
        is_perfect,
        candidates_evaluated: 1,
        used_fidelity: score.used_fidelity,
        notes: Vec::new(),
    };

    Ok(LayoutResult {
        layout,
        score: Some(score),
        diagnostics,
    })
}
