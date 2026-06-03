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

//! Device-aware routing transforms.
//!
//! This module exposes transform-level routing entry points. The SABRE
//! algorithm itself lives in [`crate::compiler::sabre`]; this layer defines the
//! routing transform contract and result shape used by compiler callers.

use crate::circuit::{Circuit, Instruction, Operation, StandardGate};
use crate::compiler::CompilerError;
use crate::compiler::sabre::{SabreConfig, SabreRoutingDiagnostics, sabre_layout_and_route};
use crate::compiler::transform::layout::{LayoutObjective, LayoutScore};
use crate::device::{Device, Layout};

/// Routed circuit and transform-level routing metadata.
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Physical circuit with inserted SWAP operations.
    pub circuit: Circuit,
    /// Whether routing changed the circuit representation.
    pub changed: bool,
    /// Initial logical-to-physical layout selected by SABRE.
    pub initial_layout: Layout,
    /// Final logical-to-physical layout after routing.
    pub final_layout: Layout,
    /// Score of the selected initial layout.
    pub layout_score: Option<LayoutScore>,
    /// Number of inserted SWAP operations.
    pub swap_count: usize,
    /// Diagnostics describing SABRE routing behavior.
    pub diagnostics: SabreRoutingDiagnostics,
}

/// Selects a SABRE initial layout and routes `circuit` for `device`.
///
/// The returned circuit is rebuilt over usable physical qubit identifiers and
/// includes inserted [`StandardGate::SWAP`] operations when the selected layout
/// alone cannot satisfy the physical topology.
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] for invalid SABRE configuration,
/// insufficient usable physical qubits, unreachable interactions in the usable
/// topology, or unsupported circuit operations such as undecomposed gates that
/// touch more than two qubits.
pub fn route_sabre(
    circuit: &Circuit,
    device: &Device,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<RoutingResult, CompilerError> {
    let result = sabre_layout_and_route(circuit, device, objective, config)?;
    let changed = circuit_changed_by_routing(circuit, &result.routing.circuit);

    Ok(RoutingResult {
        circuit: result.routing.circuit,
        changed,
        initial_layout: result.routing.initial_layout,
        final_layout: result.routing.final_layout,
        layout_score: result.layout_score,
        swap_count: result.routing.swap_count,
        diagnostics: result.routing.diagnostics,
    })
}

fn circuit_changed_by_routing(input: &Circuit, routed: &Circuit) -> bool {
    input.qubits() != routed.qubits()
        || input.global_phase() != routed.global_phase()
        || input.operations().len() != routed.operations().len()
        || input
            .operations()
            .iter()
            .zip(routed.operations())
            .any(|(left, right)| operation_changed_by_routing(left, right))
        || routed.operations().iter().any(is_swap)
}

fn operation_changed_by_routing(left: &Operation, right: &Operation) -> bool {
    left.qubits != right.qubits
        || format!("{:?}", left.params) != format!("{:?}", right.params)
        || left.label != right.label
        || format!("{:?}", left.instruction) != format!("{:?}", right.instruction)
}

fn is_swap(operation: &Operation) -> bool {
    matches!(
        operation.instruction,
        Instruction::Standard(StandardGate::SWAP)
    )
}

#[cfg(test)]
#[path = "./sabre_test.rs"]
mod routing_test;
