// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! SABRE routing transform.
//!
//! This module adapts the compiler SABRE core into a transform-level routing
//! entry point. The algorithm implementation remains in
//! [`crate::compile::sabre`]; this layer owns only the public transform result
//! shape and the `route_sabre` API.

use crate::circuit::Circuit;
use crate::compile::CompilerError;
use crate::compile::sabre::{SabreConfig, SabreRoutingDiagnostics, sabre_layout_and_route};
use crate::compile::transform::layout::{LayoutObjective, LayoutScore};
use crate::device::{Device, Layout, LogicalQubit, PhysicalQubit};

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
/// This function first runs SABRE layout refinement, then routes the original
/// forward circuit from the selected initial layout. The returned circuit is
/// rebuilt over usable physical qubit identifiers and includes inserted
/// [`StandardGate::SWAP`] operations when the selected layout alone cannot
/// satisfy the physical topology.
///
/// Equal deterministic seeds in [`SabreConfig`] produce equal cqlib routing
/// results for the same circuit and device.
///
/// # Limitations
///
/// This transform does not perform target-basis lowering, directed native-gate
/// validation, or compiler workflow selection. Callers should run required
/// decomposition and basis translation passes explicitly.
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
    let changed = routing_changed(
        circuit,
        &result.routing.circuit,
        result.routing.swap_count,
        &result.routing.initial_layout,
    );

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

fn routing_changed(
    input: &Circuit,
    routed: &Circuit,
    swap_count: usize,
    initial_layout: &Layout,
) -> bool {
    // Any inserted SWAP, physical-qubit renumbering, or global-phase change is
    // observable at the transform-result level even if the operation sequence
    // would otherwise compare equal.
    if swap_count > 0
        || input.qubits() != routed.qubits()
        || input.global_phase() != routed.global_phase()
    {
        return true;
    }

    // A no-SWAP route can still change the representation when SABRE selected a
    // non-identity initial layout: operations are emitted on physical qubit ids.
    if !input.qubits().into_iter().all(|qubit| {
        initial_layout.get_physical(LogicalQubit::from_qubit(qubit))
            == Some(PhysicalQubit::from_qubit(qubit))
    }) {
        return true;
    }

    format!("{:?}", input.operations()) != format!("{:?}", routed.operations())
}

#[cfg(test)]
#[path = "./sabre_test.rs"]
mod routing_test;
