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

//! SABRE initial layout.

use super::{
    CircuitLayoutAnalysis, LayoutObjective, LayoutResult, PhysicalLayoutGraph,
    analyze_circuit_for_layout, build_physical_layout_graph,
};
use crate::circuit::Circuit;
use crate::compiler::CompilerError;
use crate::compiler::sabre::{SabreConfig, sabre_refine_layout_prepared};
use crate::device::Device;

/// Selects an initial layout with SABRE forward/backward refinement.
///
/// This function only returns the refined initial layout. It does not insert
/// SWAP operations or rebuild a physical circuit; callers that need routing
/// should use the SABRE routing entry points after selecting a layout.
///
/// # Errors
///
/// Returns [`CompilerError::InvalidInput`] for invalid SABRE configuration,
/// insufficient usable physical qubits, unreachable interactions in the usable
/// topology, or unsupported circuit operations.
pub fn sabre_layout(
    circuit: &Circuit,
    device: &Device,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<LayoutResult, CompilerError> {
    let analysis = analyze_circuit_for_layout(circuit)?;
    let physical = build_physical_layout_graph(device)?;
    sabre_layout_prepared(circuit, &analysis, &physical, objective, config)
}

/// Selects a SABRE initial layout from already-prepared layout analysis.
///
/// The original circuit is still required because SABRE refinement uses the
/// operation dependency order to run trial routing. `analysis` and `physical`
/// are reused for scoring and physical graph facts.
pub fn sabre_layout_prepared(
    circuit: &Circuit,
    analysis: &CircuitLayoutAnalysis,
    physical: &PhysicalLayoutGraph,
    objective: &LayoutObjective,
    config: &SabreConfig,
) -> Result<LayoutResult, CompilerError> {
    sabre_refine_layout_prepared(circuit, analysis, physical, None, objective, config)
}
