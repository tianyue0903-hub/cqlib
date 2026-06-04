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

//! Shared foundations for compiler layout algorithms.
//!
//! This module provides the stable support code needed by concrete layout
//! methods: circuit interaction analysis, a compiler-local physical graph view,
//! objective scoring, and result diagnostics.
//!
//! Concrete algorithms — `trivial_layout`, `greedy_layout`, `vf2_perfect_layout`,
//! `sabre_layout` — are implemented as independent functions on top of these
//! pieces, with workflow-level code deciding which function to call.

mod analysis;
mod greedy;
mod objective;
mod physical;
mod result;
mod sabre;
mod trivial;
mod vf2;
mod vf2_engine;

pub use analysis::{
    CircuitLayoutAnalysis, Interaction, InteractionGraph, analyze_circuit_for_layout,
};
pub use greedy::{greedy_layout, greedy_layout_prepared};
pub use objective::{LayoutObjective, LayoutScore};
pub use physical::{DistanceTable, PhysicalLayoutGraph, build_physical_layout_graph};
pub use result::{LayoutDiagnostics, LayoutResult};
pub use sabre::{sabre_layout, sabre_layout_prepared};
pub use trivial::{trivial_layout, trivial_layout_prepared};
pub use vf2::{
    Vf2EdgeRequirement, Vf2LayoutConfig, vf2_perfect_layout, vf2_perfect_layout_prepared,
};

#[cfg(test)]
mod greedy_test;

#[cfg(test)]
mod layout_test;

#[cfg(test)]
mod sabre_test;

#[cfg(test)]
mod vf2_test;
