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

pub mod canonicalize;
pub mod decompose;
pub mod layout;
pub mod rewrite;
pub mod transformer;

pub use canonicalize::{
    CanonicalizeConfig, CanonicalizeResult, Canonicalizer, canonicalize_circuit,
};
pub use layout::{
    CircuitLayoutAnalysis, DistanceTable, Interaction, InteractionGraph, LayoutDiagnostics,
    LayoutObjective, LayoutResult, LayoutScore, PhysicalLayoutGraph, Vf2EdgeRequirement,
    Vf2LayoutConfig, analyze_circuit_for_layout, build_physical_layout_graph, greedy_layout,
    greedy_layout_prepared, sabre_layout, sabre_layout_prepared, trivial_layout,
    trivial_layout_prepared, vf2_perfect_layout, vf2_perfect_layout_prepared,
};
pub use rewrite::{
    KnowledgeRewriteResult, KnowledgeRewriteStats, KnowledgeRewriter, RewriteConfig, RewriteMode,
    rewrite_circuit,
};
pub use transformer::{TransformResult, Transformer};
