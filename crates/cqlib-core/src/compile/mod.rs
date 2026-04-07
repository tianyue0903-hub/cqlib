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

//! Compile pipeline entry points.
//!
//! This namespace groups hardware-aware mapping and routing infrastructure that
//! transforms logical circuits into topology-compliant circuits.
//!
//! Main responsibilities:
//! - validation and error modeling for compile-time passes
//! - strict structural mapping (VF2)
//! - heuristic routing and remapping (SABRE)
//! - hybrid orchestration (`map_with_vf2_sabre`)
//!
//! Re-exports are intentionally centralized here so consumers can import compile
//! APIs from a stable module path.

/// Error types emitted by compile and mapping workflows.
pub mod error;
/// Shared gate-graph construction utilities for compile passes.
pub(crate) mod graph;
/// Mapping/routing algorithms and related data structures.
pub mod mapping;
/// Template-matching based optimization utilities.
pub mod optimization;
/// Shared flat-circuit preparation helpers for compile passes.
pub(crate) mod prepared;
/// Shared structured-program helpers for control-flow-aware compile passes.
pub(crate) mod structured;

pub use error::CompileError;
pub use mapping::{
    FidelityMap, GaConfig, GeneticAlgMapping, SabreConfig, SabreMapping, Vf2CandidateOptions,
    Vf2CandidateScore, Vf2LayoutCandidate, Vf2Mapping, Vf2Policy, Vf2ScoreWeights, map_with_ga,
    map_with_vf2_sabre,
};
pub use optimization::{
    CliffordRzConfig, CliffordRzLevel, CliffordRzOptimization, CliffordRzStrategy, TemplateLibrary,
    TemplateMatch, TemplateMatching, TemplateOptimization, TemplateOptimizationConfig,
    CommutativeOptimization,
};
