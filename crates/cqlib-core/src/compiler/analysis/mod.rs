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

//! Compiler analyses and cached derived views.
//!
//! The `analysis` module hosts reusable, read-only facts derived from the
//! current compiler state. Analyses are intended to support workflows and
//! transforms with fast prechecks, stable reporting, and shared cost-model
//! inputs without scattering one-off scans throughout the compiler pipeline.
//!
//! # Responsibilities
//!
//! Analyses in this module:
//! - derive information from the current circuit, CFG, device, or layout
//! - do not mutate compiler state
//! - can be cached by revision in [`AnalysisStore`]
//! - provide stable data for multiple passes or user-facing reports
//!
//! The owning [`crate::compiler::CompilerContext`] lazily builds these values and
//! stores them in a type-indexed cache keyed by the current compiler revision.
//! Whenever the circuit or target-dependent state changes, the context clears the
//! cache so analyses remain coherent with the active state.
//!
//! # Current Analyses
//!
//! - [`InstructionStats`]: circuit-wide operation category counts
//! - [`QubitUsage`]: per-qubit participation ranges and categories
//! - [`BlockSummary`]: CFG block summaries for block-local transforms
//! - [`CouplingRequirements`]: global and block-local logical 2-qubit demand
//! - [`BasisAnalysis`]: normalized instruction-family usage in the current circuit
//! - [`NativeSupportAnalysis`]: target-device native support diagnostics
//! - [`CostAnalysis`]: unified logical and target-aware cost estimates
//!
//! # When To Add A New Analysis
//!
//! A new analysis belongs in this module only if at least one of the following
//! is true:
//! - it is reused by two or more passes
//! - it is expensive enough that revision-based caching is worthwhile
//! - it must appear in stable external reporting
//! - it needs to be declared explicitly as a pass prerequisite
//!
//! One-off scans used only inside a single transform should stay local to that
//! transform instead of being promoted into `analysis/`.

pub mod basis;
pub mod block_summary;
pub mod cost;
pub mod coupling_requirements;
pub mod instruction_stats;
pub mod native_support;
pub mod qubit_usage;
pub mod store;

pub use basis::{AnalysisKey, BasisAnalysis, BasisEntry, BasisKey, ContextAnalysis};
pub use block_summary::{BlockSummary, BlockSummaryEntry};
pub use cost::{CostAnalysis, LogicalCost, PhysicalCostEstimate};
pub use coupling_requirements::{CouplingKey, CouplingRequirement, CouplingRequirements};
pub use instruction_stats::InstructionStats;
pub use native_support::{NativeSupportAnalysis, NativeSupportOpEntry, NativeSupportStatus};
pub use qubit_usage::{QubitUsage, QubitUsageSummary};
pub use store::AnalysisStore;
