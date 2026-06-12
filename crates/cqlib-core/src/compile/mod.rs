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

//! Compiler pipeline for lowering and optimizing quantum circuits.
//!
//! This module is the public entry point for cqlib circuit compilation. It
//! takes a logical [`Circuit`](crate::circuit::Circuit), applies a deterministic
//! staged lowering workflow, and returns a rebuilt circuit together with
//! step-level diagnostics.
//!
//! The compiler workflow is intentionally a staged lowering pipeline rather
//! than a dynamic pass manager. High-level circuit representations are lowered
//! through stable compiler layers:
//!
//! ```text
//! logical circuit
//!   -> canonicalized logical IR
//!   -> expanded circuit-backed definitions
//!   -> decomposed unitary and multi-controlled gates
//!   -> knowledge-rule optimization
//!   -> optional physical layout and SABRE routing
//!   -> optional target-basis translation
//!   -> canonicalized output
//! ```
//!
//! # Public Entry Points
//!
//! - [`compile`] is the recommended user-facing API.
//! - [`CompileConfig`] describes optimization effort, target constraints,
//!   physical-device constraints, optional initial layout, resource policy, and
//!   the heuristic routing seed.
//! - [`CompileResult`] returns the compiled circuit and the workflow step
//!   report.
//! - [`CompilerWorkflow`] is useful when callers want to construct and inspect
//!   a workflow explicitly.
//!
//! Lower-level modules such as [`transform`], [`sabre`], [`knowledge`], and
//! [`commutation`] expose reusable compiler infrastructure. They are intended
//! for advanced users and internal composition; ordinary compilation should
//! start with [`compile`].
//!
//! # Target Constraints
//!
//! Target-basis constraints are resolved before transforms run. An explicit
//! [`CompileConfig::target_basis`] takes precedence over native gates declared
//! by [`CompileConfig::device`]. If neither is present, target-basis lowering
//! is skipped.
//!
//! Device constraints serve a separate purpose. A configured device provides
//! usable-qubit capacity, topology for layout/routing, and optionally native
//! gates for target-basis translation. The compiler currently guarantees
//! undirected physical adjacency after routing; final directed-coupling
//! legalization is a separate compiler concern.
//!
//! [`CompileConfig::initial_layout`] may be used with a device to skip
//! automatic initial-layout selection and route from a caller-supplied
//! logical-to-physical mapping. Without a device, an initial layout is invalid.
//! [`CompileConfig::seed`] affects only heuristic device layout/routing. When
//! an initial layout is supplied, the seed still controls routing trials but no
//! automatic layout candidates are generated.
//!
//! # Classical Control and High-Level Operations
//!
//! The workflow preserves classical-control structure. Transforms that support
//! control-flow bodies recurse into them and report whether they changed the IR
//! through [`TransformResult`](transform::TransformResult). The workflow does
//! not pre-scan control-flow trees to decide whether a transform should run.
//! This module does not currently lower dynamic classical control into a
//! hardware runtime instruction format.
//!
//! # Step Reports
//!
//! [`WorkflowStepReport`] records the workflow-local stage and step name, plus
//! whether the step changed the circuit, was skipped, or emitted a short
//! reason. Step names describe workflow positions such as `route.sabre` or
//! `translate.target_basis`; they are not required to equal
//! [`Transformer::name`](transform::Transformer::name).
//!
//! # Examples
//!
//! Compile a logical circuit with default logical optimization:
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Qubit};
//! use cqlib_core::compile::{CompileConfig, CompileMode, compile};
//! use cqlib_core::compile::resource::ResourcePolicy;
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! let result = compile(
//!     &circuit,
//!     CompileConfig {
//!         mode: CompileMode::Normal,
//!         target_basis: None,
//!         device: None,
//!         initial_layout: None,
//!         resource_policy: ResourcePolicy::default(),
//!         seed: None,
//!     },
//! )
//! .unwrap();
//!
//! assert_eq!(result.mode, CompileMode::Normal);
//! assert!(!result.steps.is_empty());
//! ```
//!
//! Compile to an explicit target basis:
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Instruction, Qubit, StandardGate};
//! use cqlib_core::compile::{CompileConfig, CompileMode, compile};
//! use cqlib_core::compile::resource::ResourcePolicy;
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! let result = compile(
//!     &circuit,
//!     CompileConfig {
//!         mode: CompileMode::Normal,
//!         target_basis: Some(vec![
//!             Instruction::Standard(StandardGate::H),
//!             Instruction::Standard(StandardGate::CZ),
//!         ]),
//!         device: None,
//!         initial_layout: None,
//!         resource_policy: ResourcePolicy::default(),
//!         seed: None,
//!     },
//! )
//! .unwrap();
//!
//! assert!(
//!     result
//!         .steps
//!         .iter()
//!         .any(|step| step.name == "translate.target_basis" && step.changed)
//! );
//! ```
//!
//! Compile for a device topology:
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Qubit};
//! use cqlib_core::compile::{CompileConfig, CompileMode, compile};
//! use cqlib_core::compile::resource::ResourcePolicy;
//! use cqlib_core::device::Device;
//!
//! let mut circuit = Circuit::new(3);
//! circuit.cx(Qubit::new(0), Qubit::new(2)).unwrap();
//!
//! let result = compile(
//!     &circuit,
//!     CompileConfig {
//!         mode: CompileMode::Normal,
//!         target_basis: None,
//!         device: Some(Device::line("line-3", 3).unwrap()),
//!         initial_layout: None,
//!         resource_policy: ResourcePolicy::default(),
//!         seed: Some(7),
//!     },
//! )
//! .unwrap();
//!
//! assert!(
//!     result
//!         .steps
//!         .iter()
//!         .any(|step| step.name == "route.sabre" && !step.skipped)
//! );
//! ```
//!
//! Route from a supplied initial layout:
//!
//! ```rust
//! use cqlib_core::circuit::{Circuit, Qubit};
//! use cqlib_core::compile::{CompileConfig, CompileMode, compile};
//! use cqlib_core::compile::resource::ResourcePolicy;
//! use cqlib_core::device::{Device, Layout};
//!
//! let mut circuit = Circuit::new(1);
//! circuit.h(Qubit::new(0)).unwrap();
//! let layout = Layout::from_pairs(&[(0, 2)], 3).unwrap();
//!
//! let result = compile(
//!     &circuit,
//!     CompileConfig {
//!         mode: CompileMode::Normal,
//!         target_basis: None,
//!         device: Some(Device::line("line-3", 3).unwrap()),
//!         initial_layout: Some(layout),
//!         resource_policy: ResourcePolicy::default(),
//!         seed: Some(11),
//!     },
//! )
//! .unwrap();
//!
//! assert_eq!(result.circuit.operations()[0].qubits.as_slice(), &[Qubit::new(2)]);
//! ```
//!
//! Inspect workflow step reports:
//!
//! ```rust
//! use cqlib_core::circuit::Circuit;
//! use cqlib_core::compile::{CompileConfig, CompileMode, compile};
//! use cqlib_core::compile::resource::ResourcePolicy;
//!
//! let result = compile(
//!     &Circuit::new(1),
//!     CompileConfig {
//!         mode: CompileMode::Enhanced,
//!         target_basis: None,
//!         device: None,
//!         initial_layout: None,
//!         resource_policy: ResourcePolicy::default(),
//!         seed: None,
//!     },
//! )
//! .unwrap();
//!
//! let routing = result
//!     .steps
//!     .iter()
//!     .find(|step| step.name == "route.sabre")
//!     .unwrap();
//! assert!(routing.skipped);
//! ```

pub mod commutation;
pub mod compiler;
pub mod error;
pub mod knowledge;
pub mod physical_target;
pub mod resource;
pub mod sabre;
pub mod transform;
pub mod workflow;

/// Tolerance for proving equality between compiler parameter expressions.
pub(crate) const PARAMETER_EQ_TOLERANCE: f64 = 1e-12;

/// Tolerance for treating a scalar as numerically zero.
pub(crate) const NUMERIC_ZERO_TOLERANCE: f64 = 1e-14;

/// Tolerance for checking whether a candidate phase ratio has unit norm.
pub(crate) const UNIT_PHASE_NORM_TOLERANCE: f64 = 1e-8;

pub use commutation::{
    Commutation, CommutationChecker, CommutationConfig, CommutationResult, algebraic_commutation,
    check_commutation,
};
pub use compiler::{CompileConfig, CompileMode, CompileResult, compile};
pub use error::CompilerError;
pub use sabre::{
    SabreConfig, SabreHeuristicConfig, SabreRoutingDiagnostics, SabreRoutingResult,
    normalize_initial_layout, sabre_route, validate_config, validate_reachable_interactions,
};
pub use workflow::{CompilerWorkflow, WorkflowStepReport};
