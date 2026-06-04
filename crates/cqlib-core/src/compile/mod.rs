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

pub mod commutation;
pub mod compiler;
pub mod error;
pub mod knowledge;
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
    SabreCompileResult, SabreConfig, SabreHeuristicConfig, SabreRoutingDiagnostics,
    SabreRoutingResult, sabre_layout_and_route, sabre_refine_layout, sabre_route,
};
pub use workflow::{CompilerWorkflow, WorkflowStepReport};
