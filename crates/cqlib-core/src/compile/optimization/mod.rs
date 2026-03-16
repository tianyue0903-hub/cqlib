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

//! Compile-layer optimization algorithms.
//!
//! This namespace exposes optimization passes built on the compile
//! preprocessing stack.

/// Internal Clifford+Rz optimization for control-flow-aware compile pipelines.
pub(crate) mod clifford_rz;
/// Template matching and template optimization implementation.
pub mod template;

#[allow(unused_imports)]
pub(crate) use clifford_rz::CliffordRzPass;
pub use clifford_rz::{
    CliffordRzConfig, CliffordRzLevel, CliffordRzOptimization, CliffordRzStrategy,
};
pub use template::{
    TemplateLibrary, TemplateMatch, TemplateMatching, TemplateOptimization,
    TemplateOptimizationConfig,
};
