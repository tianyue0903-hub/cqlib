// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Shared logical-qubit ancillary-resource management for compiler stages.
//!
//! This module is compiler infrastructure rather than MCX-specific synthesis
//! code. It tracks which logical qubits compiler transforms may use as
//! ancillary resources and keeps allocation policy separate from algorithms
//! that consume those resources.
//!
//! The main entry point is [`ResourceManager`]. A compiler planner asks the
//! manager for a side-effect-free [`ResourcePlan`] with
//! [`ResourceManager::preview`], compares feasible algorithm candidates, and
//! commits only the selected plan with [`ResourceManager::commit`]. The returned
//! [`ResourceLease`] reserves its qubits until the caller restores their
//! required state and calls [`ResourceManager::release`].
//!
//! Clean and dirty resources are usage contracts, not facts inferred by
//! simulating quantum state. A clean ancillary qubit must enter an algorithm in
//! `|0>` and be restored to `|0>` before release. A dirty ancillary qubit may
//! enter in an unknown, possibly entangled state, so the consuming algorithm
//! must separately prove that it restores the complete input state before
//! release.
//!
//! Input qubits are not automatically clean ancillary resources. Although a
//! circuit's input qubits conventionally start in `|0>`, the compiler does not
//! currently analyze intermediate quantum state and therefore cannot establish
//! that an input qubit is still clean at an arbitrary transform boundary.
//!
//! The compiler may create clean logical ancillary qubits before layout. After
//! [`ResourceManager::enter_post_layout`] is called, requests may reuse
//! registered resources but may not create logical qubits without physical
//! mappings. Activating unused physical capacity after layout requires
//! orchestration outside this module.
//!
//! [`ResourceManager`] records allocation and restoration contracts; it does
//! not prove the quantum semantics of a consuming algorithm. A compiler stage
//! must call [`ResourceManager::verify_idle`] at an appropriate boundary to
//! ensure that every temporary lease has been released and that the circuit and
//! resource indexes still agree.
//!
//! # Example
//!
//! ```
//! use cqlib_core::circuit::Circuit;
//! use cqlib_core::compile::resource::{
//!     AncillaRequirement, ResourceLimits, ResourceManager, ResourcePolicy, ResourceRequest,
//! };
//! use std::collections::BTreeSet;
//!
//! let mut circuit = Circuit::new(2);
//! let mut resources = ResourceManager::from_circuit(
//!     &circuit,
//!     ResourcePolicy {
//!         max_pre_layout_clean_ancillas: 1,
//!         allow_dirty_borrowing: false,
//!     },
//!     ResourceLimits::default(),
//! )?;
//! // Preview is side-effect free, so a planner may compare several candidates.
//! let plan = resources.preview(&ResourceRequest {
//!     requirement: AncillaRequirement::CleanZero,
//!     count: 1,
//!     excluded: BTreeSet::new(),
//! })?;
//! let lease = resources.commit(&mut circuit, plan)?;
//! // Pass lease.qubits() to an algorithm that restores the clean-zero contract.
//! resources.release(&lease)?;
//! resources.verify_idle(&circuit)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
mod manager;
mod model;
mod policy;

pub use error::ResourceError;
pub use manager::ResourceManager;
pub use model::{AncillaRequirement, ResourceLease, ResourcePlan, ResourceRequest};
pub use policy::{ResourceLimits, ResourcePolicy};

#[cfg(test)]
mod manager_test;
