// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

/// Policy controlling which ancillary resources the compiler may use.
///
/// Policy expresses compiler permission rather than device capacity. Use
/// [`ResourceLimits`] for hard target-derived bounds. The default policy is
/// conservative: it creates no clean ancillas and does not borrow input qubits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourcePolicy {
    /// Total number of clean logical ancillary qubits the compiler may create
    /// before layout. Released qubits remain part of this total and may be
    /// reused.
    pub max_pre_layout_clean_ancillas: usize,
    /// Whether input qubits may be borrowed under the dirty-resource contract.
    ///
    /// This permits borrowing only when the consuming algorithm accepts unknown
    /// input state and restores it exactly. It never makes input qubits eligible
    /// for clean-zero requests.
    pub allow_dirty_borrowing: bool,
}

/// Hard resource limits that may be derived from a target device.
///
/// Limits constrain the complete logical circuit, including input qubits and
/// compiler-allocated clean ancillas.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Optional maximum total number of logical qubits in the circuit.
    ///
    /// `None` means that this manager does not enforce a total-qubit bound.
    pub max_total_qubits: Option<usize>,
}
