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

//! Multi-controlled X synthesis primitives.
//!
//! This module is the algorithm layer for decomposing an MCX operation into
//! lower-level
//! [`ValueOperation`](crate::circuit::operation::ValueOperation) sequences. A
//! value operation carries self-contained parameter expressions rather than
//! circuit-local parameter-table indices. The module does not choose an
//! algorithm, allocate ancillary qubits, or normalize open controls. Those
//! responsibilities belong to the future `mc_gate` decomposition planner.
//!
//! The module currently defines the intended synthesis surface only. The
//! algorithms are public so callers can explicitly select an implementation,
//! but they are deliberately not connected to an active compiler entry point
//! until their implementations and semantic tests land.
//!
//! # Algorithm references
//!
//! - Maslov, *Advantages of using relative-phase Toffoli gates with an
//!   application to multiple control Toffoli optimization*, Phys. Rev. A 93,
//!   022311 (2016), [arXiv:1508.03273](https://arxiv.org/abs/1508.03273).
//! - Iten et al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318
//!   (2016), [arXiv:1501.06911](https://arxiv.org/abs/1501.06911).
//! - Barenco et al., *Elementary gates for quantum computation*, Phys. Rev. A
//!   52, 3457 (1995),
//!   [arXiv:quant-ph/9503016](https://arxiv.org/abs/quant-ph/9503016).
//! - Vale et al., *Circuit Decomposition of Multicontrolled Special Unitary
//!   Single-Qubit Gates*, IEEE Trans. Quantum Eng. 5 (2024),
//!   [arXiv:2302.06377](https://arxiv.org/abs/2302.06377).
//! - Huang and Palsberg, *Compiling Conditional Quantum Gates without Using
//!   Helper Qubits*, PLDI 2024,
//!   [DOI:10.1145/3656436](https://doi.org/10.1145/3656436).
//! - Khattar and Gidney, *Rise of conditionally clean ancillae for optimizing
//!   quantum circuits* (2024),
//!   [arXiv:2407.17966](https://arxiv.org/abs/2407.17966).
//!
//! The ancilla-free implementation uses the linear-size Huang-Palsberg
//! construction. Fixed small-control templates such as C3X and C4X belong in
//! the compiler knowledge-rule library rather than this algorithm module.

#![allow(dead_code)]

mod clean_v_chain;
mod conditionally_clean;
mod dirty_v_chain;
mod no_auxiliary;
mod one_clean_recursive;
mod relative_phase;
mod trivial;
mod utils;

#[cfg(test)]
mod clean_v_chain_test;
#[cfg(test)]
mod conditionally_clean_test;
#[cfg(test)]
mod dirty_v_chain_test;
#[cfg(test)]
mod no_auxiliary_test;
#[cfg(test)]
mod one_clean_recursive_test;
#[cfg(test)]
mod relative_phase_test;
#[cfg(test)]
mod trivial_test;

pub(super) const DECOMPOSE_MCX_NAME: &str = "decompose.mcx";

pub use clean_v_chain::decompose_mcx_n_clean;
pub use conditionally_clean::{
    decompose_mcx_1_clean_kg24, decompose_mcx_1_dirty, decompose_mcx_2_clean, decompose_mcx_2_dirty,
};
pub use dirty_v_chain::decompose_mcx_n_dirty;
pub use no_auxiliary::decompose_mcx_no_aux;
pub use one_clean_recursive::decompose_mcx_1_clean_b95;
pub use trivial::decompose_mcx_small;
