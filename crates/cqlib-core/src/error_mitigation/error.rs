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

use thiserror::Error;

use crate::circuit::CircuitError;

/// Shared errors raised by error-mitigation APIs.
#[derive(Debug, Error)]
pub enum ErrorMitigationError {
    #[error(transparent)]
    Circuit(#[from] CircuitError),

    #[error("virtual distillation requires at least 2 copies, got {0}")]
    InvalidCopies(usize),

    #[error("hamiltonian qubit count mismatch: expected {expected}, got {actual}")]
    HamiltonianQubitCountMismatch { expected: usize, actual: usize },

    #[error("virtual distillation denominator mean is zero")]
    ZeroDenominatorMean,
}

