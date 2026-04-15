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
use crate::qis::error::QisError;

/// Shared errors raised by error-mitigation APIs.
#[derive(Debug, Error)]
pub enum ErrorMitigationError {
    #[error(transparent)]
    Circuit(#[from] CircuitError),

    #[error(transparent)]
    Qis(#[from] QisError),

    #[error("virtual distillation requires at least 2 copies, got {0}")]
    InvalidCopies(usize),

    #[error("hamiltonian qubit count mismatch: expected {expected}, got {actual}")]
    HamiltonianQubitCountMismatch { expected: usize, actual: usize },

    #[error("virtual distillation denominator mean is zero")]
    ZeroDenominatorMean,

    #[error("fold level must be non-negative, got {0}")]
    InvalidFoldLevel(i32),

    #[error("run() must be completed before get_mitigated()")]
    RunRequiredBeforeMitigation,

    #[error("run() has already been completed for this ErrorMitigation instance")]
    AlreadyRun,

    #[error("get_mitigated() has already been completed for this ErrorMitigation instance")]
    AlreadyMitigated,

    #[error("run arguments do not match the configured mitigation method")]
    RunArgsMethodMismatch,

    #[error("processing arguments do not match the configured mitigation method")]
    ProcessArgsMethodMismatch,

    #[error("noisy results must not be empty")]
    EmptyNoisyResults,

    #[error("noisy results length mismatch: expected {expected}, got {actual}")]
    NoisyResultsLengthMismatch { expected: usize, actual: usize },

    #[error("polynomial degree {degree} must be smaller than number of data points {num_points}")]
    InvalidPolynomialDegree { degree: usize, num_points: usize },

    #[error("all noisy results must be positive for exponential extrapolation")]
    NonPositiveNoisyResults,

    #[error("exponential fit failed due to a singular linear-regression system")]
    SingularExponentialFit,

    #[error("polynomial fit failed due to a singular normal-equation matrix")]
    SingularPolynomialFit,
}
