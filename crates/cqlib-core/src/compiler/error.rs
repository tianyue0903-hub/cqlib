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

use crate::circuit::CircuitError;
use thiserror::Error;

/// Errors raised by compiler infrastructure and compiler state validation.
#[derive(Debug, Error)]
pub enum CompilerError {
    /// Conversion or validation of the circuit control-flow graph failed.
    #[error(transparent)]
    Circuit(#[from] CircuitError),
    /// The input compiler state or circuit does not satisfy a pass precondition.
    #[error("invalid compiler input: {0}")]
    InvalidInput(String),
    /// A compiler transform could not complete its declared operation.
    #[error("compiler transform '{name}' failed: {reason}")]
    TransformFailed {
        /// Stable transform or synthesis primitive name.
        name: &'static str,
        /// Human-readable diagnostic describing why the transform failed.
        reason: String,
    },
    /// A compiler pass produced a state that violates its declared contract.
    #[error("compiler invariant violation: {0}")]
    InvariantViolation(String),
}
