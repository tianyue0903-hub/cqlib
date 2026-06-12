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

use crate::circuit::Circuit;
use crate::compile::CompilerError;

/// Common output shape for compiler transforms over a circuit.
#[derive(Debug, Clone)]
pub struct TransformResult {
    /// Transformed circuit.
    pub circuit: Circuit,
    /// Whether the transform changed the compiler IR representation.
    ///
    /// A transform reports `false` when it found no applicable operation or
    /// reached the same representation. This is a transform-local contract:
    /// callers should not pre-scan circuits to infer whether a transform should
    /// run.
    pub changed: bool,
}

/// Common interface for compiler transforms that consume one circuit and produce
/// a rebuilt circuit.
///
/// # Implementing
///
/// - [`name`](Transformer::name) returns a static human-readable label for logging.
/// - [`transform`](Transformer::transform) applies the pass to a circuit.
///
/// Parameters that differ between pass instances (e.g. config, device) are bound at
/// construction time so `transform` keeps a uniform signature across all passes.
pub trait Transformer {
    /// Human-readable pass name for logging and debugging.
    fn name(&self) -> &'static str;

    /// Applies the transform to `circuit`.
    fn transform(&self, circuit: &Circuit) -> Result<TransformResult, CompilerError>;
}
