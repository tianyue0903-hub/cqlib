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

//! # Visualization Errors
//!
//! Error types shared by visualization IR building and rendering backends.
//!

use crate::circuit::error::CircuitError;
use thiserror::Error;

/// Errors raised by visualization builders and drawers.
#[derive(Debug, Error)]
pub enum VisualizationError {
    /// Circuit preprocessing failed before visualization IR construction.
    #[error("circuit preprocessing failed: {0}")]
    CircuitBuild(#[from] CircuitError),

    /// A qubit referenced by an operation is not present in the circuit qubit list.
    #[error("operation references unknown qubit Q{0}")]
    UnknownQubit(u32),

    /// A symbolic parameter index points outside the circuit parameter table.
    #[error("parameter index {index} out of bounds (len={len})")]
    ParameterIndexOutOfBounds { index: u32, len: usize },

    /// SVG parsing/rasterization failed while converting SVG-first outputs.
    #[error("svg rendering failed: {0}")]
    SvgRenderFailed(String),

    /// IO error while writing output files.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
