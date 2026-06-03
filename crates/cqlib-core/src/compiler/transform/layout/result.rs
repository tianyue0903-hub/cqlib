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

//! Result types for concrete layout algorithms.

use super::objective::LayoutScore;
use crate::device::Layout;

/// Candidate or selected layout produced by a concrete layout algorithm.
#[derive(Debug, Clone)]
pub struct LayoutResult {
    /// Logical-to-physical layout.
    pub layout: Layout,
    /// Optional score used to rank this layout.
    pub score: Option<LayoutScore>,
    /// Diagnostics describing search and scoring behavior.
    pub diagnostics: LayoutDiagnostics,
}

/// Diagnostics emitted by layout planning.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayoutDiagnostics {
    /// Whether this layout directly realizes every two-qubit interaction on an
    /// adjacent hardware edge.
    pub is_perfect: bool,
    /// Number of candidate layouts considered by the method.
    pub candidates_evaluated: usize,
    /// Whether fidelity/error data contributed to scoring.
    pub used_fidelity: bool,
    /// Human-readable notes for debug reports.
    pub notes: Vec<String>,
}

impl LayoutDiagnostics {
    /// Creates empty diagnostics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one diagnostic note.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}
