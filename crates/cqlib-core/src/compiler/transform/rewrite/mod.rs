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

//! Knowledge-based local circuit rewrite.
//!
//! This pass consumes validated compiler knowledge rules and applies them as
//! dependency-aware local rewrites.  In production mode it behaves as a
//! conservative optimizer; in lowering mode it can use decomposition rules and
//! an explicit target gate basis.

mod basis;
mod config;
mod matcher;
mod rewriter;

pub use config::{RewriteConfig, RewriteMode};
pub use rewriter::{
    KnowledgeRewriteResult, KnowledgeRewriteStats, KnowledgeRewriter, rewrite_circuit,
};

#[cfg(test)]
#[path = "./rewrite_test.rs"]
mod rewrite_test;
