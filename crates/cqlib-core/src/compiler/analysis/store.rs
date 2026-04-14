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

//! Revision-scoped analysis cache storage.
//!
//! This module provides the type-indexed cache used by
//! [`crate::compiler::CompilerContext`] to memoize analysis results for the
//! current revision. The cache is intentionally generic over analysis result
//! types and does not encode analysis-specific policy.
//!
//! Cache coherence is managed by the context: when circuit or target-dependent
//! state changes, cached entries are invalidated.

use alloc::boxed::Box;
use core::any::{Any, TypeId};
use core::fmt::Debug;
use std::collections::HashMap;

#[derive(Default)]
struct CachedAnalysis {
    revision: u64,
    value: Option<Box<dyn Any>>,
}

/// Type-indexed cache for analyses derived from the current compiler state.
///
/// The store is keyed by analysis result type and compiler revision. Whenever the
/// circuit or other analysis-invalidating state changes, the owning
/// [`crate::compiler::CompilerContext`] clears or refreshes this cache.
#[derive(Default)]
pub struct AnalysisStore {
    entries: HashMap<TypeId, CachedAnalysis>,
}

impl AnalysisStore {
    /// Returns a cached analysis if it exists for the provided revision.
    pub fn get<T: 'static>(&self, revision: u64) -> Option<&T> {
        self.entries.get(&TypeId::of::<T>()).and_then(|entry| {
            if entry.revision == revision {
                entry.value.as_deref()?.downcast_ref::<T>()
            } else {
                None
            }
        })
    }

    /// Stores an analysis result for the provided revision.
    pub fn insert<T: 'static>(&mut self, revision: u64, value: T) {
        self.entries.insert(
            TypeId::of::<T>(),
            CachedAnalysis {
                revision,
                value: Some(Box::new(value)),
            },
        );
    }

    /// Invalidates all cached analyses.
    pub fn invalidate_all(&mut self) {
        self.entries.clear();
    }

    /// Removes a specific cached analysis type.
    pub fn remove<T: 'static>(&mut self) {
        self.entries.remove(&TypeId::of::<T>());
    }
}

impl Debug for AnalysisStore {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AnalysisStore")
            .field("entries", &self.entries.len())
            .finish()
    }
}
