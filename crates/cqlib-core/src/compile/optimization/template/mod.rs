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

//! Template matching and template-based optimization for flat 1q/2q circuits.

mod library;
mod optimizer;

pub use library::TemplateLibrary;
pub use optimizer::{
    TemplateMatch, TemplateMatching, TemplateOptimization, TemplateOptimizationConfig,
};

#[cfg(test)]
mod architecture_tests {
    const LIBRARY_SOURCE: &str = include_str!("library.rs");
    const OPTIMIZER_SOURCE: &str = include_str!("optimizer.rs");
    const MAPPING_SOURCE: &str = include_str!("../../mapping/mod.rs");

    #[test]
    fn test_template_modules_no_longer_depend_on_mapping_internals() {
        assert!(!LIBRARY_SOURCE.contains("compile::mapping"));
        assert!(!OPTIMIZER_SOURCE.contains("compile::mapping"));
    }

    #[test]
    fn test_mapping_and_template_share_prepared_helpers() {
        assert!(OPTIMIZER_SOURCE.contains("compile::prepared"));
        assert!(MAPPING_SOURCE.contains("compile::prepared"));
    }
}
