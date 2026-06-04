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

//! Convenience helpers for loading rules from files or strings.

use crate::compiler::knowledge::rule::Rule;
use crate::compiler::knowledge::rule_dsl::ast::RuleDef;
use crate::compiler::knowledge::rule_dsl::lower::LowerError;
use crate::compiler::knowledge::rule_dsl::parser::{ParseError, Parser};
use std::collections::HashSet;
use std::path::Path;

/// Unified error type for loading rules.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LoadError {
    /// An I/O error occurred while reading the file.
    #[error("I/O error while loading rules: {0}")]
    Io(String),
    /// The source could not be parsed into AST nodes.
    #[error("failed to parse rule DSL: {0}")]
    Parse(ParseError),
    /// AST lowering to runtime [`Rule`] structures failed.
    #[error("failed to lower rule DSL: {0}")]
    Lower(LowerError),
    /// A rule file defines the same rule name more than once.
    #[error("duplicate rule name: {0}")]
    DuplicateRuleName(String),
}

/// Loads all rules from a file and lowers them to [`Rule`] objects.
///
/// # Errors
///
/// Returns [`LoadError::Io`] if the file cannot be read,
/// [`LoadError::Parse`] if the text is not valid DSL,
/// or [`LoadError::Lower`] if a gate name / arity is invalid.
pub fn load_rules_from_file(path: impl AsRef<Path>) -> Result<Vec<Rule>, LoadError> {
    let source =
        std::fs::read_to_string(path.as_ref()).map_err(|e| LoadError::Io(format!("{}", e)))?;
    load_rules_from_str(&source)
}

/// Loads all rules from a string and lowers them to [`Rule`] objects.
///
/// # Errors
///
/// Returns [`LoadError::Parse`] if the text is not valid DSL,
/// or [`LoadError::Lower`] if a gate name / arity is invalid.
pub fn load_rules_from_str(source: &str) -> Result<Vec<Rule>, LoadError> {
    let defs = load_rule_defs_from_str(source)?;
    validate_unique_rule_names(&defs)?;
    defs.into_iter()
        .map(RuleDef::into_rule)
        .collect::<Result<Vec<_>, _>>()
        .map_err(LoadError::Lower)
}

/// Loads rule definitions from a file without lowering them.
///
/// # Errors
///
/// Returns [`LoadError::Io`] if the file cannot be read,
/// or [`LoadError::Parse`] if the text is not valid DSL.
pub fn load_rule_defs_from_file(path: impl AsRef<Path>) -> Result<Vec<RuleDef>, LoadError> {
    let source =
        std::fs::read_to_string(path.as_ref()).map_err(|e| LoadError::Io(format!("{}", e)))?;
    load_rule_defs_from_str(&source)
}

/// Loads rule definitions from a string without lowering them.
///
/// # Errors
///
/// Returns [`LoadError::Parse`] if the text is not valid DSL.
pub fn load_rule_defs_from_str(source: &str) -> Result<Vec<RuleDef>, LoadError> {
    let mut parser = Parser::new(source).map_err(LoadError::Parse)?;
    parser.parse_rule_file().map_err(LoadError::Parse)
}

fn validate_unique_rule_names(defs: &[RuleDef]) -> Result<(), LoadError> {
    let mut seen = HashSet::with_capacity(defs.len());
    for def in defs {
        if !seen.insert(def.name.as_str()) {
            return Err(LoadError::DuplicateRuleName(def.name.clone()));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "./load_test.rs"]
mod load_test;
