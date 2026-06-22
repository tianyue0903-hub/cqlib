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

//! Python bindings for compiler knowledge rules, libraries, DSL, and matching.

pub mod library;
pub mod matcher;
pub mod rule;

use library::{
    PyRuleId, PyRuleKind, PyRuleLibrary, PyRuleMetadata, py_dump, py_dumps, py_load, py_loads,
};
use matcher::{
    PyMatchBindings, py_conditions_hold, py_instantiate_target, py_match_rule_item,
    py_rule_matches_operations,
};
use pyo3::prelude::*;
use rule::{PyCondition, PyRule, PyRuleItem, PyVerifyResult};

/// Registers knowledge bindings as `_native.compile.knowledge`.
pub(crate) fn register_knowledge_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "knowledge")?;

    m.add_class::<PyRuleItem>()?;
    m.add_class::<PyCondition>()?;
    m.add_class::<PyRule>()?;
    m.add_class::<PyVerifyResult>()?;
    m.add_class::<PyRuleId>()?;
    m.add_class::<PyRuleKind>()?;
    m.add_class::<PyRuleMetadata>()?;
    m.add_class::<PyRuleLibrary>()?;
    m.add_class::<PyMatchBindings>()?;
    m.add_function(pyo3::wrap_pyfunction!(py_loads, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_load, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_dumps, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_dump, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_match_rule_item, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_conditions_hold, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_instantiate_target, &m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_rule_matches_operations, &m)?)?;

    parent.add_submodule(&m)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item("cqlib._native.compile.knowledge", &m)?;

    Ok(())
}
