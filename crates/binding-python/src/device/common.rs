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

use cqlib_core::circuit::Qubit;
use cqlib_core::qis::pauli::Pauli;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn py_id_to_qubit(idx: usize) -> PyResult<Qubit> {
    let id = u32::try_from(idx)
        .map_err(|_| PyValueError::new_err(format!("qubit id {} overflows u32", idx)))?;
    Ok(Qubit::new(id))
}

pub fn qubit_to_py_id(qubit: Qubit) -> usize {
    qubit.id() as usize
}

pub fn parse_pauli(name: &str) -> PyResult<Pauli> {
    match name.to_ascii_uppercase().as_str() {
        "I" => Ok(Pauli::I),
        "X" => Ok(Pauli::X),
        "Y" => Ok(Pauli::Y),
        "Z" => Ok(Pauli::Z),
        _ => Err(PyValueError::new_err(format!(
            "invalid Pauli '{}', expected one of I/X/Y/Z",
            name
        ))),
    }
}

pub fn pauli_to_name(pauli: Pauli) -> &'static str {
    match pauli {
        Pauli::I => "I",
        Pauli::X => "X",
        Pauli::Y => "Y",
        Pauli::Z => "Z",
    }
}
