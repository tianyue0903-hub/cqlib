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

pub mod qasm2;
pub mod qcis;

pub use qasm2::dump::{dump as qasm2_dump, dumps as qasm2_dumps};
pub use qasm2::load::{load as qasm2_load, loads as qasm2_loads};
pub use qcis::dump::{dump as qcis_dump, dumps as qcis_dumps};
pub use qcis::load::{load as qcis_load, loads as qcis_loads};

#[cfg(test)]
#[path = "./qasm2_to_qcis_test.rs"]
mod qasm2_to_qcis_test;
