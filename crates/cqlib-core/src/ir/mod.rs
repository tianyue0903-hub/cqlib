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

//! # Quantum Circuit Intermediate Representation (IR)
//!
//! This module provides parsers and serializers for various quantum circuit formats,
//! enabling interoperability between Cqlib and other quantum computing tools.
//!
//! ## Supported Formats
//!
//! | Format | Description | Load Module | Dump Module |
//! |--------|-------------|--------------|-------------|
//! | OpenQASM 2.0 | IBM's quantum assembly language | [`qasm2::load`] | [`qasm2::dump`] |
//! | QCIS | Telecom Quantum's intermediate format | [`qcis::load`] | [`qcis::dump`] |
//!
//! ## Quick Start
//!
//! ### Loading from OpenQASM
//!
//! ```rust
//! use cqlib_core::ir::qasm2_loads;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     include "qelib1.inc";
//!     qreg q[2];
//!     h q[0];
//!     cx q[0], q[1];
//! "#;
//!
//! let circuit = qasm2_loads(qasm).unwrap();
//! ```
//!
//! ### Loading from QCIS
//!
//! ```rust
//! use cqlib_core::ir::qcis_loads;
//!
//! let qcis = r#"
//! H Q0
//! CZ Q0 Q1
//! M Q0 Q1
//! "#;
//!
//! let circuit = qcis_loads(qcis).unwrap();
//! ```
//!
//! ### Dumping to Format
//!
//! ```rust
//! use cqlib_core::{Circuit, Qubit, qasm2_dumps, qcis_dumps};
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! let qasm = qasm2_dumps(&circuit).unwrap();
//! let qcis = qcis_dumps(&circuit).unwrap();
//! ```

pub mod qasm2;
pub mod qcis;

pub use qasm2::dump::{dump as qasm2_dump, dumps as qasm2_dumps};
pub use qasm2::load::{load as qasm2_load, loads as qasm2_loads};
pub use qcis::dump::{dump as qcis_dump, dumps as qcis_dumps};
pub use qcis::load::{load as qcis_load, loads as qcis_loads};

#[cfg(test)]
#[path = "./qasm2_to_qcis_test.rs"]
mod qasm2_to_qcis_test;
