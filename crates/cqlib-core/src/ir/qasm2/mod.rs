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

//! # OpenQASM 2.0 Parser and Serializer
//!
//! This module provides bidirectional conversion between OpenQASM 2.0 format
//! and the internal `Circuit` representation.
//!
//! ## Overview
//!
//! The OpenQASM 2.0 is a popular quantum circuit description language developed by IBM.
//! This module enables:
//!
//! - **Loading**: Parsing OpenQASM 2.0 source code into internal `Circuit` representation
//! - **Dumping**: Converting internal `Circuit` back to OpenQASM 2.0 format
//!
//! ## Key Components
//!
//! - [`ast`] - Abstract Syntax Tree definitions for OpenQASM 2.0
//! - [`load`] - Parser converting OpenQASM 2.0 to internal Circuit
//! - [`dump`] - Serializer converting internal Circuit to OpenQASM 2.0
//!
//! ## Usage
//!
//! ### Loading OpenQASM
//!
//! ```rust
//! use cqlib_core::ir::qasm2::load::loads;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     include "qelib1.inc";
//!     qreg q[2];
//!     creg c[1];
//!     h q[0];
//!     cx q[0], q[1];
//!     measure q[0] -> c[0];
//! "#;
//!
//! let circuit = loads(qasm).unwrap();
//! assert_eq!(circuit.num_qubits(), 2);
//! ```
//!
//! ### Dumping to OpenQASM
//!
//! ```rust
//! use cqlib_core::ir::qasm2::dump::dumps;
//! use cqlib_core::circuit::Circuit;
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(0.try_into().unwrap()).unwrap();
//! circuit.cx(0.try_into().unwrap(), 1.try_into().unwrap()).unwrap();
//!
//! let qasm = dumps(&circuit).unwrap();
//! assert!(qasm.contains("OPENQASM 2.0"));
//! ```

pub mod ast;
pub mod dump;
pub mod load;

pub use dump::{dump, dumps};
pub use load::{load, loads};
