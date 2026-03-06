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

//! # QCIS (Quantum Circuit Intermediate Representation) Parser and Serializer
//!
//! This module provides bidirectional conversion between QCIS format and the internal
//! `Circuit` representation.
//!
//! ## Overview
//!
//! QCIS (Quantum Circuit Intermediate Representation) is a simplified quantum circuit format
//! optimized for backend execution. It uses a compact text-based format where each line
//! represents a single quantum operation.
//!
//! ## Supported Gates
//!
//! ### Native QCIS Gates
//! - **Single-qubit rotations**: X2P, X2M, Y2P, Y2M, XY2P, XY2M
//! - **Two-qubit gates**: CZ
//! - **Single-qubit gates**: RZ
//! - **Delay**: I (identity with time parameter)
//!
//! ### Standard Gates (mapped to QCIS equivalents)
//! - **Pauli gates**: X, Y, Z
//! - **Clifford gates**: H, S, T
//! - **Parametrized gates**: RX, RY, RXY
//! - **Multi-qubit**: Barrier, Measure
//!
//! ## Usage
//!
//! ### Loading QCIS
//!
//! ```rust
//! use cqlib_core::ir::qcis::load::loads;
//!
//! let qcis = r#"
//! H Q0
//! CZ Q0 Q1
//! M Q0 Q1
//! "#;
//!
//! let circuit = loads(qcis).unwrap();
//! ```
//!
//! ### Dumping to QCIS
//!
//! ```rust
//! use cqlib_core::ir::qcis::dump::dumps;
//! use cqlib_core::circuit::{Circuit, Qubit};
//!
//! let mut circuit = Circuit::new(2);
//! circuit.h(Qubit::new(0)).unwrap();
//! circuit.cz(Qubit::new(0), Qubit::new(1)).unwrap();
//!
//! let qcis = dumps(&circuit).unwrap();
//! // Output: "H Q0\nCZ Q0 Q1\n"
//! ```

pub mod dump;
pub mod load;

pub use dump::{dump, dumps};
pub use load::{load, loads};
