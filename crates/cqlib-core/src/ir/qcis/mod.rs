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

//! # QCIS Parser and Serializer
//!
//! This module converts between QCIS text and cqlib's [`Circuit`](crate::circuit::Circuit)
//! IR. It covers the circuit-level QCIS instruction set used by this crate; pulse
//! and coupling-control instructions are intentionally outside this module.
//!
//! ## Text Format
//!
//! QCIS is line oriented. Each non-empty line has an opcode, one or more qubits,
//! and optional parameters:
//!
//! ```text
//! OPCODE Q0 [Q1 ...] [param ...]
//! ```
//!
//! Qubits are written as `Q<id>` with an unsigned decimal id. Parameters are
//! parsed as cqlib parameter expressions, so numeric values and expressions such
//! as `pi`, `pi/2`, and `theta` are accepted. Lines beginning with `//` and
//! inline text after `//` are ignored.
//!
//! ## Circuit Instructions
//!
//! The QCIS instructions represented directly by this module include all
//! cqlib standard gates except identity and global phase:
//!
//! - Single-qubit gates: `H`, `S`, `SD`, `T`, `TD`, `X`, `X2P`, `X2M`, `Y`,
//!   `Y2P`, `Y2M`, and `Z`.
//! - Parameterized single-qubit gates: `RX`, `RXY`, `RY`, `RZ`, `U`, `XY`,
//!   `XY2P`, `XY2M`, and `PHASE`.
//! - Multi-qubit gates: `RXX`, `RYY`, `RZX`, `RZZ`, `SWAP`, `CX`, `CCX`, `CY`,
//!   `CZ`, `CRX`, `CRY`, `CRZ`, and `FSIM`.
//! - `I Qn t`: delay/no-op on `Qn` for `t` ticks. In QCIS, `t` must be a fixed
//!   non-negative integer count whose unit is 0.5 ns, so `t = 1` means 0.5 ns.
//!   This is represented as
//!   [`Instruction::Delay`](crate::circuit::gate::Instruction::Delay), not as
//!   cqlib's standard identity gate.
//! - `M Qn [Qm ...]`: measurement. Loading expands multi-qubit measurement to
//!   per-qubit measurement operations; dumping preserves grouped `measure_bits`
//!   operations when present.
//! - `B Qn [Qm ...]` or `Barrier Qn [Qm ...]`: barrier.
//!
//! The loader accepts `SDG` and `TDG` as aliases. The dumper normalizes them to
//! `SD` and `TD`.
//!
//! QCIS `I` is not an alias for the cqlib identity gate. To emit `I Qn t`, use
//! [`Circuit::delay`](crate::circuit::Circuit::delay). A standard identity gate
//! is rejected by the QCIS dumper to avoid losing the required duration.
//! `GPHASE` and [`StandardGate::GPhase`](crate::circuit::StandardGate::GPhase)
//! are also intentionally unsupported.
//!
//! ## Entry Points and Errors
//!
//! - [`load::load`] / [`load::from_path`] reads a QCIS file and returns
//!   [`load::QcisParseError`] on failure.
//! - [`load::loads`] / [`load::from_str`] reads QCIS source from a string and
//!   returns [`load::QcisParseError`] on failure.
//! - [`dump::dump`] / [`dump::to_path`] writes a circuit to a QCIS file and
//!   returns [`dump::QcisDumpError`] on failure.
//! - [`dump::dumps`] / [`dump::to_string`] serializes a circuit to a QCIS string
//!   and returns [`dump::QcisDumpError`] on failure.
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

pub use dump::{dump, dumps, to_path, to_string};
pub use load::{from_path, from_str, load, loads};
