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

//! C binding for Cqlib quantum computing library.
//!
//! This module provides C-compatible APIs for quantum circuit operations,
//! parameter management, IR format parsing (QCIS, OpenQASM2), and device
//! topology/properties management.
//!
//! # Module Structure
//!
//! - [`circuit`]: Quantum circuit operations and gates
//! - [`ir`]: IR format parsing (QCIS, OpenQASM2)
//! - [`device`]: Device topology and qubit properties
//! - [`qis`]: Quantum information science operations

// Allow clippy warnings for FFI functions that dereference raw pointers
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod circuit;
pub mod device;
pub mod ir;
pub mod qis;
