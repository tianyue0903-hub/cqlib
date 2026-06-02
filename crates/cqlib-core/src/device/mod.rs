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

//! # Quantum Device Module
//!
//! This module provides types for modeling quantum hardware devices.
//! It includes device topology, qubit layout mapping, noise models,
//! and execution result handling.
//!
//! ## Key Components
//!
//! - [`Device`]: Quantum device specification with qubit and edge properties
//! - [`Topology`]: Device connectivity graph
//! - [`Layout`]: Logical to physical qubit mapping
//! - [`NoiseModel`]: Noise simulation parameters
//! - [`ExecutionResult`]: Measurement outcome collection

pub mod device_impl;
pub mod error;
pub mod layout;
pub mod noise;
pub mod qubit;
pub mod result;
pub mod topology;

pub use device_impl::{Device, EdgeProp, InstructionProp, QubitProp};
pub use error::{DeviceError, LayoutError, TopologyError};
pub use layout::Layout;
pub use noise::{NoiseModel, OperationKey, ReadoutError, SingleQubitNoise, TwoQubitNoise};
pub use qubit::{LogicalQubit, PhysicalQubit};
pub use result::{ExecutionResult, Outcome, Status};
pub use topology::Topology;
