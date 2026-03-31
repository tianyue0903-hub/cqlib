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

//! # Device Error Types
//!
//! This module defines error types for quantum device operations.
//! It provides error handling for device topology validation,
//! layout mapping, and qubit connectivity issues.

use crate::circuit::Qubit;
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    InvalidOnlineQubit(Qubit),
    QubitNotInTopology(Qubit),
    EdgeNotInTopology(Qubit, Qubit),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOnlineQubit(q) => {
                write!(
                    f,
                    "Specified online qubit {} does not exist in the device topology",
                    q
                )
            }
            Self::QubitNotInTopology(q) => {
                write!(f, "Qubit {} is not in the device topology", q)
            }
            Self::EdgeNotInTopology(control, target) => {
                write!(
                    f,
                    "Edge ({}, {}) is not in the device topology",
                    control, target
                )
            }
        }
    }
}

/// Errors that can occur when creating or operating on a [`Layout`].
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// The number of logical qubits exceeds the number of physical qubits.
    TooManyLogicalQubits { logical: usize, physical: usize },
    /// A virtual qubit in `init_map` is not present in the logical qubit list.
    InvalidVirtualQubit(Qubit),
    /// A physical qubit in `init_map` is not present in the physical qubit list.
    InvalidPhysicalQubit(Qubit),
    /// Multiple virtual qubits are mapped to the same physical qubit.
    DuplicatePhysicalMapping,
    /// The requested qubit was not found in the layout.
    QubitNotFound(Qubit),
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyLogicalQubits { logical, physical } => {
                write!(
                    f,
                    "Logical qubits ({}) exceed physical qubits ({})",
                    logical, physical
                )
            }
            Self::InvalidVirtualQubit(q) => {
                write!(f, "Virtual qubit {} not in logical qubit list", q)
            }
            Self::InvalidPhysicalQubit(q) => {
                write!(f, "Physical qubit {} not in physical qubit list", q)
            }
            Self::DuplicatePhysicalMapping => {
                write!(f, "Multiple virtual qubits mapped to same physical qubit")
            }
            Self::QubitNotFound(q) => write!(f, "Qubit {} not found in layout", q),
        }
    }
}

/// Errors that can occur when operating on a Topology.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// Tried to operate on a qubit that doesn't exist in the topology.
    QubitNotFound(Qubit),
    /// Tried to operate on a coupling edge that doesn't exist.
    CouplingNotFound { control: Qubit, target: Qubit },
    /// Tried to add a duplicate qubit.
    QubitAlreadyExists(Qubit),
    /// Tried to add a duplicate coupling.
    CouplingAlreadyExists { control: Qubit, target: Qubit },
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopologyError::QubitNotFound(q) => write!(f, "Qubit {:?} not found", q),
            TopologyError::CouplingNotFound { control, target } => {
                write!(f, "Coupling ({:?} -> {:?}) not found", control, target)
            }
            TopologyError::QubitAlreadyExists(q) => write!(f, "Qubit {:?} already exists", q),
            TopologyError::CouplingAlreadyExists { control, target } => {
                write!(f, "Coupling ({:?} -> {:?}) already exists", control, target)
            }
        }
    }
}
