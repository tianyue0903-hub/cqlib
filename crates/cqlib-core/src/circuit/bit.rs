// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! # Bit and Register Module
//!
//! This module defines the fundamental units of information for quantum circuits:
//! qubits and classical bits, along with their respective registers.
//!
//! ## Key Components
//!
//! - [`Qubit`]: Represents a single quantum bit.
//! - [`QuantumRegister`]: A named collection of qubits.
//! - [`Clbit`]: Represents a single classical bit.
//! - [`ClassicalRegister`]: A named collection of classical bits.
//!
//! ## Usage
//!
//! The types in this module are typically used when defining the structure of a quantum circuit.
//! Registers act as containers that manage the lifecycle and identification of bits.

use std::sync::Arc;
use std::{fmt, hash::Hash, ops::Index};

/// Represents a single quantum bit (qubit) within a quantum circuit.
///
/// `Qubit` is the fundamental unit of information in quantum computing. It is lightweight,
/// identified by a simple index within a named register, and supports hashing and equality checks.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::bit::Qubit;
///
/// let q = Qubit::new(0, "q");
/// println!("qubit: {}", q); // qubit: q[0]
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Qubit {
    /// The unique identifier or index of the qubit within its register context.
    id: usize,
    /// The name of the register this qubit belongs to.
    register_name: Arc<String>,
}

impl fmt::Display for Qubit {
    /// Formats the qubit as "register_name[id]".
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}[{}]", self.register_name, self.id)
    }
}

impl Qubit {
    /// Creates a new `Qubit` instance.
    ///
    /// This is typically used for testing or standalone qubit creation.
    /// In a circuit context, qubits are usually created via [`QuantumRegister`].
    pub fn new(id: usize, register_name: &str) -> Qubit {
        Qubit {
            id,
            register_name: Arc::new(register_name.to_string()),
        }
    }

    /// Returns the index of the qubit within its register.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the name of the register this qubit belongs to.
    pub fn register_name(&self) -> &str {
        &self.register_name
    }
}

/// A named collection of quantum bits.
///
/// `QuantumRegister` provides a convenient way to group qubits together.
/// It manages the lifecycle and indexing of a logical set of qubits.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::bit::QuantumRegister;
///
/// // Create a 3-qubit register named "q"
/// let qreg = QuantumRegister::new("q", 3);
///
/// assert_eq!(qreg.len(), 3);
/// println!("Created: {}", qreg); // Output: QuantumRegister(name='q', size=3)
/// ```
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct QuantumRegister {
    /// The human-readable name of the register (e.g., "q", "ancilla").
    name: String,
    /// The internal vector of qubits contained in this register.
    qubits: Vec<Qubit>,
}

impl fmt::Display for QuantumRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuantumRegister(name='{}', size={})",
            self.name,
            self.qubits.len()
        )
    }
}

impl Index<usize> for QuantumRegister {
    type Output = Qubit;

    /// Allows indexing into the register to retrieve a specific qubit.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    fn index(&self, index: usize) -> &Self::Output {
        &self.qubits[index]
    }
}

impl<'a> IntoIterator for &'a QuantumRegister {
    type Item = &'a Qubit;
    type IntoIter = std::slice::Iter<'a, Qubit>;

    /// Creates an iterator over references to the qubits in the register.
    fn into_iter(self) -> Self::IntoIter {
        self.qubits.iter()
    }
}

impl QuantumRegister {
    /// Creates a new quantum register with the specified name and size.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the register.
    /// * `size` - The number of qubits to allocate.
    pub fn new(name: &str, size: usize) -> Self {
        let name_arc = Arc::new(name.to_string());
        let mut qubits = Vec::with_capacity(size);
        for i in 0..size {
            qubits.push(Qubit {
                id: i,
                register_name: name_arc.clone(),
            });
        }

        Self {
            name: name.to_string(),
            qubits,
        }
    }

    /// Returns the number of qubits in the register.
    pub fn len(&self) -> usize {
        self.qubits.len()
    }

    /// Returns `true` if the register contains no qubits.
    pub fn is_empty(&self) -> bool {
        self.qubits.is_empty()
    }

    /// Returns the name of the register.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Represents a single classical bit (clbit) within a quantum circuit.
///
/// `Clbit` is used to store measurement results from quantum operations.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::bit::Clbit;
///
/// let c = Clbit::new(0, "c");
/// println!("Classical bit: {}", c); // Output: c[0]
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Clbit {
    /// The unique identifier or index of the classical bit.
    id: usize,
    /// The name of the register this clbit belongs to.
    register_name: Arc<String>,
}

impl Clbit {
    /// Creates a new `Clbit` instance.
    pub fn new(id: usize, register_name: &str) -> Self {
        Self {
            id,
            register_name: Arc::new(register_name.to_string()),
        }
    }

    /// Returns the index of the classical bit within its register.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the name of the register this clbit belongs to.
    pub fn register_name(&self) -> &str {
        &self.register_name
    }
}

impl fmt::Display for Clbit {
    /// Formats the classical bit as "register_name[id]".
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}[{}]", self.register_name, self.id)
    }
}

/// A named collection of classical bits.
///
/// `ClassicalRegister` is used to group classical bits, typically for storing the outcomes
/// of measurements performed on a `QuantumRegister`.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::bit::ClassicalRegister;
///
/// // Create a register to store 3 measurement bits
/// let creg = ClassicalRegister::new("c", 3);
/// assert_eq!(creg.len(), 3);
/// ```
#[derive(Debug, Default, Clone)]
pub struct ClassicalRegister {
    /// The human-readable name of the register (e.g., "c", "meas").
    name: String,
    /// The internal vector of classical bits contained in this register.
    clbits: Vec<Clbit>,
}

impl fmt::Display for ClassicalRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClassicalRegister(name='{}', size={})",
            self.name,
            self.clbits.len()
        )
    }
}

impl Index<usize> for ClassicalRegister {
    type Output = Clbit;

    /// Allows indexing into the register to retrieve a specific classical bit.
    fn index(&self, index: usize) -> &Self::Output {
        &self.clbits[index]
    }
}

impl<'a> IntoIterator for &'a ClassicalRegister {
    type Item = &'a Clbit;
    type IntoIter = std::slice::Iter<'a, Clbit>;

    /// Creates an iterator over references to the classical bits in the register.
    fn into_iter(self) -> Self::IntoIter {
        self.clbits.iter()
    }
}

impl ClassicalRegister {
    /// Creates a new classical register with the specified name and size.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the register.
    /// * `size` - The number of bits to allocate.
    pub fn new(name: &str, size: usize) -> Self {
        let mut clbits = Vec::with_capacity(size);
        let name_arc = Arc::new(name.to_string());
        for i in 0..size {
            clbits.push(Clbit {
                id: i,
                register_name: name_arc.clone(),
            });
        }

        Self {
            name: name.to_string(),
            clbits,
        }
    }

    /// Returns the number of bits in the register.
    pub fn len(&self) -> usize {
        self.clbits.len()
    }

    /// Returns `true` if the register contains no bits.
    pub fn is_empty(&self) -> bool {
        self.clbits.is_empty()
    }

    /// Returns the name of the register.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
#[path = "./bit_test.rs"]
mod bit_test;
