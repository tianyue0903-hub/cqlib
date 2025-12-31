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

use std::fmt;
use std::ops::Index;

/// Represents a single quantum bit (qubit) within a quantum circuit.
///
/// `Qubit` is the fundamental unit of information in quantum computing. It is lightweight,
/// identified by a simple index, and supports hashing and equality checks, making it
/// suitable for use as a key in maps or sets (e.g., for circuit DAGs or connectivity graphs).
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::bit::Qubit;
///
/// let q = Qubit { id: 0 };
/// println!("Allocated qubit: {}", q); // Output: Q0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qubit {
    /// The unique identifier or index of the qubit within its register context.
    pub id: usize,
}

impl fmt::Display for Qubit {
    /// Formats the qubit as "Q{id}".
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Q{}", self.id)
    }
}

/// A named collection of quantum bits.
///
/// `QuantumRegister` provides a convenient way to group qubits together, similar to array
/// declarations in classical programming. It manages the lifecycle and indexing of a
/// logical set of qubits.
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
#[derive(Debug, Default, Clone)]
pub struct QuantumRegister {
    /// The human-readable name of the register (e.g., "q", "ancilla").
    pub name: String,
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
    /// Panics if the index is out of bounds (greater than or equal to `self.len()`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::bit::QuantumRegister;
    ///
    /// let qreg = QuantumRegister::new("q", 2);
    /// let q0 = qreg[0]; // Access the first qubit
    /// // let q_panic = qreg[5]; // This would panic
    /// ```
    fn index(&self, index: usize) -> &Self::Output {
        &self.qubits[index]
    }
}

impl<'a> IntoIterator for &'a QuantumRegister {
    type Item = &'a Qubit;
    type IntoIter = std::slice::Iter<'a, Qubit>;

    /// Creates an iterator over references to the qubits in the register.
    ///
    /// This allows iterating over a register without consuming ownership.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cqlib_core::circuit::bit::QuantumRegister;
    ///
    /// let qreg = QuantumRegister::new("q", 2);
    /// for qubit in &qreg {
    ///     println!("Iterating: {}", qubit);
    /// }
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        self.qubits.iter()
    }
}

impl QuantumRegister {
    /// Creates a new quantum register with the specified name and size.
    ///
    /// This initializes a vector of `Qubit` instances with IDs ranging from `0` to `size - 1`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the register.
    /// * `size` - The number of qubits to allocate.
    pub fn new(name: &str, size: usize) -> Self {
        let mut qubits = Vec::with_capacity(size);
        for i in 0..size {
            qubits.push(Qubit { id: i });
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
}

/// Represents a single classical bit (clbit) within a quantum circuit.
///
/// `Clbit` is used to store measurement results from quantum operations. Like [`Qubit`],
/// it is a lightweight identifier.
///
/// # Examples
///
/// ```rust
/// use cqlib_core::circuit::bit::Clbit;
///
/// let c = Clbit { id: 0 };
/// println!("Classical bit: {}", c); // Output: C0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Clbit {
    /// The unique identifier or index of the classical bit.
    pub id: usize,
}

impl fmt::Display for Clbit {
    /// Formats the classical bit as "C{id}".
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "C{}", self.id)
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
    pub name: String,
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
        for i in 0..size {
            clbits.push(Clbit { id: i });
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
}

#[cfg(test)]
#[path = "./bit_test.rs"]
mod bit_test;
