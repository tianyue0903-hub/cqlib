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

//! Qubit layout management for quantum circuit routing.
//!
//! This module provides [`Layout`], which maps logical (virtual) qubits to physical
//! qubits on a quantum device. Layout is essential for circuit routing algorithms
//! like SABRE that need to track how virtual qubits move across physical hardware.
//!
//! # Concepts
//!
//! - **Logical qubits**: Virtual qubits used in the quantum circuit (Q0, Q1, ...)
//! - **Physical qubits**: Actual hardware qubits on the device (P100, P101, ...)
//! - **Ancilla qubits**: Automatically generated auxiliary qubits to fill unused physical qubits
//!
//! # Example
//!
//! ```
//! use cqlib_core::circuit::Qubit;
//! use cqlib_core::device::Layout;
//!
//! // Create a layout with 2 logical qubits mapped to 3 physical qubits
//! let logical = vec![Qubit::new(0), Qubit::new(1)];
//! let physical = vec![Qubit::new(100), Qubit::new(101), Qubit::new(102)];
//!
//! let layout = Layout::new(logical, physical, None).unwrap();
//!
//! // Query mappings
//! assert!(layout.get_physical(Qubit::new(0)).is_some());
//! assert_eq!(layout.num_ancilla(), 1); // 1 ancilla auto-generated
//! ```

use crate::circuit::Qubit;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Errors that can occur when creating or operating on a [`Layout`].
#[derive(Debug, Clone, PartialEq, Eq)]
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

impl std::error::Error for LayoutError {}

/// Maps logical (virtual) qubits to physical qubits on a quantum device.
///
/// A layout represents the current assignment of virtual qubits to physical hardware.
/// It is used by routing algorithms to track qubit placement and update mappings
/// when SWAP gates are inserted.
///
/// The layout maintains bidirectional mappings:
/// - `v2p`: virtual qubit → physical qubit
/// - `p2v`: physical qubit → virtual qubit
#[derive(Debug, Clone, Default)]
pub struct Layout {
    /// Set of logical (virtual) qubits in the circuit.
    logical_qubits: HashSet<Qubit>,
    /// Set of ancilla qubits auto-generated to fill unused physical qubits.
    ancilla_qubits: HashSet<Qubit>,
    /// Set of physical qubits available on the device.
    physical_qubits: HashSet<Qubit>,
    /// Bidirectional mapping: virtual qubit → physical qubit.
    v2p: HashMap<Qubit, Qubit>,
    /// Bidirectional mapping: physical qubit → virtual qubit.
    p2v: HashMap<Qubit, Qubit>,
}

impl Layout {
    /// Creates a new layout mapping logical qubits to physical qubits.
    ///
    /// # Arguments
    ///
    /// * `logical` - List of logical (virtual) qubits to be mapped
    /// * `physical` - List of physical qubits available on the device
    /// * `init_map` - Optional initial mapping from logical to physical qubits
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] if:
    /// - `logical.len() > physical.len()`
    /// - `init_map` contains invalid virtual or physical qubits
    /// - `init_map` maps multiple virtual qubits to the same physical qubit
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::Qubit;
    /// use cqlib_core::device::Layout;
    /// use std::collections::HashMap;
    ///
    /// let logical = vec![Qubit::new(0), Qubit::new(1)];
    /// let physical = vec![Qubit::new(100), Qubit::new(101), Qubit::new(102)];
    ///
    /// // With automatic sequential mapping
    /// let layout = Layout::new(logical.clone(), physical.clone(), None).unwrap();
    ///
    /// // With custom initial mapping
    /// let mut init = HashMap::new();
    /// init.insert(Qubit::new(0), Qubit::new(100));
    /// let layout = Layout::new(logical, physical, Some(init)).unwrap();
    /// ```
    pub fn new(
        logical: Vec<Qubit>,
        physical: Vec<Qubit>,
        init_map: Option<HashMap<Qubit, Qubit>>,
    ) -> Result<Self, LayoutError> {
        // Validate: logical qubits cannot exceed physical qubits
        if logical.len() > physical.len() {
            return Err(LayoutError::TooManyLogicalQubits {
                logical: logical.len(),
                physical: physical.len(),
            });
        }

        let logical_set: HashSet<_> = logical.iter().copied().collect();
        let physical_set: HashSet<_> = physical.iter().copied().collect();

        // Validate init_map if provided
        if let Some(ref map) = init_map {
            // Check all virtual qubits in init_map are in the logical list
            for &v in map.keys() {
                if !logical_set.contains(&v) {
                    return Err(LayoutError::InvalidVirtualQubit(v));
                }
            }

            // Check all physical qubits in init_map are in the physical list
            for &p in map.values() {
                if !physical_set.contains(&p) {
                    return Err(LayoutError::InvalidPhysicalQubit(p));
                }
            }

            // Check no duplicate physical qubit mappings
            let mapped_physicals: HashSet<_> = map.values().copied().collect();
            if mapped_physicals.len() != map.len() {
                return Err(LayoutError::DuplicatePhysicalMapping);
            }
            // Note: InsufficientPhysicalQubits is mathematically impossible here:
            // - unmapped_logical = L - M
            // - ancilla = P - L
            // - remaining_physical = P - M
            // So: unmapped_logical + ancilla = (L-M) + (P-L) = P-M = remaining_physical
        }

        // Build logical qubit set
        let mut logical_qubits = HashSet::new();
        let max_logical_id = logical.iter().map(|q| q.id()).max();
        for q in &logical {
            logical_qubits.insert(*q);
        }

        // Generate ancilla qubits to fill the gap between logical and physical counts
        let num_ancilla = physical.len() - logical.len();
        let mut ancilla_qubits = HashSet::new();
        let mut ancilla_vec = Vec::new();

        let mut next_id = max_logical_id.map(|id| id + 1).unwrap_or(0);
        for _ in 0..num_ancilla {
            let ancilla = Qubit::new(next_id);
            ancilla_qubits.insert(ancilla);
            ancilla_vec.push(ancilla);
            next_id += 1;
        }

        let physical_qubits: HashSet<Qubit> = physical.iter().copied().collect();

        let mut layout = Self {
            logical_qubits,
            ancilla_qubits,
            physical_qubits,
            v2p: HashMap::new(),
            p2v: HashMap::new(),
        };

        if let Some(map) = init_map {
            // Apply initial mapping
            for (v, p) in map {
                layout.v2p.insert(v, p);
                layout.p2v.insert(p, v);
            }

            // Find unmapped physical qubits
            let mut unmapped_physicals: Vec<Qubit> = physical
                .into_iter()
                .filter(|p| !layout.p2v.contains_key(p))
                .collect();

            // Map remaining logical qubits to unmapped physical qubits
            let unmapped_logicals: Vec<Qubit> = logical
                .into_iter()
                .filter(|l| !layout.v2p.contains_key(l))
                .collect();

            for v in unmapped_logicals {
                let p = unmapped_physicals
                    .pop()
                    .expect("unmapped_physicals should not be empty (validated above)");
                layout.v2p.insert(v, p);
                layout.p2v.insert(p, v);
            }

            // Map ancilla qubits to remaining physical qubits
            for v in ancilla_vec {
                let p = unmapped_physicals
                    .pop()
                    .expect("unmapped_physicals should not be empty (validated above)");
                layout.v2p.insert(v, p);
                layout.p2v.insert(p, v);
            }

            debug_assert!(
                unmapped_physicals.is_empty(),
                "All physical qubits should be mapped"
            );
        } else {
            // Sequential mapping: logical + ancilla → physical in order
            let mut all_virtuals = logical;
            all_virtuals.extend(ancilla_vec);

            for (v, p) in all_virtuals.into_iter().zip(physical.into_iter()) {
                layout.v2p.insert(v, p);
                layout.p2v.insert(p, v);
            }
        }

        // Consistency checks
        debug_assert_eq!(layout.v2p.len(), layout.p2v.len());
        debug_assert_eq!(layout.v2p.len(), layout.physical_qubits.len());

        Ok(layout)
    }

    /// Returns the number of logical qubits.
    pub fn num_logical(&self) -> usize {
        self.logical_qubits.len()
    }

    /// Returns the number of ancilla qubits.
    pub fn num_ancilla(&self) -> usize {
        self.ancilla_qubits.len()
    }

    /// Returns the number of physical qubits.
    pub fn num_physical(&self) -> usize {
        self.physical_qubits.len()
    }

    /// Returns the physical qubit that a virtual qubit is mapped to.
    ///
    /// # Arguments
    ///
    /// * `virtual_id` - The virtual qubit to look up
    ///
    /// # Returns
    ///
    /// `Some(Qubit)` if the virtual qubit is mapped, `None` otherwise.
    pub fn get_physical(&self, virtual_id: Qubit) -> Option<Qubit> {
        self.v2p.get(&virtual_id).copied()
    }

    /// Returns the virtual qubit mapped to a physical qubit.
    ///
    /// # Arguments
    ///
    /// * `physical_id` - The physical qubit to look up
    ///
    /// # Returns
    ///
    /// `Some(Qubit)` if a virtual qubit is mapped to this physical qubit, `None` otherwise.
    pub fn get_virtual(&self, physical_id: Qubit) -> Option<Qubit> {
        self.p2v.get(&physical_id).copied()
    }

    /// Returns an iterator over all logical (virtual) qubits in the layout.
    pub fn logical_qubits(&self) -> impl Iterator<Item = Qubit> + '_ {
        self.logical_qubits.iter().copied()
    }

    /// Returns an iterator over all ancilla (auxiliary) qubits in the layout.
    pub fn ancilla_qubits(&self) -> impl Iterator<Item = Qubit> + '_ {
        self.ancilla_qubits.iter().copied()
    }

    /// Returns an iterator over all physical qubits available on the device.
    pub fn physical_qubits(&self) -> impl Iterator<Item = Qubit> + '_ {
        self.physical_qubits.iter().copied()
    }

    /// Returns the virtual-to-physical qubit mapping.
    pub fn v2p_map(&self) -> &HashMap<Qubit, Qubit> {
        &self.v2p
    }

    /// Returns the physical-to-virtual qubit mapping.
    pub fn p2v_map(&self) -> &HashMap<Qubit, Qubit> {
        &self.p2v
    }

    /// Swaps the virtual qubits mapped to two physical qubits.
    ///
    /// This is the core operation used by routing algorithms (e.g., SABRE) when
    /// inserting SWAP gates. After a SWAP gate is applied on the hardware,
    /// the virtual qubits on those physical qubits are exchanged.
    ///
    /// # Arguments
    ///
    /// * `phys_a` - First physical qubit
    /// * `phys_b` - Second physical qubit
    ///
    /// # Panics
    ///
    /// Panics if either physical qubit is not in the layout.
    ///
    /// # Example
    ///
    /// ```
    /// use cqlib_core::circuit::Qubit;
    /// use cqlib_core::device::Layout;
    ///
    /// let mut layout = Layout::new(
    ///     vec![Qubit::new(0)],
    ///     vec![Qubit::new(100), Qubit::new(101)],
    ///     None,
    /// ).unwrap();
    ///
    /// // Before swap: Q0 is on some physical qubit
    /// let phys_before = layout.get_physical(Qubit::new(0)).unwrap();
    ///
    /// // Perform SWAP on physical qubits 100 and 101
    /// layout.swap_physical(Qubit::new(100), Qubit::new(101));
    ///
    /// // After swap, Q0 is on the other physical qubit
    /// let phys_after = layout.get_physical(Qubit::new(0)).unwrap();
    /// assert_ne!(phys_before, phys_after);
    /// ```
    pub fn swap_physical(&mut self, phys_a: Qubit, phys_b: Qubit) {
        // Early return if swapping the same qubit
        if phys_a == phys_b {
            return;
        }

        // Verify physical qubits exist in layout
        assert!(
            self.physical_qubits.contains(&phys_a),
            "Physical qubit {} not in layout",
            phys_a
        );
        assert!(
            self.physical_qubits.contains(&phys_b),
            "Physical qubit {} not in layout",
            phys_b
        );

        // Get virtual qubits currently on these physical qubits
        let virt_a = self.get_virtual(phys_a);
        let virt_b = self.get_virtual(phys_b);

        // Update v2p mappings
        if let Some(v_a) = virt_a {
            self.v2p.insert(v_a, phys_b);
        }
        if let Some(v_b) = virt_b {
            self.v2p.insert(v_b, phys_a);
        }

        // Update p2v mappings based on which qubits have virtual mappings
        match (virt_a, virt_b) {
            (Some(v_a), Some(v_b)) => {
                // Both have virtual qubits: swap them
                self.p2v.insert(phys_a, v_b);
                self.p2v.insert(phys_b, v_a);
            }
            (Some(v_a), None) => {
                // Only phys_a has virtual qubit: move it to phys_b
                self.p2v.insert(phys_b, v_a);
                self.p2v.remove(&phys_a);
            }
            (None, Some(v_b)) => {
                // Only phys_b has virtual qubit: move it to phys_a
                self.p2v.insert(phys_a, v_b);
                self.p2v.remove(&phys_b);
            }
            (None, None) => {
                // Neither has virtual qubit: nothing to do
            }
        }
    }
}

#[cfg(test)]
#[path = "./layout_test.rs"]
mod layout_test;
