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

//! Shared qubit helpers.

use crate::circuit::Qubit;
use std::collections::HashSet;

/// Returns the first qubit that occurs more than once across the provided
/// groups.
///
/// Groups and qubits within each group are inspected in slice order.
pub fn find_duplicate_qubit(qubit_groups: &[&[Qubit]]) -> Option<Qubit> {
    let mut seen = HashSet::with_capacity(qubit_groups.iter().map(|qubits| qubits.len()).sum());
    qubit_groups
        .iter()
        .flat_map(|qubits| qubits.iter().copied())
        .find(|&qubit| !seen.insert(qubit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_first_duplicate_qubit_across_groups() {
        let first = [Qubit::new(0), Qubit::new(1)];
        let second = [Qubit::new(2), Qubit::new(1)];

        assert_eq!(
            find_duplicate_qubit(&[&first, &second]),
            Some(Qubit::new(1))
        );
    }

    #[test]
    fn returns_none_for_distinct_qubit_groups() {
        let first = [Qubit::new(0), Qubit::new(1)];
        let second = [Qubit::new(2), Qubit::new(3)];

        assert_eq!(find_duplicate_qubit(&[&first, &second]), None);
    }
}
