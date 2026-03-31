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

use super::*;

#[test]
fn test_layout_basic_creation() {
    let logical = vec![Qubit::new(0), Qubit::new(1)];
    let physical = vec![Qubit::new(100), Qubit::new(101), Qubit::new(102)];

    let layout = Layout::new(logical, physical, None).unwrap();
    assert_eq!(layout.num_logical(), 2);
    assert_eq!(layout.num_ancilla(), 1);
    assert_eq!(layout.num_physical(), 3);
}

#[test]
fn test_layout_with_init_map() {
    let logical = vec![Qubit::new(0), Qubit::new(1)];
    let physical = vec![Qubit::new(100), Qubit::new(101), Qubit::new(102)];
    let init_map: HashMap<Qubit, Qubit> = [(Qubit::new(0), Qubit::new(100))].into_iter().collect();

    let layout = Layout::new(logical, physical, Some(init_map)).unwrap();
    assert_eq!(layout.get_physical(Qubit::new(0)), Some(Qubit::new(100)));
}

#[test]
fn test_layout_too_many_logical_error() {
    let logical = vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let physical = vec![Qubit::new(100), Qubit::new(101)];

    let result = Layout::new(logical, physical, None);
    assert!(matches!(
        result.unwrap_err(),
        LayoutError::TooManyLogicalQubits {
            logical: 3,
            physical: 2
        }
    ));
}

#[test]
fn test_layout_invalid_virtual_qubit_error() {
    let logical = vec![Qubit::new(0)];
    let physical = vec![Qubit::new(100), Qubit::new(101)];
    // Qubit 99 is not in logical list
    let init_map: HashMap<Qubit, Qubit> = [(Qubit::new(99), Qubit::new(100))].into_iter().collect();

    let result = Layout::new(logical, physical, Some(init_map));
    assert!(matches!(
        result.unwrap_err(),
        LayoutError::InvalidVirtualQubit(q) if q == Qubit::new(99)
    ));
}

#[test]
fn test_layout_invalid_physical_qubit_error() {
    let logical = vec![Qubit::new(0)];
    let physical = vec![Qubit::new(100), Qubit::new(101)];
    // Qubit 999 is not in physical list
    let init_map: HashMap<Qubit, Qubit> = [(Qubit::new(0), Qubit::new(999))].into_iter().collect();

    let result = Layout::new(logical, physical, Some(init_map));
    assert!(matches!(
        result.unwrap_err(),
        LayoutError::InvalidPhysicalQubit(q) if q == Qubit::new(999)
    ));
}

#[test]
fn test_layout_duplicate_physical_mapping_error() {
    let logical = vec![Qubit::new(0), Qubit::new(1)];
    let physical = vec![Qubit::new(100), Qubit::new(101), Qubit::new(102)];
    // Both logical qubits mapped to same physical qubit
    let init_map: HashMap<Qubit, Qubit> = [
        (Qubit::new(0), Qubit::new(100)),
        (Qubit::new(1), Qubit::new(100)),
    ]
    .into_iter()
    .collect();

    let result = Layout::new(logical, physical, Some(init_map));
    assert!(matches!(
        result.unwrap_err(),
        LayoutError::DuplicatePhysicalMapping
    ));
}

#[test]
fn test_layout_sequential_mapping() {
    // Test sequential mapping without init_map
    let logical = vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let physical = vec![
        Qubit::new(100),
        Qubit::new(101),
        Qubit::new(102),
        Qubit::new(103),
    ];

    let layout = Layout::new(logical, physical, None).unwrap();

    // 3 logical + 1 ancilla = 4 virtual mapped to 4 physical
    assert_eq!(layout.num_logical(), 3);
    assert_eq!(layout.num_ancilla(), 1);
    assert_eq!(layout.num_physical(), 4);

    // Check sequential mapping: Q0->100, Q1->101, Q2->102, ancilla->103
    assert_eq!(layout.get_physical(Qubit::new(0)), Some(Qubit::new(100)));
    assert_eq!(layout.get_physical(Qubit::new(1)), Some(Qubit::new(101)));
    assert_eq!(layout.get_physical(Qubit::new(2)), Some(Qubit::new(102)));
    assert_eq!(layout.get_physical(Qubit::new(3)), Some(Qubit::new(103))); // ancilla

    // Check reverse mapping
    assert_eq!(layout.get_virtual(Qubit::new(100)), Some(Qubit::new(0)));
    assert_eq!(layout.get_virtual(Qubit::new(101)), Some(Qubit::new(1)));
    assert_eq!(layout.get_virtual(Qubit::new(102)), Some(Qubit::new(2)));
    assert_eq!(layout.get_virtual(Qubit::new(103)), Some(Qubit::new(3)));
}

#[test]
fn test_layout_swap_physical() {
    let logical = vec![Qubit::new(0), Qubit::new(1)];
    let physical = vec![Qubit::new(100), Qubit::new(101), Qubit::new(102)];

    let mut layout = Layout::new(logical, physical, None).unwrap();

    // Get initial mappings
    let phys_100_virt = layout.get_virtual(Qubit::new(100));
    let phys_101_virt = layout.get_virtual(Qubit::new(101));

    // Swap physical qubits 100 and 101
    layout
        .swap_physical(Qubit::new(100), Qubit::new(101))
        .unwrap();

    // After swap, virtual qubits should be exchanged
    assert_eq!(layout.get_virtual(Qubit::new(100)), phys_101_virt);
    assert_eq!(layout.get_virtual(Qubit::new(101)), phys_100_virt);
}

#[test]
fn test_layout_swap_same_qubit() {
    let logical = vec![Qubit::new(0)];
    let physical = vec![Qubit::new(100), Qubit::new(101)];

    let mut layout = Layout::new(logical, physical, None).unwrap();

    // Get initial mapping
    let original_virt = layout.get_virtual(Qubit::new(100));

    // Swapping with itself should be a no-op
    layout
        .swap_physical(Qubit::new(100), Qubit::new(100))
        .unwrap();

    // Mapping should be unchanged
    assert_eq!(layout.get_virtual(Qubit::new(100)), original_virt);
}

#[test]
fn test_layout_swap_two_mapped() {
    // With 1 logical + 1 ancilla = 2 virtual qubits mapped to 2 physical qubits
    let logical = vec![Qubit::new(0)];
    let physical = vec![Qubit::new(100), Qubit::new(101)];

    let mut layout = Layout::new(logical, physical, None).unwrap();

    // Get initial mappings
    let phys_a = Qubit::new(100);
    let phys_b = Qubit::new(101);
    let virt_a = layout.get_virtual(phys_a);
    let virt_b = layout.get_virtual(phys_b);

    // Both physical qubits should have virtual qubits (one logical, one ancilla)
    assert!(virt_a.is_some());
    assert!(virt_b.is_some());

    // Swap
    layout.swap_physical(phys_a, phys_b).unwrap();

    // Virtual qubits should be exchanged
    assert_eq!(layout.get_virtual(phys_a), virt_b);
    assert_eq!(layout.get_virtual(phys_b), virt_a);
    // v2p mappings should also be updated
    if let Some(v_a) = virt_a {
        assert_eq!(layout.get_physical(v_a), Some(phys_b));
    }
    if let Some(v_b) = virt_b {
        assert_eq!(layout.get_physical(v_b), Some(phys_a));
    }
}

#[test]
fn test_layout_get_physical() {
    let logical = vec![Qubit::new(0), Qubit::new(1)];
    let physical = vec![Qubit::new(100), Qubit::new(101), Qubit::new(102)];

    let layout = Layout::new(logical, physical, None).unwrap();

    // Logical qubits should be mapped
    assert!(layout.get_physical(Qubit::new(0)).is_some());
    assert!(layout.get_physical(Qubit::new(1)).is_some());

    // Ancilla qubits should also be mapped
    assert!(layout.get_physical(Qubit::new(2)).is_some()); // ancilla

    // Non-existent qubit should return None
    assert_eq!(layout.get_physical(Qubit::new(999)), None);
}

#[test]
fn test_ancilla_id_generation_no_conflict_with_physical() {
    // Test scenario where physical qubit IDs might overlap with ancilla IDs
    // Logical: [0, 1] (max id = 1), so ancilla starts from 2
    // Physical: [2, 3, 4]
    // This means ancilla(2) will have same ID as physical(2)

    let logical = vec![Qubit::new(0), Qubit::new(1)];
    let physical = vec![Qubit::new(2), Qubit::new(3), Qubit::new(4)];

    let layout = Layout::new(logical, physical, None).unwrap();

    // Verify ancilla was created with ID 2
    assert!(layout.get_physical(Qubit::new(2)).is_some());

    // The ancilla Qubit(2) maps to physical Qubit(4) (sequential assignment)
    // Q0 -> P2, Q1 -> P3, Ancilla(2) -> P4
    assert_eq!(layout.get_physical(Qubit::new(0)), Some(Qubit::new(2)));
    assert_eq!(layout.get_physical(Qubit::new(1)), Some(Qubit::new(3)));
    assert_eq!(layout.get_physical(Qubit::new(2)), Some(Qubit::new(4))); // ancilla

    // Verify reverse mapping: physical qubits map to virtual qubits
    assert_eq!(layout.get_virtual(Qubit::new(2)), Some(Qubit::new(0))); // logical
    assert_eq!(layout.get_virtual(Qubit::new(3)), Some(Qubit::new(1))); // logical
    assert_eq!(layout.get_virtual(Qubit::new(4)), Some(Qubit::new(2))); // ancilla
}
