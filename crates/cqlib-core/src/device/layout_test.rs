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
//
// Modified from the original work.

use super::*;
use std::collections::BTreeMap;

#[test]
fn test_layout_basic_creation_leaves_extra_physical_vacant() {
    let logical = vec![LogicalQubit::new(0), LogicalQubit::new(1)];
    let physical = vec![
        PhysicalQubit::new(100),
        PhysicalQubit::new(101),
        PhysicalQubit::new(102),
    ];

    let layout = Layout::new(logical, physical, None).unwrap();

    assert_eq!(layout.num_logical(), 2);
    assert_eq!(layout.num_physical(), 3);
    assert_eq!(layout.num_vacant_physical(), 1);
    assert_eq!(
        layout.vacant_physical_qubits().collect::<Vec<_>>(),
        vec![PhysicalQubit::new(102)]
    );
}

#[test]
fn test_layout_with_init_map_maps_remaining_logical_in_input_order() {
    let logical = vec![LogicalQubit::new(0), LogicalQubit::new(1)];
    let physical = vec![
        PhysicalQubit::new(100),
        PhysicalQubit::new(101),
        PhysicalQubit::new(102),
    ];
    let init_map = [(LogicalQubit::new(1), PhysicalQubit::new(102))]
        .into_iter()
        .collect::<BTreeMap<_, _>>();

    let layout = Layout::new(logical, physical, Some(init_map)).unwrap();

    assert_eq!(
        layout.get_physical(LogicalQubit::new(0)),
        Some(PhysicalQubit::new(100))
    );
    assert_eq!(
        layout.get_physical(LogicalQubit::new(1)),
        Some(PhysicalQubit::new(102))
    );
    assert_eq!(
        layout.vacant_physical_qubits().collect::<Vec<_>>(),
        vec![PhysicalQubit::new(101)]
    );
}

#[test]
fn from_pairs_maps_only_supplied_logical_qubits_and_leaves_vacancies() {
    let layout = Layout::from_pairs(&[(2, 3), (0, 1)], 5).unwrap();

    assert_eq!(layout.num_logical(), 2);
    assert_eq!(layout.num_physical(), 5);
    assert_eq!(layout.num_vacant_physical(), 3);
    assert_eq!(
        layout.get_physical(LogicalQubit::new(2)),
        Some(PhysicalQubit::new(3))
    );
    assert_eq!(
        layout.get_physical(LogicalQubit::new(0)),
        Some(PhysicalQubit::new(1))
    );
    assert_eq!(layout.get_physical(LogicalQubit::new(1)), None);
}

#[test]
fn from_pairs_rejects_duplicate_logical_qubits() {
    let error = Layout::from_pairs(&[(0, 0), (0, 1)], 2).unwrap_err();

    assert!(matches!(
        error,
        LayoutError::DuplicateLogicalQubit(q) if q == LogicalQubit::new(0)
    ));
}

#[test]
fn from_pairs_rejects_duplicate_physical_qubits() {
    let error = Layout::from_pairs(&[(0, 1), (1, 1)], 2).unwrap_err();

    assert!(matches!(
        error,
        LayoutError::PhysicalQubitAlreadyOccupied(q) if q == PhysicalQubit::new(1)
    ));
}

#[test]
fn from_pairs_rejects_physical_qubit_outside_declared_range() {
    let error = Layout::from_pairs(&[(0, 2)], 2).unwrap_err();

    assert!(matches!(
        error,
        LayoutError::InvalidPhysicalQubit(q) if q == PhysicalQubit::new(2)
    ));
}

#[test]
fn test_layout_too_many_logical_error() {
    let result = Layout::new(
        vec![
            LogicalQubit::new(0),
            LogicalQubit::new(1),
            LogicalQubit::new(2),
        ],
        vec![PhysicalQubit::new(100), PhysicalQubit::new(101)],
        None,
    );

    assert!(matches!(
        result.unwrap_err(),
        LayoutError::TooManyLogicalQubits {
            logical: 3,
            physical: 2
        }
    ));
}

#[test]
fn test_layout_duplicate_logical_error() {
    let logical = vec![LogicalQubit::new(0), LogicalQubit::new(0)];
    let physical = vec![PhysicalQubit::new(100), PhysicalQubit::new(101)];

    assert!(matches!(
        Layout::new(logical, physical, None).unwrap_err(),
        LayoutError::DuplicateLogicalQubit(q) if q == LogicalQubit::new(0)
    ));
}

#[test]
fn test_layout_duplicate_physical_error() {
    let logical = vec![LogicalQubit::new(0)];
    let physical = vec![PhysicalQubit::new(100), PhysicalQubit::new(100)];

    assert!(matches!(
        Layout::new(logical, physical, None).unwrap_err(),
        LayoutError::DuplicatePhysicalQubit(q) if q == PhysicalQubit::new(100)
    ));
}

#[test]
fn test_layout_invalid_logical_qubit_error() {
    let init_map = [(LogicalQubit::new(99), PhysicalQubit::new(100))]
        .into_iter()
        .collect();

    let result = Layout::new(
        vec![LogicalQubit::new(0)],
        vec![PhysicalQubit::new(100), PhysicalQubit::new(101)],
        Some(init_map),
    );

    assert!(matches!(
        result.unwrap_err(),
        LayoutError::InvalidLogicalQubit(q) if q == LogicalQubit::new(99)
    ));
}

#[test]
fn test_layout_invalid_physical_qubit_error() {
    let init_map = [(LogicalQubit::new(0), PhysicalQubit::new(999))]
        .into_iter()
        .collect();

    let result = Layout::new(
        vec![LogicalQubit::new(0)],
        vec![PhysicalQubit::new(100), PhysicalQubit::new(101)],
        Some(init_map),
    );

    assert!(matches!(
        result.unwrap_err(),
        LayoutError::InvalidPhysicalQubit(q) if q == PhysicalQubit::new(999)
    ));
}

#[test]
fn test_layout_init_map_rejects_occupied_physical() {
    let init_map = [
        (LogicalQubit::new(0), PhysicalQubit::new(100)),
        (LogicalQubit::new(1), PhysicalQubit::new(100)),
    ]
    .into_iter()
    .collect();

    let result = Layout::new(
        vec![LogicalQubit::new(0), LogicalQubit::new(1)],
        vec![PhysicalQubit::new(100), PhysicalQubit::new(101)],
        Some(init_map),
    );

    assert!(matches!(
        result.unwrap_err(),
        LayoutError::PhysicalQubitAlreadyOccupied(q) if q == PhysicalQubit::new(100)
    ));
}

#[test]
fn test_layout_bind_and_unbind() {
    let mut layout = Layout::new(
        vec![LogicalQubit::new(0)],
        vec![PhysicalQubit::new(100), PhysicalQubit::new(101)],
        None,
    )
    .unwrap();

    assert!(layout.is_physical_vacant(PhysicalQubit::new(101)));
    layout
        .bind(LogicalQubit::new(1), PhysicalQubit::new(101))
        .unwrap();
    assert_eq!(layout.num_logical(), 2);
    assert_eq!(layout.num_vacant_physical(), 0);
    assert_eq!(
        layout.get_logical(PhysicalQubit::new(101)),
        Some(LogicalQubit::new(1))
    );

    assert_eq!(
        layout.unbind(LogicalQubit::new(1)).unwrap(),
        PhysicalQubit::new(101)
    );
    assert!(layout.is_physical_vacant(PhysicalQubit::new(101)));
}

#[test]
fn test_layout_bind_errors() {
    let mut layout = Layout::new(
        vec![LogicalQubit::new(0)],
        vec![PhysicalQubit::new(100), PhysicalQubit::new(101)],
        None,
    )
    .unwrap();

    assert!(matches!(
        layout.bind(LogicalQubit::new(0), PhysicalQubit::new(101)),
        Err(LayoutError::LogicalQubitAlreadyBound(q)) if q == LogicalQubit::new(0)
    ));
    assert!(matches!(
        layout.bind(LogicalQubit::new(1), PhysicalQubit::new(100)),
        Err(LayoutError::PhysicalQubitAlreadyOccupied(q)) if q == PhysicalQubit::new(100)
    ));
    assert!(matches!(
        layout.bind(LogicalQubit::new(1), PhysicalQubit::new(999)),
        Err(LayoutError::InvalidPhysicalQubit(q)) if q == PhysicalQubit::new(999)
    ));
}

#[test]
fn test_layout_unbind_rejects_unmapped_logical() {
    let mut layout = Layout::new(
        vec![LogicalQubit::new(0)],
        vec![PhysicalQubit::new(100)],
        None,
    )
    .unwrap();

    assert!(matches!(
        layout.unbind(LogicalQubit::new(99)),
        Err(LayoutError::LogicalQubitNotBound(q)) if q == LogicalQubit::new(99)
    ));
}

#[test]
fn test_layout_swap_two_occupied_physical() {
    let mut layout = Layout::new(
        vec![LogicalQubit::new(0), LogicalQubit::new(1)],
        vec![PhysicalQubit::new(100), PhysicalQubit::new(101)],
        None,
    )
    .unwrap();

    layout
        .swap_physical(PhysicalQubit::new(100), PhysicalQubit::new(101))
        .unwrap();

    assert_eq!(
        layout.get_physical(LogicalQubit::new(0)),
        Some(PhysicalQubit::new(101))
    );
    assert_eq!(
        layout.get_physical(LogicalQubit::new(1)),
        Some(PhysicalQubit::new(100))
    );
}

#[test]
fn test_layout_swap_occupied_and_vacant_physical() {
    let mut layout = Layout::new(
        vec![LogicalQubit::new(0)],
        vec![PhysicalQubit::new(100), PhysicalQubit::new(101)],
        None,
    )
    .unwrap();

    layout
        .swap_physical(PhysicalQubit::new(100), PhysicalQubit::new(101))
        .unwrap();

    assert_eq!(
        layout.get_physical(LogicalQubit::new(0)),
        Some(PhysicalQubit::new(101))
    );
    assert!(layout.is_physical_vacant(PhysicalQubit::new(100)));
}

#[test]
fn test_layout_swap_vacant_physical_is_noop() {
    let mut layout = Layout::new(
        vec![LogicalQubit::new(0)],
        vec![
            PhysicalQubit::new(100),
            PhysicalQubit::new(101),
            PhysicalQubit::new(102),
        ],
        None,
    )
    .unwrap();

    layout
        .swap_physical(PhysicalQubit::new(101), PhysicalQubit::new(102))
        .unwrap();
    layout
        .swap_physical(PhysicalQubit::new(100), PhysicalQubit::new(100))
        .unwrap();

    assert_eq!(
        layout.get_physical(LogicalQubit::new(0)),
        Some(PhysicalQubit::new(100))
    );
}

#[test]
fn test_layout_swap_rejects_invalid_physical() {
    let mut layout = Layout::new(
        vec![LogicalQubit::new(0)],
        vec![PhysicalQubit::new(100)],
        None,
    )
    .unwrap();

    assert!(matches!(
        layout.swap_physical(PhysicalQubit::new(100), PhysicalQubit::new(999)),
        Err(LayoutError::InvalidPhysicalQubit(q)) if q == PhysicalQubit::new(999)
    ));
}

#[test]
fn test_layout_vacant_physical_iteration_is_sorted() {
    let layout = Layout::new(
        vec![LogicalQubit::new(0)],
        vec![
            PhysicalQubit::new(103),
            PhysicalQubit::new(101),
            PhysicalQubit::new(102),
        ],
        None,
    )
    .unwrap();

    assert_eq!(
        layout.vacant_physical_qubits().collect::<Vec<_>>(),
        vec![PhysicalQubit::new(101), PhysicalQubit::new(102)]
    );
}
