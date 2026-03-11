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
fn test_topology_creation() {
    let qubits = vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let couplings = vec![
        (Qubit::new(0), Qubit::new(1), "CX".to_string()),
        (Qubit::new(1), Qubit::new(2), "CX".to_string()),
    ];

    let topology = Topology::new(qubits, couplings).unwrap();

    assert_eq!(topology.num_qubits(), 3);
    assert_eq!(topology.num_couplings(), 2);
}

#[test]
fn test_is_connected() {
    let qubits = vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let couplings = vec![
        (Qubit::new(0), Qubit::new(1), "CX".to_string()),
        (Qubit::new(1), Qubit::new(2), "CX".to_string()),
    ];

    let topology = Topology::new(qubits, couplings).unwrap();

    assert!(topology.is_connected(Qubit::new(0), Qubit::new(1)));
    assert!(topology.is_connected(Qubit::new(1), Qubit::new(2)));
    assert!(!topology.is_connected(Qubit::new(0), Qubit::new(2)));
}

#[test]
fn test_neighbors() {
    let qubits = vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)];
    let couplings = vec![
        (Qubit::new(0), Qubit::new(1), "CX".to_string()),
        (Qubit::new(0), Qubit::new(2), "CX".to_string()),
    ];

    let topology = Topology::new(qubits, couplings).unwrap();

    let neighbors: Vec<_> = topology.neighbors(Qubit::new(0)).collect();
    assert_eq!(neighbors.len(), 2);
}

#[test]
fn test_degree() {
    let qubits = vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)];
    let couplings = vec![
        (Qubit::new(0), Qubit::new(1), "CX".to_string()),
        (Qubit::new(0), Qubit::new(2), "CX".to_string()),
        (Qubit::new(0), Qubit::new(3), "CX".to_string()),
    ];

    let topology = Topology::new(qubits, couplings).unwrap();

    assert_eq!(topology.degree(&Qubit::new(0)), 3);
}

#[test]
fn test_contains_qubit() {
    let qubits = vec![Qubit::new(0), Qubit::new(1)];
    let topology = Topology::new(qubits, vec![]).unwrap();

    assert!(topology.contains_qubit(&Qubit::new(0)));
    assert!(!topology.contains_qubit(&Qubit::new(2)));
}

#[test]
fn test_get_coupling_name() {
    let qubits = vec![Qubit::new(0), Qubit::new(1)];
    let couplings = vec![(Qubit::new(0), Qubit::new(1), "CX".to_string())];
    let topology = Topology::new(qubits, couplings).unwrap();

    assert_eq!(
        topology.get_coupling_name(Qubit::new(0), Qubit::new(1)),
        Some("CX".to_string())
    );
}

#[test]
fn test_add_qubits() {
    let mut topology = Topology::new(vec![Qubit::new(0)], vec![]).unwrap();
    topology
        .add_qubits(vec![Qubit::new(1), Qubit::new(2)])
        .unwrap();
    assert_eq!(topology.num_qubits(), 3);
}

#[test]
fn test_add_qubits_duplicate_error() {
    let mut topology = Topology::new(vec![Qubit::new(0)], vec![]).unwrap();
    let result = topology.add_qubits(vec![Qubit::new(0), Qubit::new(1)]);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TopologyError::QubitAlreadyExists(_)
    ));
}

#[test]
fn test_add_couplings() {
    let mut topology = Topology::new(vec![Qubit::new(0), Qubit::new(1)], vec![]).unwrap();
    topology
        .add_couplings(vec![(Qubit::new(0), Qubit::new(1), "CX".to_string())])
        .unwrap();
    assert_eq!(topology.num_couplings(), 1);
}

#[test]
fn test_add_couplings_missing_qubit_error() {
    let mut topology = Topology::new(vec![Qubit::new(0)], vec![]).unwrap();
    let result = topology.add_couplings(vec![(Qubit::new(0), Qubit::new(1), "CX".to_string())]);
    assert!(result.is_err());
}

#[test]
fn test_remove_qubits() {
    let mut topology = Topology::new(
        vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(1), Qubit::new(2), "CX".to_string()),
        ],
    )
    .unwrap();

    topology.remove_qubits(vec![Qubit::new(1)]).unwrap();

    assert_eq!(topology.num_qubits(), 2);
    assert!(!topology.is_connected(Qubit::new(0), Qubit::new(1)));
}

#[test]
fn test_remove_qubits_not_found_error() {
    let mut topology = Topology::new(vec![Qubit::new(0)], vec![]).unwrap();
    let result = topology.remove_qubits(vec![Qubit::new(1)]);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TopologyError::QubitNotFound(_)
    ));
}

#[test]
fn test_remove_couplings() {
    let mut topology = Topology::new(
        vec![Qubit::new(0), Qubit::new(1), Qubit::new(2)],
        vec![
            (Qubit::new(0), Qubit::new(1), "CX".to_string()),
            (Qubit::new(1), Qubit::new(2), "CX".to_string()),
        ],
    )
    .unwrap();

    topology
        .remove_couplings(vec![(Qubit::new(0), Qubit::new(1))])
        .unwrap();

    assert_eq!(topology.num_couplings(), 1);
    assert!(!topology.is_connected(Qubit::new(0), Qubit::new(1)));
}

#[test]
fn test_remove_couplings_not_found_error() {
    let mut topology = Topology::new(vec![Qubit::new(0), Qubit::new(1)], vec![]).unwrap();
    let result = topology.remove_couplings(vec![(Qubit::new(0), Qubit::new(1))]);
    assert!(result.is_err());
}
