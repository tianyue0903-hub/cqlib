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
    let qubits = vec![
        PhysicalQubit::new(0),
        PhysicalQubit::new(1),
        PhysicalQubit::new(2),
    ];
    let couplings = vec![
        (
            PhysicalQubit::new(0),
            PhysicalQubit::new(1),
            "CX".to_string(),
        ),
        (
            PhysicalQubit::new(1),
            PhysicalQubit::new(2),
            "CX".to_string(),
        ),
    ];

    let topology = Topology::new(qubits, couplings).unwrap();

    assert_eq!(topology.num_qubits(), 3);
    assert_eq!(topology.num_couplings(), 2);
}

#[test]
fn test_supports_directed_coupling() {
    let qubits = vec![
        PhysicalQubit::new(0),
        PhysicalQubit::new(1),
        PhysicalQubit::new(2),
    ];
    let couplings = vec![
        (
            PhysicalQubit::new(0),
            PhysicalQubit::new(1),
            "CX".to_string(),
        ),
        (
            PhysicalQubit::new(1),
            PhysicalQubit::new(2),
            "CX".to_string(),
        ),
    ];

    let topology = Topology::new(qubits, couplings).unwrap();

    assert!(topology.supports_directed_coupling(PhysicalQubit::new(0), PhysicalQubit::new(1)));
    assert!(topology.supports_directed_coupling(PhysicalQubit::new(1), PhysicalQubit::new(2)));
    assert!(!topology.supports_directed_coupling(PhysicalQubit::new(1), PhysicalQubit::new(0)));
    assert!(!topology.supports_directed_coupling(PhysicalQubit::new(0), PhysicalQubit::new(2)));
}

#[test]
fn test_successors() {
    let qubits = vec![
        PhysicalQubit::new(0),
        PhysicalQubit::new(1),
        PhysicalQubit::new(2),
    ];
    let couplings = vec![
        (
            PhysicalQubit::new(0),
            PhysicalQubit::new(1),
            "CX".to_string(),
        ),
        (
            PhysicalQubit::new(0),
            PhysicalQubit::new(2),
            "CX".to_string(),
        ),
    ];

    let topology = Topology::new(qubits, couplings).unwrap();

    let successors: Vec<_> = topology.successors(PhysicalQubit::new(0)).collect();
    assert_eq!(successors.len(), 2);
}

#[test]
fn test_out_degree() {
    let qubits = vec![
        PhysicalQubit::new(0),
        PhysicalQubit::new(1),
        PhysicalQubit::new(2),
        PhysicalQubit::new(3),
    ];
    let couplings = vec![
        (
            PhysicalQubit::new(0),
            PhysicalQubit::new(1),
            "CX".to_string(),
        ),
        (
            PhysicalQubit::new(0),
            PhysicalQubit::new(2),
            "CX".to_string(),
        ),
        (
            PhysicalQubit::new(0),
            PhysicalQubit::new(3),
            "CX".to_string(),
        ),
    ];

    let topology = Topology::new(qubits, couplings).unwrap();

    assert_eq!(topology.out_degree(&PhysicalQubit::new(0)), 3);
}

#[test]
fn test_contains_qubit() {
    let qubits = vec![PhysicalQubit::new(0), PhysicalQubit::new(1)];
    let topology = Topology::new(qubits, vec![]).unwrap();

    assert!(topology.contains_qubit(&PhysicalQubit::new(0)));
    assert!(!topology.contains_qubit(&PhysicalQubit::new(2)));
}

#[test]
fn test_get_coupling_name() {
    let qubits = vec![PhysicalQubit::new(0), PhysicalQubit::new(1)];
    let couplings = vec![(
        PhysicalQubit::new(0),
        PhysicalQubit::new(1),
        "CX".to_string(),
    )];
    let topology = Topology::new(qubits, couplings).unwrap();

    assert_eq!(
        topology.get_coupling_name(PhysicalQubit::new(0), PhysicalQubit::new(1)),
        Some("CX".to_string())
    );
}

#[test]
fn test_add_qubits() {
    let mut topology = Topology::new(vec![PhysicalQubit::new(0)], vec![]).unwrap();
    topology
        .add_qubits(vec![PhysicalQubit::new(1), PhysicalQubit::new(2)])
        .unwrap();
    assert_eq!(topology.num_qubits(), 3);
}

#[test]
fn test_add_qubits_duplicate_error() {
    let mut topology = Topology::new(vec![PhysicalQubit::new(0)], vec![]).unwrap();
    let result = topology.add_qubits(vec![PhysicalQubit::new(0), PhysicalQubit::new(1)]);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TopologyError::QubitAlreadyExists(_)
    ));
}

#[test]
fn test_add_couplings() {
    let mut topology =
        Topology::new(vec![PhysicalQubit::new(0), PhysicalQubit::new(1)], vec![]).unwrap();
    topology
        .add_couplings(vec![(
            PhysicalQubit::new(0),
            PhysicalQubit::new(1),
            "CX".to_string(),
        )])
        .unwrap();
    assert_eq!(topology.num_couplings(), 1);
}

#[test]
fn test_add_couplings_missing_qubit_error() {
    let mut topology = Topology::new(vec![PhysicalQubit::new(0)], vec![]).unwrap();
    let result = topology.add_couplings(vec![(
        PhysicalQubit::new(0),
        PhysicalQubit::new(1),
        "CX".to_string(),
    )]);
    assert!(result.is_err());
}

#[test]
fn test_remove_qubits() {
    let mut topology = Topology::new(
        vec![
            PhysicalQubit::new(0),
            PhysicalQubit::new(1),
            PhysicalQubit::new(2),
        ],
        vec![
            (
                PhysicalQubit::new(0),
                PhysicalQubit::new(1),
                "CX".to_string(),
            ),
            (
                PhysicalQubit::new(1),
                PhysicalQubit::new(2),
                "CX".to_string(),
            ),
        ],
    )
    .unwrap();

    topology.remove_qubits(vec![PhysicalQubit::new(1)]).unwrap();

    assert_eq!(topology.num_qubits(), 2);
    assert!(!topology.supports_directed_coupling(PhysicalQubit::new(0), PhysicalQubit::new(1)));
}

#[test]
fn test_remove_qubits_not_found_error() {
    let mut topology = Topology::new(vec![PhysicalQubit::new(0)], vec![]).unwrap();
    let result = topology.remove_qubits(vec![PhysicalQubit::new(1)]);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TopologyError::QubitNotFound(_)
    ));
}

#[test]
fn test_remove_couplings() {
    let mut topology = Topology::new(
        vec![
            PhysicalQubit::new(0),
            PhysicalQubit::new(1),
            PhysicalQubit::new(2),
        ],
        vec![
            (
                PhysicalQubit::new(0),
                PhysicalQubit::new(1),
                "CX".to_string(),
            ),
            (
                PhysicalQubit::new(1),
                PhysicalQubit::new(2),
                "CX".to_string(),
            ),
        ],
    )
    .unwrap();

    topology
        .remove_couplings(vec![(PhysicalQubit::new(0), PhysicalQubit::new(1))])
        .unwrap();

    assert_eq!(topology.num_couplings(), 1);
    assert!(!topology.supports_directed_coupling(PhysicalQubit::new(0), PhysicalQubit::new(1)));
}

#[test]
fn test_remove_couplings_not_found_error() {
    let mut topology =
        Topology::new(vec![PhysicalQubit::new(0), PhysicalQubit::new(1)], vec![]).unwrap();
    let result = topology.remove_couplings(vec![(PhysicalQubit::new(0), PhysicalQubit::new(1))]);
    assert!(result.is_err());
}

#[test]
fn new_rejects_duplicate_qubits() {
    let q0 = PhysicalQubit::new(0);

    assert_eq!(
        Topology::new(vec![q0, q0], vec![]).unwrap_err(),
        TopologyError::QubitAlreadyExists(q0)
    );
}

#[test]
fn line_rejects_duplicate_qubits() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);

    assert_eq!(
        Topology::line(vec![q0, q1, q0]).unwrap_err(),
        TopologyError::QubitAlreadyExists(q0)
    );
}

#[test]
fn line_creates_directed_couplings() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let topology = Topology::line(vec![q0, q1, q2]).unwrap();

    assert_eq!(topology.num_qubits(), 3);
    assert_eq!(topology.num_couplings(), 2);
    assert!(topology.supports_directed_coupling(q0, q1));
    assert!(topology.supports_directed_coupling(q1, q2));
    assert!(!topology.supports_directed_coupling(q1, q0));
}

#[test]
fn add_qubits_rejects_batch_duplicates_without_mutation() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let mut topology = Topology::new(vec![q0], vec![]).unwrap();

    assert_eq!(
        topology.add_qubits([q1, q1]).unwrap_err(),
        TopologyError::QubitAlreadyExists(q1)
    );
    assert_eq!(topology.num_qubits(), 1);
    assert!(!topology.contains_qubit(&q1));
}

#[test]
fn new_rejects_duplicate_couplings() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let coupling = (q0, q1, "CX".to_string());

    assert_eq!(
        Topology::new(vec![q0, q1], vec![coupling.clone(), coupling]).unwrap_err(),
        TopologyError::CouplingAlreadyExists {
            control: q0,
            target: q1,
        }
    );
}

#[test]
fn add_couplings_rejects_duplicates_without_mutation() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let mut topology = Topology::new(vec![q0, q1, q2], vec![(q0, q1, "CX".to_string())]).unwrap();

    assert_eq!(
        topology
            .add_couplings([(q1, q2, "CX".to_string()), (q0, q1, "CX".to_string())])
            .unwrap_err(),
        TopologyError::CouplingAlreadyExists {
            control: q0,
            target: q1,
        }
    );
    assert_eq!(topology.num_couplings(), 1);
    assert!(!topology.supports_directed_coupling(q1, q2));

    let mut topology = Topology::new(vec![q0, q1], vec![]).unwrap();
    assert_eq!(
        topology
            .add_couplings([(q0, q1, "CX".to_string()), (q0, q1, "CZ".to_string())])
            .unwrap_err(),
        TopologyError::CouplingAlreadyExists {
            control: q0,
            target: q1,
        }
    );
    assert_eq!(topology.num_couplings(), 0);
}

#[test]
fn couplings_reject_self_loops() {
    let q0 = PhysicalQubit::new(0);

    assert_eq!(
        Topology::new(vec![q0], vec![(q0, q0, "CX".to_string())]).unwrap_err(),
        TopologyError::SelfCoupling { qubit: q0 }
    );
}

#[test]
fn reverse_couplings_are_distinct() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let topology = Topology::new(
        vec![q0, q1],
        vec![(q0, q1, "CX".to_string()), (q1, q0, "CX".to_string())],
    )
    .unwrap();

    assert_eq!(topology.num_couplings(), 2);
    assert!(topology.supports_directed_coupling(q0, q1));
    assert!(topology.supports_directed_coupling(q1, q0));
}

#[test]
fn predecessors_and_degrees_follow_edge_direction() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let topology = Topology::new(
        vec![q0, q1, q2],
        vec![(q0, q1, "CX".to_string()), (q2, q1, "CX".to_string())],
    )
    .unwrap();

    let mut predecessors: Vec<_> = topology.predecessors(q1).collect();
    predecessors.sort();
    assert_eq!(predecessors, vec![q0, q2]);
    assert_eq!(topology.in_degree(&q1), 2);
    assert_eq!(topology.out_degree(&q1), 0);
}

#[test]
fn supports_coupling_either_direction_checks_both_directions() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let topology = Topology::new(vec![q0, q1, q2], vec![(q0, q1, "CX".to_string())]).unwrap();

    assert!(topology.supports_coupling_either_direction(q0, q1));
    assert!(topology.supports_coupling_either_direction(q1, q0));
    assert!(!topology.supports_coupling_either_direction(q0, q2));
}

#[test]
fn neighbors_undirected_merges_successors_and_predecessors() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let q3 = PhysicalQubit::new(3);
    let topology = Topology::new(
        vec![q0, q1, q2, q3],
        vec![
            (q0, q1, "CX".to_string()),
            (q1, q0, "CX".to_string()),
            (q2, q1, "CZ".to_string()),
        ],
    )
    .unwrap();

    assert_eq!(
        topology.neighbors_undirected(q1).collect::<Vec<_>>(),
        vec![q0, q2]
    );
    assert!(
        topology
            .neighbors_undirected(q3)
            .collect::<Vec<_>>()
            .is_empty()
    );
}

#[test]
fn undirected_edges_collapses_reverse_couplings() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let topology = Topology::new(
        vec![q0, q1, q2],
        vec![
            (q1, q0, "CX".to_string()),
            (q0, q1, "CX".to_string()),
            (q2, q1, "CZ".to_string()),
        ],
    )
    .unwrap();

    assert_eq!(
        topology.undirected_edges().collect::<Vec<_>>(),
        vec![(q0, q1), (q1, q2)]
    );
}

#[test]
fn remove_qubits_rejects_batch_duplicates_without_mutation() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let mut topology = Topology::new(vec![q0, q1], vec![]).unwrap();

    assert_eq!(
        topology.remove_qubits([q1, q1]).unwrap_err(),
        TopologyError::DuplicateQubitRemoval(q1)
    );
    assert_eq!(topology.num_qubits(), 2);
    assert!(topology.contains_qubit(&q1));
}

#[test]
fn remove_couplings_rejects_batch_duplicates_without_mutation() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let mut topology = Topology::new(vec![q0, q1], vec![(q0, q1, "CX".to_string())]).unwrap();

    assert_eq!(
        topology.remove_couplings([(q0, q1), (q0, q1)]).unwrap_err(),
        TopologyError::DuplicateCouplingRemoval {
            control: q0,
            target: q1,
        }
    );
    assert_eq!(topology.num_couplings(), 1);
    assert!(topology.supports_directed_coupling(q0, q1));
}
