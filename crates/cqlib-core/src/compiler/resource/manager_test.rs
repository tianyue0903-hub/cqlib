// This code is part of Cqlib.
//
// (C) Copyright China Telecom Quantum Group 2025-2026
//
// This code is licensed under the Apache License, Version 2.0.
// You may obtain a copy of this license in the LICENSE.txt file in
// the root directory of this source tree or at
// http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::{
    AncillaRequirement, ResourceError, ResourceLimits, ResourceManager, ResourcePolicy,
    ResourceRequest,
};
use crate::circuit::{Circuit, Qubit};
use std::collections::BTreeSet;

#[test]
fn from_circuit_registers_input_qubits() {
    let circuit = Circuit::from_qubits(vec![Qubit::new(4), Qubit::new(1)]).unwrap();
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 2,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    assert_eq!(plan.qubits(), &[Qubit::new(1), Qubit::new(4)]);
    manager.verify_consistency(&circuit).unwrap();
}

#[test]
fn from_circuit_rejects_initial_capacity_overflow() {
    let circuit = Circuit::new(2);
    let error = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy::default(),
        ResourceLimits {
            max_total_qubits: Some(1),
        },
    )
    .unwrap_err();

    assert_eq!(
        error,
        ResourceError::CapacityExceeded {
            limit: 1,
            requested_total: 2,
        }
    );
}

#[test]
fn preview_is_side_effect_free() {
    let mut circuit = Circuit::new(0);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 1,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let request = ResourceRequest {
        requirement: AncillaRequirement::CleanZero,
        count: 1,
        excluded: BTreeSet::from([]),
    };

    let first = manager.preview(&request).unwrap();
    let second = manager.preview(&request).unwrap();

    assert_eq!(first, second);
    assert_eq!(circuit.num_qubits(), 0);
    manager.commit(&mut circuit, first).unwrap();
}

#[test]
fn clean_commit_allocates_new_qubits() {
    let mut circuit = Circuit::from_qubits(vec![Qubit::new(1), Qubit::new(4)]).unwrap();
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 2,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 2,
            excluded: BTreeSet::from([]),
        })
        .unwrap();

    assert_eq!(plan.qubits(), &[Qubit::new(5), Qubit::new(6)]);
    assert_eq!(plan.requirement(), AncillaRequirement::CleanZero);
    assert_eq!(plan.num_new_qubits(), 2);
    let lease = manager.commit(&mut circuit, plan).unwrap();
    assert_eq!(lease.qubits(), &[Qubit::new(5), Qubit::new(6)]);
    assert_eq!(
        circuit.qubits(),
        vec![Qubit::new(1), Qubit::new(4), Qubit::new(5), Qubit::new(6)]
    );
}

#[test]
fn clean_release_allows_reuse() {
    let mut circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 1,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let first_plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    let first_lease = manager.commit(&mut circuit, first_plan).unwrap();
    manager.release(&first_lease).unwrap();

    let reused = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    assert_eq!(reused.qubits(), &[Qubit::new(1)]);
    assert_eq!(reused.num_new_qubits(), 0);
}

#[test]
fn clean_request_never_borrows_input_qubits() {
    let circuit = Circuit::new(3);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap_err();

    assert_eq!(
        error,
        ResourceError::InsufficientResources {
            requirement: "clean-zero",
            requested: 1,
            available: 0,
        }
    );
}

#[test]
fn clean_request_respects_policy_limit() {
    let circuit = Circuit::new(0);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 1,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 2,
            excluded: BTreeSet::from([]),
        })
        .unwrap_err();

    assert_eq!(
        error,
        ResourceError::InsufficientResources {
            requirement: "clean-zero",
            requested: 2,
            available: 1,
        }
    );
}

#[test]
fn clean_request_respects_total_capacity() {
    let circuit = Circuit::new(2);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 1,
            allow_dirty_borrowing: false,
        },
        ResourceLimits {
            max_total_qubits: Some(2),
        },
    )
    .unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap_err();

    assert_eq!(
        error,
        ResourceError::CapacityExceeded {
            limit: 2,
            requested_total: 3,
        }
    );
}

#[test]
fn dirty_request_borrows_non_excluded_input_qubits() {
    let circuit = Circuit::new(3);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 2,
            excluded: BTreeSet::from([]),
        })
        .unwrap();

    assert_eq!(plan.qubits(), &[Qubit::new(0), Qubit::new(1)]);
    assert_eq!(plan.requirement(), AncillaRequirement::Dirty);
    assert_eq!(plan.num_new_qubits(), 0);
}

#[test]
fn dirty_request_skips_excluded_input_qubits() {
    let circuit = Circuit::new(3);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([Qubit::new(0), Qubit::new(2)]),
        })
        .unwrap();

    assert_eq!(plan.qubits(), &[Qubit::new(1)]);
}

#[test]
fn dirty_request_prefers_existing_compiler_ancillas() {
    let mut circuit = Circuit::new(2);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 1,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let clean_plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    let clean_lease = manager.commit(&mut circuit, clean_plan).unwrap();
    manager.release(&clean_lease).unwrap();

    let dirty_plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();

    assert_eq!(dirty_plan.qubits(), &[Qubit::new(2)]);
}

#[test]
fn dirty_request_does_not_allocate_new_qubits() {
    let circuit = Circuit::new(0);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 3,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap_err();

    assert!(matches!(
        error,
        ResourceError::InsufficientResources {
            requirement: "dirty",
            requested: 1,
            available: 0,
        }
    ));
    assert_eq!(circuit.num_qubits(), 0);
}

#[test]
fn dirty_request_rejected_when_policy_forbids_borrowing() {
    let circuit = Circuit::new(2);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap_err();

    assert_eq!(
        error,
        ResourceError::InsufficientResources {
            requirement: "dirty",
            requested: 1,
            available: 0,
        }
    );
}

#[test]
fn active_lease_prevents_overlapping_use() {
    let mut circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let first_plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    manager.commit(&mut circuit, first_plan).unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap_err();

    assert_eq!(
        error,
        ResourceError::InsufficientResources {
            requirement: "dirty",
            requested: 1,
            available: 0,
        }
    );
}

#[test]
fn stale_plan_is_rejected_after_commit() {
    let mut circuit = Circuit::new(2);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let request = ResourceRequest {
        requirement: AncillaRequirement::Dirty,
        count: 1,
        excluded: BTreeSet::from([]),
    };
    let committed = manager.preview(&request).unwrap();
    let stale = manager.preview(&request).unwrap();
    manager.commit(&mut circuit, committed).unwrap();

    let error = manager.commit(&mut circuit, stale).unwrap_err();

    assert_eq!(
        error,
        ResourceError::StalePlan {
            plan_revision: 0,
            current_revision: 1,
        }
    );
}

#[test]
fn stale_plan_is_rejected_after_release() {
    let mut circuit = Circuit::new(2);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let first_plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    let first_lease = manager.commit(&mut circuit, first_plan).unwrap();
    let stale = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    manager.release(&first_lease).unwrap();

    let error = manager.commit(&mut circuit, stale).unwrap_err();

    assert_eq!(
        error,
        ResourceError::StalePlan {
            plan_revision: 1,
            current_revision: 2,
        }
    );
}

#[test]
fn release_rejects_duplicate_release() {
    let mut circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    let lease = manager.commit(&mut circuit, plan).unwrap();
    manager.release(&lease).unwrap();

    let error = manager.release(&lease).unwrap_err();

    assert_eq!(
        error,
        ResourceError::UnknownLease {
            lease_id: lease.id()
        }
    );
}

#[test]
fn post_layout_clean_request_reuses_pre_layout_compiler_ancilla() {
    let mut circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 1,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    let lease = manager.commit(&mut circuit, plan).unwrap();
    manager.release(&lease).unwrap();
    manager.enter_post_layout(&circuit).unwrap();

    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();

    assert_eq!(plan.qubits(), &[Qubit::new(1)]);
    assert_eq!(plan.num_new_qubits(), 0);
    assert_eq!(circuit.qubits(), vec![Qubit::new(0), Qubit::new(1)]);
}

#[test]
fn post_layout_clean_request_does_not_allocate_new_qubits() {
    let circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 1,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    manager.enter_post_layout(&circuit).unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap_err();

    assert_eq!(
        error,
        ResourceError::InsufficientResources {
            requirement: "clean-zero",
            requested: 1,
            available: 0,
        }
    );
    assert_eq!(circuit.num_qubits(), 1);
}

#[test]
fn verify_consistency_detects_circuit_resource_mismatch() {
    let mut circuit = Circuit::new(1);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    circuit.add_qubits(vec![Qubit::new(1)]).unwrap();

    let error = manager.verify_consistency(&circuit).unwrap_err();

    assert!(matches!(
        error,
        ResourceError::CircuitResourceMismatch { .. }
    ));
}

#[test]
fn verify_idle_rejects_active_lease() {
    let mut circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    manager.commit(&mut circuit, plan).unwrap();

    let error = manager.verify_idle(&circuit).unwrap_err();

    assert_eq!(error, ResourceError::ActiveLeaseRemaining { count: 1 });
}

#[test]
fn allocation_rejects_qubit_id_overflow() {
    let circuit = Circuit::from_qubits(vec![Qubit::new(u32::MAX)]).unwrap();
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 1,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::CleanZero,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap_err();

    assert_eq!(error, ResourceError::QubitIdOverflow);
}

#[test]
fn selection_order_is_deterministic() {
    let circuit = Circuit::from_qubits(vec![Qubit::new(7), Qubit::new(2), Qubit::new(5)]).unwrap();
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let first = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 2,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    let second = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 2,
            excluded: BTreeSet::from([]),
        })
        .unwrap();

    assert_eq!(first.qubits(), &[Qubit::new(2), Qubit::new(5)]);
    assert_eq!(first, second);
}

#[test]
fn zero_count_request_is_rejected() {
    let circuit = Circuit::new(1);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 0,
            excluded: BTreeSet::from([]),
        })
        .unwrap_err();

    assert!(matches!(error, ResourceError::InvalidRequest { .. }));
}

#[test]
fn request_rejects_unknown_excluded_qubit() {
    let circuit = Circuit::new(1);
    let manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();

    let error = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([Qubit::new(9)]),
        })
        .unwrap_err();

    assert!(matches!(error, ResourceError::InvalidRequest { .. }));
}

#[test]
fn commit_rejects_circuit_resource_mismatch() {
    let mut circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    circuit.add_qubits(vec![Qubit::new(9)]).unwrap();

    let error = manager.commit(&mut circuit, plan).unwrap_err();

    assert!(matches!(
        error,
        ResourceError::CircuitResourceMismatch { .. }
    ));
    let unchanged = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    assert_eq!(unchanged.qubits(), &[Qubit::new(0)]);
}

#[test]
fn enter_post_layout_rejects_active_lease() {
    let mut circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let plan = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    manager.commit(&mut circuit, plan).unwrap();

    let error = manager.enter_post_layout(&circuit).unwrap_err();

    assert_eq!(error, ResourceError::ActiveLeaseRemaining { count: 1 });
}

#[test]
fn enter_post_layout_rejects_circuit_resource_mismatch() {
    let mut circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    circuit.add_qubits(vec![Qubit::new(9)]).unwrap();

    let error = manager.enter_post_layout(&circuit).unwrap_err();

    assert!(matches!(
        error,
        ResourceError::CircuitResourceMismatch { .. }
    ));
}

#[test]
fn commit_rejects_plan_from_another_manager() {
    let circuit = Circuit::new(1);
    let source = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let mut destination = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: false,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let plan = source
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    let mut circuit = circuit;

    let error = destination.commit(&mut circuit, plan).unwrap_err();

    assert_eq!(error, ResourceError::ForeignPlan);
    destination.verify_idle(&circuit).unwrap();
}

#[test]
fn release_rejects_lease_from_another_manager() {
    let mut first_circuit = Circuit::new(1);
    let mut first = ResourceManager::from_circuit(
        &first_circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let first_plan = first
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    let first_lease = first.commit(&mut first_circuit, first_plan).unwrap();

    let mut second_circuit = Circuit::new(1);
    let mut second = ResourceManager::from_circuit(
        &second_circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let second_plan = second
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    let second_lease = second.commit(&mut second_circuit, second_plan).unwrap();

    let error = second.release(&first_lease).unwrap_err();

    assert_eq!(
        error,
        ResourceError::UnknownLease {
            lease_id: first_lease.id()
        }
    );
    assert_eq!(
        second.verify_idle(&second_circuit).unwrap_err(),
        ResourceError::ActiveLeaseRemaining { count: 1 }
    );
    second.release(&second_lease).unwrap();
    second.verify_idle(&second_circuit).unwrap();
}

#[test]
fn stale_plan_is_rejected_after_entering_post_layout() {
    let mut circuit = Circuit::new(1);
    let mut manager = ResourceManager::from_circuit(
        &circuit,
        ResourcePolicy {
            max_pre_layout_clean_ancillas: 0,
            allow_dirty_borrowing: true,
        },
        ResourceLimits::default(),
    )
    .unwrap();
    let stale = manager
        .preview(&ResourceRequest {
            requirement: AncillaRequirement::Dirty,
            count: 1,
            excluded: BTreeSet::from([]),
        })
        .unwrap();
    manager.enter_post_layout(&circuit).unwrap();

    let error = manager.commit(&mut circuit, stale).unwrap_err();

    assert!(matches!(error, ResourceError::StalePlan { .. }));
}
