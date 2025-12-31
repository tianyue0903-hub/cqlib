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

use super::*;

#[test]
fn test_qubit_creation_and_display() {
    let q0 = Qubit { id: 0 };
    let q1 = Qubit { id: 1 };

    assert_eq!(q0.id, 0);
    assert_eq!(q1.id, 1);
    assert_ne!(q0, q1);

    assert_eq!(format!("{}", q0), "Q0");
    assert_eq!(format!("{}", q1), "Q1");
}

#[test]
fn test_clbit_creation_and_display() {
    let c0 = Clbit { id: 0 };
    let c1 = Clbit { id: 1 };

    assert_eq!(c0.id, 0);
    assert_eq!(c1.id, 1);
    assert_ne!(c0, c1);

    assert_eq!(format!("{}", c0), "C0");
    assert_eq!(format!("{}", c1), "C1");
}

#[test]
fn test_quantum_register_creation() {
    let size = 3;
    let name = "qreg";
    let qreg = QuantumRegister::new(name, size);

    assert_eq!(qreg.name, name);
    assert_eq!(qreg.len(), size);
    assert!(!qreg.is_empty());

    // Verify qubits are initialized with correct IDs 0..size
    for i in 0..size {
        assert_eq!(qreg[i].id, i);
    }
}

#[test]
fn test_classical_register_creation() {
    let size = 2;
    let name = "creg";
    let creg = ClassicalRegister::new(name, size);

    assert_eq!(creg.name, name);
    assert_eq!(creg.len(), size);
    assert!(!creg.is_empty());

    // Verify clbits are initialized with correct IDs 0..size
    for i in 0..size {
        assert_eq!(creg[i].id, i);
    }
}

#[test]
fn test_quantum_register_indexing() {
    let qreg = QuantumRegister::new("q", 2);
    let q0 = qreg[0];
    let q1 = qreg[1];

    assert_eq!(q0.id, 0);
    assert_eq!(q1.id, 1);
}

#[test]
#[should_panic]
fn test_quantum_register_index_out_of_bounds() {
    let qreg = QuantumRegister::new("q", 1);
    let _ = qreg[1]; // Should panic
}

#[test]
fn test_quantum_register_iteration() {
    let size = 3;
    let qreg = QuantumRegister::new("q", size);

    let mut count = 0;
    for (i, qubit) in (&qreg).into_iter().enumerate() {
        assert_eq!(qubit.id, i);
        count += 1;
    }
    assert_eq!(count, size);
}

#[test]
fn test_registers_equality_behavior() {
    // Current behavior: IDs are 0-based index per register.
    // So qubits from different registers with same index are "equal" in struct equality.
    let qreg1 = QuantumRegister::new("q1", 1);
    let qreg2 = QuantumRegister::new("q2", 1);

    assert_eq!(qreg1[0], qreg2[0]);

    // Hash consistency
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(qreg1[0]);
    assert!(set.contains(&qreg2[0]));
}

#[test]
fn test_empty_register() {
    let qreg = QuantumRegister::new("empty", 0);
    assert!(qreg.is_empty());
    assert_eq!(qreg.len(), 0);
}

#[test]
fn test_register_display() {
    let qreg = QuantumRegister::new("q", 3);
    assert_eq!(format!("{}", qreg), "QuantumRegister(name='q', size=3)");

    let creg = ClassicalRegister::new("c", 2);
    assert_eq!(format!("{}", creg), "ClassicalRegister(name='c', size=2)");
}
