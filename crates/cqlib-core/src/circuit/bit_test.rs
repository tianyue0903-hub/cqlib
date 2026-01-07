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
    let q0 = Qubit::new(0, "q");
    let q1 = Qubit::new(1, "q");

    assert_eq!(q0.id(), 0);
    assert_eq!(q1.id(), 1);
    assert_eq!(q0.register_name(), "q");
    assert_ne!(q0, q1);

    assert_eq!(format!("{}", q0), "q[0]");
    assert_eq!(format!("{}", q1), "q[1]");
}

#[test]
fn test_clbit_creation_and_display() {
    let c0 = Clbit::new(0, "c");
    let c1 = Clbit::new(1, "c");

    assert_eq!(c0.id(), 0);
    assert_eq!(c1.id(), 1);
    assert_eq!(c0.register_name(), "c");
    assert_ne!(c0, c1);

    assert_eq!(format!("{}", c0), "c[0]");
    assert_eq!(format!("{}", c1), "c[1]");
}

#[test]
fn test_quantum_register_creation() {
    let size = 3;
    let name = "qreg";
    let qreg = QuantumRegister::new(name, size);

    assert_eq!(qreg.name(), name);
    assert_eq!(qreg.len(), size);
    assert!(!qreg.is_empty());

    // Verify qubits are initialized with correct IDs 0..size
    for i in 0..size {
        assert_eq!(qreg[i].id(), i);
        assert_eq!(qreg[i].register_name(), name);
    }
}

#[test]
fn test_classical_register_creation() {
    let size = 2;
    let name = "creg";
    let creg = ClassicalRegister::new(name, size);

    assert_eq!(creg.name(), name);
    assert_eq!(creg.len(), size);
    assert!(!creg.is_empty());

    // Verify clbits are initialized with correct IDs 0..size
    for i in 0..size {
        assert_eq!(creg[i].id(), i);
        assert_eq!(creg[i].register_name(), name);
    }
}

#[test]
fn test_quantum_register_indexing() {
    let qreg = QuantumRegister::new("q", 2);
    // Note: Index returns &Qubit, and Qubit is no longer Copy.
    // We can clone if we need an owned Qubit, or just use the reference.
    let q0 = &qreg[0];
    let q1 = &qreg[1];

    assert_eq!(q0.id(), 0);
    assert_eq!(q1.id(), 1);
}

#[test]
#[should_panic]
fn test_quantum_register_index_out_of_bounds() {
    let qreg = QuantumRegister::new("q", 1);
    let _ = &qreg[1]; // Should panic
}

#[test]
fn test_quantum_register_iteration() {
    let size = 3;
    let qreg = QuantumRegister::new("q", size);

    let mut count = 0;
    // &qreg iterates over &Qubit
    for (i, qubit) in (&qreg).into_iter().enumerate() {
        assert_eq!(qubit.id(), i);
        assert_eq!(qubit.register_name(), "q");
        count += 1;
    }
    assert_eq!(count, size);
}

#[test]
fn test_registers_equality_behavior() {
    // New behavior: Qubits from different registers (even with same index)
    // are DIFFERENT because they carry the register name.
    let qreg1 = QuantumRegister::new("q1", 1);
    let qreg2 = QuantumRegister::new("q2", 1);

    // Previously this was equal (Q0 == Q0). Now it is q1[0] != q2[0].
    assert_ne!(qreg1[0], qreg2[0]);

    // Hash consistency
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(qreg1[0].clone()); // Need to clone to own the key
    assert!(!set.contains(&qreg2[0])); // Should NOT contain the qubit from other register
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
