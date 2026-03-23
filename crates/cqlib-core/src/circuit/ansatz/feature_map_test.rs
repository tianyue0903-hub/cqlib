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
use crate::circuit::ansatz::Ansatz;
use crate::circuit::gate::StandardGate;

#[test]
fn test_angle_encoding() {
    let ansatz = AngleEncoding::new(3, StandardGate::RX);
    assert_eq!(ansatz.num_qubits(), 3);
    assert_eq!(ansatz.num_parameters(), 3);

    let circuit = ansatz.build_circuit("x").unwrap();
    assert_eq!(circuit.num_qubits(), 3);

    // Check parameters
    assert_eq!(circuit.parameters().len(), 3);
    let syms = circuit.symbols();
    assert!(syms.contains("x_0"));
    assert!(syms.contains("x_1"));
    assert!(syms.contains("x_2"));

    // Check circuit operations: 3 RX gates
    let ops = circuit.operations();
    assert_eq!(ops.len(), 3);
}

#[test]
fn test_zz_feature_map() {
    // 2 qubits, 1 layer, Linear entanglement
    let ansatz = ZZFeatureMap::new(2)
        .reps(1)
        .entanglement(EntanglementTopology::Linear);

    assert_eq!(ansatz.num_qubits(), 2);
    // Features are just x_0, x_1
    assert_eq!(ansatz.num_parameters(), 2);

    let circuit = ansatz.build_circuit("f").unwrap();
    // Check parameters
    // We expect 3 parameters to be registered:
    // 1. f_0 * 2 (or 2 * f_0 depending on expression formatting)
    // 2. f_1 * 2 (or 2 * f_1)
    // 3. (π - f_0) * (π - f_1) * 4
    let params: Vec<_> = circuit.parameters().iter().map(|p| p.to_string()).collect();
    assert_eq!(params.len(), 3);
    // Check for either format of the multiplication (with or without spaces)
    assert!(
        params
            .iter()
            .any(|p| p == "f_0 * 2" || p == "2 * f_0" || p == "2*f_0"),
        "Expected 'f_0 * 2', '2 * f_0', or '2*f_0', got: {:?}",
        params
    );
    assert!(
        params
            .iter()
            .any(|p| p == "f_1 * 2" || p == "2 * f_1" || p == "2*f_1"),
        "Expected 'f_1 * 2', '2 * f_1', or '2*f_1', got: {:?}",
        params
    );
    assert!(
        params
            .iter()
            .any(|p| p.contains("π") && p.contains("f_0") && p.contains("f_1") && p.contains("4")),
        "Expected parameter containing π, f_0, f_1, and 4, got: {:?}",
        params
    );

    let syms = circuit.symbols();
    assert!(syms.contains("f_0"));
    assert!(syms.contains("f_1"));

    // Check circuit operations
    // Layer 1:
    // 2 H gates
    // 2 RZ gates
    // 1 ZZ interaction (CNOT, RZ, CNOT = 3 gates)
    // Total ops: 2 + 2 + 3 = 7
    let ops = circuit.operations();
    assert_eq!(ops.len(), 7);
}
