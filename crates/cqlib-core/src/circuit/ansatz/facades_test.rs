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
use crate::qis::pauli::PauliString;

#[test]
fn test_real_amplitudes() {
    let ansatz = real_amplitudes(3, 2, EntanglementTopology::Linear);
    let circuit = ansatz.build_circuit("theta").unwrap();

    // 3 qubits
    assert_eq!(circuit.num_qubits(), 3);

    // Reps = 2, so 3 layers of RY.
    // Each layer has 3 qubits = 9 parameters total.
    assert_eq!(ansatz.num_parameters(), 9);
    assert_eq!(circuit.parameters().len(), 9);

    // Operations:
    // Layer 0: 3 RY, 2 CX
    // Layer 1: 3 RY, 2 CX
    // Layer 2: 3 RY
    // Total ops: 9 RY + 4 CX = 13
    assert_eq!(circuit.operations().len(), 13);
}

#[test]
fn test_efficient_su2() {
    let ansatz = efficient_su2(2, 1, EntanglementTopology::Full);
    let circuit = ansatz.build_circuit("p").unwrap();

    // 2 qubits
    assert_eq!(circuit.num_qubits(), 2);

    // Reps = 1, so 2 layers of [RY, RZ].
    // Each layer: 2 qubits * 2 gates = 4 parameters.
    // Total 2 layers * 4 = 8 parameters.
    assert_eq!(ansatz.num_parameters(), 8);
    assert_eq!(circuit.parameters().len(), 8);
}

#[test]
fn test_zz_feature_map_basic() {
    let fm = zz_feature_map(3, 1, EntanglementTopology::Full);
    let circuit = fm.build_circuit("x").unwrap();

    assert_eq!(fm.num_qubits(), 3);
    // Always num_qubits parameters regardless of reps
    assert_eq!(fm.num_parameters(), 3);
    // Feature symbols x_0, x_1, x_2 must all be present; π is also a symbol
    // (used internally in the ZZ angle formula ∝ (π − x_i)(π − x_j)).
    let syms = circuit.symbols();
    assert!(syms.contains("x_0") && syms.contains("x_1") && syms.contains("x_2"));

    // 1 rep, Full topology (3 pairs: (0,1),(0,2),(1,2)):
    // H×3 + RZ×3 + 3×(CNOT,RZ,CNOT) = 6 + 9 = 15
    assert_eq!(circuit.operations().len(), 15);
}

#[test]
fn test_zz_feature_map_reps_multiplies_depth() {
    let fm_1rep = zz_feature_map(2, 1, EntanglementTopology::Full);
    let fm_2rep = zz_feature_map(2, 2, EntanglementTopology::Full);

    let c1 = fm_1rep.build_circuit("x").unwrap();
    let c2 = fm_2rep.build_circuit("x").unwrap();

    // Reps doubles the number of operations
    assert_eq!(c2.operations().len(), c1.operations().len() * 2);
    // Parameters never change — always num_qubits
    assert_eq!(fm_1rep.num_parameters(), fm_2rep.num_parameters());
}

#[test]
fn test_zz_feature_map_linear_topology() {
    // Linear topology: only adjacent pairs (0,1),(1,2) for 3 qubits
    let fm = zz_feature_map(3, 1, EntanglementTopology::Linear);
    let circuit = fm.build_circuit("x").unwrap();

    // H×3 + RZ×3 + 2×(CNOT,RZ,CNOT) = 6 + 6 = 12
    assert_eq!(circuit.operations().len(), 12);
}

#[test]
fn test_pauli_feature_map_facade_z_zz() {
    let paulis = vec![
        ("Z".parse::<PauliString>().unwrap(), "Z".to_string()),
        ("ZZ".parse::<PauliString>().unwrap(), "ZZ".to_string()),
    ];
    let fm = pauli_feature_map(2, 1, paulis, EntanglementTopology::Full);
    let circuit = fm.build_circuit("x").unwrap();

    assert_eq!(fm.num_qubits(), 2);
    assert_eq!(fm.num_parameters(), 2);
    // Feature symbols x_0, x_1 must be present; π also appears in ZZ angle formula.
    let syms = circuit.symbols();
    assert!(syms.contains("x_0") && syms.contains("x_1"));

    // 1 rep, 2 qubits, Z+ZZ with Full topology:
    // H×2 + Z[0](CNOT-less, 1 gate) + Z[1](1 gate) + ZZ[0,1](CNOT,RZ,CNOT = 3 gates) = 2+2+3 = 7
    assert_eq!(circuit.operations().len(), 7);
}

#[test]
fn test_pauli_feature_map_facade_zzz_3local() {
    // 3 qubits with ZZZ: all C(3,3)=1 triple → 1 evolution per rep
    let paulis = vec![("ZZZ".parse::<PauliString>().unwrap(), "ZZZ".to_string())];
    let fm = pauli_feature_map(3, 1, paulis, EntanglementTopology::Full);
    let circuit = fm.build_circuit("x").unwrap();

    assert_eq!(fm.num_qubits(), 3);
    assert_eq!(fm.num_parameters(), 3);

    // 1 rep: H×3 + ZZZ(CNOT,CNOT,RZ,CNOT,CNOT = 5 gates) = 3 + 5 = 8
    assert_eq!(circuit.operations().len(), 8);
}

#[test]
fn test_pauli_feature_map_facade_invalid_zero_qubits() {
    let fm = pauli_feature_map(0, 1, vec![], EntanglementTopology::Full);
    assert!(fm.build_circuit("x").is_err());
}
