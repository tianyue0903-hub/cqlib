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
