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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::gate::StandardGate;

    #[test]
    fn test_circuit_inverse_structure() {
        let mut circuit = Circuit::new(2);
        let q0 = Qubit::new(0);
        let q1 = Qubit::new(1);

        // H(0) -> S(0) -> CX(0, 1)
        circuit.h(q0).unwrap();
        circuit.s(q0).unwrap();
        circuit.cx(q0, q1).unwrap();

        let inv = circuit.inverse().unwrap();

        assert_eq!(inv.data.len(), 3);

        // Expected: CX(0,1) -> SDG(0) -> H(0)

        // Op 0: CX
        let op0 = &inv.data[0];
        if let Instruction::Standard(g) = op0.instruction {
            assert_eq!(g, StandardGate::CX);
        } else {
            panic!("Expected StandardGate");
        }
        assert_eq!(op0.qubits[0], q0);
        assert_eq!(op0.qubits[1], q1);

        // Op 1: SDG
        let op1 = &inv.data[1];
        if let Instruction::Standard(g) = op1.instruction {
            assert_eq!(g, StandardGate::SDG);
        } else {
            panic!("Expected StandardGate");
        }

        // Op 2: H
        let op2 = &inv.data[2];
        if let Instruction::Standard(g) = op2.instruction {
            assert_eq!(g, StandardGate::H);
        } else {
            panic!("Expected StandardGate");
        }
    }

    #[test]
    fn test_circuit_inverse_parametric() {
        let mut circuit = Circuit::new(1);
        let q0 = Qubit::new(0);
        let theta = Parameter::from("theta");

        // RX(theta)
        circuit.rx(q0, theta.clone()).unwrap();

        let inv = circuit.inverse().unwrap();
        let op0 = &inv.data[0];

        // Expected: RX(-theta)
        // Check params
        // resolving params from circuit
        let p = match &op0.params[0] {
            CircuitParam::Index(i) => inv.parameters[*i as usize].clone(),
            _ => panic!("Expected symbolic param"),
        };

        // p should be -1.0 * theta
        // We can verify by evaluating
        let mut bind = std::collections::HashMap::new();
        bind.insert("theta".to_string(), 1.0);
        assert_eq!(p.evaluate(&Some(bind)).unwrap(), -1.0);
    }

    #[test]
    fn test_inverse_global_phase() {
        let mut circuit = Circuit::new(1);
        circuit.global_phase = CircuitParam::Fixed(0.5);
        let inv = circuit.inverse().unwrap();
        match inv.global_phase {
            CircuitParam::Fixed(v) => assert_eq!(v, -0.5),
            _ => panic!("Expected fixed phase"),
        }
    }

    #[test]
    fn test_irreversible() {
        let mut circuit = Circuit::new(1);
        circuit.measure(Qubit::new(0)).unwrap();
        assert!(matches!(
            circuit.inverse(),
            Err(CircuitError::IrreversibleOperation)
        ));
    }
}
