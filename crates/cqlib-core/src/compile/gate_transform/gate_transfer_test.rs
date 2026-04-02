use super::*;
use crate::circuit::Qubit;
use crate::circuit::gate::MCGate;
use crate::circuit::gate::control_flow::ConditionView;
use crate::circuit::gate::{CircuitGate, FrozenCircuit, UnitaryGate};
use crate::circuit::param::ParameterValue;
use crate::circuit::parameter::impls::Parameter;
use num::complex::Complex;
use num::complex::ComplexFloat;
use std::sync::Arc;

#[test]
fn test_gate_transform_basic() {
    // Create a simple circuit with CZ gate
    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    circuit.h(q0).unwrap();
    circuit.cz(q0, q1).unwrap();

    // Create instruction set targeting CX
    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX],
        None,
    );

    let mut gt = GateTransform::new(iset);
    let result = gt.execute(&circuit);

    // Verify the result contains CX not CZ
    let ops = result.operations();
    let has_cx = ops.iter().any(|op| {
        if let Instruction::Standard(sgate) = &op.instruction {
            *sgate == StandardGate::CX
        } else {
            false
        }
    });
    let has_cz = ops.iter().any(|op| {
        if let Instruction::Standard(sgate) = &op.instruction {
            *sgate == StandardGate::CZ
        } else {
            false
        }
    });

    assert!(has_cx, "Result should contain CX gate");
    assert!(!has_cz, "Result should not contain CZ gate");
}

fn generate_full_circuit() -> Circuit {
    // Create a 3-qubit circuit
    let mut circuit = Circuit::new(3);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    // --- Single Qubit Gates --- //
    // Pauli gates
    circuit.x(q0).unwrap();
    circuit.y(q0).unwrap();
    circuit.z(q0).unwrap();
    circuit.i(q0).unwrap();

    // Clifford gates
    circuit.h(q0).unwrap();
    circuit.s(q1).unwrap();
    circuit.sdg(q2).unwrap();
    circuit.t(q0).unwrap();
    circuit.tdg(q1).unwrap();

    // // Sqrt gates
    circuit.x2p(q2).unwrap();
    circuit.x2m(q0).unwrap();
    circuit.y2p(q1).unwrap();
    circuit.y2m(q2).unwrap();

    // --- Directives --- //
    circuit.measure(q0).unwrap();
    circuit.measure(q1).unwrap();
    circuit.measure(q2).unwrap();

    // // Parametric gates with parameters
    circuit.rx(q0, 0.5).unwrap();
    circuit.ry(q1, 0.5).unwrap();
    circuit.rz(q2, 0.5).unwrap();
    circuit.phase(q0, 0.5).unwrap();
    circuit.xy(q1, 0.5).unwrap();
    circuit.xy2p(q2, 0.5).unwrap();
    circuit.xy2m(q0, 0.5).unwrap();
    circuit.rxy(q1, 0.5, 0.5).unwrap(); // Two parameters
    circuit.u(q2, 0.5, 0.5, 0.5).unwrap(); // Three parameters

    // --- Two Qubit Gates --- //
    // Non-parametric
    circuit.cx(q0, q1).unwrap();
    circuit.cy(q1, q2).unwrap();
    circuit.cz(q0, q2).unwrap();
    circuit.swap(q0, q1).unwrap();

    // --- Directives --- //
    circuit.barrier(vec![q0, q1, q2]).unwrap();

    // Parametric
    circuit.rxx(q0, q1, 0.5).unwrap();
    circuit.ryy(q1, q2, 0.5).unwrap();
    circuit.rzz(q0, q2, 0.5).unwrap();
    circuit.rzx(q0, q1, 0.5).unwrap();
    circuit.crx(q1, q2, 0.5).unwrap();
    circuit.cry(q0, q1, 0.5).unwrap();
    circuit.crz(q1, q2, 0.5).unwrap();
    // circuit.fsim(q0, q2, 0.5, 0.5).unwrap(); // Two parameters

    // --- Three Qubit Gates --- //
    circuit.ccx(q0, q1, q2).unwrap();
    circuit.ccx(q0, q2, q1).unwrap();

    // --- Directives --- //
    circuit.reset(q0).unwrap();
    circuit.reset(q1).unwrap();
    circuit.reset(q2).unwrap();

    circuit
}

/// Check if all gates in the circuit are in the instruction set or are directives
fn check_all_gates_in_instruction_set(circuit: &Circuit, instruction_set: &InstructionSet) -> bool {
    for op in circuit.operations() {
        match &op.instruction {
            Instruction::Standard(sgate) => {
                // For single-qubit gates, check if they can be decomposed
                // For multi-qubit gates, check if they can be transformed
                if sgate.num_qubits() == 1 {
                    // Single-qubit gates should be decomposable
                    // We'll assume the instruction set can handle them
                    if !instruction_set.single_qubit_gates.contains(sgate) {
                        return false;
                    }
                } else if sgate.num_qubits() == 2 {
                    // Two-qubit gates should be transformable
                    if !instruction_set.double_qubit_gate.contains(sgate) {
                        return false;
                    }
                } else {
                    // CCX is a special case
                    return false;
                }
            }
            Instruction::UnitaryGate(_) => {
                // Unitary gates are not allowed
                return false;
            }
            _ => {
                // Other instruction types are allowed
            }
        }
    }
    true
}

#[test]
fn test_gate_transform_identity_elimination() {
    // Create a circuit where single-qubit gates cancel out
    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);
    circuit.h(q0).unwrap();
    circuit.h(q0).unwrap();

    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX],
        None,
    );

    let mut gt = GateTransform::new(iset);
    let result = gt.execute(&circuit);

    // H·H = I, so no gates should remain
    // Note: Current implementation doesn't perform identity elimination
    // This test is kept for future implementation
    assert_eq!(
        result.operations().len(),
        0,
        "H·H should cancel to identity"
    );
}

fn complex_inner_product(vec1: &[Complex<f64>], vec2: &[Complex<f64>]) -> Complex<f64> {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| a.conj() * b)
        .sum()
}

fn is_matrix_differ_by_phase(matrix1: &Array2<Complex<f64>>, matrix2: &Array2<Complex<f64>>) -> bool {
    let vec1: Vec<Complex<f64>> = matrix1.iter().copied().collect();
    let vec2: Vec<Complex<f64>> = matrix2.iter().copied().collect();
    let inner: Complex<f64> = complex_inner_product(&vec1, &vec2);
    let inner_abs: f64 = inner.abs();
    let vec1_norm: f64 = complex_inner_product(&vec1, &vec1).re.sqrt();
    let vec2_norm: f64 = complex_inner_product(&vec2, &vec2).re.sqrt();

    let cos_vec = inner_abs / (vec1_norm * vec2_norm);
    (cos_vec - 1.0).abs() < 1e-10
}

fn gate_transfer_circuit_test(iset: &InstructionSet, circuit: &Circuit) {
    let mut gt = GateTransform::new(iset.clone());
    let result = gt.execute(&circuit);
    assert!(
        check_all_gates_in_instruction_set(&result, &mut iset.clone()),
        "Gate transfer contains other Standard Gate not in iset."
    );

    let result_matrix = result.to_matrix(None);
    let circuit_matrix = circuit.to_matrix(None);

    if !is_matrix_differ_by_phase(&result_matrix, &circuit_matrix) {
        eprintln!("Assertion failed! Result circuit operations:");
        for (i, op) in result.operations().iter().enumerate() {
            eprintln!(
                "  [{}] {:?} on qubits {:?}, with parameter {:?}",
                i, op.instruction, op.qubits, op.params
            );
        }
        eprintln!("Original circuit operations:");
        for (i, op) in circuit.operations().iter().enumerate() {
            eprintln!("  [{}] {:?} on qubits {:?}", i, op.instruction, op.qubits);
        }
    }

    assert!(
        is_matrix_differ_by_phase(&result_matrix, &circuit_matrix),
        "Gate transfer should not change circuit size {}",
        result.operations().len()
    );
}

fn gate_transfer_symbolic_circuit_test(
    iset: &InstructionSet,
    circuit: &Circuit,
    bindings: HashMap<String, f64>,
) {
    let mut gt = GateTransform::new(iset.clone());
    let result = gt.execute(circuit);

    let bound_result = result.assign_parameters(&Some(bindings.clone())).unwrap();
    let bound_circuit = circuit.assign_parameters(&Some(bindings)).unwrap();

    let result_matrix = bound_result.to_matrix(None);
    let circuit_matrix = bound_circuit.to_matrix(None);

    assert!(
        is_matrix_differ_by_phase(&result_matrix, &circuit_matrix),
        "Symbolic gate transfer should preserve circuit semantics"
    );
}

#[test]
fn test_gate_transfer_full_circuit_with_cx() {
    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_cz() {
    let iset = InstructionSet::new(
        vec![
            StandardGate::X2M,
            StandardGate::X2P,
            StandardGate::Y2P,
            StandardGate::Y2M,
            StandardGate::RZ,
        ],
        vec![StandardGate::CZ, StandardGate::CX],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_cy() {
    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
        vec![StandardGate::CY],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_dynamic_iset() {
    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
        vec![StandardGate::CY],
        None,
    );
    let new_iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
        vec![StandardGate::CX],
        None,
    );
    let mut gt = GateTransform::new(iset.clone());
    gt.set_instruction_set(new_iset.clone());

    let circuit = generate_full_circuit();
    let result = gt.execute(&circuit);
    assert!(
        check_all_gates_in_instruction_set(&result, &mut new_iset.clone()),
        "Gate transfer contains other Standard Gate not in iset."
    );

    let result_matrix = result.to_matrix(None);
    let circuit_matrix = circuit.to_matrix(None);

    if !is_matrix_differ_by_phase(&result_matrix, &circuit_matrix) {
        eprintln!("Assertion failed! Result circuit operations:");
        for (i, op) in result.operations().iter().enumerate() {
            eprintln!(
                "  [{}] {:?} on qubits {:?}, with parameter {:?}",
                i, op.instruction, op.qubits, op.params
            );
        }
        eprintln!("Original circuit operations:");
        for (i, op) in circuit.operations().iter().enumerate() {
            eprintln!("  [{}] {:?} on qubits {:?}", i, op.instruction, op.qubits);
        }
    }

    assert!(
        is_matrix_differ_by_phase(&result_matrix, &circuit_matrix),
        "Gate transfer should not change circuit size {}",
        result.operations().len()
    );
}

#[test]
fn test_gate_transfer_full_circuit_with_rxx() {
    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
        vec![StandardGate::RXX],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_ryy() {
    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
        vec![StandardGate::RYY],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_rzz() {
    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
        vec![StandardGate::RZZ],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_rzx() {
    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX, StandardGate::RY],
        vec![StandardGate::RZX],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_cx_hrz() {
    let iset = InstructionSet::new(
        vec![StandardGate::H, StandardGate::RZ],
        vec![StandardGate::CX],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_fsim() {
    let iset = InstructionSet::new(
        vec![
            StandardGate::RX,
            StandardGate::RZ,
            StandardGate::RY,
            StandardGate::H,
        ],
        vec![StandardGate::FSIM],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_cz_u() {
    let iset = InstructionSet::new(vec![StandardGate::U], vec![StandardGate::CZ], None);
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

#[test]
fn test_gate_transfer_full_circuit_with_cx_rxx() {
    let iset = InstructionSet::new(
        vec![
            StandardGate::RX,
            StandardGate::RZ,
            StandardGate::RY,
            StandardGate::H,
        ],
        vec![StandardGate::CZ, StandardGate::RZZ],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_full_circuit());
}

fn generate_circuit_with_special_operation() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    let mut sub_circuit = Circuit::new(2);
    sub_circuit.cx(q0, q1).unwrap();

    let frozen_sub_circuit = FrozenCircuit::new(sub_circuit);
    let mut ugate = UnitaryGate::new("TestUnitary", 2);
    ugate = ugate
        .with_matrix(StandardGate::CX.matrix(&[]).into_owned())
        .unwrap();
    ugate = ugate.with_circuit(Arc::new(frozen_sub_circuit));

    let cgate = CircuitGate::new(
        "TestCircuit",
        FrozenCircuit::new(generate_full_circuit()),
    )
    .unwrap();

    let mut circuit = generate_full_circuit();
    circuit
        .append(
            Instruction::UnitaryGate(Box::new(ugate)),
            vec![q0, q1],
            std::iter::empty(),
            None,
        )
        .unwrap();
    circuit
        .append(
            Instruction::CircuitGate(Box::new(cgate)),
            vec![q0, q2, q1],
            std::iter::empty(),
            None,
        )
        .unwrap();
    let mc_gate = MCGate::new(2, StandardGate::X);
    circuit
        .append(
            Instruction::McGate(Box::new(mc_gate)),
            vec![q0, q1, q2],
            std::iter::empty(),
            None,
        )
        .unwrap();

    circuit
}

fn generate_circuit_with_control_flow() -> Circuit {
    let mut circuit = generate_full_circuit();
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);

    circuit.h(q0).unwrap();
    circuit.h(q1).unwrap();

    circuit.measure(q0).unwrap();
    circuit.measure(q1).unwrap();

    let condition1 = ConditionView::new(q0, 1);

    let if1_true_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![q1],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![q1, q2],
            params: smallvec![],
            label: None,
        },
    ];

    let if1_false_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::X),
            qubits: smallvec![q1],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::Z),
            qubits: smallvec![q2],
            params: smallvec![],
            label: None,
        },
    ];

    circuit
        .if_else(condition1, if1_true_body, Some(if1_false_body))
        .unwrap();

    let condition2 = ConditionView::new(q0, 1);
    let while_body = vec![
        Operation {
            instruction: Instruction::Standard(StandardGate::H),
            qubits: smallvec![q1],
            params: smallvec![],
            label: None,
        },
        Operation {
            instruction: Instruction::Standard(StandardGate::CX),
            qubits: smallvec![q2, q1],
            params: smallvec![],
            label: None,
        },
    ];

    circuit.while_loop(condition2, while_body).unwrap();

    circuit
}

#[test]
fn test_gate_transfer_full_circuit_with_unitary_and_circuit_gate() {
    let iset = InstructionSet::new(
        vec![
            StandardGate::RX,
            StandardGate::RZ,
            StandardGate::RY,
            StandardGate::H,
        ],
        vec![StandardGate::CX, StandardGate::RZZ],
        None,
    );
    gate_transfer_circuit_test(&iset, &generate_circuit_with_special_operation());
}

#[test]
fn test_gate_transform_with_control_flow() {
    let iset = InstructionSet::new(
        vec![
            StandardGate::RX,
            StandardGate::RZ,
            StandardGate::RY,
            StandardGate::H,
        ],
        vec![StandardGate::CX, StandardGate::RZZ],
        None,
    );

    let circuit = generate_circuit_with_control_flow();
    let dag = CircuitDag::from_circuit(&circuit).expect("Failed to create CircuitDag");
    assert!(
        dag.num_blocks() > 1,
        "Control flow circuit should have multiple blocks"
    );

    let mut gt = GateTransform::new(iset.clone());
    let result = gt.execute(&circuit);

    let result_matrix = result.to_matrix(None);
    let circuit_matrix = circuit.to_matrix(None);
    assert!(
        is_matrix_differ_by_phase(&result_matrix, &circuit_matrix),
        "Gate transfer should not change circuit size {}",
        result.operations().len()
    );

    let result_dag = CircuitDag::from_circuit(&result).expect("Failed to create CircuitDag");
    assert_eq!(
        dag.num_blocks(),
        result_dag.num_blocks(),
        "Gate transfer should not change circuit block number"
    );

    for (_idx, block) in result_dag.blocks() {
        let mut old_subc = Circuit::new(3);
        let mut new_subc = Circuit::new(3);
        let operations = block.operations.clone();
        for op in &operations {
            if let Instruction::Standard(gate) = &op.instruction {
                if gate.num_qubits() == 1 {
                    assert!(
                        iset.single_qubit_gates.contains(gate),
                        "Single qubit gate {} do not in instruction set",
                        gate
                    );
                } else if gate.num_qubits() == 2 {
                    assert!(
                        iset.double_qubit_gate.contains(gate),
                        "Double qubit gate {} do not in instruction set",
                        gate
                    );
                }
            }
            let params = op
                .params
                .iter()
                .map(|x| match x {
                    CircuitParam::Fixed(f) => ParameterValue::Fixed(*f),
                    _ => ParameterValue::Fixed(0.0),
                })
                .collect::<Vec<_>>();
            new_subc
                .append(op.instruction.clone(), op.qubits.clone(), params, None)
                .unwrap();
        }

        for op in dag.data[_idx].operations.iter() {
            let params = op
                .params
                .iter()
                .map(|x| match x {
                    CircuitParam::Fixed(f) => ParameterValue::Fixed(*f),
                    _ => ParameterValue::Fixed(0.0),
                })
                .collect::<Vec<_>>();
            old_subc
                .append(op.instruction.clone(), op.qubits.clone(), params, None)
                .unwrap();
        }

        let result_matrix = old_subc.to_matrix(None);
        let circuit_matrix = new_subc.to_matrix(None);
        assert!(
            is_matrix_differ_by_phase(&result_matrix, &circuit_matrix),
            "Gate transfer should not change circuit size {}",
            result.operations().len()
        );
    }
}

#[test]
fn test_gate_transform_preserves_symbolic_rzz_parameters() {
    let iset = InstructionSet::new(
        vec![StandardGate::RZ, StandardGate::RX],
        vec![StandardGate::CX],
        None,
    );

    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let theta = Parameter::symbol("theta");

    circuit.rzz(q0, q1, theta).unwrap();

    let mut gt = GateTransform::new(iset.clone());
    let result = gt.execute(&circuit);

    assert!(result.symbols().contains("theta"));
    assert!(result
        .operations()
        .iter()
        .any(|op| op.params.iter().any(|param| matches!(param, CircuitParam::Index(_)))));

    let mut bindings = HashMap::new();
    bindings.insert("theta".to_string(), 0.37);
    gate_transfer_symbolic_circuit_test(&iset, &circuit, bindings);
}

#[test]
fn test_gate_transform_symbolic_single_qubit_rx_to_target_basis() {
    let iset = InstructionSet::new(
        vec![StandardGate::H, StandardGate::RZ],
        vec![StandardGate::CX],
        None,
    );

    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);
    let theta = Parameter::symbol("theta");
    circuit.rx(q0, theta).unwrap();

    let mut gt = GateTransform::new(iset.clone());
    let result = gt.execute(&circuit);

    assert!(check_all_gates_in_instruction_set(&result, &iset));
    assert!(result.symbols().contains("theta"));
    assert!(result
        .operations()
        .iter()
        .any(|op| op.params.iter().any(|param| matches!(param, CircuitParam::Index(_)))));

    let mut bindings = HashMap::new();
    bindings.insert("theta".to_string(), 0.37);
    gate_transfer_symbolic_circuit_test(&iset, &circuit, bindings);
}

#[test]
fn test_gate_transform_symbolic_u_recursively_rewrites_to_target_basis() {
    let iset = InstructionSet::new(
        vec![StandardGate::H, StandardGate::RZ],
        vec![StandardGate::CX],
        None,
    );

    let mut circuit = Circuit::new(1);
    let q0 = Qubit::new(0);
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let lambda = Parameter::symbol("lambda");
    circuit.u(q0, theta, phi, lambda).unwrap();

    let mut gt = GateTransform::new(iset.clone());
    let result = gt.execute(&circuit);

    assert!(check_all_gates_in_instruction_set(&result, &iset));
    assert!(result.symbols().contains("theta"));
    assert!(result.symbols().contains("phi"));
    assert!(result.symbols().contains("lambda"));
    assert!(!result.operations().iter().any(|op| {
        matches!(
            op.instruction,
            Instruction::Standard(StandardGate::U)
                | Instruction::Standard(StandardGate::RX)
                | Instruction::Standard(StandardGate::RY)
                | Instruction::Standard(StandardGate::RXY)
                | Instruction::Standard(StandardGate::XY)
                | Instruction::Standard(StandardGate::XY2P)
                | Instruction::Standard(StandardGate::XY2M)
        )
    }));

    let mut bindings = HashMap::new();
    bindings.insert("theta".to_string(), 0.11);
    bindings.insert("phi".to_string(), -0.23);
    bindings.insert("lambda".to_string(), 0.41);
    gate_transfer_symbolic_circuit_test(&iset, &circuit, bindings);
}

#[test]
fn test_gate_transform_preserves_mixed_symbolic_single_and_double_qubit_parameters() {
    let iset = InstructionSet::new(
        vec![StandardGate::H, StandardGate::RZ],
        vec![StandardGate::CX],
        None,
    );

    let mut circuit = Circuit::new(2);
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");
    let lambda = Parameter::symbol("lambda");
    circuit.u(q0, theta.clone(), phi.clone(), lambda.clone()).unwrap();
    circuit.rzz(q0, q1, theta.clone() + phi.clone()).unwrap();

    let mut gt = GateTransform::new(iset.clone());
    let result = gt.execute(&circuit);

    assert!(check_all_gates_in_instruction_set(&result, &iset));
    assert!(result.symbols().contains("theta"));
    assert!(result.symbols().contains("phi"));
    assert!(result.symbols().contains("lambda"));
    assert!(result
        .operations()
        .iter()
        .any(|op| op.params.iter().any(|param| matches!(param, CircuitParam::Index(_)))));

    let mut bindings = HashMap::new();
    bindings.insert("theta".to_string(), 0.29);
    bindings.insert("phi".to_string(), -0.17);
    bindings.insert("lambda".to_string(), 0.63);
    gate_transfer_symbolic_circuit_test(&iset, &circuit, bindings);
}
