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

use crate::circuit::dag::CircuitDag;
use crate::circuit::gate::Instruction;
use crate::circuit::{Circuit, Qubit};

/// Helper function to create a simple circuit for testing
fn create_test_circuit() -> Circuit {
    let mut circuit = Circuit::new(3);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.cx(Qubit::new(1), Qubit::new(2)).unwrap();
    circuit
}

/// Test basic conversion from Circuit to CircuitDag
#[test]
fn test_from_circuit_basic() {
    let circuit = create_test_circuit();
    let dag = CircuitDag::from_circuit(&circuit);

    // Verify qubits are preserved
    assert_eq!(dag.qubits.len(), 3);
    assert!(dag.qubits.contains(&Qubit::new(0)));
    assert!(dag.qubits.contains(&Qubit::new(1)));
    assert!(dag.qubits.contains(&Qubit::new(2)));

    // Verify operations count (3 nodes)
    assert_eq!(dag.data.node_count(), 3);
}

/// Test conversion from CircuitDag back to Circuit
#[test]
fn test_to_circuit_basic() {
    let circuit = create_test_circuit();
    let dag = CircuitDag::from_circuit(&circuit);
    let recovered = dag.to_circuit();

    // Verify qubit count
    assert_eq!(circuit.num_qubits(), recovered.num_qubits());

    // Verify operation count
    assert_eq!(circuit.operations().len(), recovered.operations().len());
}

/// Test roundtrip: Circuit -> CircuitDag -> Circuit
#[test]
fn test_roundtrip() {
    let original = create_test_circuit();
    let dag = CircuitDag::from_circuit(&original);
    let recovered = dag.to_circuit();

    // Operations should match
    assert_eq!(original.operations().len(), recovered.operations().len());
}

/// Test CircuitDag with parameterized gates
#[test]
fn test_parametric_circuit() {
    use crate::circuit::parameter::impls::Parameter;

    let mut circuit = Circuit::new(2);
    let theta = Parameter::symbol("theta");
    circuit.rx(Qubit::new(0), theta.clone()).unwrap();
    circuit.h(Qubit::new(1)).unwrap();

    let dag = CircuitDag::from_circuit(&circuit);
    assert_eq!(dag.parameters.len(), 1);

    let recovered = dag.to_circuit();
    assert_eq!(recovered.parameters().len(), 1);
}

/// Test CircuitDag with symbols
#[test]
fn test_symbols() {
    use crate::circuit::parameter::impls::Parameter;

    let mut circuit = Circuit::new(2);
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");

    // Use different parameters with different symbols
    circuit.rx(Qubit::new(0), theta.clone()).unwrap();
    circuit.ry(Qubit::new(1), phi.clone()).unwrap();

    let dag = CircuitDag::from_circuit(&circuit);

    // Verify symbols are preserved
    assert!(dag.symbols.contains(&"theta".to_string()));
    assert!(dag.symbols.contains(&"phi".to_string()));
    assert_eq!(dag.symbols.len(), 2);

    // Test roundtrip preserves symbols
    let recovered = dag.to_circuit();
    let recovered_symbols: Vec<_> = recovered.symbols().iter().cloned().collect();
    assert!(recovered_symbols.contains(&"theta".to_string()));
    assert!(recovered_symbols.contains(&"phi".to_string()));
}

/// Test CircuitDag with same symbol used multiple times
#[test]
fn test_duplicate_symbols() {
    use crate::circuit::parameter::impls::Parameter;

    let mut circuit = Circuit::new(2);
    let theta = Parameter::symbol("theta");

    // Use same parameter on multiple gates
    circuit.rx(Qubit::new(0), theta.clone()).unwrap();
    circuit.ry(Qubit::new(1), theta.clone()).unwrap();

    let dag = CircuitDag::from_circuit(&circuit);

    // Symbol should only appear once due to deduplication
    assert!(dag.symbols.contains(&"theta".to_string()));
    assert_eq!(dag.symbols.len(), 1);
}

/// Test CircuitDag with complex parameter expressions
#[test]
fn test_complex_parameters() {
    use crate::circuit::parameter::impls::Parameter;

    let mut circuit = Circuit::new(2);
    let theta = Parameter::symbol("theta");
    let phi = Parameter::symbol("phi");

    // Use expression with multiple symbols
    let expr = theta.clone() + phi.clone();
    circuit.rx(Qubit::new(0), expr).unwrap();

    let dag = CircuitDag::from_circuit(&circuit);

    // Both symbols should be present
    assert!(dag.symbols.contains(&"theta".to_string()));
    assert!(dag.symbols.contains(&"phi".to_string()));

    let recovered = dag.to_circuit();
    let recovered_symbols: Vec<_> = recovered.symbols().iter().cloned().collect();
    assert!(recovered_symbols.contains(&"theta".to_string()));
    assert!(recovered_symbols.contains(&"phi".to_string()));
}

/// Test CircuitDag with global phase
#[test]
fn test_global_phase() {
    use crate::circuit::parameter::impls::Parameter;

    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();

    // Set a fixed global phase
    let phase = Parameter::from(std::f64::consts::PI);
    circuit.set_global_phase(phase);

    let dag = CircuitDag::from_circuit(&circuit);
    let recovered = dag.to_circuit();

    // Check global phase is preserved
    let original_phase = circuit.global_phase();
    let recovered_phase = recovered.global_phase();
    assert_eq!(
        original_phase.evaluate(&None).unwrap(),
        recovered_phase.evaluate(&None).unwrap()
    );
}

/// Test CircuitDag with symbolic global phase
#[test]
fn test_symbolic_global_phase() {
    use crate::circuit::parameter::impls::Parameter;

    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();

    // Set a symbolic global phase
    let phase = Parameter::symbol("phi");
    circuit.set_global_phase(phase);

    let dag = CircuitDag::from_circuit(&circuit);
    let recovered = dag.to_circuit();

    // Global phase should be preserved (as parameter)
    let recovered_phase = recovered.global_phase();
    assert!(recovered_phase.get_symbols().contains(&"phi".to_string()));
}

/// Test CircuitDag edge construction (dependency tracking)
#[test]
fn test_dag_edges() {
    let circuit = create_test_circuit();
    let dag = CircuitDag::from_circuit(&circuit);

    // The circuit has 3 operations:
    // 1. H(0)
    // 2. CX(0, 1)
    // 3. CX(1, 2)
    //
    // Edges should exist for:
    // - H(0) -> CX(0,1) (qubit 0 dependency)
    // - CX(0,1) -> CX(1,2) (qubit 1 dependency)
    //
    // Total edges: 2
    assert_eq!(dag.data.edge_count(), 2);
}

/// Test CircuitDag preserves operation order via topological sort
#[test]
fn test_topological_order() {
    let circuit = create_test_circuit();
    let dag = CircuitDag::from_circuit(&circuit);
    let recovered = dag.to_circuit();

    // Verify all operations are the same type
    for (orig, recov) in circuit
        .operations()
        .iter()
        .zip(recovered.operations().iter())
    {
        // Instructions should match
        assert_eq!(
            format!("{:?}", orig.instruction),
            format!("{:?}", recov.instruction)
        );
    }
}

/// Test empty circuit conversion
#[test]
fn test_empty_circuit() {
    let circuit = Circuit::new(2);
    let dag = CircuitDag::from_circuit(&circuit);

    assert_eq!(dag.qubits.len(), 2);
    assert_eq!(dag.data.node_count(), 0);
    assert_eq!(dag.data.edge_count(), 0);

    let recovered = dag.to_circuit();
    assert_eq!(recovered.num_qubits(), 2);
    assert_eq!(recovered.operations().len(), 0);
}

/// Test single qubit circuit
#[test]
fn test_single_qubit_circuit() {
    let mut circuit = Circuit::new(1);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.x(Qubit::new(0)).unwrap();
    circuit.y(Qubit::new(0)).unwrap();

    let dag = CircuitDag::from_circuit(&circuit);

    // 3 nodes, 2 edges (H->X, X->Y)
    assert_eq!(dag.data.node_count(), 3);
    assert_eq!(dag.data.edge_count(), 2);

    let recovered = dag.to_circuit();
    assert_eq!(recovered.operations().len(), 3);
}

/// Test circuit with measurements
#[test]
fn test_measurements() {
    use crate::circuit::gate::Directive;

    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.measure(Qubit::new(0)).unwrap();
    circuit.cx(Qubit::new(0), Qubit::new(1)).unwrap();
    circuit.measure(Qubit::new(1)).unwrap();

    let dag = CircuitDag::from_circuit(&circuit);

    // Verify operations include measurements
    let ops: Vec<_> = dag.data.node_indices().map(|i| &dag.data[i]).collect();
    let measure_count = ops
        .iter()
        .filter(|op| matches!(op.instruction, Instruction::Directive(Directive::Measure)))
        .count();
    assert_eq!(measure_count, 2);

    let recovered = dag.to_circuit();
    assert_eq!(recovered.operations().len(), 4);
}

/// Test parallel operations on different qubits
#[test]
fn test_parallel_operations() {
    let mut circuit = Circuit::new(2);
    // These operations are independent (act on different qubits)
    circuit.h(Qubit::new(0)).unwrap();
    circuit.h(Qubit::new(1)).unwrap();

    let dag = CircuitDag::from_circuit(&circuit);

    // 2 nodes, 0 edges (no dependencies)
    assert_eq!(dag.data.node_count(), 2);
    assert_eq!(dag.data.edge_count(), 0);

    let recovered = dag.to_circuit();
    assert_eq!(recovered.operations().len(), 2);
}

/// Test circuit with barriers
#[test]
fn test_barriers() {
    use crate::circuit::gate::Directive;

    let mut circuit = Circuit::new(2);
    circuit.h(Qubit::new(0)).unwrap();
    circuit.barrier(vec![Qubit::new(0), Qubit::new(1)]).unwrap();
    circuit.h(Qubit::new(1)).unwrap();

    let dag = CircuitDag::from_circuit(&circuit);

    // Verify barrier is preserved
    let ops: Vec<_> = dag.data.node_indices().map(|i| &dag.data[i]).collect();
    let barrier_count = ops
        .iter()
        .filter(|op| matches!(op.instruction, Instruction::Directive(Directive::Barrier)))
        .count();
    assert_eq!(barrier_count, 1);

    let recovered = dag.to_circuit();
    assert_eq!(recovered.operations().len(), 3);
}

/// Test circuit with no symbols (fixed parameters only)
#[test]
fn test_no_symbols() {
    let mut circuit = Circuit::new(2);
    // Use fixed parameter values (no symbols)
    circuit.rx(Qubit::new(0), 0.5).unwrap();
    circuit.ry(Qubit::new(1), 1.57).unwrap();

    let dag = CircuitDag::from_circuit(&circuit);

    // No symbols should be present
    assert!(dag.symbols.is_empty());

    let recovered = dag.to_circuit();
    assert!(recovered.symbols().is_empty());
}
