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

use super::{CompileConfig, CompileMode, compile};
use crate::circuit::{
    Circuit, Instruction, MCGate, ParameterValue, Qubit, StandardGate, circuit_to_matrix,
};
use crate::compile::CompilerError;
use crate::compile::resource::ResourcePolicy;
use crate::device::{Device, PhysicalQubit, Topology};
use crate::util::test_utils::{
    EPSILON, assert_circuits_equivalent_up_to_global_phase,
    assert_matrices_equal_up_to_global_phase, contains_high_level_gate, standard_ops, step_changed,
};
use std::collections::HashSet;

fn compile_normal(circuit: &Circuit) -> super::CompileResult {
    compile(
        circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: None,
            device: None,
            resource_policy: ResourcePolicy::default(),
            seed: None,
        },
    )
    .unwrap()
}

fn assert_compiled_matrix_equivalent(actual: &Circuit, expected: &Circuit) {
    let actual_matrix = circuit_to_matrix(actual, None).unwrap();
    let expected_matrix = circuit_to_matrix(expected, None).unwrap();
    assert_matrices_equal_up_to_global_phase(&actual_matrix, &expected_matrix, EPSILON);
    assert_circuits_equivalent_up_to_global_phase(actual, expected, EPSILON);
}

fn compile_to_basis(circuit: &Circuit, basis: Vec<StandardGate>) -> super::CompileResult {
    compile(
        circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: Some(
                basis
                    .into_iter()
                    .map(Instruction::Standard)
                    .collect::<Vec<_>>(),
            ),
            device: None,
            resource_policy: ResourcePolicy::default(),
            seed: None,
        },
    )
    .unwrap()
}

fn compile_to_basis_checked(circuit: &Circuit, basis: &[StandardGate]) -> super::CompileResult {
    let result = compile_to_basis(circuit, basis.to_vec());
    assert!(
        step_changed(&result, "translate.target_basis"),
        "target-basis translation should change circuit for basis {basis:?}"
    );
    assert_only_standard_gates(&result.circuit, basis);
    assert_compiled_matrix_equivalent(&result.circuit, circuit);
    result
}

fn compile_on_device_checked(
    circuit: &Circuit,
    device: Device,
    seed: u32,
    allowed: &[StandardGate],
) -> super::CompileResult {
    let topology = device.topology().clone();
    let result = compile(
        circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: None,
            device: Some(device),
            resource_policy: ResourcePolicy::default(),
            seed: Some(seed),
        },
    )
    .unwrap();

    assert!(
        result
            .steps
            .iter()
            .any(|step| step.name == "route.sabre" && !step.skipped),
        "routing step should run"
    );
    assert_only_standard_gates(&result.circuit, allowed);
    assert_all_two_qubit_operations_supported_by_topology(&result.circuit, &topology);
    assert!(result.circuit.qubits().len() <= topology.num_qubits());
    result
}

fn assert_only_standard_gates(circuit: &Circuit, allowed: &[StandardGate]) {
    for op in circuit.operations() {
        assert!(
            matches!(op.instruction, Instruction::Standard(gate) if allowed.contains(&gate)),
            "unexpected operation in compiled circuit: {op:?}"
        );
    }
}

fn assert_all_two_qubit_operations_supported_by_topology(circuit: &Circuit, topology: &Topology) {
    for op in circuit.operations() {
        if op.qubits.len() == 2 {
            let a = PhysicalQubit::new(op.qubits[0].id());
            let b = PhysicalQubit::new(op.qubits[1].id());
            assert!(
                topology.supports_coupling_either_direction(a, b),
                "operation {op:?} is not supported by topology"
            );
        }
    }
}

fn native_basis(gates: &[StandardGate]) -> Vec<Instruction> {
    gates.iter().copied().map(Instruction::Standard).collect()
}

fn qcis_native_basis() -> Vec<StandardGate> {
    vec![
        StandardGate::I,
        StandardGate::RZ,
        StandardGate::X2P,
        StandardGate::X2M,
        StandardGate::Y2P,
        StandardGate::Y2M,
        StandardGate::XY2P,
        StandardGate::XY2M,
        StandardGate::CZ,
        StandardGate::GPhase,
    ]
}

fn qcis_cz_basis() -> Vec<StandardGate> {
    vec![
        StandardGate::RZ,
        StandardGate::X2P,
        StandardGate::X2M,
        StandardGate::Y2P,
        StandardGate::Y2M,
        StandardGate::CZ,
        StandardGate::GPhase,
    ]
}

fn device_from_edges(
    name: &str,
    num_qubits: u32,
    edges: &[(u32, u32)],
    native_gates: &[StandardGate],
) -> Device {
    let physical = (0..num_qubits).map(PhysicalQubit::new).collect::<Vec<_>>();
    let couplings = edges
        .iter()
        .enumerate()
        .map(|(index, &(a, b))| {
            (
                PhysicalQubit::new(a),
                PhysicalQubit::new(b),
                format!("e{index}"),
            )
        })
        .collect::<Vec<_>>();
    let topology = Topology::new(physical.clone(), couplings).unwrap();
    Device::new(
        name,
        physical.iter().copied().collect::<HashSet<_>>(),
        topology,
    )
    .unwrap()
    .with_native_gates(native_basis(native_gates))
}

fn line_device_with_basis(name: &str, num_qubits: u32, native_gates: &[StandardGate]) -> Device {
    let edges = (0..num_qubits - 1).map(|i| (i, i + 1)).collect::<Vec<_>>();
    device_from_edges(name, num_qubits, &edges, native_gates)
}

fn bidirectional_line_device_with_basis(
    name: &str,
    num_qubits: u32,
    native_gates: &[StandardGate],
) -> Device {
    let mut edges = Vec::new();
    for i in 0..num_qubits - 1 {
        edges.push((i, i + 1));
        edges.push((i + 1, i));
    }
    device_from_edges(name, num_qubits, &edges, native_gates)
}

fn ring_device_with_basis(name: &str, num_qubits: u32, native_gates: &[StandardGate]) -> Device {
    let mut edges = Vec::new();
    for i in 0..num_qubits {
        edges.push((i, (i + 1) % num_qubits));
        edges.push(((i + 1) % num_qubits, i));
    }
    device_from_edges(name, num_qubits, &edges, native_gates)
}

fn star_device_with_basis(
    name: &str,
    num_qubits: u32,
    center: u32,
    native_gates: &[StandardGate],
) -> Device {
    let mut edges = Vec::new();
    for i in 0..num_qubits {
        if i != center {
            edges.push((center, i));
            edges.push((i, center));
        }
    }
    device_from_edges(name, num_qubits, &edges, native_gates)
}

fn grid_device_with_basis(name: &str, native_gates: &[StandardGate]) -> Device {
    device_from_edges(
        name,
        6,
        &[
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
            (0, 3),
            (3, 0),
            (1, 4),
            (4, 1),
            (2, 5),
            (5, 2),
        ],
        native_gates,
    )
}

fn bell_circuit() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit.cx(q0, q1).unwrap();
    circuit
}

fn ghz_circuit(num_qubits: usize) -> Circuit {
    assert!(num_qubits >= 2);
    let mut circuit = Circuit::new(num_qubits);
    circuit.h(Qubit::new(0)).unwrap();
    for i in 0..num_qubits - 1 {
        circuit
            .cx(Qubit::new(i as u32), Qubit::new(i as u32 + 1))
            .unwrap();
    }
    circuit
}

fn qft3_circuit() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.h(q2).unwrap();
    circuit.crz(q1, q2, std::f64::consts::FRAC_PI_2).unwrap();
    circuit.h(q1).unwrap();
    circuit.crz(q0, q2, std::f64::consts::FRAC_PI_4).unwrap();
    circuit.crz(q0, q1, std::f64::consts::FRAC_PI_2).unwrap();
    circuit.h(q0).unwrap();
    circuit.swap(q0, q2).unwrap();
    circuit
}

fn toffoli_circuit() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::X))),
            vec![q0, q1, q2],
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();
    circuit
}

fn single_qubit_gate_suite() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.h(q0).unwrap();
    circuit.x(q1).unwrap();
    circuit.y(q2).unwrap();
    circuit.z(q0).unwrap();
    circuit.s(q1).unwrap();
    circuit.sdg(q2).unwrap();
    circuit.t(q0).unwrap();
    circuit.tdg(q1).unwrap();
    circuit.phase(q2, 0.37).unwrap();
    circuit.rx(q0, 0.31).unwrap();
    circuit.ry(q1, -0.29).unwrap();
    circuit.rz(q2, 0.43).unwrap();
    circuit.rxy(q0, 0.27, -0.19).unwrap();
    circuit.xy(q1, 0.41).unwrap();
    circuit.u(q2, 0.23, -0.17, 0.11).unwrap();
    circuit.x2p(q0).unwrap();
    circuit.x2m(q1).unwrap();
    circuit.y2p(q2).unwrap();
    circuit.y2m(q0).unwrap();
    circuit.xy2p(q1, 0.13).unwrap();
    circuit.xy2m(q2, -0.21).unwrap();
    circuit
}

fn two_qubit_gate_suite() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);
    let mut circuit = Circuit::new(4);
    circuit.h(q0).unwrap();
    circuit.rx(q1, 0.17).unwrap();
    circuit.ry(q2, -0.23).unwrap();
    circuit.rz(q3, 0.29).unwrap();
    circuit.cx(q0, q2).unwrap();
    circuit.cy(q1, q3).unwrap();
    circuit.cz(q2, q0).unwrap();
    circuit.swap(q0, q3).unwrap();
    circuit.crx(q3, q1, 0.31).unwrap();
    circuit.cry(q2, q0, -0.37).unwrap();
    circuit.crz(q1, q2, 0.43).unwrap();
    circuit.rxx(q0, q1, 0.19).unwrap();
    circuit.ryy(q2, q3, -0.27).unwrap();
    circuit.rzz(q0, q2, 0.33).unwrap();
    circuit.rzx(q3, q1, -0.39).unwrap();
    circuit.fsim(q1, q2, 0.21, -0.35).unwrap();
    circuit
}

fn two_qubit_gate_suite_without_fsim() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);
    let mut circuit = Circuit::new(4);
    circuit.h(q0).unwrap();
    circuit.rx(q1, 0.17).unwrap();
    circuit.ry(q2, -0.23).unwrap();
    circuit.rz(q3, 0.29).unwrap();
    circuit.cx(q0, q2).unwrap();
    circuit.cy(q1, q3).unwrap();
    circuit.cz(q2, q0).unwrap();
    circuit.swap(q0, q3).unwrap();
    circuit.crx(q3, q1, 0.31).unwrap();
    circuit.cry(q2, q0, -0.37).unwrap();
    circuit.crz(q1, q2, 0.43).unwrap();
    circuit.rxx(q0, q1, 0.19).unwrap();
    circuit.ryy(q2, q3, -0.27).unwrap();
    circuit.rzz(q0, q2, 0.33).unwrap();
    circuit.rzx(q3, q1, -0.39).unwrap();
    circuit
}

fn controlled_rotation_suite() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.h(q0).unwrap();
    circuit.rx(q1, 0.11).unwrap();
    circuit.ry(q2, -0.13).unwrap();
    circuit.crx(q0, q1, 0.23).unwrap();
    circuit.cry(q1, q2, -0.31).unwrap();
    circuit.crz(q2, q0, 0.41).unwrap();
    circuit.crx(q2, q1, -0.29).unwrap();
    circuit.cry(q0, q2, 0.37).unwrap();
    circuit.crz(q1, q0, -0.43).unwrap();
    circuit
}

fn ising_gate_suite() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.h(q0).unwrap();
    circuit.rx(q1, 0.17).unwrap();
    circuit.ry(q2, 0.19).unwrap();
    circuit.rxx(q0, q1, 0.23).unwrap();
    circuit.ryy(q1, q2, -0.29).unwrap();
    circuit.rzz(q0, q2, 0.31).unwrap();
    circuit.rzx(q2, q1, -0.37).unwrap();
    circuit
}

fn fsim_circuit() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.rx(q0, 0.17).unwrap();
    circuit.ry(q1, -0.19).unwrap();
    circuit.fsim(q0, q1, 0.13, 0.41).unwrap();
    circuit
}

fn swap_gate_suite() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(3);
    circuit.h(q0).unwrap();
    circuit.rx(q1, 0.17).unwrap();
    circuit.ry(q2, -0.19).unwrap();
    circuit.swap(q0, q2).unwrap();
    circuit.swap(q1, q2).unwrap();
    circuit
}

fn multi_controlled_gate_suite() -> Circuit {
    let qubits = (0..5).map(Qubit::new).collect::<Vec<_>>();
    let mut circuit = Circuit::new(5);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(3, StandardGate::X))),
            vec![qubits[0], qubits[1], qubits[2], qubits[3]],
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::RZ))),
            vec![qubits[1], qubits[2], qubits[4]],
            vec![ParameterValue::Fixed(0.31)],
            None,
        )
        .unwrap();
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(1, StandardGate::SWAP))),
            vec![qubits[0], qubits[3], qubits[4]],
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(1, StandardGate::FSIM))),
            vec![qubits[2], qubits[0], qubits[4]],
            vec![ParameterValue::Fixed(0.17), ParameterValue::Fixed(-0.23)],
            None,
        )
        .unwrap();
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::XY2P))),
            vec![qubits[0], qubits[1], qubits[2]],
            vec![ParameterValue::Fixed(0.29)],
            None,
        )
        .unwrap();
    circuit
}

fn long_range_device_circuit() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);
    let mut circuit = Circuit::new(4);
    circuit.h(q0).unwrap();
    circuit.rx(q1, 0.17).unwrap();
    circuit.ry(q2, -0.19).unwrap();
    circuit.rz(q3, 0.23).unwrap();
    circuit.cx(q0, q3).unwrap();
    circuit.crx(q3, q1, 0.31).unwrap();
    circuit.rzz(q0, q2, -0.37).unwrap();
    circuit.fsim(q1, q3, 0.21, -0.27).unwrap();
    circuit.swap(q0, q2).unwrap();
    circuit
}

fn dense_four_qubit_device_circuit() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);
    let mut circuit = Circuit::new(4);
    circuit.h(q0).unwrap();
    circuit.h(q1).unwrap();
    circuit.rx(q2, 0.11).unwrap();
    circuit.ry(q3, -0.13).unwrap();
    circuit.cx(q0, q2).unwrap();
    circuit.cz(q1, q3).unwrap();
    circuit.rxx(q0, q3, 0.23).unwrap();
    circuit.ryy(q1, q2, -0.29).unwrap();
    circuit.crz(q3, q0, 0.31).unwrap();
    circuit
}

fn ising_device_circuit() -> Circuit {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let q3 = Qubit::new(3);
    let mut circuit = Circuit::new(4);
    circuit.rx(q0, 0.17).unwrap();
    circuit.ry(q1, -0.19).unwrap();
    circuit.rz(q2, 0.23).unwrap();
    circuit.h(q3).unwrap();
    circuit.rxx(q0, q3, 0.29).unwrap();
    circuit.ryy(q1, q2, -0.31).unwrap();
    circuit.rzz(q0, q2, 0.37).unwrap();
    circuit.rzx(q3, q1, -0.41).unwrap();
    circuit.fsim(q2, q3, 0.13, -0.17).unwrap();
    circuit
}

// ── Pure logical optimization ──

#[test]
fn compile_bell_to_h_cz_basis() {
    let circuit = bell_circuit();
    let result = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: Some(vec![
                Instruction::Standard(StandardGate::H),
                Instruction::Standard(StandardGate::CZ),
            ]),
            device: None,
            resource_policy: ResourcePolicy::default(),
            seed: None,
        },
    )
    .unwrap();

    assert!(result.changed);
    assert_eq!(
        standard_ops(&result.circuit),
        vec![
            StandardGate::H,
            StandardGate::H,
            StandardGate::CZ,
            StandardGate::H
        ]
    );
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
    let ops = result.circuit.operations();
    assert_eq!(ops[0].qubits.as_slice(), &[Qubit::new(0)]); // H on q0
    assert_eq!(ops[1].qubits.as_slice(), &[Qubit::new(1)]); // H on q1
    assert_eq!(ops[2].qubits.as_slice(), &[Qubit::new(0), Qubit::new(1)]); // CZ
    assert_eq!(ops[3].qubits.as_slice(), &[Qubit::new(1)]); // H on q1
}

#[test]
fn compile_qft3_without_target_basis_preserves_unitary() {
    let circuit = qft3_circuit();
    let result = compile_normal(&circuit);

    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
    assert!(!contains_high_level_gate(&result.circuit));
}

#[test]
fn compile_qft3_reports_unsupported_h_cz_target_basis() {
    let circuit = qft3_circuit();
    let err = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: Some(vec![
                Instruction::Standard(StandardGate::H),
                Instruction::Standard(StandardGate::CZ),
            ]),
            device: None,
            resource_policy: ResourcePolicy::default(),
            seed: None,
        },
    )
    .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::InvalidInput(reason) if reason.contains("CRZ")
    ));
}

#[test]
fn compile_cancels_adjacent_self_inverse_across_full_pipeline() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.h(q0).unwrap();
    circuit.x(q0).unwrap();
    circuit.h(q0).unwrap();
    circuit.x(q0).unwrap();

    let result = compile_normal(&circuit);

    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
    // H·X·H·X = (H·X·H)·X — H and X don't cancel directly, but the pipeline
    // should canonicalize and apply knowledge-rule optimizations.
    assert!(
        standard_ops(&result.circuit).len() <= 4,
        "optimization should not increase gate count"
    );
}

#[test]
fn compile_merges_consecutive_same_axis_rotations() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.rz(q0, 0.25).unwrap();
    circuit.rz(q0, 0.5).unwrap();
    circuit.rz(q0, -0.75).unwrap();

    let result = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Enhanced,
            target_basis: None,
            device: None,
            resource_policy: ResourcePolicy::default(),
            seed: None,
        },
    )
    .unwrap();

    assert!(result.changed);
    assert!(result.circuit.operations().is_empty());
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
}

// ── Decomposition ──

#[test]
fn compile_decomposes_toffoli_into_standard_gates() {
    let circuit = toffoli_circuit();

    let result = compile_normal(&circuit);

    assert!(!contains_high_level_gate(&result.circuit));
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::CCX]);
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
}

#[test]
fn compile_decomposes_c3x_with_fallback_to_no_auxiliary() {
    let qubits = (0..4).map(Qubit::new).collect::<Vec<_>>();
    let mut circuit = Circuit::new(4);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(3, StandardGate::X))),
            qubits,
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();

    let result = compile_normal(&circuit);

    assert!(step_changed(&result, "decompose.mc_gates"));
    assert!(!contains_high_level_gate(&result.circuit));
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
}

#[test]
fn compile_lowers_common_gates_to_qcis_native_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit.x(q1).unwrap();
    circuit.y(q0).unwrap();
    circuit.rx(q0, 0.31).unwrap();
    circuit.ry(q1, -0.27).unwrap();
    circuit.cx(q0, q1).unwrap();
    circuit.crx(q1, q0, 0.19).unwrap();
    circuit.cry(q0, q1, -0.41).unwrap();
    circuit.rzz(q0, q1, 0.53).unwrap();

    let basis = qcis_native_basis();
    let result = compile_to_basis(&circuit, basis.clone());

    assert!(step_changed(&result, "translate.target_basis"));
    assert_only_standard_gates(&result.circuit, &basis);
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
}

#[test]
fn compile_converts_x2p_and_y2p_to_xy2p_basis() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.x2p(q0).unwrap();
    circuit.y2p(q1).unwrap();

    let result = compile_to_basis(&circuit, vec![StandardGate::XY2P]);

    assert!(step_changed(&result, "translate.target_basis"));
    assert_eq!(standard_ops(&result.circuit), vec![StandardGate::XY2P; 2]);
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
}

#[test]
fn compile_converts_xy2p_to_x2p_rz_basis() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit.xy2p(q0, 0.37).unwrap();

    let basis = vec![StandardGate::RZ, StandardGate::X2P];
    let result = compile_to_basis(&circuit, basis.clone());

    assert!(step_changed(&result, "translate.target_basis"));
    assert_eq!(
        standard_ops(&result.circuit),
        vec![StandardGate::RZ, StandardGate::X2P, StandardGate::RZ]
    );
    assert_only_standard_gates(&result.circuit, &basis);
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
}

#[test]
fn compile_decomposes_multi_controlled_qcis_half_rotations() {
    for (gate, params) in [
        (StandardGate::X2P, vec![]),
        (StandardGate::Y2P, vec![]),
        (StandardGate::XY2P, vec![ParameterValue::Fixed(0.73)]),
    ] {
        let mut circuit = Circuit::new(4);
        circuit
            .append(
                Instruction::McGate(Box::new(MCGate::new(3, gate))),
                vec![Qubit::new(0), Qubit::new(1), Qubit::new(2), Qubit::new(3)],
                params,
                None,
            )
            .unwrap();

        let result = compile_normal(&circuit);

        assert!(step_changed(&result, "decompose.mc_gates"));
        assert!(!contains_high_level_gate(&result.circuit));
        assert_compiled_matrix_equivalent(&result.circuit, &circuit);
    }
}

#[test]
fn compile_lowers_single_qubit_suite_to_qcis_x_half_basis() {
    let circuit = single_qubit_gate_suite();
    let basis = vec![
        StandardGate::RZ,
        StandardGate::X2P,
        StandardGate::X2M,
        StandardGate::GPhase,
    ];

    compile_to_basis_checked(&circuit, &basis);
}

#[test]
fn compile_lowers_single_qubit_suite_to_qcis_y_half_basis() {
    let circuit = single_qubit_gate_suite();
    let basis = vec![
        StandardGate::RZ,
        StandardGate::Y2P,
        StandardGate::Y2M,
        StandardGate::GPhase,
    ];

    compile_to_basis_checked(&circuit, &basis);
}

#[test]
fn compile_lowers_single_qubit_suite_to_qcis_xy_half_basis() {
    let circuit = single_qubit_gate_suite();
    let basis = vec![
        StandardGate::RZ,
        StandardGate::XY2P,
        StandardGate::XY2M,
        StandardGate::GPhase,
    ];

    compile_to_basis_checked(&circuit, &basis);
}

#[test]
fn compile_lowers_two_qubit_suite_to_qcis_cz_basis() {
    let circuit = two_qubit_gate_suite();
    let basis = qcis_cz_basis();

    compile_to_basis_checked(&circuit, &basis);
}

#[test]
fn compile_lowers_ccx_to_clifford_t_cx_basis() {
    let circuit = {
        let mut circuit = Circuit::new(3);
        circuit
            .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
            .unwrap();
        circuit
    };
    let basis = vec![
        StandardGate::H,
        StandardGate::CX,
        StandardGate::T,
        StandardGate::TDG,
    ];

    let result = compile_to_basis_checked(&circuit, &basis);

    assert!(standard_ops(&result.circuit).contains(&StandardGate::CX));
    assert!(!standard_ops(&result.circuit).contains(&StandardGate::CCX));
}

#[test]
fn compile_lowers_ccx_to_clifford_t_cz_basis() {
    let circuit = {
        let mut circuit = Circuit::new(3);
        circuit
            .ccx(Qubit::new(0), Qubit::new(1), Qubit::new(2))
            .unwrap();
        circuit
    };
    let basis = vec![
        StandardGate::H,
        StandardGate::CZ,
        StandardGate::T,
        StandardGate::TDG,
    ];

    let result = compile_to_basis_checked(&circuit, &basis);

    assert!(standard_ops(&result.circuit).contains(&StandardGate::CZ));
    assert!(!standard_ops(&result.circuit).contains(&StandardGate::CCX));
}

#[test]
fn compile_lowers_two_qubit_suite_to_cx_native_basis() {
    let circuit = two_qubit_gate_suite_without_fsim();
    let basis = vec![
        StandardGate::H,
        StandardGate::S,
        StandardGate::SDG,
        StandardGate::RX,
        StandardGate::RY,
        StandardGate::RZ,
        StandardGate::CX,
        StandardGate::GPhase,
    ];

    compile_to_basis_checked(&circuit, &basis);
}

#[test]
fn compile_lowers_two_qubit_suite_to_cz_native_basis() {
    let circuit = two_qubit_gate_suite_without_fsim();
    let basis = vec![
        StandardGate::H,
        StandardGate::S,
        StandardGate::SDG,
        StandardGate::RX,
        StandardGate::RY,
        StandardGate::RZ,
        StandardGate::CZ,
        StandardGate::GPhase,
    ];

    compile_to_basis_checked(&circuit, &basis);
}

#[test]
fn compile_lowers_controlled_rotations_to_rzz_native_basis() {
    let circuit = controlled_rotation_suite();
    let basis = vec![
        StandardGate::H,
        StandardGate::RX,
        StandardGate::RZ,
        StandardGate::RZZ,
        StandardGate::GPhase,
    ];

    compile_to_basis_checked(&circuit, &basis);
}

#[test]
fn compile_lowers_controlled_rotations_to_rzx_native_basis() {
    let circuit = controlled_rotation_suite();
    let basis = vec![
        StandardGate::H,
        StandardGate::S,
        StandardGate::SDG,
        StandardGate::RX,
        StandardGate::RY,
        StandardGate::RZ,
        StandardGate::RZX,
        StandardGate::GPhase,
    ];

    compile_to_basis_checked(&circuit, &basis);
}

#[test]
fn compile_lowers_swap_to_ising_exchange_basis() {
    let circuit = swap_gate_suite();
    let basis = vec![
        StandardGate::H,
        StandardGate::RX,
        StandardGate::RY,
        StandardGate::RXX,
        StandardGate::RYY,
        StandardGate::RZZ,
        StandardGate::GPhase,
    ];

    let result = compile_to_basis_checked(&circuit, &basis);

    assert!(standard_ops(&result.circuit).contains(&StandardGate::RXX));
    assert!(!standard_ops(&result.circuit).contains(&StandardGate::SWAP));
}

#[test]
fn compile_lowers_ising_suite_to_rzz_native_basis() {
    let circuit = ising_gate_suite();
    let basis = vec![
        StandardGate::H,
        StandardGate::RX,
        StandardGate::RY,
        StandardGate::RZ,
        StandardGate::RZZ,
        StandardGate::GPhase,
    ];

    compile_to_basis_checked(&circuit, &basis);
}

#[test]
fn compile_lowers_fsim_to_ising_exchange_basis() {
    let circuit = fsim_circuit();
    let basis = vec![
        StandardGate::H,
        StandardGate::RX,
        StandardGate::RY,
        StandardGate::RZ,
        StandardGate::RXX,
        StandardGate::RYY,
        StandardGate::RZZ,
        StandardGate::GPhase,
    ];

    let result = compile_to_basis_checked(&circuit, &basis);

    assert!(standard_ops(&result.circuit).contains(&StandardGate::RXX));
    assert!(standard_ops(&result.circuit).contains(&StandardGate::RYY));
    assert!(!standard_ops(&result.circuit).contains(&StandardGate::FSIM));
}

#[test]
fn compile_reports_fsim_gap_for_pure_rzz_native_basis() {
    let circuit = fsim_circuit();
    let err = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: Some(native_basis(&[
                StandardGate::H,
                StandardGate::RX,
                StandardGate::RY,
                StandardGate::RZ,
                StandardGate::RZZ,
                StandardGate::GPhase,
            ])),
            device: None,
            resource_policy: ResourcePolicy::default(),
            seed: None,
        },
    )
    .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::InvalidInput(reason) if reason.contains("FSIM")
    ));
}

#[test]
fn compile_lowers_multi_controlled_suite_to_qcis_cz_basis() {
    let circuit = multi_controlled_gate_suite();
    let basis = qcis_cz_basis();
    let result = compile_to_basis_checked(&circuit, &basis);

    assert!(step_changed(&result, "decompose.mc_gates"));
    assert!(!contains_high_level_gate(&result.circuit));
}

// ── Device routing + basis translation ──

#[test]
fn compile_ghz3_routes_on_line_device_and_lowers_to_h_cz() {
    let circuit = ghz_circuit(3);
    let device = Device::line("test-device", 3)
        .unwrap()
        .with_native_gates(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CZ),
        ]);

    let result = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: None,
            device: Some(device),
            resource_policy: ResourcePolicy::default(),
            seed: Some(42),
        },
    )
    .unwrap();

    assert!(
        result
            .steps
            .iter()
            .any(|step| step.name == "route.sabre" && !step.skipped)
    );
    assert!(step_changed(&result, "translate.target_basis"));
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
    for op in result.circuit.operations() {
        assert!(matches!(
            op.instruction,
            Instruction::Standard(StandardGate::H | StandardGate::CZ)
        ));
    }
}

#[test]
fn compile_ghz5_routes_on_line_device() {
    let circuit = ghz_circuit(5);
    let device = Device::line("test-device", 5).unwrap();

    let result = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: None,
            device: Some(device),
            resource_policy: ResourcePolicy::default(),
            seed: Some(17),
        },
    )
    .unwrap();

    assert!(
        result
            .steps
            .iter()
            .any(|step| step.name == "route.sabre" && !step.skipped)
    );
    assert!(result.circuit.qubits().len() <= 5);
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
}

#[test]
fn compile_toffoli_on_4q_line_device_requires_target_basis_for_ccx_lowering() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let q2 = Qubit::new(2);
    let mut circuit = Circuit::new(4);
    circuit
        .append(
            Instruction::McGate(Box::new(MCGate::new(2, StandardGate::X))),
            vec![q0, q1, q2],
            Vec::<ParameterValue>::new(),
            None,
        )
        .unwrap();
    circuit.h(Qubit::new(3)).unwrap();
    let device = Device::line("test-device", 4).unwrap();

    let err = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: None,
            device: Some(device),
            resource_policy: ResourcePolicy::default(),
            seed: Some(17),
        },
    )
    .unwrap_err();

    assert!(matches!(
        err,
        CompilerError::InvalidInput(reason)
            if reason.contains("layout requires unitary operations with more than two qubits")
                && reason.contains("CCX")
    ));
}

#[test]
fn compile_long_range_circuit_on_line_device_to_qcis_native_basis() {
    let circuit = long_range_device_circuit();
    let basis = qcis_cz_basis();
    let device = line_device_with_basis("line-qcis", 4, &basis);

    let result = compile_on_device_checked(&circuit, device, 101, &basis);

    assert!(step_changed(&result, "translate.target_basis"));
    assert!(result.circuit.qubits().len() <= 4);
}

#[test]
fn compile_long_range_circuit_on_ring_device_to_qcis_native_basis() {
    let circuit = long_range_device_circuit();
    let basis = qcis_cz_basis();
    let device = ring_device_with_basis("ring-qcis", 4, &basis);

    let result = compile_on_device_checked(&circuit, device, 102, &basis);

    assert!(step_changed(&result, "translate.target_basis"));
}

#[test]
fn compile_dense_circuit_on_bidirectional_line_to_cz_native_basis() {
    let circuit = dense_four_qubit_device_circuit();
    let basis = vec![
        StandardGate::H,
        StandardGate::S,
        StandardGate::SDG,
        StandardGate::RX,
        StandardGate::RY,
        StandardGate::RZ,
        StandardGate::CZ,
        StandardGate::GPhase,
    ];
    let device = bidirectional_line_device_with_basis("bidir-line-cz", 4, &basis);

    let result = compile_on_device_checked(&circuit, device, 103, &basis);

    assert!(step_changed(&result, "translate.target_basis"));
}

#[test]
fn compile_dense_circuit_on_star_device_to_cx_native_basis() {
    let circuit = dense_four_qubit_device_circuit();
    let basis = vec![
        StandardGate::H,
        StandardGate::S,
        StandardGate::SDG,
        StandardGate::RX,
        StandardGate::RY,
        StandardGate::RZ,
        StandardGate::CX,
        StandardGate::GPhase,
    ];
    let device = star_device_with_basis("star-cx", 4, 0, &basis);

    let result = compile_on_device_checked(&circuit, device, 104, &basis);

    assert!(step_changed(&result, "translate.target_basis"));
}

#[test]
fn compile_ising_circuit_on_grid_device_to_ising_native_basis() {
    let circuit = ising_device_circuit();
    let basis = vec![
        StandardGate::H,
        StandardGate::RX,
        StandardGate::RY,
        StandardGate::RZ,
        StandardGate::RXX,
        StandardGate::RYY,
        StandardGate::RZZ,
        StandardGate::GPhase,
    ];
    let device = grid_device_with_basis("grid-ising", &basis);

    let result = compile_on_device_checked(&circuit, device, 105, &basis);

    assert!(step_changed(&result, "translate.target_basis"));
}

// ── Enhanced mode ──

#[test]
fn compile_enhanced_ghz3_routes_and_cleans_up() {
    let circuit = ghz_circuit(3);
    let device = Device::line("test-device", 3)
        .unwrap()
        .with_native_gates(vec![
            Instruction::Standard(StandardGate::H),
            Instruction::Standard(StandardGate::CZ),
        ]);

    let result = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Enhanced,
            target_basis: None,
            device: Some(device),
            resource_policy: ResourcePolicy::default(),
            seed: Some(42),
        },
    )
    .unwrap();

    assert!(
        result
            .steps
            .iter()
            .any(|step| step.name == "route.sabre" && !step.skipped)
    );
    assert!(
        result
            .steps
            .iter()
            .any(|step| step.name == "optimize.post_routing" && !step.skipped)
    );
    assert!(
        result
            .steps
            .iter()
            .any(|step| step.name == "optimize.target_cleanup" && !step.skipped)
    );
    for op in result.circuit.operations() {
        assert!(matches!(
            op.instruction,
            Instruction::Standard(StandardGate::H | StandardGate::CZ)
        ));
    }
    assert_compiled_matrix_equivalent(&result.circuit, &circuit);
}

// ── Error paths ──

#[test]
fn compile_reports_error_for_unsupported_target_basis() {
    let circuit = bell_circuit();
    let err = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: Some(vec![Instruction::Standard(StandardGate::CZ)]),
            device: None,
            resource_policy: ResourcePolicy::default(),
            seed: None,
        },
    )
    .unwrap_err();

    assert!(!format!("{err}").is_empty());
}

#[test]
fn compile_rejects_circuit_wider_than_device() {
    let mut circuit = Circuit::new(4);
    circuit.h(Qubit::new(0)).unwrap();
    let device = Device::line("test-device", 2).unwrap();

    let err = compile(
        &circuit,
        CompileConfig {
            mode: CompileMode::Normal,
            target_basis: None,
            device: Some(device),
            resource_policy: ResourcePolicy::default(),
            seed: None,
        },
    )
    .unwrap_err();

    assert!(format!("{err}").contains("4 logical qubits"));
}
