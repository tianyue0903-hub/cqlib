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
use crate::circuit::gate::standard_gate::StandardGate;

#[test]
fn test_instruction_prop() {
    let instr = Instruction::Standard(StandardGate::H);
    let prop = InstructionProp::new(instr.clone(), 0.01).with_length(10.0);

    assert!(matches!(
        prop.instruction(),
        Instruction::Standard(StandardGate::H)
    ));
    assert_eq!(prop.error_rate(), 0.01);
    assert_eq!(prop.length(), Some(10.0));
}

#[test]
fn test_qubit_prop() {
    let instr = Instruction::Standard(StandardGate::H);
    let instr_prop = InstructionProp::new(instr, 0.01).with_length(10.0);

    let prop = QubitProp::new(0.05)
        .with_prob_meas0_prep1(0.06)
        .with_prob_meas1_prep0(0.04)
        .with_t1(50.0)
        .with_t2(30.0)
        .with_frequency(5.0)
        .with_native_instruction(instr_prop);

    assert_eq!(prop.readout_error(), 0.05);
    assert_eq!(prop.t1(), Some(50.0));
    assert_eq!(prop.t2(), Some(30.0));
    assert_eq!(prop.frequency(), Some(5.0));
    assert_eq!(prop.native_instructions().len(), 1);
    assert_eq!(prop.native_instructions()[0].error_rate(), 0.01);
}

#[test]
fn test_edge_prop() {
    let instr = Instruction::Standard(StandardGate::CX);
    let instr_prop = InstructionProp::new(instr, 0.02).with_length(200.0);

    let prop = EdgeProp::new().with_native_instruction(instr_prop);
    assert_eq!(prop.native_instructions().len(), 1);
    assert_eq!(prop.native_instructions()[0].error_rate(), 0.02);
}

#[test]
fn test_device_creation_and_defaults() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let topo = Topology::new(vec![q0, q1], vec![(q0, q1, "cx".to_string())]);

    let mut device = Device::new(
        "test_device".to_string(),
        HashSet::from_iter([q0, q1]),
        topo.unwrap(),
    )
    .unwrap()
    .with_default_t1(40.0)
    .with_default_t2(20.0)
    .with_default_readout_error(0.03)
    .with_default_single_qubit_error(0.001)
    .with_default_two_qubit_error(0.01);

    assert_eq!(device.name(), "test_device");
    assert_eq!(device.default_single_qubit_error(), Some(0.001));
    assert_eq!(device.default_two_qubit_error(), Some(0.01));

    // Fallbacks for q0 (no specific properties set yet)
    assert_eq!(device.get_t1(q0), Some(40.0));
    assert_eq!(device.get_t2(q0), Some(20.0));
    assert_eq!(device.get_readout_error(q0), Some(0.03));

    // Add specific properties for q0
    let q0_prop = QubitProp::new(0.02).with_t1(60.0).with_t2(30.0);
    assert!(device.add_qubit_properties(q0, q0_prop).is_ok());

    assert_eq!(device.get_t1(q0), Some(60.0));
    assert_eq!(device.get_t2(q0), Some(30.0));
    assert_eq!(device.get_readout_error(q0), Some(0.02));

    // Fallbacks still apply to q1
    assert_eq!(device.get_t1(q1), Some(40.0));
}

#[test]
fn test_device_errors() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2); // Not in topology
    let topo = Topology::new(vec![q0, q1], vec![(q0, q1, "0-1".to_string())]);

    let mut device = Device::new(
        "test_device".to_string(),
        HashSet::from_iter([q0, q1]),
        topo.unwrap(),
    )
    .unwrap();

    let prop = QubitProp::new(0.05);
    let err = device.add_qubit_properties(q2, prop);
    assert_eq!(err.unwrap_err(), DeviceError::QubitNotInTopology(q2));

    let edge_prop = EdgeProp::new();
    let err = device.add_edge_properties(q1, q0, edge_prop).unwrap_err();
    assert_eq!(err, DeviceError::EdgeNotInTopology(q1, q0));
}

#[test]
fn invalid_qubits_must_be_registered_with_device() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let topology = Topology::new(vec![q0, q1], vec![(q0, q1, "CX".to_string())]).unwrap();
    let device = Device::new("test-device", HashSet::from([q0, q1]), topology).unwrap();

    assert_eq!(
        device
            .clone()
            .with_invalid_qubits(HashSet::from([q2]))
            .unwrap_err(),
        DeviceError::QubitNotInDevice(q2)
    );

    let mut device = device.with_invalid_qubits(HashSet::from([q1])).unwrap();
    assert_eq!(
        device.set_invalid_qubits(HashSet::from([q2])).unwrap_err(),
        DeviceError::QubitNotInDevice(q2)
    );
    assert_eq!(device.invalid_qubits().collect::<Vec<_>>(), vec![q1]);
}

#[test]
fn usable_qubits_exclude_invalid_qubits_in_stable_order() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let topology = Topology::new(
        vec![q0, q1, q2],
        vec![(q0, q1, "CX".to_string()), (q1, q2, "CX".to_string())],
    )
    .unwrap();
    let device = Device::new("test-device", HashSet::from([q2, q0, q1]), topology)
        .unwrap()
        .with_invalid_qubits(HashSet::from([q1]))
        .unwrap();

    assert_eq!(device.qubits().collect::<Vec<_>>(), vec![q0, q1, q2]);
    assert_eq!(device.invalid_qubits().collect::<Vec<_>>(), vec![q1]);
    assert_eq!(device.usable_qubits().collect::<Vec<_>>(), vec![q0, q2]);
    assert!(device.is_usable_qubit(q0));
    assert!(!device.is_usable_qubit(q1));
    assert!(!device.is_usable_qubit(PhysicalQubit::new(99)));
    assert_eq!(device.num_usable_qubits(), 2);
}

#[test]
fn single_qubit_error_uses_native_instruction_then_default() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let topology = Topology::new(vec![q0, q1, q2], vec![]).unwrap();
    let mut device = Device::new("test-device", HashSet::from([q0, q1, q2]), topology)
        .unwrap()
        .with_default_single_qubit_error(0.01)
        .with_invalid_qubits(HashSet::from([q2]))
        .unwrap();
    device
        .add_qubit_properties(
            q0,
            QubitProp::new(0.02).with_native_instruction(InstructionProp::new(
                Instruction::Standard(StandardGate::H),
                0.001,
            )),
        )
        .unwrap();

    assert_eq!(
        device.single_qubit_error(q0, &Instruction::Standard(StandardGate::H)),
        Some(0.001)
    );
    assert_eq!(
        device.single_qubit_error(q0, &Instruction::Standard(StandardGate::X)),
        Some(0.01)
    );
    assert_eq!(
        device.single_qubit_error(q1, &Instruction::Standard(StandardGate::H)),
        Some(0.01)
    );
    assert_eq!(
        device.single_qubit_error(q2, &Instruction::Standard(StandardGate::H)),
        None
    );
    assert_eq!(
        device.single_qubit_error(
            PhysicalQubit::new(99),
            &Instruction::Standard(StandardGate::H)
        ),
        None
    );
}

#[test]
fn two_qubit_and_edge_errors_respect_direction_and_usability() {
    let q0 = PhysicalQubit::new(0);
    let q1 = PhysicalQubit::new(1);
    let q2 = PhysicalQubit::new(2);
    let topology = Topology::new(
        vec![q0, q1, q2],
        vec![
            (q0, q1, "CX".to_string()),
            (q1, q0, "CX".to_string()),
            (q1, q2, "CX".to_string()),
        ],
    )
    .unwrap();
    let mut device = Device::new("test-device", HashSet::from([q0, q1, q2]), topology)
        .unwrap()
        .with_default_two_qubit_error(0.07);
    device
        .add_edge_properties(
            q0,
            q1,
            EdgeProp::new()
                .with_native_instruction(InstructionProp::new(
                    Instruction::Standard(StandardGate::CZ),
                    0.03,
                ))
                .with_native_instruction(InstructionProp::new(
                    Instruction::Standard(StandardGate::CX),
                    0.02,
                )),
        )
        .unwrap();

    assert_eq!(
        device.two_qubit_error(q0, q1, &Instruction::Standard(StandardGate::CX)),
        Some(0.02)
    );
    assert_eq!(
        device.two_qubit_error(q0, q1, &Instruction::Standard(StandardGate::SWAP)),
        Some(0.07)
    );
    assert_eq!(
        device.two_qubit_error(q1, q0, &Instruction::Standard(StandardGate::CX)),
        Some(0.07)
    );
    assert_eq!(
        device.two_qubit_error(q0, q2, &Instruction::Standard(StandardGate::CX)),
        None
    );
    assert_eq!(device.edge_error(q0, q1), Some(0.02));
    assert_eq!(device.edge_error(q1, q0), Some(0.07));

    device.set_invalid_qubits(HashSet::from([q2])).unwrap();
    assert_eq!(
        device.two_qubit_error(q1, q2, &Instruction::Standard(StandardGate::CX)),
        None
    );
    assert_eq!(device.edge_error(q1, q2), None);
}
