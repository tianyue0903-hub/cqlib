use super::*;
use crate::circuit::gate::{ClassicalDataOp, Instruction};
use crate::circuit::{Circuit, ClassicalExpr, ClassicalType, ParameterValue, Qubit, StandardGate};
use crate::ir::qasm3_loads;

fn assert_round_trip_standard_gates(source: &str, expected: &[StandardGate]) {
    let circuit = qasm3_loads(source).unwrap();
    let actual = circuit
        .operations()
        .iter()
        .filter_map(|op| match op.instruction {
            Instruction::Standard(gate) => Some(gate),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn dumps_stdgate_only_circuit_without_extra_definitions() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.h(q0).unwrap();
    circuit.cx(q0, q1).unwrap();
    circuit.phase(q1, std::f64::consts::PI / 2.0).unwrap();
    circuit.u(q0, 0.1, 0.2, 0.3).unwrap();

    let qasm = dumps(&circuit).unwrap();

    assert_eq!(
        qasm,
        r#"OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;

h q[0];
cx q[0],q[1];
p(1.5707963267948966) q[1];
u3(0.1,0.2,0.3) q[0];
"#
    );
    assert_round_trip_standard_gates(
        &qasm,
        &[
            StandardGate::H,
            StandardGate::CX,
            StandardGate::Phase,
            StandardGate::U,
        ],
    );
}

#[test]
fn dumps_extension_gate_definitions_and_keeps_call_names() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.x2p(q0).unwrap();
    circuit.x2m(q1).unwrap();
    circuit.y2p(q0).unwrap();
    circuit.y2m(q1).unwrap();
    circuit.xy2p(q0, 0.25).unwrap();
    circuit.xy2m(q1, 0.5).unwrap();

    let qasm = dumps(&circuit).unwrap();

    assert_eq!(
        qasm,
        r#"OPENQASM 3.0;
include "stdgates.inc";

gate x2p q { rx(pi/2) q; }

gate x2m q { rx(-pi/2) q; }

gate y2p q { ry(pi/2) q; }

gate y2m q { ry(-pi/2) q; }

gate xy2p(phi) q { rz(-phi) q; x2p q; rz(phi) q; }

gate xy2m(phi) q { rz(-phi) q; x2m q; rz(phi) q; }

qubit[2] q;

x2p q[0];
x2m q[1];
y2p q[0];
y2m q[1];
xy2p(0.25) q[0];
xy2m(0.5) q[1];
"#
    );
    assert_round_trip_standard_gates(
        &qasm,
        &[
            StandardGate::X2P,
            StandardGate::X2M,
            StandardGate::Y2P,
            StandardGate::Y2M,
            StandardGate::XY2P,
            StandardGate::XY2M,
        ],
    );
}

#[test]
fn dumps_ising_gate_definitions_and_keeps_call_names() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    circuit.rxx(q0, q1, 0.1).unwrap();
    circuit.ryy(q0, q1, 0.2).unwrap();
    circuit.rzz(q0, q1, 0.3).unwrap();
    circuit.rzx(q0, q1, 0.4).unwrap();

    let qasm = dumps(&circuit).unwrap();

    assert_eq!(
        qasm,
        r#"OPENQASM 3.0;
include "stdgates.inc";

gate rxx(theta) a,b { h a; h b; cx a,b; rz(theta) b; cx a,b; h a; h b; }

gate ryy(theta) a,b { rx(pi/2) a; rx(pi/2) b; cx a,b; rz(theta) b; cx a,b; rx(-pi/2) a; rx(-pi/2) b; }

gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }

gate rzx(theta) a,b { h b; cx a,b; rz(theta) b; cx a,b; h b; }

qubit[2] q;

rxx(0.1) q[0],q[1];
ryy(0.2) q[0],q[1];
rzz(0.3) q[0],q[1];
rzx(0.4) q[0],q[1];
"#
    );
    assert_round_trip_standard_gates(
        &qasm,
        &[
            StandardGate::RXX,
            StandardGate::RYY,
            StandardGate::RZZ,
            StandardGate::RZX,
        ],
    );
}

#[test]
fn dumps_gphase_as_statement() {
    let mut circuit = Circuit::new(0);
    circuit
        .append(
            Instruction::Standard(StandardGate::GPhase),
            std::iter::empty::<Qubit>(),
            [ParameterValue::Fixed(0.25)],
            None,
        )
        .unwrap();

    let qasm = dumps(&circuit).unwrap();

    assert_eq!(
        qasm,
        r#"OPENQASM 3.0;
include "stdgates.inc";

qubit[0] q;

gphase(0.25);
"#
    );
    let loaded = qasm3_loads(&qasm).unwrap();
    assert_eq!(loaded.operations().len(), 0);
    assert!((loaded.global_phase().evaluate(&None).unwrap() - 0.25).abs() < 1e-10);
}

#[test]
fn dumps_measurement_reset_and_barrier() {
    let q0 = Qubit::new(0);
    let q1 = Qubit::new(1);
    let mut circuit = Circuit::new(2);
    let bits = circuit.var(ClassicalType::bit_vec(2).unwrap());
    circuit.measure_bits_into([q0, q1], bits).unwrap();
    circuit.barrier(vec![q0, q1]).unwrap();
    circuit.reset(q1).unwrap();

    let qasm = dumps(&circuit).unwrap();

    assert_eq!(
        qasm,
        r#"OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c0;
bit[2] v0;

c0 = measure q;
barrier q[0],q[1];
reset q[1];
"#
    );

    let loaded = qasm3_loads(&qasm).unwrap();
    assert!(matches!(
        loaded.operations()[0].instruction,
        Instruction::ClassicalData(ClassicalDataOp::MeasureBits { .. })
    ));
}

#[test]
fn dumps_single_bit_measurement_round_trip() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    let bit = circuit.var(ClassicalType::Bit);
    circuit.measure_into(q0, bit).unwrap();

    let qasm = dumps(&circuit).unwrap();

    assert_eq!(
        qasm,
        r#"OPENQASM 3.0;
include "stdgates.inc";

qubit q;
bit c0;
bit v0;

c0 = measure q;
"#
    );
    let loaded = qasm3_loads(&qasm).unwrap();
    assert!(matches!(
        loaded.operations()[0].instruction,
        Instruction::ClassicalData(ClassicalDataOp::MeasureBit { .. })
    ));
}

#[test]
fn rejects_general_store() {
    let mut circuit = Circuit::new(0);
    let bit = circuit.var(ClassicalType::Bit);
    circuit
        .store(bit, ClassicalExpr::bit_literal(true))
        .unwrap();

    let err = dumps(&circuit).unwrap_err();

    assert!(matches!(
        err,
        Qasm3DumpError::UnsupportedClassicalData(message)
            if message.contains("general store")
    ));
}

#[test]
fn rejects_delay() {
    let q0 = Qubit::new(0);
    let mut circuit = Circuit::new(1);
    circuit
        .append(Instruction::Delay, [q0], [ParameterValue::Fixed(1.0)], None)
        .unwrap();

    let err = dumps(&circuit).unwrap_err();

    assert!(matches!(
        err,
        Qasm3DumpError::UnsupportedInstruction(message) if message == "delay"
    ));
}
