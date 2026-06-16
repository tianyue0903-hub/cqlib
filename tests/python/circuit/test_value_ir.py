import numpy as np

from cqlib.circuit import (
    Circuit,
    ClassicalControlOp,
    ValueControlBody,
    ValueOperation,
)


def test_operations_are_self_contained_value_operations():
    circuit = Circuit(1)
    circuit.rx(0, 0.25)

    operation = circuit[0]

    assert isinstance(operation, ValueOperation)
    assert operation.params == [0.25]


def test_dynamic_control_flow_round_trips_through_value_ir():
    body_circuit = Circuit(1)
    body_circuit.x(0)

    circuit = Circuit(1)
    measured = circuit.measure(0)
    control = ClassicalControlOp.if_(
        measured.expr().to_bool(),
        ValueControlBody(body_circuit.operations),
    )
    circuit.append_control(control)

    circuit.validate()
    assert len(circuit) == 2
    assert circuit[1].instruction.is_classical_control


def test_matrix_is_exposed_as_a_circuit_method_only():
    circuit = Circuit(1)
    circuit.h(0)

    assert np.allclose(circuit.to_matrix(), np.array([[1, 1], [1, -1]]) / np.sqrt(2))

    import cqlib.circuit as circuit_module

    assert not hasattr(circuit_module, "circuit_to_matrix")
    assert not hasattr(circuit_module, "ControlFlow")
