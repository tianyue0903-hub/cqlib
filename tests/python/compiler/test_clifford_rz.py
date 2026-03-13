# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http:#www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Clifford+Rz optimizer tests.

Test coverage:
- numeric linear rewrite and matrix preservation
- unsupported-gate segment splitting
- symbolic RZ boundary preservation
- recursive optimization inside if-else and while-loop bodies
- nested control-flow structure preservation
"""

import numpy as np

from cqlib.circuit import Circuit, ConditionView, ControlFlow, Directive, Parameter, Qubit, StandardGate
from cqlib.compiler import CliffordRzOptimization


def _op_names(ops) -> list[str]:
    return [op.instruction.name for op in ops]


class TestCliffordRzOptimization:
    """Tests Python Clifford+Rz optimization coverage."""

    def test_linear_rewrite_preserves_matrix(self) -> None:
        """Cancels a numeric X-RZ-X-RZ pattern and preserves the unitary."""
        circuit = Circuit(1)
        circuit.x(0)
        circuit.rz(0, np.pi / 4)
        circuit.x(0)
        circuit.rz(0, np.pi / 4)

        optimizer = CliffordRzOptimization(level="heavy")
        optimized = optimizer.execute(circuit)

        assert len(list(optimized.operations)) == 0
        assert np.allclose(optimized.to_matrix(), circuit.to_matrix())

    def test_unsupported_gate_splits_supported_chunks(self) -> None:
        """Cancels supported gates around an unsupported CZ boundary."""
        circuit = Circuit(2)
        circuit.h(0)
        circuit.h(0)
        circuit.cz(0, 1)
        circuit.x(1)
        circuit.x(1)

        optimizer = CliffordRzOptimization()
        optimized = optimizer.execute(circuit)
        ops = list(optimized.operations)

        assert _op_names(ops) == ["CZ"]

    def test_symbolic_rz_is_hard_boundary(self) -> None:
        """Keeps symbolic RZ untouched and prevents cross-boundary cancellation."""
        circuit = Circuit(1)
        theta = Parameter("theta")
        circuit.x(0)
        circuit.rz(0, theta)
        circuit.x(0)

        optimizer = CliffordRzOptimization()
        optimized = optimizer.execute(circuit)
        ops = list(optimized.operations)

        assert _op_names(ops) == ["X", "RZ", "X"]
        assert len(optimized.parameters) == 1
        assert optimized.parameters[0] == theta

    def test_if_else_bodies_are_optimized_recursively(self) -> None:
        """Optimizes true and false branches without changing the control-flow node."""
        circuit = Circuit(3)
        circuit.measure(0)
        circuit.if_else(
            ConditionView(Qubit(0), 1),
            [(StandardGate.H, [1]), (StandardGate.H, [1]), (StandardGate.X, [1])],
            [(StandardGate.X, [1]), (StandardGate.X, [1]), (StandardGate.CZ, [1, 2])],
        )

        optimizer = CliffordRzOptimization(level="heavy")
        optimized = optimizer.execute(circuit)
        ops = list(optimized.operations)

        assert len(ops) == 2
        assert ops[0].instruction.name == "Measure"

        control_flow = ops[1].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_if_else
        gate = control_flow.as_if_else
        assert _op_names(gate.true_body) == ["X"]
        assert _op_names(gate.false_body) == ["CZ"]

    def test_while_loop_preserves_directives_and_optimizes_segments(self) -> None:
        """Preserves barriers while canceling supported gates on both sides."""
        circuit = Circuit(2)
        circuit.measure(0)
        circuit.while_loop(
            ConditionView(Qubit(0), 1),
            [
                (StandardGate.X, [1]),
                (StandardGate.X, [1]),
                (Directive.barrier(), [1]),
                (StandardGate.H, [1]),
                (StandardGate.H, [1]),
            ],
        )

        optimizer = CliffordRzOptimization(level="heavy")
        optimized = optimizer.execute(circuit)
        ops = list(optimized.operations)

        assert len(ops) == 2
        assert ops[0].instruction.name == "Measure"

        control_flow = ops[1].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_while_loop
        gate = control_flow.as_while_loop
        assert _op_names(gate.body) == ["Barrier"]

    def test_nested_control_flow_structure_is_preserved(self) -> None:
        """Keeps nested control flow intact while optimizing inside nested branches."""
        inner_true = Circuit(3)
        inner_true.h(2)
        inner_true.h(2)
        inner_true.x(2)

        inner_false = Circuit(3)
        inner_false.x(2)
        inner_false.x(2)
        inner_false.cz(0, 2)

        nested_if = ControlFlow.if_else(
            ConditionView(Qubit(1), 1),
            list(inner_true.operations),
            list(inner_false.operations),
        )

        circuit = Circuit(3)
        circuit.measure(0)
        circuit.measure(1)
        circuit.while_loop(
            ConditionView(Qubit(0), 1),
            [
                (nested_if, [0, 1, 2]),
                (StandardGate.H, [2]),
                (StandardGate.H, [2]),
            ],
        )

        optimizer = CliffordRzOptimization(level="heavy")
        optimized = optimizer.execute(circuit)
        ops = list(optimized.operations)

        assert len(ops) == 3
        assert _op_names(ops[:2]) == ["Measure", "Measure"]
        outer = ops[2].instruction.control_flow
        assert outer is not None
        assert outer.is_while_loop
        assert len(outer.as_while_loop.body) == 1

        nested = outer.as_while_loop.body[0].instruction.control_flow
        assert nested is not None
        assert nested.is_if_else
        assert _op_names(nested.as_if_else.true_body) == ["X"]
        assert _op_names(nested.as_if_else.false_body) == ["CZ"]
