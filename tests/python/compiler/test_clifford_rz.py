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
- built-in light/heavy flows
- ordered custom strategy flows and validation
- Hadamard/single-qubit/two-qubit cleanup
- phase-polynomial and global-Rz reductions
- recursive optimization in control-flow bodies
- full supported Clifford+Rz gate coverage
"""

import numpy as np
import pytest

from cqlib.circuit import Circuit, ConditionView, ControlFlow, Directive, Parameter, Qubit, StandardGate
from cqlib.compiler import CliffordRzOptimization


def _op_names(ops) -> list[str]:
    return [op.instruction.name for op in ops]


def _matrix_with_global_phase(circuit: Circuit) -> np.ndarray:
    phase = circuit.global_phase.evaluate({})
    return np.exp(1j * phase) * circuit.to_matrix()


def _assert_same_matrix(lhs: Circuit, rhs: Circuit) -> None:
    assert np.allclose(_matrix_with_global_phase(lhs), _matrix_with_global_phase(rhs))


class TestCliffordRzOptimization:
    """Tests Python Clifford+Rz optimization coverage."""

    def test_light_flow_exposes_effective_strategies(self) -> None:
        optimizer = CliffordRzOptimization(level="light")
        assert optimizer.level == "light"
        assert optimizer.strategies == ["hadamard", "single_qubit", "two_qubit"]

    def test_heavy_flow_exposes_effective_strategies(self) -> None:
        optimizer = CliffordRzOptimization(level="heavy")
        assert optimizer.level == "heavy"
        assert optimizer.strategies == [
            "hadamard",
            "single_qubit",
            "two_qubit",
            "phase_polynomial",
            "global_rz",
            "single_qubit",
            "two_qubit",
        ]

    def test_custom_flow_requires_non_empty_strategies(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty strategies list"):
            CliffordRzOptimization(level="custom")

    def test_custom_flow_rejects_unknown_strategy(self) -> None:
        with pytest.raises(ValueError, match="unknown CliffordRz strategy"):
            CliffordRzOptimization(level="custom", strategies=["not_a_strategy"])

    def test_builtin_levels_reject_explicit_strategies(self) -> None:
        with pytest.raises(ValueError, match="strategies can only be provided"):
            CliffordRzOptimization(level="light", strategies=["hadamard"])

    def test_custom_flow_preserves_order_and_duplicates(self) -> None:
        optimizer = CliffordRzOptimization(
            level="custom",
            strategies=["single_qubit", "single_qubit", "two_qubit"],
        )
        assert optimizer.level == "custom"
        assert optimizer.strategies == ["single_qubit", "single_qubit", "two_qubit"]

    def test_linear_rewrite_preserves_matrix(self) -> None:
        circuit = Circuit(1)
        circuit.x(0)
        circuit.rz(0, np.pi / 4)
        circuit.x(0)
        circuit.rz(0, np.pi / 4)

        optimized = CliffordRzOptimization(level="heavy").execute(circuit)

        assert len(list(optimized.operations)) == 0
        _assert_same_matrix(circuit, optimized)

    def test_phase_alias_merges_as_rz(self) -> None:
        circuit = Circuit(1)
        circuit.phase(0, 0.2)
        circuit.phase(0, 0.3)

        optimized = CliffordRzOptimization(level="light").execute(circuit)
        ops = list(optimized.operations)

        assert _op_names(ops) == ["RZ"]
        _assert_same_matrix(circuit, optimized)

    def test_two_qubit_cleanup_cancels_consecutive_cx(self) -> None:
        circuit = Circuit(2)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        circuit.x(0)

        optimized = CliffordRzOptimization(level="light").execute(circuit)

        assert _op_names(list(optimized.operations)) == ["X"]

    def test_hadamard_strategy_flips_cx_and_enables_cancellation(self) -> None:
        circuit = Circuit(2)
        circuit.h(0)
        circuit.h(1)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.h(1)
        circuit.cx(1, 0)
        circuit.cx(1, 0)

        custom = CliffordRzOptimization(level="custom", strategies=["two_qubit"])
        optimized_without_h = custom.execute(circuit)

        custom = CliffordRzOptimization(level="custom", strategies=["hadamard", "two_qubit"])
        optimized_with_h = custom.execute(circuit)

        assert len(list(optimized_with_h.operations)) < len(list(optimized_without_h.operations))
        _assert_same_matrix(circuit, optimized_with_h)

    def test_light_vs_heavy_flow(self) -> None:
        circuit = Circuit(2)
        circuit.rz(1, 0.2)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        circuit.rz(0, 0.4)

        light = CliffordRzOptimization(level="light").execute(circuit)
        heavy = CliffordRzOptimization(level="heavy").execute(circuit)

        assert len(list(heavy.operations)) < len(list(light.operations))
        _assert_same_matrix(circuit, light)
        _assert_same_matrix(circuit, heavy)

    def test_global_rz_custom_flow_reduces_supported_component(self) -> None:
        circuit = Circuit(2)
        circuit.rz(0, 0.2)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        circuit.rz(1, 0.2)

        optimizer = CliffordRzOptimization(level="custom", strategies=["global_rz"])
        optimized = optimizer.execute(circuit)

        assert len(list(optimized.operations)) < len(list(circuit.operations))
        _assert_same_matrix(circuit, optimized)

    def test_supported_clifford_gate_set_is_optimized(self) -> None:
        circuit = Circuit(2)
        circuit.y(0)
        circuit.x2p(0)
        circuit.x2m(0)
        circuit.y2p(1)
        circuit.y2m(1)
        circuit.cy(0, 1)
        circuit.cz(0, 1)
        circuit.swap(0, 1)
        circuit.phase(0, 0.125)
        circuit.rz(1, 0.375)

        optimized = CliffordRzOptimization(level="heavy").execute(circuit)
        _assert_same_matrix(circuit, optimized)

    def test_symbolic_rz_is_hard_boundary(self) -> None:
        circuit = Circuit(1)
        theta = Parameter("theta")
        circuit.x(0)
        circuit.rz(0, theta)
        circuit.x(0)

        optimized = CliffordRzOptimization().execute(circuit)
        ops = list(optimized.operations)

        assert _op_names(ops) == ["X", "RZ", "X"]
        assert len(optimized.parameters) == 1
        assert optimized.parameters[0] == theta

    def test_if_else_bodies_are_optimized_recursively(self) -> None:
        circuit = Circuit(3)
        circuit.measure(0)
        circuit.if_else(
            ConditionView(Qubit(0), 1),
            [(StandardGate.H, [1]), (StandardGate.H, [1]), (StandardGate.X, [1])],
            [(StandardGate.X, [1]), (StandardGate.X, [1]), (StandardGate.CZ, [1, 2])],
        )

        optimized = CliffordRzOptimization(level="heavy").execute(circuit)
        ops = list(optimized.operations)

        assert len(ops) == 2
        assert ops[0].instruction.name == "Measure"

        control_flow = ops[1].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_if_else
        gate = control_flow.as_if_else
        assert _op_names(gate.true_body) == ["X"]
        assert _op_names(gate.false_body) == ["H", "CX", "H"]

    def test_while_loop_preserves_directives_and_optimizes_segments(self) -> None:
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

        optimized = CliffordRzOptimization(level="heavy").execute(circuit)
        ops = list(optimized.operations)

        assert len(ops) == 2
        assert ops[0].instruction.name == "Measure"

        control_flow = ops[1].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_while_loop
        gate = control_flow.as_while_loop
        assert _op_names(gate.body) == ["Barrier"]

    def test_nested_control_flow_structure_is_preserved(self) -> None:
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

        optimized = CliffordRzOptimization(level="heavy").execute(circuit)
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
        assert _op_names(nested.as_if_else.false_body) == ["H", "CX", "H"]
