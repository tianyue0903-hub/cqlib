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
VF2 compiler mapping tests.

Test coverage:
- strict VF2 mapping behavior and fallback layout search
- candidate schema/ranking/determinism/options behavior
- topology-size and strict-mode validation errors
- randomized stress checks for topology-valid mapped circuits
"""

import random

import pytest

from cqlib.circuit import Circuit, ConditionView, ControlFlow, Directive, Parameter, Qubit, StandardGate
from cqlib.compiler import (
    vf2_find_initial_layout,
    vf2_find_initial_layout_candidates,
    vf2_is_subgraph_isomorphic,
    vf2_map,
)

from cqlib.device import Topology

from . import assert_all_2q_on_topology
from . import assert_ops_on_topology_recursive
from . import random_circuit


def _triangle_circuit() -> Circuit:
    """Builds a 3-qubit triangle interaction circuit."""
    circuit = Circuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(0, 2)
    return circuit


class TestVf2LayoutBehavior:
    """Tests strict/fallback behavior for initial layout and mapping."""

    def test_find_initial_layout_fallback_top1(self):
        """Returns top-1 fallback layout when strict subgraph match is impossible."""
        topology = Topology.line([0, 1, 2])
        circuit = _triangle_circuit()

        assert vf2_is_subgraph_isomorphic(circuit, topology) is False

        layout = vf2_find_initial_layout(circuit, topology)
        assert layout is not None
        assert len(layout) == 3
        assert len(set(layout)) == 3

    def test_vf2_map_remains_strict(self):
        """Raises error in strict map mode when routing would be required."""
        topology = Topology.line([0, 1, 2])
        circuit = _triangle_circuit()

        with pytest.raises(ValueError):
            vf2_map(circuit, topology)

    def test_vf2_map_rejects_topology_too_small(self):
        """Raises error when topology has fewer physical qubits than logical width."""
        topology = Topology.line([0, 1])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        with pytest.raises(ValueError):
            vf2_map(circuit, topology)

    def test_vf2_map_if_else_preserves_structure(self):
        """Maps `if_else` with one static layout and preserves the control-flow node."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.measure(0)
        circuit.if_else(
            ConditionView(Qubit(0), 1),
            [(StandardGate.X, [1])],
            [(StandardGate.CX, [1, 2])],
        )

        mapped = vf2_map(circuit, topology)
        mapped_ops = list(mapped.operations)
        assert_ops_on_topology_recursive(mapped_ops, topology)

        control_flow = mapped_ops[1].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_if_else
        gate = control_flow.as_if_else
        assert gate.condition.qubit.index == mapped_ops[0].qubits[0].index
        assert len(gate.true_body) == 1
        assert len(gate.false_body) == 1

    def test_vf2_map_while_loop_preserves_structure(self):
        """Maps `while_loop` with one static layout and preserves the control-flow node."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.measure(0)
        circuit.while_loop(
            ConditionView(Qubit(0), 1),
            [(StandardGate.CX, [1, 2])],
        )

        mapped = vf2_map(circuit, topology)
        mapped_ops = list(mapped.operations)
        assert_ops_on_topology_recursive(mapped_ops, topology)

        control_flow = mapped_ops[1].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_while_loop
        gate = control_flow.as_while_loop
        assert gate.condition.qubit.index == mapped_ops[0].qubits[0].index
        assert len(gate.body) == 1

    def test_if_else_isomorphic(self):
        """Includes `if_else` body interactions in the global static-layout check."""
        topology = Topology.line([0, 1, 2, 3])
        circuit = Circuit(4)
        circuit.measure(0)
        circuit.if_else(
            ConditionView(Qubit(0), 1),
            [(StandardGate.CX, [1, 2])],
            [(StandardGate.CX, [2, 3])],
        )

        assert vf2_is_subgraph_isomorphic(circuit, topology) is True

    def test_vf2_map_preserves_global_phase_with_control_flow(self):
        """Preserves source global phase when statically mapping control-flow circuits."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.set_global_phase(0.5)
        circuit.measure(0)
        circuit.if_else(
            ConditionView(Qubit(0), 1),
            [(StandardGate.CX, [1, 2])],
        )

        mapped = vf2_map(circuit, topology)
        assert mapped.global_phase.evaluate() == pytest.approx(0.5)
        assert_ops_on_topology_recursive(list(mapped.operations), topology)

    def test_vf2_map_supports_nested_control_flow_objects(self):
        """Preserves nested control-flow bodies exposed through the Python binding."""
        topology = Topology.line([0, 1, 2])

        inner_true = Circuit(3)
        inner_true.cx(0, 1)
        inner_false = Circuit(3)
        inner_false.cx(1, 2)
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
                (Directive.measure(), [2]),
            ],
        )

        mapped = vf2_map(circuit, topology)
        mapped_ops = list(mapped.operations)
        assert_ops_on_topology_recursive(mapped_ops, topology)

        control_flow = mapped_ops[2].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_while_loop
        assert control_flow.as_while_loop.body[1].name == "Measure"
        nested = control_flow.as_while_loop.body[0].instruction.control_flow
        assert nested is not None
        assert nested.is_if_else


class TestVf2CandidateSearch:
    """Tests VF2 candidate API schema and candidate-selection options."""

    def test_candidates_topk_and_schema(self):
        """Returns bounded top-k results with expected candidate and score keys."""
        topology = Topology.line([0, 1, 2, 3])
        circuit = _triangle_circuit()
        circuit.h(0)
        circuit.x(2)

        candidates = vf2_find_initial_layout_candidates(circuit, topology, top_k=3)
        assert 0 < len(candidates) <= 3

        for candidate in candidates:
            assert {"region", "layout", "score"} <= set(candidate.keys())
            assert isinstance(candidate["region"], list)
            assert isinstance(candidate["layout"], list)
            assert len(candidate["region"]) == 3
            assert len(candidate["layout"]) == 3

            score = candidate["score"]
            assert {"total", "fidelity", "topology_fit", "gate_distribution"} <= set(
                score.keys()
            )
            for key in ("total", "fidelity", "topology_fit", "gate_distribution"):
                assert 0.0 <= score[key] <= 1.0

    def test_candidates_sorted_deterministically(self):
        """Produces deterministic candidate ordering for fixed input and topology."""
        topology = Topology.line([0, 1, 2, 3])
        circuit = _triangle_circuit()

        c1 = vf2_find_initial_layout_candidates(circuit, topology, top_k=5)
        c2 = vf2_find_initial_layout_candidates(circuit, topology, top_k=5)

        sig1 = [(c["layout"], c["score"]["total"]) for c in c1]
        sig2 = [(c["layout"], c["score"]["total"]) for c in c2]
        assert sig1 == sig2

    def test_candidates_custom_weights(self):
        """Applies custom candidate weights and keeps total scores descending."""
        topology = Topology.line([0, 1, 2, 3])
        circuit = _triangle_circuit()
        circuit.h(1)

        candidates = vf2_find_initial_layout_candidates(
            circuit,
            topology,
            top_k=4,
            w_fidelity=0.8,
            w_topology=0.1,
            w_gate_distribution=0.1,
        )
        assert len(candidates) > 0
        totals = [c["score"]["total"] for c in candidates]
        assert totals == sorted(totals, reverse=True)

    def test_candidates_topk_effective_on_strict_case(self):
        """Returns more than one candidate in a strictly embeddable small case."""
        topology = Topology(
            [0, 1, 2, 3], [(0, 1, "G0"), (1, 2, "G1"), (2, 3, "G2"), (3, 0, "G3")]
        )
        circuit = Circuit(2)
        circuit.cx(0, 1)

        candidates = vf2_find_initial_layout_candidates(circuit, topology, top_k=4)
        assert 1 < len(candidates) <= 4

    def test_candidates_respect_max_matches_per_subgraph(self):
        """Respects match cap and returns bounded candidate count."""
        topology = Topology(
            [0, 1, 2, 3], [(0, 1, "G0"), (1, 2, "G1"), (2, 3, "G2"), (3, 0, "G3")]
        )
        circuit = Circuit(2)
        circuit.cx(0, 1)

        candidates = vf2_find_initial_layout_candidates(
            circuit,
            topology,
            top_k=8,
            max_matches_per_subgraph=1,
        )
        assert len(candidates) <= 1

    def test_candidates_topk_zero_returns_empty(self):
        """Returns an empty list when top_k is explicitly set to zero."""
        topology = Topology([0, 1, 2], [(0, 1, "G0"), (1, 2, "G1")])
        circuit = Circuit(2)
        circuit.cx(0, 1)

        candidates = vf2_find_initial_layout_candidates(circuit, topology, top_k=0)
        assert candidates == []

    def test_candidates_non_positive_weights_are_stable(self):
        """Handles non-positive candidate weights without crashing."""
        topology = Topology(
            [0, 1, 2, 3], [(0, 1, "G0"), (1, 2, "G1"), (2, 3, "G2"), (3, 0, "G3")]
        )
        circuit = Circuit(2)
        circuit.cx(0, 1)

        candidates = vf2_find_initial_layout_candidates(
            circuit,
            topology,
            top_k=4,
            w_fidelity=-1.0,
            w_topology=0.0,
            w_gate_distribution=-5.0,
        )
        assert 1 <= len(candidates) <= 4
        for candidate in candidates:
            assert {"region", "layout", "score"} <= set(candidate.keys())


class TestVf2RandomizedStress:
    """Tests randomized VF2 strict mapping behavior under repeated runs."""

    def test_random_circuit(self):
        """Keeps operation count and edge validity for fitted random circuits."""
        repeats = 1000
        num_qubits = random.randint(5, 10)
        line_topology = Topology.line(list(range(num_qubits)))
        full_topology = Topology(
            list(range(num_qubits)),
            [(i, j, "") for i in range(num_qubits) for j in range(i + 1, num_qubits)],
        )

        for _ in range(repeats):
            circuit = random_circuit(num_qubits)

            if vf2_is_subgraph_isomorphic(circuit, line_topology):
                fitted_topo = line_topology
            else:
                fitted_topo = full_topology

            mapped_circuit = vf2_map(circuit, fitted_topo)
            assert_all_2q_on_topology(mapped_circuit, fitted_topo)
            assert len(circuit.operations) == len(mapped_circuit.operations)
