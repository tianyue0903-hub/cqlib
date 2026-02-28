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

import random
import pytest

from cqlib.circuit import Circuit
from cqlib.compiler import (
    Topology,
    vf2_find_initial_layout,
    vf2_find_initial_layout_candidates,
    vf2_is_subgraph_isomorphic,
    vf2_map,
)

try:
    from . import assert_all_2q_on_topology
    from . import random_circuit
except (
    ImportError
):  # Direct script execution: python tests/python/compiler/test_mapping_workflow.py
    from __init__ import assert_all_2q_on_topology
    from __init__ import random_circuit


def _triangle_circuit() -> Circuit:
    circuit = Circuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(0, 2)
    return circuit


def test_vf2_find_initial_layout_fallback_top1():
    topology = Topology.line([0, 1, 2])
    circuit = _triangle_circuit()

    assert vf2_is_subgraph_isomorphic(circuit, topology) is False

    layout = vf2_find_initial_layout(circuit, topology)
    assert layout is not None
    assert len(layout) == 3
    assert len(set(layout)) == 3


def test_vf2_map_remains_strict():
    topology = Topology.line([0, 1, 2])
    circuit = _triangle_circuit()

    with pytest.raises(ValueError):
        vf2_map(circuit, topology)


def test_vf2_find_initial_layout_candidates_topk_and_schema():
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


def test_vf2_candidates_sorted_deterministically():
    topology = Topology.line([0, 1, 2, 3])
    circuit = _triangle_circuit()

    c1 = vf2_find_initial_layout_candidates(circuit, topology, top_k=5)
    c2 = vf2_find_initial_layout_candidates(circuit, topology, top_k=5)

    sig1 = [(c["layout"], c["score"]["total"]) for c in c1]
    sig2 = [(c["layout"], c["score"]["total"]) for c in c2]
    assert sig1 == sig2


def test_vf2_candidates_custom_weights():
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


def test_vf2_candidates_topk_effective_on_strict_case():
    topology = Topology([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 0)])
    circuit = Circuit(2)
    circuit.cx(0, 1)

    candidates = vf2_find_initial_layout_candidates(circuit, topology, top_k=4)
    assert 1 < len(candidates) <= 4


def test_vf2_candidates_respect_max_matches_per_subgraph():
    topology = Topology([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 0)])
    circuit = Circuit(2)
    circuit.cx(0, 1)

    candidates = vf2_find_initial_layout_candidates(
        circuit,
        topology,
        top_k=8,
        max_matches_per_subgraph=1,
    )
    assert len(candidates) <= 1


def test_random_circuit():
    repeats = 1000
    num_qubits = random.randint(5, 10)
    line_topology = Topology.line(list(range(num_qubits)))
    full_topology = Topology(
        list(range(num_qubits)),
        [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)],
    )

    for _ in range(repeats):
        circuit = random_circuit(num_qubits)

        fitted_topo = None
        if vf2_is_subgraph_isomorphic(circuit, line_topology):
            fitted_topo = line_topology
        else:
            fitted_topo = full_topology

        mapped_circuit = vf2_map(circuit, fitted_topo)
        assert_all_2q_on_topology(mapped_circuit, fitted_topo)
        assert len(circuit.operations) == len(mapped_circuit.operations)
