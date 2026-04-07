# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Compiler workflow mapping tests.

Test coverage:
- topology API properties and connectivity checks
- SabreConfig policy parsing and validation behavior
- VF2 standalone and hybrid VF2+SABRE workflow paths
- fidelity-map validation (value, qubit existence, edge existence)
- topology-size failure modes and randomized routing stress behavior
"""

import random

import pytest

from cqlib.circuit import (
    Circuit,
    ConditionView,
    ControlFlow,
    Directive,
    Parameter,
    Qubit,
    StandardGate,
)
from cqlib.compiler import (
    SabreConfig,
    map_with_vf2_sabre,
    vf2_find_initial_layout,
    vf2_is_subgraph_isomorphic,
    vf2_map,
)

from cqlib.device import Topology
from . import assert_all_2q_on_topology
from . import assert_ops_on_topology_recursive
from . import count_gate
from . import count_swaps_recursive
from . import directive_names_recursive
from . import random_circuit


def _ops_signature(circuit: Circuit) -> tuple:
    """Builds a deterministic operation signature for mapped-circuit comparisons."""
    return tuple(
        (op.name, tuple(q.index for q in op.qubits)) for op in circuit.operations
    )


def _swap_edges(circuit: Circuit) -> list[tuple[int, int]]:
    """Extracts normalized SWAP edges from a circuit."""
    edges = []
    for op in circuit.operations:
        if op.name.upper() != "SWAP":
            continue
        q0, q1 = (q.index for q in op.qubits)
        edges.append(tuple(sorted((q0, q1))))
    return edges


class TestCompilerTopologyApi:
    """Tests topology construction, properties, and connectivity helpers."""

    def test_topology_properties_and_connectivity(self):
        """Reports qubit/coupling counts and direct-edge connectivity correctly."""
        topology = Topology([0, 2, 4], [(0, 2, "G1"), (2, 4, "G2")])
        assert topology.num_qubits == 3
        assert topology.num_couplings == 2

        assert topology.is_connected(0, 2) or topology.is_connected(2, 0)
        assert topology.is_connected(2, 4) or topology.is_connected(4, 2)
        assert not (topology.is_connected(0, 4) or topology.is_connected(4, 0))


class TestSabreConfigApi:
    """Tests SabreConfig construction, aliases, and validation behavior."""

    def test_sabreconfig_policy_aliases_are_accepted(self):
        """Accepts documented alias policy names and maps circuits successfully."""
        aliases = ["direct", "vf2_then_sabre", "vf2_initial_only", "off", "none"]
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        for alias in aliases:
            config = SabreConfig(vf2_policy=alias, seed=7)
            mapped = map_with_vf2_sabre(circuit, topology, config=config)
            assert_all_2q_on_topology(mapped, topology)

    def test_sabreconfig_rejects_invalid_policy(self):
        """Raises ValueError for unknown policy strings."""
        with pytest.raises(ValueError):
            SabreConfig(vf2_policy="unexpected_policy")


class TestCompilerWorkflowScenarios:
    """Tests normal-path compiler workflows using VF2 and SABRE combinations."""

    def test_vf2_standalone_and_direct_pipeline(self):
        """Uses strict VF2 path and confirms zero-SWAP direct pipeline behavior."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit([10, 20, 30])
        circuit.cx(10, 20)
        circuit.cx(20, 30)

        ok = vf2_is_subgraph_isomorphic(circuit, topology)
        assert ok is True

        layout = vf2_find_initial_layout(circuit, topology)
        assert layout is not None and len(layout) == 3

        direct_mapped = vf2_map(circuit, topology)
        assert len(direct_mapped.operations) == len(circuit.operations)

        config = SabreConfig(vf2_policy="direct_then_sabre", seed=7)
        mapped = map_with_vf2_sabre(circuit, topology, config=config)

        assert len(mapped.operations) == len(circuit.operations)
        assert count_gate(mapped, "SWAP") == 0
        assert_all_2q_on_topology(mapped, topology)

    def test_sabre_fallback_initial_only(self):
        """Routes with SABRE fallback under `initial_only` when strict VF2 fails."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(0, 2)

        ok = vf2_is_subgraph_isomorphic(circuit, topology)
        assert ok is False

        config = SabreConfig(
            vf2_policy="initial_only",
            seed=42,
            initial_iterations=3,
            repeat_iterations=1,
            swap_iterations=2,
        )
        mapped = map_with_vf2_sabre(circuit, topology, config=config)

        assert len(mapped.operations) >= len(circuit.operations)
        assert_all_2q_on_topology(mapped, topology)

    def test_map_with_vf2_sabre_default_config(self):
        """Uses default config when `config=None` and still returns valid mapping."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        mapped = map_with_vf2_sabre(circuit, topology, config=None)

        assert len(mapped.operations) == len(circuit.operations)
        assert_all_2q_on_topology(mapped, topology)

    def test_hybrid_mapping_supports_if_else(self):
        """Routes `if_else` bodies and preserves the control-flow structure."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.measure(0)
        circuit.if_else(
            ConditionView(Qubit(0), 1),
            [(StandardGate.CX, [0, 1])],
            [(StandardGate.CX, [1, 2])],
        )
        circuit.cx(0, 2)

        config = SabreConfig(vf2_policy="initial_only", repeat_iterations=0, seed=9)
        mapped = map_with_vf2_sabre(circuit, topology, config=config)

        mapped_ops = list(mapped.operations)
        assert_ops_on_topology_recursive(mapped_ops, topology)
        control_flow = mapped_ops[1].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_if_else
        assert count_swaps_recursive(mapped_ops) > 0

    def test_hybrid_mapping_supports_while_loop(self):
        """Routes `while_loop` bodies and preserves the control-flow structure."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.measure(0)
        circuit.while_loop(
            ConditionView(Qubit(0), 1),
            [
                (StandardGate.CX, [0, 1]),
                (StandardGate.CX, [1, 2]),
                (StandardGate.CX, [0, 2]),
            ],
        )
        circuit.cx(0, 2)

        config = SabreConfig(vf2_policy="initial_only", repeat_iterations=0, seed=9)
        mapped = map_with_vf2_sabre(circuit, topology, config=config)

        mapped_ops = list(mapped.operations)
        assert_ops_on_topology_recursive(mapped_ops, topology)
        control_flow = mapped_ops[1].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_while_loop
        assert len(control_flow.as_while_loop.body) > 3

    def test_hybrid_mapping_preserves_symbolic_global_phase_in_control_flow(self):
        """Preserves source global phase on routed control-flow circuits."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        theta = Parameter("theta")
        circuit.set_global_phase(theta)
        circuit.measure(0)
        circuit.while_loop(
            ConditionView(Qubit(0), 1),
            [
                (Directive.measure(), [1]),
                (Directive.reset(), [2]),
                (StandardGate.CX, [0, 2]),
            ],
        )
        circuit.cx(0, 2)

        config = SabreConfig(vf2_policy="initial_only", repeat_iterations=0, seed=9)
        mapped = map_with_vf2_sabre(circuit, topology, config=config)

        assert mapped.global_phase == theta
        assert_ops_on_topology_recursive(list(mapped.operations), topology)

    def test_hybrid_mapping_preserves_directives_inside_control_flow_body(self):
        """Keeps directive operations inside mapped control-flow bodies."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.measure(0)
        circuit.while_loop(
            ConditionView(Qubit(0), 1),
            [
                (Directive.measure(), [1]),
                (Directive.reset(), [2]),
                (StandardGate.CX, [0, 2]),
            ],
        )
        circuit.cx(0, 2)

        mapped = map_with_vf2_sabre(
            circuit,
            topology,
            config=SabreConfig(vf2_policy="initial_only", repeat_iterations=0, seed=9),
        )

        control_flow = list(mapped.operations)[1].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_while_loop
        assert directive_names_recursive(control_flow.as_while_loop.body) == [
            "Measure",
            "Reset",
        ]

    def test_hybrid_mapping_supports_nested_control_flow_objects(self):
        """Preserves nested control-flow bodies exposed by the Python binding."""
        topology = Topology.line([0, 1, 2])

        inner_loop_body = Circuit(3)
        inner_loop_body.cx(0, 2)
        nested_while = ControlFlow.while_loop(
            ConditionView(Qubit(1), 1),
            list(inner_loop_body.operations),
        )

        circuit = Circuit(3)
        circuit.measure(0)
        circuit.measure(1)
        circuit.if_else(
            ConditionView(Qubit(0), 1),
            [(nested_while, [0, 1, 2])],
            [(StandardGate.CX, [1, 2])],
        )
        circuit.cx(0, 2)

        mapped = map_with_vf2_sabre(
            circuit,
            topology,
            config=SabreConfig(vf2_policy="initial_only", repeat_iterations=0, seed=11),
        )

        mapped_ops = list(mapped.operations)
        assert_ops_on_topology_recursive(mapped_ops, topology)
        control_flow = mapped_ops[2].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_if_else
        nested = control_flow.as_if_else.true_body[0].instruction.control_flow
        assert nested is not None
        assert nested.is_while_loop


class TestCompilerWorkflowValidation:
    """Tests error handling for fidelity and topology validation paths."""

    def test_fidelity_validation_and_defaults(self):
        """Accepts reverse fidelity key and rejects out-of-range fidelity values."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit([11, 22])
        circuit.cx(11, 22)

        valid_map = {(1, 0): 0.95}
        cfg = SabreConfig(vf2_policy="disabled", seed=10)
        mapped = map_with_vf2_sabre(
            circuit, topology, fidelity_map=valid_map, config=cfg
        )
        assert_all_2q_on_topology(mapped, topology)

        invalid_map = {(0, 1): 1.2}
        with pytest.raises(ValueError):
            map_with_vf2_sabre(circuit, topology, fidelity_map=invalid_map, config=cfg)

    def test_fidelity_map_rejects_unknown_qubit(self):
        """Raises error when fidelity map references qubit absent from topology."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(2)
        circuit.cx(0, 1)
        cfg = SabreConfig(vf2_policy="disabled", seed=11)

        with pytest.raises(ValueError):
            map_with_vf2_sabre(
                circuit,
                topology,
                fidelity_map={(0, 99): 0.9},
                config=cfg,
            )

    def test_fidelity_map_rejects_non_existent_topology_edge(self):
        """Raises error when fidelity map references non-adjacent topology edge."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(2)
        circuit.cx(0, 1)
        cfg = SabreConfig(vf2_policy="disabled", seed=12)

        with pytest.raises(ValueError):
            map_with_vf2_sabre(
                circuit,
                topology,
                fidelity_map={(0, 2): 0.9},
                config=cfg,
            )

    def test_hybrid_mapping_rejects_topology_too_small(self):
        """Raises error when topology cannot host logical circuit width."""
        topology = Topology.line([0, 1])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        cfg = SabreConfig(vf2_policy="disabled", seed=13)

        with pytest.raises(ValueError):
            map_with_vf2_sabre(circuit, topology, config=cfg)

    def test_mapping_preserves_measure_barrier_and_reset(self):
        """Preserves supported directives instead of rejecting them."""
        topology = Topology.line([0, 1])
        circuit = Circuit(2)
        circuit.h(0)
        circuit.barrier([0, 1])
        circuit.measure(0)
        circuit.reset(1)

        mapped = map_with_vf2_sabre(circuit, topology, config=SabreConfig(seed=5))
        directive_names = [op.name for op in mapped.operations if op.instruction.is_directive]
        assert directive_names == ["Barrier", "Measure", "Reset"]

class TestSabreFidelityEnhancements:
    """Tests the SABRE fidelity-aware additions and new configuration controls."""

    def test_sabreconfig_exposes_new_fidelity_controls(self):
        """Accepts and reports new VF2-seed and fidelity-objective configuration fields."""
        config = SabreConfig(
            vf2_seed_top_k=5,
            vf2_seed_weight_fidelity=0.7,
            vf2_seed_weight_topology=0.2,
            vf2_seed_weight_gate_distribution=0.1,
            swap_fidelity_weight=0.4,
            gate_cost_weight=2.0,
            predicted_fidelity_weight=0.8,
        )
        text = repr(config)
        assert "vf2_seed_top_k=5" in text
        assert "vf2_seed_weight_fidelity=0.7" in text
        assert "vf2_seed_weight_topology=0.2" in text
        assert "vf2_seed_weight_gate_distribution=0.1" in text
        assert "swap_fidelity_weight=0.4" in text
        assert "gate_cost_weight=2" in text
        assert "predicted_fidelity_weight=0.8" in text

    def test_vf2_seed_weight_changes_initial_only_mapping_result(self):
        """Changing VF2 seed weights changes initial-only SABRE mapping behavior."""
        topology = Topology.line([0, 1, 2, 3, 4, 5])
        circuit = Circuit([10, 20, 30, 40])
        circuit.cx(20, 10)
        circuit.cx(20, 30)
        circuit.cx(10, 20)
        circuit.cx(30, 20)
        circuit.cx(20, 40)
        circuit.cx(10, 20)
        circuit.h(10)
        circuit.h(20)
        circuit.cx(20, 10)

        fidelity_map = {
            (0, 1): 0.68,
            (1, 2): 0.74,
            (2, 3): 0.62,
            (3, 4): 0.55,
            (4, 5): 0.58,
        }

        common_kwargs = dict(
            vf2_policy="initial_only",
            seed=321,
            initial_iterations=1,
            repeat_iterations=0,
            swap_iterations=2,
            vf2_seed_top_k=8,
            swap_fidelity_weight=0.0,
            gate_cost_weight=1.0,
            predicted_fidelity_weight=0.0,
        )
        fidelity_seed_cfg = SabreConfig(
            vf2_seed_weight_fidelity=0.9,
            vf2_seed_weight_topology=0.05,
            vf2_seed_weight_gate_distribution=0.05,
            **common_kwargs,
        )
        topology_seed_cfg = SabreConfig(
            vf2_seed_weight_fidelity=0.05,
            vf2_seed_weight_topology=0.9,
            vf2_seed_weight_gate_distribution=0.05,
            **common_kwargs,
        )

        mapped_fidelity_seed = map_with_vf2_sabre(
            circuit,
            topology,
            fidelity_map=fidelity_map,
            config=fidelity_seed_cfg,
        )
        mapped_topology_seed = map_with_vf2_sabre(
            circuit,
            topology,
            fidelity_map=fidelity_map,
            config=topology_seed_cfg,
        )

        assert_all_2q_on_topology(mapped_fidelity_seed, topology)
        assert_all_2q_on_topology(mapped_topology_seed, topology)
        assert _ops_signature(mapped_fidelity_seed) != _ops_signature(
            mapped_topology_seed
        )

    def test_local_swap_prefers_high_fidelity_edge(self):
        """On equal-distance choices, SABRE chooses SWAPs on higher-fidelity edges."""
        topology = Topology([0, 1, 2], [(0, 1, "G0"), (0, 2, "G1")])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(0, 2)

        fidelity_map = {(0, 1): 0.2, (0, 2): 0.9}
        config = SabreConfig(
            vf2_policy="disabled",
            seed=3,
            initial_iterations=1,
            repeat_iterations=0,
            swap_iterations=16,
            swap_fidelity_weight=1.0,
            gate_cost_weight=1.0,
            predicted_fidelity_weight=0.0,
        )

        mapped = map_with_vf2_sabre(
            circuit,
            topology,
            fidelity_map=fidelity_map,
            config=config,
        )

        swap_edges = _swap_edges(mapped)
        assert swap_edges, "expected at least one SWAP in this routing case"
        assert all(edge == (0, 2) for edge in swap_edges)
        assert_all_2q_on_topology(mapped, topology)

    def test_weight_ratio_changes_global_routing_choice(self):
        """Changing cost-vs-fidelity objective weights changes selected routed circuit."""
        topology = Topology.line([0, 1, 2, 3])
        circuit = Circuit(4)
        circuit.cx(3, 1)
        circuit.cx(0, 2)
        circuit.cx(1, 2)
        circuit.cx(1, 0)
        circuit.cx(2, 3)
        circuit.cx(1, 3)

        fidelity_map = {(0, 1): 0.8, (1, 2): 0.96, (2, 3): 0.28}

        cost_priority_cfg = SabreConfig(
            vf2_policy="disabled",
            seed=77,
            initial_iterations=3,
            repeat_iterations=2,
            swap_iterations=4,
            swap_fidelity_weight=0.0,
            gate_cost_weight=1.0,
            predicted_fidelity_weight=0.0,
        )
        fidelity_priority_cfg = SabreConfig(
            vf2_policy="disabled",
            seed=77,
            initial_iterations=3,
            repeat_iterations=2,
            swap_iterations=4,
            swap_fidelity_weight=0.0,
            gate_cost_weight=0.1,
            predicted_fidelity_weight=5.0,
        )

        mapped_cost = map_with_vf2_sabre(
            circuit,
            topology,
            fidelity_map=fidelity_map,
            config=cost_priority_cfg,
        )
        mapped_fidelity = map_with_vf2_sabre(
            circuit,
            topology,
            fidelity_map=fidelity_map,
            config=fidelity_priority_cfg,
        )

        assert_all_2q_on_topology(mapped_cost, topology)
        assert_all_2q_on_topology(mapped_fidelity, topology)
        assert _ops_signature(mapped_cost) != _ops_signature(mapped_fidelity)


class TestCompilerRandomizedStress:
    """Tests randomized hybrid mapping stability over repeated runs."""

    def test_random_circuit(self):
        """Maintains topology validity and non-negative op growth in random stress."""
        repeats = 1000
        num_qubits = 5
        topology = Topology.line(list(range(num_qubits)))
        cases_with_increase = 0

        sabre_cfg = SabreConfig(
            vf2_policy="disabled",
            initial_iterations=2,
            repeat_iterations=3,
            swap_iterations=2,
        )
        for _ in range(repeats):
            circuit = random_circuit(num_qubits)
            input_ops = len(circuit.operations)

            if vf2_is_subgraph_isomorphic(circuit, topology):
                vf2_mapped = vf2_map(circuit, topology)
                assert len(vf2_mapped.operations) == input_ops
                continue
            else:
                vf2_mapped = circuit

            mapped = map_with_vf2_sabre(vf2_mapped, topology, config=sabre_cfg)

            increase = len(mapped.operations) - input_ops
            if increase > 0:
                cases_with_increase += 1

            assert increase >= 0
            assert_all_2q_on_topology(mapped, topology)

        assert cases_with_increase >= 0
