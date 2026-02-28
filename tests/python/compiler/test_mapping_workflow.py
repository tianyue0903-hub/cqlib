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

from cqlib.circuit import Circuit
from cqlib.compiler import (
    SabreConfig,
    Topology,
    map_with_vf2_sabre,
    vf2_find_initial_layout,
    vf2_is_subgraph_isomorphic,
    vf2_map,
)

from . import assert_all_2q_on_topology
from . import count_gate
from . import random_circuit


class TestCompilerTopologyApi:
    """Tests topology construction, properties, and connectivity helpers."""

    def test_topology_properties_and_connectivity(self):
        """Reports qubit/coupling counts and direct-edge connectivity correctly."""
        topology = Topology([0, 2, 4], [(0, 2), (2, 4, "CZ")])
        assert topology.num_qubits == 3
        assert topology.num_couplings == 2

        assert topology.is_connected(0, 2) or topology.is_connected(2, 0)
        assert topology.is_connected(2, 4) or topology.is_connected(4, 2)
        assert not (topology.is_connected(0, 4) or topology.is_connected(4, 0))

    def test_topology_rejects_overflow_qubit_id(self):
        """Rejects qubit ids that overflow internal `u32` representation."""
        overflow_id = 1 << 40

        with pytest.raises(ValueError):
            Topology([overflow_id], [])

        topology = Topology.line([0, 1])
        with pytest.raises(ValueError):
            topology.is_connected(0, overflow_id)


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


class TestCompilerWorkflowValidation:
    """Tests error handling for fidelity and topology validation paths."""

    def test_fidelity_validation_and_defaults(self):
        """Accepts reverse fidelity key and rejects out-of-range fidelity values."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit([11, 22])
        circuit.cx(11, 22)

        valid_map = {(1, 0): 0.95}
        cfg = SabreConfig(vf2_policy="disabled", seed=10)
        mapped = map_with_vf2_sabre(circuit, topology, fidelity_map=valid_map, config=cfg)
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


class TestCompilerRandomizedStress:
    """Tests randomized hybrid mapping stability over repeated runs."""

    def test_random_circuit(self):
        """Maintains topology validity and non-negative op growth in random stress."""
        rng = random.Random(20260227)
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
            circuit = random_circuit(num_qubits, rng)
            input_ops = len(circuit.operations)

            if vf2_is_subgraph_isomorphic(circuit, topology):
                vf2_mapped = vf2_map(circuit, topology)
                assert len(vf2_mapped.operations) == input_ops
            else:
                vf2_mapped = circuit

            mapped = map_with_vf2_sabre(vf2_mapped, topology, config=sabre_cfg)

            increase = len(mapped.operations) - input_ops
            if increase > 0:
                cases_with_increase += 1

            assert increase >= 0
            assert_all_2q_on_topology(mapped, topology)

        assert cases_with_increase >= 0
