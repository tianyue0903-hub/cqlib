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
Genetic Algorithm (GA) mapping tests.

Test coverage:
- GaConfig construction, defaults, and parameter validation
- Basic GA mapping workflow
- Invalid qubits exclusion and topology fragmentation handling
- Fidelity map integration
- Determinism with fixed seeds
- Comparison with SABRE mapping results
- Various topology and circuit combinations
"""

import pytest

from cqlib.circuit import Circuit, ConditionView, Qubit, StandardGate
from cqlib.compiler import GaConfig, SabreConfig, map_with_ga
from cqlib.device import Topology

from . import (
    assert_all_2q_on_topology,
    assert_ops_on_topology_recursive,
    count_swaps_recursive,
)


def _ops_signature(circuit: Circuit) -> tuple:
    """Builds a deterministic operation signature for mapped-circuit comparisons."""
    return tuple(
        (op.name, tuple(q.index for q in op.qubits)) for op in circuit.operations
    )


def _control_flow_ga_config(seed: int) -> GaConfig:
    """Builds a deterministic GA config for structured-control-flow routing tests."""
    return GaConfig(
        population=6,
        update_iters=2,
        sabre_config=SabreConfig(vf2_policy="initial_only", repeat_iterations=0, seed=seed),
        seed=seed,
    )


class TestGaConfigApi:
    """Tests GaConfig construction, defaults, and parameter validation."""

    def test_gaconfig_default_values(self):
        """GaConfig uses sensible defaults when no parameters are provided."""
        config = GaConfig()
        assert config.population == 10
        assert config.select_prob == 0.4
        assert config.crossover_prob == 0.4
        assert config.mutation_prob == 0.25
        assert config.forced_mutation_prob == 0.05
        assert config.crossover_qubit_number == 3
        assert config.update_iters == 5
        assert config.seed == -1

    def test_gaconfig_custom_values(self):
        """GaConfig accepts and stores custom parameter values."""
        config = GaConfig(
            population=20,
            select_prob=0.5,
            crossover_prob=0.6,
            mutation_prob=0.3,
            forced_mutation_prob=0.1,
            crossover_qubit_number=5,
            update_iters=10,
            seed=42,
        )
        assert config.population == 20
        assert config.select_prob == 0.5
        assert config.crossover_prob == 0.6
        assert config.mutation_prob == 0.3
        assert config.forced_mutation_prob == 0.1
        assert config.crossover_qubit_number == 5
        assert config.update_iters == 10
        assert config.seed == 42

    def test_gaconfig_repr(self):
        """GaConfig __repr__ contains key configuration fields."""
        config = GaConfig(population=15, seed=123)
        text = repr(config)
        assert "GaConfig" in text
        assert "population=15" in text
        assert "seed=123" in text


class TestGaMappingBasicWorkflow:
    """Tests basic GA mapping workflow scenarios."""

    def test_map_with_ga_default_config(self):
        """GA mapping works with default configuration."""
        topology = Topology.line([0, 1, 2, 3])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        mapped = map_with_ga(circuit, topology)

        assert len(mapped.operations) >= len(circuit.operations)
        assert_all_2q_on_topology(mapped, topology)

    def test_map_with_ga_custom_config(self):
        """GA mapping respects custom configuration parameters."""
        topology = Topology.line([0, 1, 2, 3, 4])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(0, 2)

        config = GaConfig(
            population=8,
            update_iters=3,
            crossover_prob=0.5,
            mutation_prob=0.3,
            seed=42,
        )
        mapped = map_with_ga(circuit, topology, config=config)

        assert len(mapped.operations) >= len(circuit.operations)
        assert_all_2q_on_topology(mapped, topology)

    def test_map_with_ga_non_contiguous_qubit_ids(self):
        """GA mapping handles non-contiguous qubit IDs in both circuit and topology."""
        topology = Topology([100, 200, 300, 400], [(100, 200, "CX"), (200, 300, "CX"), (300, 400, "CX")])
        circuit = Circuit([10, 20, 30])
        circuit.cx(10, 20)
        circuit.cx(20, 30)

        config = GaConfig(seed=42)
        mapped = map_with_ga(circuit, topology, config=config)
        assert_all_2q_on_topology(mapped, topology)


class TestGaMappingInvalidQubits:
    """Tests GA mapping behavior with invalid/broken qubits."""

    def test_map_with_ga_avoids_invalid_qubits(self):
        """GA mapping avoids qubits marked as invalid."""
        topology = Topology.line([0, 1, 2, 3, 4, 5])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        # Mark qubit 2 as invalid (breaks the line into two segments)
        invalid_qubits = {2}
        config = GaConfig(seed=42)

        mapped = map_with_ga(circuit, topology, config=config, invalid_qubits=invalid_qubits)

        assert_all_2q_on_topology(mapped, topology)
        # Verify mapped qubits are not in the invalid set
        for op in mapped.operations:
            for q in op.qubits:
                assert q.index not in invalid_qubits, f"Used invalid qubit {q.index}"

class TestGaMappingFidelityMap:
    """Tests GA mapping with fidelity map integration."""

    def test_map_with_ga_with_fidelity_map(self):
        """GA mapping accepts and uses fidelity map for optimization."""
        topology = Topology.line([0, 1, 2, 3, 4])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        # Define fidelity map with varying edge fidelities
        fidelity_map = {
            (0, 1): 0.5,
            (1, 2): 0.99,
            (2, 3): 0.99,
            (3, 4): 0.5,
        }

        config = GaConfig(
            population=20,
            update_iters=10,
            crossover_prob=0.3,
            mutation_prob=0.3,
            seed=42,
        )
            
        mapped = map_with_ga(circuit, topology, config=config, fidelity_map=fidelity_map)
        # mapped qubits should be [1,2,3]
        assert 1 in [q.index for q in mapped.qubits]
        assert 2 in [q.index for q in mapped.qubits]
        assert 3 in [q.index for q in mapped.qubits]
        assert_all_2q_on_topology(mapped, topology)


class TestGaMappingDeterminism:
    """Tests GA mapping determinism with fixed seeds."""

    def test_map_with_ga_deterministic_with_fixed_seed(self):
        """GA mapping produces identical results with the same seed."""
        topology = Topology.line([0, 1, 2, 3, 4])
        circuit = Circuit(4)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.cx(0, 3)

        config = GaConfig(seed=999, population=6, update_iters=3)

        mapped1 = map_with_ga(circuit, topology, config=config)
        mapped2 = map_with_ga(circuit, topology, config=config)

        assert _ops_signature(mapped1) == _ops_signature(mapped2)

class TestGaMappingTopologyVariations:
    """Tests GA mapping on various topology structures."""

    def test_map_with_ga_star_topology(self):
        """GA mapping on star topology produces valid results."""
        # Star topology: center qubit 0 connected to all others
        topology = Topology(
            [0, 1, 2, 3, 4],
            [(0, 1, "CX"), (0, 2, "CX"), (0, 3, "CX"), (0, 4, "CX")]
        )

        circuit = Circuit(5)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.cx(0, 4)

        config = GaConfig(seed=42)
        mapped = map_with_ga(circuit, topology, config=config)

        assert_all_2q_on_topology(mapped, topology)

    def test_map_with_ga_grid_topology(self):
        """GA mapping on grid topology produces valid results."""
        # 2x3 grid topology
        # 0 - 1 - 2
        # |   |   |
        # 3 - 4 - 5
        topology = Topology(
            [0, 1, 2, 3, 4, 5],
            [
                (0, 1, "CX"), (1, 2, "CX"),
                (0, 3, "CX"), (1, 4, "CX"), (2, 5, "CX"),
                (3, 4, "CX"), (4, 5, "CX"),
            ]
        )

        circuit = Circuit(6)
        circuit.cx(0, 2)
        circuit.cx(3, 5)
        circuit.cx(0, 5)

        config = GaConfig(seed=42, population=8)
        mapped = map_with_ga(circuit, topology, config=config)
        assert_all_2q_on_topology(mapped, topology)

    def test_map_with_ga_ring_topology(self):
        """GA mapping on ring topology produces valid results."""
        topology = Topology(
            [0, 1, 2, 3],
            [(0, 1, "CX"), (1, 2, "CX"), (2, 3, "CX"), (3, 0, "CX")]
        )

        circuit = Circuit(4)
        circuit.cx(0, 2)
        circuit.cx(1, 3)

        config = GaConfig(seed=42)
        mapped = map_with_ga(circuit, topology, config=config)

        assert_all_2q_on_topology(mapped, topology)


class TestGaMappingControlFlow:
    """Tests GA mapping on control-flow circuits."""

    def test_map_with_ga_routes_if_else_and_continuation(self):
        """GA routes `if_else` branches and preserves the control-flow node."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(3)
        circuit.measure(0)
        circuit.if_else(
            ConditionView(Qubit(0), 1),
            [(StandardGate.CX, [0, 1])],
            [(StandardGate.CX, [1, 2])],
        )
        circuit.cx(0, 2)

        mapped = map_with_ga(circuit, topology, config=_control_flow_ga_config(seed=11))

        mapped_ops = list(mapped.operations)
        assert_ops_on_topology_recursive(mapped_ops, topology)

        control_flow_op = next(
            op for op in mapped_ops if op.instruction.control_flow is not None
        )
        control_flow = control_flow_op.instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_if_else
        assert count_swaps_recursive(mapped_ops) > 0

    def test_map_with_ga_routes_while_loop_and_continuation(self):
        """GA routes `while_loop` bodies and preserves the loop structure."""
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

        mapped = map_with_ga(circuit, topology, config=_control_flow_ga_config(seed=12))

        mapped_ops = list(mapped.operations)
        assert_ops_on_topology_recursive(mapped_ops, topology)

        control_flow_op = next(
            op for op in mapped_ops if op.instruction.control_flow is not None
        )
        control_flow = control_flow_op.instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_while_loop
        assert len(control_flow.as_while_loop.body) > 3


class TestGaMappingErrorHandling:
    """Tests GA mapping error handling scenarios."""

    def test_map_with_ga_topology_too_small(self):
        """GA mapping raises error when topology cannot accommodate circuit."""
        topology = Topology.line([0, 1])  # Only 2 qubits
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        config = GaConfig(seed=42)

        with pytest.raises(ValueError, match="Topology has insufficient qubits"):
            map_with_ga(circuit, topology, config=config)

    def test_map_with_ga_disconnected_topology_error(self):
        """GA mapping handles disconnected topology appropriately."""
        # Disconnected topology: two separate components
        topology = Topology([0, 1, 2, 3], [(0, 1, "CX"), (2, 3, "CX")])
        circuit = Circuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        config = GaConfig(seed=42)

        # Should either fail or route using only connected component
        with pytest.raises(ValueError):
            map_with_ga(circuit, topology, config=config)


class TestGaMappingComplexCircuits:
    """Tests GA mapping on complex circuit structures."""

    def test_map_with_ga_all_to_all_circuit(self):
        """GA mapping handles circuits with all-to-all connectivity pattern."""
        topology = Topology.line([0, 1, 2, 3, 4])

        circuit = Circuit(5)
        # All-to-all pattern
        for i in range(5):
            for j in range(i + 1, 5):
                circuit.cx(i, j)

        config = GaConfig(population=10, update_iters=5, seed=2024)
        mapped = map_with_ga(circuit, topology, config=config)

        assert_all_2q_on_topology(mapped, topology)
        # Should have added SWAPs for routing
        assert len(mapped.operations) >= len(circuit.operations)

    def test_map_with_ga_heavy_circuit(self):
        """GA mapping handles circuits with many gates."""
        topology = Topology.line([0, 1, 2, 3, 4, 5])

        circuit = Circuit(4)
        for _ in range(10):
            circuit.cx(0, 1)
            circuit.cx(1, 2)
            circuit.cx(2, 3)
            circuit.cx(0, 2)
            circuit.cx(1, 3)

        config = GaConfig(population=8, update_iters=3, seed=42)
        mapped = map_with_ga(circuit, topology, config=config)

        assert_all_2q_on_topology(mapped, topology)

    def test_map_with_ga_single_qubit_gates_preserved(self):
        """GA mapping preserves single-qubit gates."""
        topology = Topology.line([0, 1, 2])
        circuit = Circuit(2)
        circuit.h(0)
        circuit.x(1)
        circuit.cx(0, 1)
        circuit.z(0)

        config = GaConfig(seed=42)
        mapped = map_with_ga(circuit, topology, config=config)

        assert_all_2q_on_topology(mapped, topology)
        single_qubit_count = sum(1 for op in mapped.operations if op.num_qubits == 1)
        assert single_qubit_count == 3  
