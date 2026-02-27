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
    SabreConfig,
    map_with_vf2_sabre,
    vf2_find_initial_layout,
    vf2_is_subgraph_isomorphic,
    vf2_map,
)


def _count_gate(circuit: Circuit, gate_name: str) -> int:
    return sum(1 for op in circuit.operations if op.name.upper() == gate_name.upper())


def _assert_all_2q_on_topology(circuit: Circuit, topology: Topology) -> None:
    for op in circuit.operations:
        if op.num_qubits != 2:
            continue
        q0, q1 = op.qubits
        c0 = topology.is_connected(q0.index, q1.index)
        c1 = topology.is_connected(q1.index, q0.index)
        assert c0 or c1, f"2q op {op.name} on non-edge ({q0.index}, {q1.index})"


def random_circuit(
    num_qubits: int,
    rng: random.Random,
    min_ops: int = 8,
    max_ops: int = 16,
) -> Circuit:
    circuit = Circuit(num_qubits)
    num_ops = rng.randint(min_ops, max_ops)

    for _ in range(num_ops):
        if rng.random() < 0.4:
            q = rng.randrange(num_qubits)
            rng.choice([circuit.h, circuit.x, circuit.z])(q)
        else:
            q0, q1 = rng.sample(range(num_qubits), 2)
            circuit.cx(q0, q1)

    if not any(op.num_qubits == 2 for op in circuit.operations):
        q0, q1 = rng.sample(range(num_qubits), 2)
        circuit.cx(q0, q1)

    return circuit


def test_vf2_standalone_and_direct_pipeline():
    print("[workflow] case-1: standalone VF2 + direct_then_sabre pipeline")

    topology = Topology.line([0, 1, 2])
    circuit = Circuit([10, 20, 30])
    circuit.cx(10, 20)
    circuit.cx(20, 30)

    ok = vf2_is_subgraph_isomorphic(circuit, topology)
    print(f"[workflow] vf2_is_subgraph_isomorphic: {ok}")
    assert ok is True

    layout = vf2_find_initial_layout(circuit, topology)
    print(f"[workflow] vf2_find_initial_layout: {layout}")
    assert layout is not None and len(layout) == 3

    direct_mapped = vf2_map(circuit, topology)
    print(f"[workflow] vf2_map op_count: {len(direct_mapped.operations)}")
    assert len(direct_mapped.operations) == len(circuit.operations)

    config = SabreConfig(vf2_policy="direct_then_sabre", seed=7)
    mapped = map_with_vf2_sabre(circuit, topology, config=config)
    print(
        f"[workflow] pipeline op_count={len(mapped.operations)}, swap_count={_count_gate(mapped, 'SWAP')}"
    )

    assert len(mapped.operations) == len(circuit.operations)
    assert _count_gate(mapped, "SWAP") == 0
    _assert_all_2q_on_topology(mapped, topology)


def test_sabre_fallback_initial_only_with_prints():
    print("[workflow] case-2: initial_only policy and SABRE fallback")

    topology = Topology.line([0, 1, 2])
    circuit = Circuit(3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(0, 2)

    ok = vf2_is_subgraph_isomorphic(circuit, topology)
    print(f"[workflow] vf2_is_subgraph_isomorphic: {ok}")
    assert ok is False

    config = SabreConfig(
        vf2_policy="initial_only",
        seed=42,
        initial_iterations=3,
        repeat_iterations=1,
        swap_iterations=2,
    )
    mapped = map_with_vf2_sabre(circuit, topology, config=config)
    print(
        "[workflow] mapped stats:",
        {
            "input_ops": len(circuit.operations),
            "output_ops": len(mapped.operations),
            "swap_count": _count_gate(mapped, "SWAP"),
        },
    )

    assert len(mapped.operations) >= len(circuit.operations)
    _assert_all_2q_on_topology(mapped, topology)


def test_fidelity_validation_and_defaults_with_prints():
    print("[workflow] case-3: fidelity map validation/default behavior")

    topology = Topology.line([0, 1, 2])
    circuit = Circuit([11, 22])
    circuit.cx(11, 22)

    valid_map = {(1, 0): 0.95}  # reverse key is accepted
    cfg = SabreConfig(vf2_policy="disabled", seed=10)
    mapped = map_with_vf2_sabre(circuit, topology, fidelity_map=valid_map, config=cfg)
    print(f"[workflow] valid fidelity map route op_count={len(mapped.operations)}")
    _assert_all_2q_on_topology(mapped, topology)

    invalid_map = {(0, 1): 1.2}
    with pytest.raises(ValueError):
        map_with_vf2_sabre(circuit, topology, fidelity_map=invalid_map, config=cfg)
    print("[workflow] invalid fidelity rejected as expected")


def test_random_circuit():
    print("[workflow] case-4: random circuit case")

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
        _assert_all_2q_on_topology(mapped, topology)

    print(
        "[workflow] random summary:",
        {
            "repeats": repeats,
            "cases_with_increase": cases_with_increase,
        },
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", __file__]))
