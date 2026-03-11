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
from cqlib.circuit import Circuit
from cqlib.device import Topology
import time


def count_gate(circuit: Circuit, gate_name: str) -> int:
    return sum(1 for op in circuit.operations if op.name.upper() == gate_name.upper())


def assert_all_2q_on_topology(circuit: Circuit, topology: Topology) -> None:
    for op in circuit.operations:
        if op.num_qubits != 2:
            continue
        q0, q1 = op.qubits
        c0 = topology.is_connected(q0.index, q1.index)
        c1 = topology.is_connected(q1.index, q0.index)
        assert c0 or c1, f"2q op {op.name} on non-edge ({q0.index}, {q1.index})"


def assert_ops_on_topology_recursive(ops, topology: Topology) -> None:
    for op in ops:
        if op.num_qubits == 2:
            q0, q1 = op.qubits
            assert topology.is_connected(q0.index, q1.index) or topology.is_connected(
                q1.index, q0.index
            )
        control_flow = op.instruction.control_flow
        if control_flow is None:
            continue
        if control_flow.is_if_else:
            gate = control_flow.as_if_else
            assert_ops_on_topology_recursive(gate.true_body, topology)
            if gate.false_body is not None:
                assert_ops_on_topology_recursive(gate.false_body, topology)
        if control_flow.is_while_loop:
            gate = control_flow.as_while_loop
            assert_ops_on_topology_recursive(gate.body, topology)


def count_swaps_recursive(ops) -> int:
    total = 0
    for op in ops:
        if op.name.upper() == "SWAP":
            total += 1
        control_flow = op.instruction.control_flow
        if control_flow is None:
            continue
        if control_flow.is_if_else:
            gate = control_flow.as_if_else
            total += count_swaps_recursive(gate.true_body)
            if gate.false_body is not None:
                total += count_swaps_recursive(gate.false_body)
        if control_flow.is_while_loop:
            gate = control_flow.as_while_loop
            total += count_swaps_recursive(gate.body)
    return total


def directive_names_recursive(ops) -> list[str]:
    names = []
    for op in ops:
        if op.instruction.is_directive:
            names.append(op.name)
        control_flow = op.instruction.control_flow
        if control_flow is None:
            continue
        if control_flow.is_if_else:
            gate = control_flow.as_if_else
            names.extend(directive_names_recursive(gate.true_body))
            if gate.false_body is not None:
                names.extend(directive_names_recursive(gate.false_body))
        if control_flow.is_while_loop:
            gate = control_flow.as_while_loop
            names.extend(directive_names_recursive(gate.body))
    return names


def random_circuit(
    num_qubits: int,
    rng: random.Random = None,
    min_ops: int = 8,
    max_ops: int = 16,
) -> Circuit:
    if not rng:
        rng = random.Random(time.time())

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
        for _ in range(2 * num_qubits):
            q0, q1 = rng.sample(range(num_qubits), 2)
            circuit.cx(q0, q1)

    return circuit


def show_circuit(circuit: Circuit):
    edges = []
    for op in circuit.operations:
        if op.num_qubits != 2:
            continue
        q0, q1 = op.qubits
        edges.append((q0, q1))
    return edges
