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
from cqlib.compiler import Topology
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
        for _ in range(2*num_qubits):
            q0, q1 = rng.sample(range(num_qubits), 2)
            circuit.cx(q0, q1)

    return circuit

def show_circuit(circuit: Circuit):
    edges = []
    for op in circuit.operations:
        if op.num_qubits != 2:
            continue
        q0, q1 = op.qubits
        edges.append((q0,q1))
    return edges