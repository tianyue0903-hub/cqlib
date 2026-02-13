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
Tests for Circuit to Gate conversion.

Test coverage:
- Circuit to gate conversion
- Using circuit gates in other circuits
- Parametric circuit to gate
- Circuit gate decomposition
"""

import pytest
import numpy as np
from cqlib.circuit import Circuit, Parameter
from cqlib.circuit.gates import CircuitGate


class TestCircuitToGate:
    """Test circuit to gate conversion"""

    def test_to_gate_single_qubit(self):
        """Single qubit circuit to gate"""
        c = Circuit(1)
        c.h(0)
        c.x(0)

        gate = c.to_gate("HX")
        assert gate is not None
        assert isinstance(gate, CircuitGate)
        assert gate.num_qubits == 1

    def test_to_gate_multi_qubit(self):
        """Multi-qubit circuit to gate"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)

        gate = c.to_gate("BellPrep")
        assert gate is not None
        assert isinstance(gate, CircuitGate)
        assert gate.num_qubits == 2

    def test_to_gate_with_parameters(self):
        """Parametric circuit to gate"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta)

        gate = c.to_gate("RxGate")
        assert gate is not None
        assert isinstance(gate, CircuitGate)
        assert gate.num_params == 1

    def test_to_gate_empty_circuit(self):
        """Empty circuit to gate"""
        c = Circuit(2)
        gate = c.to_gate("Empty")
        assert gate is not None
        assert gate.num_qubits == 2

    def test_to_gate_three_qubit(self):
        """Three qubit circuit to gate"""
        c = Circuit(3)
        c.h(0)
        c.cx(0, 1)
        c.cx(1, 2)

        gate = c.to_gate("GHZ")
        assert gate is not None
        assert gate.num_qubits == 3


class TestUseCircuitGate:
    """Test using circuit gate in other circuits"""

    def test_use_circuit_gate(self):
        """Use circuit gate in main circuit"""
        sub_circuit = Circuit(2)
        sub_circuit.h(0)
        sub_circuit.cx(0, 1)

        bell_gate = sub_circuit.to_gate("Bell")

        main_circuit = Circuit(4)
        main_circuit.circuit_gate(bell_gate, [0, 1])
        main_circuit.circuit_gate(bell_gate, [2, 3])

        assert len(main_circuit) == 2

    def test_use_circuit_gate_different_qubits(self):
        """Use circuit gate on different qubit subsets"""
        sub_circuit = Circuit(2)
        sub_circuit.h(0)
        sub_circuit.cx(0, 1)

        bell_gate = sub_circuit.to_gate("Bell")

        main_circuit = Circuit(4)
        main_circuit.circuit_gate(bell_gate, [1, 2])

        assert len(main_circuit) == 1
        # Verify the operation uses correct qubits
        op = main_circuit[0]
        assert op.qubits[0].index == 1
        assert op.qubits[1].index == 2

    def test_nested_circuit_gates(self):
        """Nested circuit gates"""
        inner = Circuit(1)
        inner.h(0)
        inner_gate = inner.to_gate("H")

        middle = Circuit(1)
        middle.circuit_gate(inner_gate, [0])
        middle_gate = middle.to_gate("Middle")

        outer = Circuit(1)
        outer.circuit_gate(middle_gate, [0])

        assert len(outer) == 1


class TestCircuitGateDecompose:
    """Test circuit gate decomposition"""

    def test_decompose_circuit_gate(self):
        """Decompose circuit gate into primitives"""
        sub_circuit = Circuit(2)
        sub_circuit.h(0)
        sub_circuit.cx(0, 1)

        bell_gate = sub_circuit.to_gate("Bell")

        main_circuit = Circuit(4)
        main_circuit.circuit_gate(bell_gate, [0, 1])

        decomposed = main_circuit.decompose()
        # Decomposed circuit should have at least 2 operations (H + CX)
        assert len(decomposed) >= 2

    def test_decompose_preserves_order(self):
        """Decomposition preserves operation order"""
        sub_circuit = Circuit(2)
        sub_circuit.x(0)
        sub_circuit.h(0)
        sub_circuit.cx(0, 1)

        gate = sub_circuit.to_gate("Ordered")

        main_circuit = Circuit(2)
        main_circuit.circuit_gate(gate, [0, 1])

        decomposed = main_circuit.decompose()
        # Verify operations are in correct order
        op_names = [op.name for op in decomposed]
        assert op_names[0] == "X"
        assert op_names[1] == "H"
        assert op_names[2] == "CX"


class TestCircuitGateInverse:
    """Test circuit gate inverse"""

    def test_circuit_gate_inverse(self):
        """Inverse of circuit gate"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)

        gate = c.to_gate("Bell")
        inv_gate = gate.inverse()

        assert inv_gate is not None
        assert inv_gate.num_qubits == 2

    def test_circuit_gate_self_inverse(self):
        """Circuit gate composed of self-inverse gates"""
        c = Circuit(1)
        c.h(0)
        c.h(0)  # H @ H = I

        gate = c.to_gate("HH")
        inv_gate = gate.inverse()
        assert inv_gate is not None


class TestCircuitGateProperties:
    """Test CircuitGate properties"""

    def test_circuit_gate_num_qubits(self):
        """CircuitGate num_qubits property"""
        c = Circuit(3)
        c.h(0)

        gate = c.to_gate("Test")
        assert gate.num_qubits == 3

    def test_circuit_gate_num_params(self):
        """CircuitGate num_params property"""
        # Non-parametric circuit
        c1 = Circuit(1)
        c1.x(0)
        gate1 = c1.to_gate("X")
        assert gate1.num_params == 0

        # Parametric circuit
        theta = Parameter("theta")
        c2 = Circuit(1)
        c2.rx(0, theta)
        gate2 = c2.to_gate("Rx")
        assert gate2.num_params == 1

    def test_circuit_gate_symbols(self):
        """CircuitGate symbols method"""
        theta = Parameter("theta")
        phi = Parameter("phi")
        c = Circuit(1)
        c.rx(0, theta)
        c.rz(0, phi)

        gate = c.to_gate("TwoParam")
        symbols = gate.symbols()
        assert len(symbols) == 2
        assert "theta" in symbols
        assert "phi" in symbols
