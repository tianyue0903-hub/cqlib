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

"""Basic circuit tests demonstrating all supported quantum gate operations."""

import numpy as np
import pytest

from cqlib.circuit import Circuit, Parameter, Qubit
from cqlib.circuit.gates import (
    H, X, Y, Z, I, S, SDG, T, TDG,
    RX, RY, RZ, U, Phase, GPhase,
    CX, CY, CZ, SWAP,
    RXX, RYY, RZZ, RZX, RXY,
    CRX, CRY, CRZ,
    CCX,
    X2P, X2M, Y2P, Y2M, XY, XY2P, XY2M,
    FSIM,
    StandardGate,
)


class TestCircuitCreation:
    """Test circuit creation with different methods."""

    def test_create_with_num_qubits(self):
        """Create circuit with number of qubits."""
        c = Circuit(3)
        assert c.num_qubits == 3
        assert len(c.qubits) == 3

    def test_create_with_qubit_indices(self):
        """Create circuit with specific qubit indices."""
        c = Circuit([0, 2, 4])
        assert c.num_qubits == 3
        assert [q.index for q in c.qubits] == [0, 2, 4]

    def test_create_with_qubit_objects(self):
        """Create circuit with Qubit objects."""
        qubits = [Qubit(0), Qubit(2), Qubit(4)]
        c = Circuit(qubits)
        assert c.num_qubits == 3
        assert [q.index for q in c.qubits] == [0, 2, 4]


class TestSingleQubitGates:
    """Test single qubit gates."""

    def test_identity_gate(self):
        """Test I (Identity) gate."""
        c = Circuit(1)
        c.i(0)
        assert len(c.operations) == 1
        assert c.operations[0].name == "I"

    def test_pauli_gates(self):
        """Test Pauli X, Y, Z gates."""
        c = Circuit(1)
        c.x(0)
        c.y(0)
        c.z(0)
        assert len(c.operations) == 3
        assert c.operations[0].name == "X"
        assert c.operations[1].name == "Y"
        assert c.operations[2].name == "Z"

    def test_hadamard_gate(self):
        """Test H (Hadamard) gate."""
        c = Circuit(1)
        c.h(0)
        assert len(c.operations) == 1
        assert c.operations[0].name == "H"

    def test_phase_gates(self):
        """Test S, S-dagger, T, T-dagger gates."""
        c = Circuit(1)
        c.s(0)
        c.sdg(0)
        c.t(0)
        c.tdg(0)
        assert len(c.operations) == 4
        assert c.operations[0].name == "S"
        assert c.operations[1].name == "SDG"
        assert c.operations[2].name == "T"
        assert c.operations[3].name == "TDG"

    def test_sqrt_pauli_gates(self):
        """Test sqrt(X) and sqrt(Y) gates."""
        c = Circuit(1)
        c.x2p(0)  # √X
        c.x2m(0)  # √X†
        c.y2p(0)  # √Y
        c.y2m(0)  # √Y†
        assert len(c.operations) == 4
        assert c.operations[0].name == "X2P"
        assert c.operations[1].name == "X2M"
        assert c.operations[2].name == "Y2P"
        assert c.operations[3].name == "Y2M"

    def test_xy_gates(self):
        """Test XY rotation gates."""
        c = Circuit(1)
        c.xy(0, 0.5)
        c.xy2p(0, 0.3)
        c.xy2m(0, 0.4)
        assert len(c.operations) == 3


class TestParametricSingleQubitGates:
    """Test parametric single qubit rotation gates."""

    def test_rx_gate_with_float(self):
        """Test RX gate with float parameter."""
        c = Circuit(1)
        c.rx(0, np.pi / 2)
        assert len(c.operations) == 1
        assert c.operations[0].name == "RX"

    def test_rx_gate_with_parameter(self):
        """Test RX gate with symbolic parameter."""
        c = Circuit(1)
        theta = Parameter("theta")
        c.rx(0, theta)
        assert len(c.operations) == 1

    def test_ry_gate(self):
        """Test RY gate."""
        c = Circuit(1)
        c.ry(0, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "RY"

    def test_rz_gate(self):
        """Test RZ gate."""
        c = Circuit(1)
        c.rz(0, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "RZ"

    def test_phase_gate(self):
        """Test Phase (P) gate."""
        c = Circuit(1)
        c.phase(0, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "Phase"

    def test_u_gate(self):
        """Test U (universal single-qubit) gate."""
        c = Circuit(1)
        c.u(0, 0.1, 0.2, 0.3)
        assert len(c.operations) == 1
        assert c.operations[0].name == "U"

    def test_rxy_gate(self):
        """Test RXY (XY plane rotation) gate."""
        c = Circuit(1)
        c.rxy(0, 0.5, 0.3)
        assert len(c.operations) == 1
        assert c.operations[0].name == "RXY"


class TestTwoQubitGates:
    """Test two qubit gates."""

    def test_cnot_gate(self):
        """Test CNOT (CX) gate."""
        c = Circuit(2)
        c.cx(0, 1)
        assert len(c.operations) == 1
        assert c.operations[0].name == "CX"

    def test_cy_gate(self):
        """Test CY gate."""
        c = Circuit(2)
        c.cy(0, 1)
        assert len(c.operations) == 1
        assert c.operations[0].name == "CY"

    def test_cz_gate(self):
        """Test CZ gate."""
        c = Circuit(2)
        c.cz(0, 1)
        assert len(c.operations) == 1
        assert c.operations[0].name == "CZ"

    def test_swap_gate(self):
        """Test SWAP gate."""
        c = Circuit(2)
        c.swap(0, 1)
        assert len(c.operations) == 1
        assert c.operations[0].name == "SWAP"


class TestTwoQubitParametricGates:
    """Test two qubit parametric gates."""

    def test_rxx_gate(self):
        """Test RXX (Ising XX) gate."""
        c = Circuit(2)
        c.rxx(0, 1, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "RXX"

    def test_ryy_gate(self):
        """Test RYY (Ising YY) gate."""
        c = Circuit(2)
        c.ryy(0, 1, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "RYY"

    def test_rzz_gate(self):
        """Test RZZ (Ising ZZ) gate."""
        c = Circuit(2)
        c.rzz(0, 1, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "RZZ"

    def test_rzx_gate(self):
        """Test RZX (Ising ZX) gate."""
        c = Circuit(2)
        c.rzx(0, 1, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "RZX"

    def test_fsim_gate(self):
        """Test fSim (Fermionic Simulation) gate."""
        c = Circuit(2)
        c.fsim(0, 1, 0.5, 0.3)
        assert len(c.operations) == 1
        assert c.operations[0].name == "FSIM"


class TestControlledRotationGates:
    """Test controlled rotation gates."""

    def test_crx_gate(self):
        """Test CRX gate."""
        c = Circuit(2)
        c.crx(0, 1, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "CRX"

    def test_cry_gate(self):
        """Test CRY gate."""
        c = Circuit(2)
        c.cry(0, 1, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "CRY"

    def test_crz_gate(self):
        """Test CRZ gate."""
        c = Circuit(2)
        c.crz(0, 1, 0.5)
        assert len(c.operations) == 1
        assert c.operations[0].name == "CRZ"


class TestMultiControlledGates:
    """Test multi-controlled gates."""

    def test_ccx_gate(self):
        """Test CCX (Toffoli) gate."""
        c = Circuit(3)
        c.ccx(0, 1, 2)
        assert len(c.operations) == 1
        assert c.operations[0].name == "CCX"

    def test_multi_control(self):
        """Test multi_control method."""
        c = Circuit(4)
        c.multi_control(H, controls=[0, 1], targets=[2])
        assert len(c.operations) == 1


class TestDirectives:
    """Test non-unitary directives."""

    def test_measure(self):
        """Test measurement operation."""
        c = Circuit(1)
        c.measure(0)
        assert len(c.operations) == 1
        assert c.operations[0].name == "Measure"

    def test_reset(self):
        """Test reset operation."""
        c = Circuit(1)
        c.reset(0)
        assert len(c.operations) == 1
        assert c.operations[0].name == "Reset"

    def test_barrier(self):
        """Test barrier operation."""
        c = Circuit(3)
        c.barrier([0, 1, 2])
        assert len(c.operations) == 1
        assert c.operations[0].name == "Barrier"


class TestCircuitOperations:
    """Test circuit-level operations."""

    def test_circuit_inverse(self):
        """Test circuit inversion."""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c_inv = c.inverse()
        assert len(c_inv.operations) == 2

    def test_operations_property(self):
        """Test accessing operations property."""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        ops = c.operations
        assert len(ops) == 2


class TestComplexCircuits:
    """Test complex circuit constructions."""

    def test_bell_state_circuit(self):
        """Create a Bell state circuit."""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        assert len(c.operations) == 2
        assert c.operations[0].name == "H"
        assert c.operations[1].name == "CX"

    def test_ghz_state_circuit(self):
        """Create a GHZ state circuit."""
        c = Circuit(3)
        c.h(0)
        c.cx(0, 1)
        c.cx(0, 2)
        assert len(c.operations) == 3

    def test_qft_circuit(self):
        """Create a simple QFT-like circuit."""
        c = Circuit(3)
        c.h(0)
        c.crz(1, 0, np.pi / 2)
        c.crz(2, 0, np.pi / 4)
        c.h(1)
        c.crz(2, 1, np.pi / 2)
        c.h(2)
        assert len(c.operations) == 6


class TestGateProperties:
    """Test gate properties and methods."""

    def test_standard_gate_matrix(self):
        """Test getting matrix from standard gate."""
        h_matrix = H.matrix()
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        np.testing.assert_allclose(h_matrix, expected, atol=1e-10)

    def test_standard_gate_control(self):
        """Test adding control to standard gate."""
        cx = X.control(1)
        assert cx.num_ctrl_qubits == 1
        assert cx.num_qubits == 2

    def test_standard_gate_inverse(self):
        """Test gate inverse."""
        s_inv = S.inverse()
        assert s_inv.num_qubits == 1

    def test_parametric_gate_binding(self):
        """Test binding parameters to parametric gates."""
        rx_gate = RX(0.5)
        assert rx_gate.num_params == 1

    def test_gate_num_qubits(self):
        """Test gate qubit counts."""
        assert H.num_qubits == 1
        assert X.num_qubits == 1
        assert CX.num_qubits == 2
        assert CCX.num_qubits == 3


class TestParameterOperations:
    """Test parameter operations."""

    def test_parameter_creation(self):
        """Test creating symbolic parameters."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        assert str(theta) == "theta"
        assert str(phi) == "phi"

    def test_parameter_arithmetic(self):
        """Test parameter arithmetic operations."""
        theta = Parameter("theta")
        expr = theta + 1
        assert expr is not None

    def test_parameter_in_circuit(self):
        """Test using parameters in circuits."""
        c = Circuit(1)
        theta = Parameter("theta")
        c.rx(0, theta)
        c.ry(0, theta + 0.5)
        assert len(c.operations) == 2


class TestCircuitDataAccess:
    """Test accessing circuit data through operations."""

    def test_operation_qubits(self):
        """Test accessing operation qubits."""
        c = Circuit(2)
        c.cx(0, 1)
        op = c.operations[0]
        qubits = op.qubits
        assert len(qubits) == 2
        assert qubits[0].index == 0
        assert qubits[1].index == 1

    def test_operation_instruction(self):
        """Test accessing operation instruction."""
        c = Circuit(1)
        c.h(0)
        op = c.operations[0]
        inst = op.instruction
        assert inst.is_standard
        assert inst.name == "H"

    def test_operation_params(self):
        """Test accessing operation parameters."""
        c = Circuit(1)
        c.rx(0, 0.5)
        op = c.operations[0]
        params = op.params
        assert len(params) == 1
        assert abs(params[0] - 0.5) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
