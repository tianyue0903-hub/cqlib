from numpy import allclose
from cqlib.circuit import Circuit
from cqlib.compiler import CommutativeOptimization
from . import random_circuit


class TestOperationCommutative:
    """Test commutative method on operations"""

    def test_identity_commutes_with_everything(self):
        """Identity gate should commute with any gate"""
        c = Circuit(1)
        c.i(0)  # Identity
        c.x(0)  # Pauli X

        ops = list(c.operations)
        assert len(ops) == 2
        assert CommutativeOptimization.is_commutative(ops[0], ops[1])
        assert CommutativeOptimization.is_commutative(ops[1], ops[0])

    def test_disjoint_qubits_commute(self):
        """Operations on different qubits always commute"""
        c = Circuit(2)
        c.h(0)
        c.x(1)

        ops = list(c.operations)
        assert len(ops) == 2
        assert CommutativeOptimization.is_commutative(ops[0], ops[1])
        assert CommutativeOptimization.is_commutative(ops[1], ops[0])

    def test_same_qubit_non_commuting_gates(self):
        """H and X on same qubit don't commute"""
        c = Circuit(1)
        c.h(0)
        c.x(0)

        ops = list(c.operations)
        assert len(ops) == 2
        assert not CommutativeOptimization.is_commutative(ops[0], ops[1])
        assert not CommutativeOptimization.is_commutative(ops[1], ops[0])

    def test_identical_operations_commute(self):
        """Same gate on same qubit should commute"""
        c = Circuit(1)
        c.x(0)
        c.x(0)

        ops = list(c.operations)
        assert len(ops) == 2
        assert CommutativeOptimization.is_commutative(ops[0], ops[1])

    def test_z_commutes_with_controlled_z(self):
        """Z gate commutes with CZ on overlapping qubits"""
        c = Circuit(2)
        c.cz(0, 1)
        c.z(1)

        ops = list(c.operations)
        assert len(ops) == 2
        # CZ and Z on same target should commute (both diagonal)
        assert CommutativeOptimization.is_commutative(ops[0], ops[1])
        assert CommutativeOptimization.is_commutative(ops[1], ops[0])

    def test_x_not_commutes_with_controlled_x(self):
        """X gate does not commute with CX on overlapping qubits"""
        c = Circuit(2)
        c.cx(0, 1)
        c.x(0)

        ops = list(c.operations)
        assert len(ops) == 2
        # CX and X on same control should NOT commute
        assert not CommutativeOptimization.is_commutative(ops[0], ops[1])
        assert not CommutativeOptimization.is_commutative(ops[1], ops[0])

    def test_commutative_with_parametric_gates(self):
        """Test commutativity with parametric gates"""
        c = Circuit(1)
        c.rx(0, 0.5)
        c.ry(0, 0.5)

        ops = list(c.operations)
        assert len(ops) == 2
        assert not CommutativeOptimization.is_commutative(ops[0], ops[1])

    def test_measure_returns_not_commutative(self):
        """Measurement should not commute with standard gates"""
        c = Circuit(1)
        c.h(0)
        c.measure(0)

        ops = list(c.operations)
        assert len(ops) == 2
        # H and Measure don't commute (Measure is non-unitary)
        assert not CommutativeOptimization.is_commutative(ops[0], ops[1])


class TestCommutativeOptimization:
    """Tests normal-path compiler workflows using CommutativeOptimization."""

    def test_two_x_gates_cancelled(self):
        """Two X gates on the same qubit cancel each other."""
        cir = Circuit(1)
        cir.x(0)
        cir.x(0)
        co = CommutativeOptimization(
            para=["x"], depara=["x"], keep_phase=True, keep_order=False
        )
        opt_cir = co.execute(cir)
        assert len(opt_cir) == 0

    def test_random_circuit(self):
        """Tests random circuit generation and optimization."""
        for _ in range(20):
            cir = random_circuit(5)
            co = CommutativeOptimization(
                para=[], depara=["x", "y", "z"], keep_phase=True, keep_order=False
            )
            opt_cir = co.execute(cir)
            assert allclose(cir.to_matrix(), opt_cir.to_matrix())
            assert len(opt_cir) <= len(cir)
