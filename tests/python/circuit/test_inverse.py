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
测试 Circuit 逆运算

测试范围：
- 单量子比特门逆
- 多量子比特门逆
- 包含Barrier的逆
- 非酉操作不可逆
"""

import pytest
import numpy as np
from cqlib.circuit import Circuit


class TestCircuitInverseBasic:
    """测试基础逆运算"""

    def test_inverse_single_qubit_gates(self):
        """单量子比特门逆"""
        c = Circuit(1)
        c.h(0)
        c.x(0)

        c_inv = c.inverse()
        assert len(c_inv) == 2

    def test_inverse_pauli_gates(self):
        """Pauli门逆"""
        c = Circuit(1)
        c.x(0)
        c.y(0)
        c.z(0)

        c_inv = c.inverse()
        assert len(c_inv) == 3

    def test_inverse_cnot(self):
        """CNOT逆（自逆）"""
        c = Circuit(2)
        c.cx(0, 1)

        c_inv = c.inverse()
        assert len(c_inv) == 1

    def test_inverse_bell_state(self):
        """Bell态电路逆"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)

        c_inv = c.inverse()
        assert len(c_inv) == 2


class TestCircuitInverseWithBarrier:
    """测试含Barrier的逆"""

    def test_barrier_persists_in_inverse(self):
        """Barrier在逆中保留"""
        c = Circuit(2)
        c.h(0)
        c.barrier([0, 1])
        c.cx(0, 1)

        c_inv = c.inverse()
        assert len(c_inv) == 3
        # Barrier名称可能是 "Barrier" 或 "Directive(Barrier)"
        barrier_found = any("Barrier" in op.name for op in c_inv)
        assert barrier_found


class TestCircuitInverseNonUnitary:
    """测试非酉操作不可逆"""

    def test_measure_not_invertible(self):
        """测量不可逆"""
        c = Circuit(1)
        c.h(0)
        c.measure(0)

        with pytest.raises(Exception):
            c.inverse()

    def test_reset_not_invertible(self):
        """Reset不可逆"""
        c = Circuit(1)
        c.reset(0)

        with pytest.raises(Exception):
            c.inverse()


class TestCircuitInverseTwice:
    """测试两次逆运算"""

    def test_inverse_twice(self):
        """两次逆回到原电路"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)

        c_inv_inv = c.inverse().inverse()
        assert len(c_inv_inv) == len(c)


class TestCircuitInverseEmpty:
    """测试空电路逆"""

    def test_empty_circuit_inverse(self):
        """空电路逆"""
        c = Circuit(1)
        c_inv = c.inverse()
        assert len(c_inv) == 0
