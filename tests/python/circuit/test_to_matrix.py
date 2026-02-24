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
测试 Circuit 矩阵表示

测试范围：
- 单量子比特门矩阵
- 多量子比特门矩阵
- 自定义qubit顺序
- 验证幺正性
"""

import pytest
import numpy as np
from cqlib.circuit import Circuit, circuit_to_matrix


class TestCircuitToMatrix:
    """测试电路矩阵"""

    def test_hadamard_matrix(self):
        """Hadamard门矩阵"""
        c = Circuit(1)
        c.h(0)

        mat = c.to_matrix()
        expected = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        assert np.allclose(mat, expected)

    def test_pauli_x_matrix(self):
        """Pauli-X门矩阵"""
        c = Circuit(1)
        c.x(0)

        mat = c.to_matrix()
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.allclose(mat, expected)

    def test_identity_matrix(self):
        """空电路矩阵是单位矩阵"""
        c = Circuit(1)
        mat = c.to_matrix()
        expected = np.eye(2, dtype=complex)
        assert np.allclose(mat, expected)

    def test_cnot_matrix(self):
        """CNOT门矩阵 - 注意qubit顺序可能是反转的"""
        c = Circuit(2)
        c.cx(0, 1)

        mat = c.to_matrix()
        # 实际的矩阵（基于当前实现的qubit顺序）
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ], dtype=complex)
        assert np.allclose(mat, expected)

    def test_bell_state_matrix(self):
        """Bell态电路矩阵"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)

        mat = c.to_matrix()
        # 验证幺正性
        identity = mat @ mat.conj().T
        assert np.allclose(identity, np.eye(4))


class TestCircuitToMatrixCustomOrder:
    """测试自定义qubit顺序"""

    def test_custom_qubit_order(self):
        """自定义qubit顺序"""
        c = Circuit(2)
        c.cx(0, 1)

        mat_default = c.to_matrix()
        mat_swapped = c.to_matrix([1, 0])

        # 两种顺序产生不同矩阵
        assert not np.allclose(mat_default, mat_swapped)


class TestCircuitToMatrixFunction:
    """测试circuit_to_matrix函数"""

    def test_circuit_to_matrix_single_qubit(self):
        """单量子比特电路矩阵"""
        c = Circuit(1)
        c.h(0)

        mat = circuit_to_matrix(c)
        expected = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        assert np.allclose(mat, expected)
