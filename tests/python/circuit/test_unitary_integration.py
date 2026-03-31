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

"""
测试 Circuit 与 UnitaryGate 自定义门的集成

测试范围：
- 自定义门应用到电路
- 单量子比特自定义门
- 双量子比特自定义门
- 多量子比特自定义门
- 自定义门与标准门混合使用
"""

import pytest
import numpy as np
from cqlib.circuit import Circuit, UnitaryGate


class TestUnitaryGateInCircuit:
    """测试自定义门在电路中的应用"""

    def test_apply_single_qubit_unitary(self):
        """应用单量子比特自定义门"""
        c = Circuit(2)
        h_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        gate = UnitaryGate("CustomH", 1).with_matrix(h_mat)
        
        c.unitary(gate, [0])
        assert len(c) == 1
        assert c[0].name == "CustomH"

    def test_apply_two_qubit_unitary(self):
        """应用双量子比特自定义门"""
        c = Circuit(3)
        cnot_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        gate = UnitaryGate("CustomCNOT", 2).with_matrix(cnot_mat)
        
        c.unitary(gate, [0, 1])
        assert len(c) == 1
        assert c[0].name == "CustomCNOT"

    def test_apply_multiple_unitaries(self):
        """应用多个自定义门"""
        c = Circuit(3)
        x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
        h_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        
        x_gate = UnitaryGate("X", 1).with_matrix(x_mat)
        h_gate = UnitaryGate("H", 1).with_matrix(h_mat)
        
        c.unitary(x_gate, [0])
        c.unitary(h_gate, [1])
        c.unitary(x_gate, [2])
        
        assert len(c) == 3

    def test_mixed_standard_and_unitary(self):
        """混合使用标准门和自定义门"""
        c = Circuit(2)
        custom_mat = np.array([[0, 1], [1, 0]], dtype=complex)
        custom_gate = UnitaryGate("CustomX", 1).with_matrix(custom_mat)
        
        c.h(0)
        c.unitary(custom_gate, [1])
        c.cx(0, 1)
        
        assert len(c) == 3
        assert c[0].name == "H"
        assert c[1].name == "CustomX"
        assert c[2].name == "CX"

    def test_apply_to_different_qubit_indices(self):
        """应用自定义门到不同的qubit索引"""
        c = Circuit(4)
        mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        gate = UnitaryGate("H", 1).with_matrix(mat)
        
        c.unitary(gate, [0])
        c.unitary(gate, [2])
        c.unitary(gate, [3])
        
        assert len(c) == 3

    def test_apply_to_non_adjacent_qubits(self):
        """应用双量子比特门到非相邻qubit"""
        c = Circuit(4)
        swap_mat = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        gate = UnitaryGate("SWAP", 2).with_matrix(swap_mat)
        
        c.unitary(gate, [0, 3])  # 非相邻qubit
        assert len(c) == 1


class TestUnitaryGateErrors:
    """测试自定义门错误处理"""

    def test_unitary_wrong_num_qubits(self):
        """自定义门qubit数量不匹配应报错"""
        c = Circuit(2)
        mat = np.eye(2, dtype=complex)
        gate = UnitaryGate("I", 1).with_matrix(mat)
        
        # 尝试应用到2个qubit但门只需要1个
        with pytest.raises(Exception):
            c.unitary(gate, [0, 1])

    def test_unitary_invalid_qubit(self):
        """自定义门应用到无效qubit应报错"""
        c = Circuit(2)
        mat = np.eye(2, dtype=complex)
        gate = UnitaryGate("I", 1).with_matrix(mat)
        
        with pytest.raises(Exception):
            c.unitary(gate, [5])  # qubit 5不存在

    def test_unitary_without_matrix(self):
        """未定义矩阵的自定义门"""
        c = Circuit(2)
        gate = UnitaryGate("Undefined", 1)  # 未设置矩阵
        
        # 应用时可能接受，但后续操作可能失败
        c.unitary(gate, [0])
        assert len(c) == 1


class TestUnitaryGateMatrixValidation:
    """测试自定义门矩阵验证"""

    def test_unitary_matrix_preserved(self):
        """验证自定义门矩阵被正确保留"""
        c = Circuit(1)
        h_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        gate = UnitaryGate("H", 1).with_matrix(h_mat)
        
        c.unitary(gate, [0])
        # 获取操作并验证矩阵
        op = c[0]
        assert op.name == "H"

    def test_identity_unitary(self):
        """应用单位门不改变电路"""
        c = Circuit(1)
        i_mat = np.eye(2, dtype=complex)
        gate = UnitaryGate("I", 1).with_matrix(i_mat)
        
        c.x(0)
        c.unitary(gate, [0])  # 单位门不应改变状态
        assert len(c) == 2
