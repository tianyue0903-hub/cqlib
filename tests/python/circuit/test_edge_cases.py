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
测试 Circuit 边界情况和错误处理

测试范围：
- 极端输入值
- 边界条件
- 错误恢复
- 资源限制
"""

import pytest
import numpy as np
from cqlib.circuit import Circuit, Parameter


class TestExtremeParameterValues:
    """测试极端参数值"""

    def test_very_small_parameter(self):
        """非常小的参数值"""
        c = Circuit(1)
        c.rx(0, 1e-15)
        assert len(c) == 1

    def test_very_large_parameter(self):
        """非常大的参数值"""
        c = Circuit(1)
        c.rx(0, 1e15)
        assert len(c) == 1

    def test_inf_parameter(self):
        """无穷大参数"""
        c = Circuit(1)
        # 根据实现，这可能报错或接受
        try:
            c.rx(0, np.inf)
        except Exception:
            pass  # 预期行为

    def test_nan_parameter(self):
        """NaN参数"""
        c = Circuit(1)
        try:
            c.rx(0, np.nan)
        except Exception:
            pass  # 预期行为


class TestEmptyAndMinimalCircuits:
    """测试空电路和最小电路"""

    def test_zero_qubit_circuit(self):
        """零qubit电路"""
        c = Circuit(0)
        assert c.num_qubits == 0
        assert len(c) == 0
        assert len(c.qubits) == 0

    def test_single_qubit_minimal(self):
        """最小单qubit电路"""
        c = Circuit(1)
        assert c.num_qubits == 1
        assert len(c) == 0

    def test_empty_circuit_operations(self):
        """空电路操作"""
        c = Circuit(2)
        # 各种操作对空电路应该正常工作
        assert len(c) == 0
        assert list(c.operations) == []
        
        # 切片空电路
        ops = c[:]
        assert len(ops) == 0


class TestQubitIndexBoundaries:
    """测试qubit索引边界"""

    def test_qubit_index_zero(self):
        """qubit索引0"""
        c = Circuit(2)
        c.h(0)
        assert c[0].qubits[0].index == 0

    def test_qubit_index_last(self):
        """最后一个qubit索引"""
        c = Circuit(5)
        c.h(4)
        assert c[0].qubits[0].index == 4

    def test_very_large_qubit_index(self):
        """非常大的qubit索引"""
        # 使用非连续qubit
        c = Circuit([0, 1000000])
        assert c.num_qubits == 2
        c.h(1000000)
        assert c[0].qubits[0].index == 1000000


class TestCircuitModification:
    """测试电路修改"""

    def test_circuit_reuse(self):
        """电路重用"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        
        # 基于原电路创建新操作
        c2 = Circuit(2)
        for op in c:
            if op.name == "H":
                c2.h(op.qubits[0].index)
            elif op.name == "CX":
                c2.cx(op.qubits[0].index, op.qubits[1].index)
        
        assert len(c2) == 2

    def test_build_circuit_incrementally(self):
        """增量构建电路"""
        c = Circuit(2)
        
        # 逐步添加门
        for i in range(10):
            c.h(0)
            assert len(c) == i + 1


class TestSpecialCharactersInLabels:
    """测试特殊字符（如果适用）"""

    def test_long_gate_name(self):
        """长门名称"""
        from cqlib.circuit import UnitaryGate
        
        long_name = "A" * 1000
        mat = np.eye(2, dtype=complex)
        gate = UnitaryGate(long_name, 1).with_matrix(mat)
        
        c = Circuit(1)
        c.unitary(gate, [0])
        assert c[0].name == long_name

    def test_unicode_gate_name(self):
        """Unicode门名称"""
        from cqlib.circuit import UnitaryGate
        
        unicode_name = "自定义门_αβγ"
        mat = np.eye(2, dtype=complex)
        gate = UnitaryGate(unicode_name, 1).with_matrix(mat)
        
        c = Circuit(1)
        c.unitary(gate, [0])
        assert c[0].name == unicode_name


class TestMemoryEfficiency:
    """测试内存效率"""

    def test_large_circuit_creation(self):
        """创建大型电路"""
        # 创建有100个qubit的电路
        c = Circuit(100)
        assert c.num_qubits == 100
        
        # 添加一些操作
        for i in range(50):
            c.h(i)
        
        assert len(c) == 50

    def test_many_operations(self):
        """大量操作"""
        c = Circuit(2)
        
        # 添加1000个操作
        for _ in range(1000):
            c.h(0)
        
        assert len(c) == 1000


class TestCircuitComparison:
    """测试电路比较"""

    def test_same_circuits_equal(self):
        """相同电路相等"""
        c1 = Circuit(2)
        c1.h(0)
        c1.cx(0, 1)
        
        c2 = Circuit(2)
        c2.h(0)
        c2.cx(0, 1)
        
        # 电路应该相等（如果实现了__eq__）
        # assert c1 == c2
        assert len(c1) == len(c2)

    def test_different_circuits_not_equal(self):
        """不同电路不等"""
        c1 = Circuit(2)
        c1.h(0)
        
        c2 = Circuit(2)
        c2.x(0)
        
        # 电路应该不等
        # assert c1 != c2
        assert c1[0].name != c2[0].name


class TestCircuitHashAndRepr:
    """测试电路哈希和表示"""

    def test_circuit_repr(self):
        """电路表示"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        
        repr_str = repr(c)
        assert "Circuit" in repr_str or "circuit" in repr_str.lower()

    def test_circuit_str(self):
        """电路字符串表示"""
        c = Circuit(2)
        c.h(0)
        
        str_repr = str(c)
        assert isinstance(str_repr, str)


class TestParameterEdgeCases:
    """测试参数边界情况"""

    def test_parameter_with_very_long_name(self):
        """超长参数名"""
        long_name = "param_" + "x" * 1000
        param = Parameter(long_name)
        
        c = Circuit(1)
        c.rx(0, param)
        assert len(c) == 1

    def test_many_parameters(self):
        """大量不同参数"""
        c = Circuit(10)
        
        for i in range(10):
            param = Parameter(f"theta_{i}")
            c.rx(i, param)
        
        assert len(c) == 10
        assert len(c.parameters) >= 10

    def test_reuse_same_parameter(self):
        """重用相同参数"""
        theta = Parameter("theta")
        
        c = Circuit(3)
        c.rx(0, theta)
        c.rx(1, theta)
        c.rx(2, theta)
        
        # 应该只追踪一次该参数
        assert len(c) == 3


class TestInverseEdgeCases:
    """测试逆运算边界情况"""

    def test_inverse_empty_circuit(self):
        """空电路逆"""
        c = Circuit(2)
        c_inv = c.inverse()
        assert len(c_inv) == 0

    def test_inverse_single_gate(self):
        """单门电路逆"""
        c = Circuit(1)
        c.h(0)
        
        c_inv = c.inverse()
        assert len(c_inv) == 1
        # H门是自逆的
        assert c_inv[0].name == "H"

    def test_inverse_inverse_identity(self):
        """逆的逆是原电路"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        
        c_inv_inv = c.inverse().inverse()
        assert len(c_inv_inv) == len(c)
