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
测试 Circuit 创建功能

测试范围：
- 通过qubit数量创建
- 通过qubit索引列表创建
- 通过Qubit对象列表创建
- 空电路创建
- 重复qubit的错误处理
"""

import pytest
from cqlib.circuit import Circuit, Qubit


class TestCircuitCreationByNumQubits:
    """测试通过qubit数量创建电路"""

    def test_create_single_qubit(self):
        """创建单qubit电路"""
        c = Circuit(1)
        assert c.num_qubits == 1
        assert len(c.qubits) == 1
        assert c.qubits[0].index == 0

    def test_create_multiple_qubits(self):
        """创建多qubit电路"""
        c = Circuit(5)
        assert c.num_qubits == 5
        assert len(c.qubits) == 5
        for i, q in enumerate(c.qubits):
            assert q.index == i

    def test_create_zero_qubits(self):
        """创建0 qubit电路"""
        c = Circuit(0)
        assert c.num_qubits == 0
        assert len(c.qubits) == 0


class TestCircuitCreationByIndices:
    """测试通过qubit索引列表创建电路"""

    def test_create_with_indices(self):
        """使用索引列表创建"""
        c = Circuit([0, 1, 2])
        assert c.num_qubits == 3
        for i, q in enumerate(c.qubits):
            assert q.index == i

    def test_create_with_non_contiguous_indices(self):
        """使用非连续索引创建"""
        c = Circuit([0, 2, 4, 6])
        assert c.num_qubits == 4
        indices = [q.index for q in c.qubits]
        assert indices == [0, 2, 4, 6]

    def test_create_with_unordered_indices(self):
        """使用无序索引创建"""
        c = Circuit([3, 1, 2, 0])
        assert c.num_qubits == 4
        indices = [q.index for q in c.qubits]
        assert indices == [3, 1, 2, 0]


class TestCircuitCreationByQubitObjects:
    """测试通过Qubit对象列表创建电路"""

    def test_create_with_qubit_objects(self):
        """使用Qubit对象创建"""
        qubits = [Qubit(0), Qubit(1), Qubit(2)]
        c = Circuit(qubits)
        assert c.num_qubits == 3

    def test_create_with_mixed_qubits(self):
        """使用混合索引的Qubit对象"""
        qubits = [Qubit(5), Qubit(10), Qubit(15)]
        c = Circuit(qubits)
        assert c.num_qubits == 3
        indices = [q.index for q in c.qubits]
        assert indices == [5, 10, 15]


class TestCircuitCreationErrors:
    """测试电路创建的错误处理"""

    def test_duplicate_qubit_indices(self):
        """重复qubit索引应报错"""
        with pytest.raises(Exception):
            Circuit([0, 1, 1, 2])

    def test_duplicate_qubit_objects(self):
        """重复Qubit对象应报错"""
        with pytest.raises(Exception):
            Circuit([Qubit(0), Qubit(1), Qubit(0)])


class TestCircuitProperties:
    """测试电路基本属性"""

    def test_circuit_width(self, single_qubit_circuit):
        """测试电路宽度"""
        c = single_qubit_circuit
        assert c.num_qubits == 1

    def test_circuit_qubits_immutable(self, two_qubit_circuit):
        """测试qubits列表"""
        c = two_qubit_circuit
        qubits = c.qubits
        assert len(qubits) == 2
        # 验证Qubit对象
        for q in qubits:
            assert isinstance(q, Qubit)

    def test_empty_circuit_operations(self, empty_circuit):
        """空电路的操作列表为空"""
        c = empty_circuit
        assert len(c) == 0
        assert list(c.operations) == []
