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
测试电路索引和迭代

测试范围：
- 通过索引访问操作
- 负索引访问
- 切片访问
- 迭代操作
- 长度计算
"""

import pytest


class TestCircuitIndexing:
    """测试电路索引访问"""

    def test_get_first_operation(self, bell_state_circuit):
        """获取第一个操作"""
        c = bell_state_circuit
        op = c[0]
        assert op.name == "H"

    def test_get_second_operation(self, bell_state_circuit):
        """获取第二个操作"""
        c = bell_state_circuit
        op = c[1]
        assert op.name == "CX"

    def test_get_last_operation_with_negative_index(self, bell_state_circuit):
        """使用负索引获取最后一个操作"""
        c = bell_state_circuit
        op = c[-1]
        assert op.name == "CX"

    def test_get_first_with_negative_index(self, bell_state_circuit):
        """使用负索引获取第一个操作"""
        c = bell_state_circuit
        op = c[-2]
        assert op.name == "H"


class TestCircuitIndexingErrors:
    """测试电路索引错误"""

    def test_index_out_of_range_positive(self, bell_state_circuit):
        """正索引越界"""
        c = bell_state_circuit
        with pytest.raises(IndexError):
            _ = c[10]

    def test_index_out_of_range_negative(self, bell_state_circuit):
        """负索引越界"""
        c = bell_state_circuit
        with pytest.raises(IndexError):
            _ = c[-10]

    def test_index_on_empty_circuit(self, empty_circuit):
        """空电路索引"""
        c = empty_circuit
        with pytest.raises(IndexError):
            _ = c[0]


class TestCircuitSlicing:
    """测试电路切片"""

    def test_slice_all_operations(self, single_qubit_circuit):
        """切片获取所有操作"""
        c = single_qubit_circuit
        c.h(0)
        c.x(0)
        c.y(0)

        ops = c[:]
        assert len(ops) == 3

    def test_slice_first_two(self, single_qubit_circuit):
        """切片获取前两个操作"""
        c = single_qubit_circuit
        c.h(0)
        c.x(0)
        c.y(0)

        ops = c[0:2]
        assert len(ops) == 2
        assert ops[0].name == "H"
        assert ops[1].name == "X"

    def test_slice_from_index(self, single_qubit_circuit):
        """从指定索引切片"""
        c = single_qubit_circuit
        c.h(0)
        c.x(0)
        c.y(0)

        ops = c[1:]
        assert len(ops) == 2
        assert ops[0].name == "X"
        assert ops[1].name == "Y"

    def test_slice_to_index(self, single_qubit_circuit):
        """切片到指定索引"""
        c = single_qubit_circuit
        c.h(0)
        c.x(0)
        c.y(0)

        ops = c[:2]
        assert len(ops) == 2

    def test_slice_with_step(self, single_qubit_circuit):
        """带步长的切片"""
        c = single_qubit_circuit
        c.h(0)
        c.x(0)
        c.y(0)
        c.z(0)

        ops = c[::2]
        assert len(ops) == 2
        assert ops[0].name == "H"
        assert ops[1].name == "Y"

    def test_negative_step_slice(self, single_qubit_circuit):
        """负步长切片（反向）"""
        c = single_qubit_circuit
        c.h(0)
        c.x(0)
        c.y(0)

        ops = c[::-1]
        assert len(ops) == 3
        assert ops[0].name == "Y"
        assert ops[2].name == "H"


class TestCircuitIteration:
    """测试电路迭代"""

    def test_iterate_operations(self, bell_state_circuit):
        """迭代电路操作"""
        c = bell_state_circuit
        ops = list(c.operations)
        assert len(ops) == 2
        assert ops[0].name == "H"
        assert ops[1].name == "CX"

    def test_iterate_empty_circuit(self, empty_circuit):
        """迭代空电路"""
        c = empty_circuit
        ops = list(c.operations)
        assert len(ops) == 0

    def test_for_loop_iteration(self, bell_state_circuit):
        """使用for循环迭代"""
        c = bell_state_circuit
        names = []
        for op in c.operations:
            names.append(op.name)
        assert names == ["H", "CX"]


class TestCircuitLength:
    """测试电路长度"""

    def test_len_empty_circuit(self, empty_circuit):
        """空电路长度"""
        assert len(empty_circuit) == 0

    def test_len_single_operation(self, single_qubit_circuit):
        """单操作电路"""
        c = single_qubit_circuit
        c.h(0)
        assert len(c) == 1

    def test_len_multiple_operations(self, bell_state_circuit):
        """多操作电路"""
        assert len(bell_state_circuit) == 2

    def test_len_after_operations(self, single_qubit_circuit):
        """添加操作后长度变化"""
        c = single_qubit_circuit
        assert len(c) == 0
        c.h(0)
        assert len(c) == 1
        c.x(0)
        assert len(c) == 2


class TestOperationProperties:
    """测试操作属性"""

    def test_operation_name(self, bell_state_circuit):
        """测试操作名称"""
        c = bell_state_circuit
        op = c[0]
        assert hasattr(op, "name")
        assert isinstance(op.name, str)

    def test_operation_qubits(self, bell_state_circuit):
        """测试操作qubits"""
        c = bell_state_circuit
        op = c[1]  # CNOT
        assert hasattr(op, "qubits")
        assert len(op.qubits) == 2

    def test_operation_params(self, single_qubit_circuit, theta_param):
        """测试操作参数"""
        c = single_qubit_circuit
        c.rx(0, theta_param)
        op = c[0]
        assert hasattr(op, "params")
        assert op.num_params == 1
