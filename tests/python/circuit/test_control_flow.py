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
测试 Circuit 控制流功能 (if-else 和 while-loop)

测试范围：
- if_else 基本功能
- while_loop 基本功能
- 使用固定参数的控制流
- 使用符号参数的控制流
- 嵌套控制流
- 错误处理
"""

import pytest
import numpy as np
from cqlib.circuit import Circuit, Qubit, Parameter, ConditionView, StandardGate


class TestIfElseBasic:
    """测试 if_else 基本功能"""

    def test_if_else_simple(self):
        """测试简单的 if-else 语句，验证操作类型和内容"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        # 如果 qubit 0 测量为 1，则对 qubit 1 应用 X 门
        circuit.if_else(condition, [(StandardGate.X, [1])])

        # 验证电路长度
        assert len(circuit) == 3  # x, measure, if_else

        # 验证操作类型
        ops = list(circuit.operations)
        assert ops[0].name == "X"
        assert ops[0].qubits[0].index == 0
        assert ops[1].name == "Measure"
        assert ops[1].qubits[0].index == 0
        assert ops[2].instruction.is_control_flow

        # 验证控制流内容
        control_flow = ops[2].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_if_else
        if_else_gate = control_flow.as_if_else
        assert if_else_gate is not None
        assert if_else_gate.condition.qubit.index == 0
        assert if_else_gate.condition.target == 1

    def test_if_else_with_false_body(self):
        """测试带有 false body 的 if-else，验证两个分支的内容"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        # 如果 qubit 0 测量为 1，则对 qubit 1 应用 X 门；否则应用 Z 门
        circuit.if_else(
            condition,
            [(StandardGate.X, [1])],  # true body
            [(StandardGate.Z, [1])],  # false body
        )

        assert len(circuit) == 3

        # 验证控制流有两个分支
        ops = list(circuit.operations)
        control_flow = ops[2].instruction.control_flow
        if_else_gate = control_flow.as_if_else

        # 验证 true body 包含 X 门
        true_ops = if_else_gate.true_body
        assert len(true_ops) == 1
        assert true_ops[0].instruction.name == "X"

        # 验证 false body 包含 Z 门
        false_ops = if_else_gate.false_body
        assert len(false_ops) == 1
        assert false_ops[0].instruction.name == "Z"

    def test_if_else_multiple_operations(self):
        """测试包含多个操作的 if-else body，验证每个操作的 qubits"""
        circuit = Circuit(3)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        # true body 包含多个操作
        true_body = [(StandardGate.H, [1]), (StandardGate.CX, [1, 2])]
        circuit.if_else(condition, true_body)

        assert len(circuit) == 3

        # 验证控制流 body 中的操作
        ops = list(circuit.operations)
        control_flow = ops[2].instruction.control_flow
        if_else_gate = control_flow.as_if_else
        true_ops = if_else_gate.true_body

        # 验证第一个操作是 H 门，作用在 qubit 1
        assert true_ops[0].instruction.name == "H"
        assert [q.index for q in true_ops[0].qubits] == [1]

        # 验证第二个操作是 CX 门，作用在 qubit 1 和 2
        assert true_ops[1].instruction.name == "CX"
        assert [q.index for q in true_ops[1].qubits] == [1, 2]


class TestWhileLoopBasic:
    """测试 while_loop 基本功能"""

    def test_while_loop_simple(self):
        """测试简单的 while 循环，验证循环体内容"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        # 当 qubit 0 测量为 1 时，对 qubit 1 应用 H 门
        circuit.while_loop(condition, [(StandardGate.H, [1])])

        assert len(circuit) == 3  # x, measure, while_loop

        # 验证 while 循环结构
        ops = list(circuit.operations)
        control_flow = ops[2].instruction.control_flow
        assert control_flow is not None
        assert control_flow.is_while_loop

        while_loop_gate = control_flow.as_while_loop
        assert while_loop_gate is not None
        assert while_loop_gate.condition.qubit.index == 0
        assert while_loop_gate.condition.target == 1

        # 验证循环体包含 H 门
        body_ops = while_loop_gate.body
        assert len(body_ops) == 1
        assert body_ops[0].instruction.name == "H"
        assert [q.index for q in body_ops[0].qubits] == [1]

    def test_while_loop_multiple_operations(self):
        """测试包含多个操作的 while body，验证每个操作的类型和参数"""
        circuit = Circuit(3)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        body = [(StandardGate.H, [1]), (StandardGate.CX, [1, 2]), (StandardGate.T, [2])]
        circuit.while_loop(condition, body)

        assert len(circuit) == 3

        # 验证循环体中的每个操作
        ops = list(circuit.operations)
        control_flow = ops[2].instruction.control_flow
        while_loop_gate = control_flow.as_while_loop
        body_ops = while_loop_gate.body

        assert len(body_ops) == 3

        # 验证第一个操作
        assert body_ops[0].instruction.name == "H"
        assert [q.index for q in body_ops[0].qubits] == [1]

        # 验证第二个操作
        assert body_ops[1].instruction.name == "CX"
        assert [q.index for q in body_ops[1].qubits] == [1, 2]

        # 验证第三个操作
        assert body_ops[2].instruction.name == "T"
        assert [q.index for q in body_ops[2].qubits] == [2]


class TestControlFlowWithParameters:
    """测试带参数的控制流"""

    def test_if_else_with_fixed_parameters(self):
        """测试使用固定参数的 if-else"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        # 使用固定参数
        true_body = [(StandardGate.RX, [1], [np.pi / 2])]
        circuit.if_else(condition, true_body)

        assert len(circuit) == 3

    def test_if_else_with_symbolic_parameters(self):
        """测试使用符号参数的 if-else"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        theta = Parameter("theta")

        condition = ConditionView(Qubit(0), 1)
        # 使用符号参数
        true_body = [(StandardGate.RX, [1], [theta])]
        circuit.if_else(condition, true_body)

        assert len(circuit) == 3
        # 验证参数已添加到电路
        assert len(circuit.parameters) == 1

    def test_while_loop_with_fixed_parameters(self):
        """测试使用固定参数的 while 循环"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        body = [(StandardGate.RX, [1], [0.5]), (StandardGate.RY, [1], [0.3])]
        circuit.while_loop(condition, body)

        assert len(circuit) == 3

    def test_while_loop_with_symbolic_parameters(self):
        """测试使用符号参数的 while 循环"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        theta = Parameter("theta")
        phi = Parameter("phi")

        condition = ConditionView(Qubit(0), 1)
        body = [(StandardGate.RX, [1], [theta]), (StandardGate.RY, [1], [phi])]
        circuit.while_loop(condition, body)

        assert len(circuit) == 3
        # 验证参数已添加到电路
        assert len(circuit.parameters) == 2


class TestControlFlowWithQubitList:
    """测试使用 Qubit 对象列表的控制流"""

    def test_if_else_with_qubit_list(self):
        """测试使用 Qubit 对象列表的 if-else"""
        circuit = Circuit(3)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        # 使用 Qubit 对象列表
        true_body = [(StandardGate.CX, [Qubit(1), Qubit(2)], None)]
        circuit.if_else(condition, true_body)

        assert len(circuit) == 3

    def test_while_loop_with_qubit_list(self):
        """测试使用 Qubit 对象列表的 while 循环"""
        circuit = Circuit(3)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        body = [(StandardGate.SWAP, [Qubit(1), Qubit(2)], None)]
        circuit.while_loop(condition, body)

        assert len(circuit) == 3


class TestControlFlowComplex:
    """测试复杂的控制流场景"""

    def test_multiple_control_flows(self):
        """测试多个控制流语句"""
        circuit = Circuit(3)

        # 初始化
        circuit.h(0)
        circuit.measure(0)

        # 第一个 if-else
        condition1 = ConditionView(Qubit(0), 1)
        circuit.if_else(condition1, [(StandardGate.X, [1], None)])

        # 第二个 if-else
        condition2 = ConditionView(Qubit(0), 0)
        circuit.if_else(condition2, [(StandardGate.Z, [2], None)])

        assert len(circuit) == 4  # h, measure, if_else, if_else

    def test_control_flow_with_various_gates(self):
        """测试使用不同类型门的控制流"""
        circuit = Circuit(3)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)

        # 使用各种门类型
        true_body = [
            (StandardGate.H, [1], None),
            (StandardGate.S, [1], None),
            (StandardGate.T, [1], None),
            (StandardGate.CX, [1, 2], None),
        ]
        circuit.if_else(condition, true_body)

        assert len(circuit) == 3

    def test_control_flow_with_mc_gate(self):
        """测试使用多控制门的控制流"""
        from cqlib.circuit.gates import McGate

        circuit = Circuit(4)
        circuit.x(0)
        circuit.measure(0)

        # 创建一个 Toffoli 门 (CCX)
        ccx = McGate(2, StandardGate.X)

        condition = ConditionView(Qubit(0), 1)
        true_body = [(ccx, [1, 2, 3], None)]  # 两个控制位，一个目标位
        circuit.if_else(condition, true_body)

        assert len(circuit) == 3


class TestConditionView:
    """测试 ConditionView 功能"""

    def test_condition_view_creation(self):
        """测试 ConditionView 创建"""
        condition = ConditionView(Qubit(0), 1)
        assert condition.qubit.index == 0
        assert condition.target == 1

    def test_condition_view_different_targets(self):
        """测试不同 target 值的 ConditionView"""
        condition0 = ConditionView(Qubit(0), 0)
        condition1 = ConditionView(Qubit(0), 1)

        assert condition0.target == 0
        assert condition1.target == 1


class TestControlFlowErrors:
    """测试控制流的错误处理"""

    def test_if_else_invalid_qubit_index(self):
        """测试使用无效 qubit 索引的 if-else"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        # qubit 索引 5 超出了电路范围
        with pytest.raises(Exception):
            circuit.if_else(condition, [(StandardGate.X, [5], None)])

    def test_while_loop_invalid_qubit_index(self):
        """测试使用无效 qubit 索引的 while 循环"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        with pytest.raises(Exception):
            circuit.while_loop(condition, [(StandardGate.X, [5], None)])

    def test_if_else_empty_body(self):
        """测试空 body 的 if-else"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        # 空 body 应该被允许
        circuit.if_else(condition, [])

        assert len(circuit) == 3

    def test_while_loop_empty_body(self):
        """测试空 body 的 while 循环"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        # 空 body 应该被允许
        circuit.while_loop(condition, [])

        assert len(circuit) == 3


class TestControlFlowDecomposition:
    """测试控制流的分解功能"""

    def test_if_else_decompose(self):
        """测试包含 if-else 的电路分解"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        circuit.if_else(condition, [(StandardGate.X, [1], None)])

        # 分解应该保持控制流结构
        decomposed = circuit.decompose()
        assert decomposed is not None

    def test_while_loop_decompose(self):
        """测试包含 while 循环的电路分解"""
        circuit = Circuit(2)
        circuit.x(0)
        circuit.measure(0)

        condition = ConditionView(Qubit(0), 1)
        circuit.while_loop(condition, [(StandardGate.H, [1], None)])

        decomposed = circuit.decompose()
        assert decomposed is not None
