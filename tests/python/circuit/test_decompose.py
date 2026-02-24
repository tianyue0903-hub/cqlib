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
测试 Circuit 分解功能

测试范围：
- 电路门分解
- 嵌套电路分解
- 分解后电路验证
"""

import pytest
import numpy as np
from cqlib.circuit import Circuit


class TestCircuitDecomposeBasic:
    """测试基础电路分解"""

    def test_decompose_empty_circuit(self):
        """分解空电路"""
        c = Circuit(2)
        decomposed = c.decompose()
        assert len(decomposed) == 0

    def test_decompose_single_gate(self):
        """分解单门电路"""
        c = Circuit(1)
        c.h(0)
        
        decomposed = c.decompose()
        # Hadamard是基本门，不应改变
        assert len(decomposed) == 1
        assert decomposed[0].name == "H"

    def test_decompose_multiple_gates(self):
        """分解多门电路"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c.x(1)
        
        decomposed = c.decompose()
        # 这些门都是基本门
        assert len(decomposed) == 3


class TestCircuitGateDecompose:
    """测试电路门分解"""

    def test_decompose_circuit_gate(self):
        """分解电路门"""
        # 创建子电路
        sub = Circuit(2)
        sub.h(0)
        sub.cx(0, 1)
        
        # 转换为门
        gate = sub.to_gate("Bell")
        
        # 在主电路中使用
        main = Circuit(4)
        main.circuit_gate(gate, [0, 1])
        
        assert len(main) == 1
        
        # 分解后应该展开为基本门
        decomposed = main.decompose()
        assert len(decomposed) >= 2  # 至少H和CX

    def test_decompose_multiple_circuit_gates(self):
        """分解多个电路门"""
        sub = Circuit(2)
        sub.h(0)
        sub.cx(0, 1)
        gate = sub.to_gate("Bell")
        
        main = Circuit(4)
        main.circuit_gate(gate, [0, 1])
        main.circuit_gate(gate, [2, 3])
        
        assert len(main) == 2
        
        decomposed = main.decompose()
        # 应该展开为4个基本门（2个子电路 x 2个门）
        assert len(decomposed) >= 4

    def test_decompose_mixed_gates(self):
        """分解混合门电路"""
        sub = Circuit(2)
        sub.h(0)
        sub.cx(0, 1)
        gate = sub.to_gate("Bell")
        
        main = Circuit(3)
        main.x(0)
        main.circuit_gate(gate, [0, 1])
        main.y(2)
        
        assert len(main) == 3
        
        decomposed = main.decompose()
        # X, H, CX, Y
        assert len(decomposed) >= 4


class TestNestedCircuitDecompose:
    """测试嵌套电路分解"""

    def test_decompose_nested_circuit_gates(self):
        """分解嵌套电路门"""
        # 最底层电路
        inner = Circuit(2)
        inner.h(0)
        inner.cx(0, 1)
        inner_gate = inner.to_gate("Inner")
        
        # 中层电路
        middle = Circuit(2)
        middle.circuit_gate(inner_gate, [0, 1])
        middle.x(0)
        middle_gate = middle.to_gate("Middle")
        
        # 顶层电路
        outer = Circuit(2)
        outer.circuit_gate(middle_gate, [0, 1])
        
        assert len(outer) == 1
        
        # 分解应该展开所有层级
        decomposed = outer.decompose()
        # 应该展开为 H, CX, X
        assert len(decomposed) >= 3


class TestDecomposeWithParameters:
    """测试带参数的电路分解"""

    def test_decompose_parametric_circuit_gate(self):
        """分解参数化电路门"""
        from cqlib.circuit import Parameter
        
        theta = Parameter("theta")
        sub = Circuit(1)
        sub.rx(0, theta)
        gate = sub.to_gate("RxGate")
        
        main = Circuit(2)
        main.circuit_gate(gate, [0])
        main.circuit_gate(gate, [1])
        
        decomposed = main.decompose()
        # 应该保持参数
        assert len(decomposed) == 2


class TestDecomposeIdempotency:
    """测试分解幂等性"""

    def test_decompose_twice(self):
        """两次分解结果相同"""
        sub = Circuit(2)
        sub.h(0)
        sub.cx(0, 1)
        gate = sub.to_gate("Bell")
        
        main = Circuit(2)
        main.circuit_gate(gate, [0, 1])
        
        decomposed1 = main.decompose()
        decomposed2 = decomposed1.decompose()
        
        # 两次分解应该产生相同结果
        assert len(decomposed1) == len(decomposed2)


class TestDecomposePreservesOrder:
    """测试分解保持顺序"""

    def test_decompose_preserves_operation_order(self):
        """分解保持操作顺序"""
        sub1 = Circuit(1)
        sub1.x(0)
        gate1 = sub1.to_gate("XGate")
        
        sub2 = Circuit(1)
        sub2.h(0)
        gate2 = sub2.to_gate("HGate")
        
        main = Circuit(1)
        main.circuit_gate(gate1, [0])
        main.circuit_gate(gate2, [0])
        
        decomposed = main.decompose()
        
        # 顺序应该是 X, H
        assert decomposed[0].name == "X"
        assert decomposed[1].name == "H"


class TestDecomposeNonDecomposable:
    """测试不可分解的电路"""

    def test_decompose_standard_gates_only(self):
        """只有标准门的电路分解"""
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c.s(1)
        c.tdg(0)
        
        decomposed = c.decompose()
        # 应该保持不变
        assert len(decomposed) == 4
        names = [op.name for op in decomposed]
        assert names == ["H", "CX", "S", "TDG"]

    def test_decompose_with_barriers(self):
        """带Barrier的电路分解"""
        sub = Circuit(2)
        sub.h(0)
        sub.cx(0, 1)
        gate = sub.to_gate("Bell")
        
        main = Circuit(2)
        main.circuit_gate(gate, [0, 1])
        main.barrier([0, 1])
        main.x(0)
        
        decomposed = main.decompose()
        # Barrier应该保留
        barrier_found = any("Barrier" in op.name for op in decomposed)
        assert barrier_found
