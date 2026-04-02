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
测试非酉操作

测试范围：
- 测量操作 (measure)
- Barrier操作
- Reset操作
- Delay操作
"""

import pytest


class TestMeasureOperation:
    """测试测量操作"""

    def test_measure_single_qubit(self, single_qubit_circuit):
        """测量单个qubit"""
        c = single_qubit_circuit
        c.h(0)
        c.measure(0)
        assert len(c) == 2
        assert c[1].name == "Measure"

    def test_measure_multiple_qubits(self, two_qubit_circuit):
        """测量多个qubit"""
        c = two_qubit_circuit
        c.h(0)
        c.cx(0, 1)
        c.measure(0)
        c.measure(1)
        assert len(c) == 4

    def test_measure_after_gates(self, single_qubit_circuit):
        """在一系列门之后测量"""
        c = single_qubit_circuit
        c.x(0)
        c.y(0)
        c.z(0)
        c.measure(0)
        assert len(c) == 4
        assert c[3].name == "Measure"

    def test_measure_makes_circuit_non_unitary(self, single_qubit_circuit):
        """测量使电路非幺正"""
        c = single_qubit_circuit
        c.h(0)
        c.measure(0)
        # 包含测量的电路不能求逆
        with pytest.raises(Exception):
            c.inverse()


class TestBarrierOperation:
    """测试Barrier操作"""

    def test_barrier_single_qubit(self, single_qubit_circuit):
        """在单个qubit上设置Barrier"""
        c = single_qubit_circuit
        c.h(0)
        c.barrier([0])
        assert len(c) == 2
        assert c[1].name == "Barrier"

    def test_barrier_multiple_qubits(self, two_qubit_circuit):
        """在多个qubit上设置Barrier"""
        c = two_qubit_circuit
        c.h(0)
        c.barrier([0, 1])
        c.cx(0, 1)
        assert len(c) == 3
        assert c[1].name == "Barrier"

    def test_barrier_persists_in_inverse(self, two_qubit_circuit):
        """Barrier在逆电路中保留"""
        c = two_qubit_circuit
        c.h(0)
        c.barrier([0, 1])
        c.cx(0, 1)

        c_inv = c.inverse()
        assert len(c_inv) == 3
        # Barrier应该在逆电路中
        barrier_found = any(op.name == "Barrier" for op in c_inv)
        assert barrier_found

    def test_empty_barrier(self, two_qubit_circuit):
        """空Barrier列表"""
        c = two_qubit_circuit
        c.barrier([])
        # 根据实现，这可能添加一个空barrier或无操作


class TestResetOperation:
    """测试Reset操作"""

    def test_reset_single_qubit(self, single_qubit_circuit):
        """重置单个qubit"""
        c = single_qubit_circuit
        c.x(0)
        c.reset(0)
        assert len(c) == 2
        assert c[1].name == "Reset"

    def test_reset_after_measure(self, single_qubit_circuit):
        """测量后重置"""
        c = single_qubit_circuit
        c.h(0)
        c.measure(0)
        c.reset(0)
        assert len(c) == 3

    def test_reset_makes_circuit_non_unitary(self, single_qubit_circuit):
        """Reset使电路非幺正"""
        c = single_qubit_circuit
        c.reset(0)
        # 包含Reset的电路不能求逆
        with pytest.raises(Exception):
            c.inverse()


class TestDelayOperation:
    """测试Delay操作"""

    def test_delay_with_float(self, single_qubit_circuit):
        """使用浮点数延迟"""
        c = single_qubit_circuit
        c.delay(0, 100.0)
        assert len(c) == 1

    def test_delay_with_parameter(self, single_qubit_circuit, theta_param):
        """使用符号参数延迟"""
        c = single_qubit_circuit
        c.delay(0, theta_param)
        assert len(c) == 1

    def test_delay_between_gates(self, single_qubit_circuit):
        """在门之间插入延迟"""
        c = single_qubit_circuit
        c.x(0)
        c.delay(0, 10.0)
        c.y(0)
        assert len(c) == 3


class TestNonUnitaryCombination:
    """测试非酉操作组合"""

    def test_measure_and_reset(self, single_qubit_circuit):
        """测量后重置"""
        c = single_qubit_circuit
        c.h(0)
        c.measure(0)
        c.reset(0)
        c.x(0)
        assert len(c) == 4

    def test_barrier_with_non_unitary(self, two_qubit_circuit):
        """Barrier与非酉操作组合"""
        c = two_qubit_circuit
        c.h(0)
        c.barrier([0, 1])
        c.measure(0)
        c.barrier([0, 1])
        c.reset(1)
        assert len(c) == 5
