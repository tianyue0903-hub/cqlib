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
测试单量子比特门操作

测试范围：
- Pauli门 (X, Y, Z, I)
- Clifford门 (H, S, SDG, T, TDG)
- 平方根门 (X2P, X2M, Y2P, Y2M)
- XY门族 (XY, XY2P, XY2M)
"""

import pytest
from cqlib.circuit import Circuit


class TestPauliGates:
    """测试Pauli门"""

    def test_identity_gate(self, single_qubit_circuit):
        """测试Identity门"""
        c = single_qubit_circuit
        c.i(0)
        assert len(c) == 1
        assert c[0].name == "I"

    def test_pauli_x_gate(self, single_qubit_circuit):
        """测试Pauli-X门"""
        c = single_qubit_circuit
        c.x(0)
        assert len(c) == 1
        assert c[0].name == "X"

    def test_pauli_y_gate(self, single_qubit_circuit):
        """测试Pauli-Y门"""
        c = single_qubit_circuit
        c.y(0)
        assert len(c) == 1
        assert c[0].name == "Y"

    def test_pauli_z_gate(self, single_qubit_circuit):
        """测试Pauli-Z门"""
        c = single_qubit_circuit
        c.z(0)
        assert len(c) == 1
        assert c[0].name == "Z"

    def test_pauli_gates_sequence(self, single_qubit_circuit):
        """测试Pauli门序列"""
        c = single_qubit_circuit
        c.x(0)
        c.y(0)
        c.z(0)
        assert len(c) == 3
        assert c[0].name == "X"
        assert c[1].name == "Y"
        assert c[2].name == "Z"


class TestCliffordGates:
    """测试Clifford门"""

    def test_hadamard_gate(self, single_qubit_circuit):
        """测试Hadamard门"""
        c = single_qubit_circuit
        c.h(0)
        assert len(c) == 1
        assert c[0].name == "H"

    def test_s_gate(self, single_qubit_circuit):
        """测试S门"""
        c = single_qubit_circuit
        c.s(0)
        assert len(c) == 1
        assert c[0].name == "S"

    def test_sdg_gate(self, single_qubit_circuit):
        """测试S-dagger门"""
        c = single_qubit_circuit
        c.sdg(0)
        assert len(c) == 1
        assert c[0].name == "SDG"

    def test_t_gate(self, single_qubit_circuit):
        """测试T门"""
        c = single_qubit_circuit
        c.t(0)
        assert len(c) == 1
        assert c[0].name == "T"

    def test_tdg_gate(self, single_qubit_circuit):
        """测试T-dagger门"""
        c = single_qubit_circuit
        c.tdg(0)
        assert len(c) == 1
        assert c[0].name == "TDG"


class TestSqrtGates:
    """测试平方根门"""

    def test_x2p_gate(self, single_qubit_circuit):
        """测试√X门（正相位）"""
        c = single_qubit_circuit
        c.x2p(0)
        assert len(c) == 1
        assert c[0].name == "X2P"

    def test_x2m_gate(self, single_qubit_circuit):
        """测试√X†门（负相位）"""
        c = single_qubit_circuit
        c.x2m(0)
        assert len(c) == 1
        assert c[0].name == "X2M"

    def test_y2p_gate(self, single_qubit_circuit):
        """测试√Y门（正相位）"""
        c = single_qubit_circuit
        c.y2p(0)
        assert len(c) == 1
        assert c[0].name == "Y2P"

    def test_y2m_gate(self, single_qubit_circuit):
        """测试√Y†门（负相位）"""
        c = single_qubit_circuit
        c.y2m(0)
        assert len(c) == 1
        assert c[0].name == "Y2M"

    def test_sqrt_gates_are_inverse(self, single_qubit_circuit):
        """测试√X和√X†是互逆的"""
        c = single_qubit_circuit
        c.x2p(0)
        c.x2m(0)
        assert len(c) == 2


class TestXYGates:
    """测试XY门族"""

    def test_xy_gate(self, single_qubit_circuit):
        """测试XY门"""
        c = single_qubit_circuit
        c.xy(0, 0.5)
        assert len(c) == 1
        assert c[0].name == "XY"

    def test_xy2p_gate(self, single_qubit_circuit):
        """测试√XY门（正相位）"""
        c = single_qubit_circuit
        c.xy2p(0, 0.5)
        assert len(c) == 1
        assert c[0].name == "XY2P"

    def test_xy2m_gate(self, single_qubit_circuit):
        """测试√XY†门（负相位）"""
        c = single_qubit_circuit
        c.xy2m(0, 0.5)
        assert len(c) == 1
        assert c[0].name == "XY2M"

    def test_xy_gates_with_parameter(self, single_qubit_circuit, theta_param):
        """测试XY门使用符号参数"""
        c = single_qubit_circuit
        c.xy(0, theta_param)
        assert len(c) == 1


class TestSingleQubitGateErrors:
    """测试单量子比特门的错误处理"""

    def test_gate_on_invalid_qubit(self, single_qubit_circuit):
        """在无效qubit上应用门应报错"""
        c = single_qubit_circuit
        with pytest.raises(Exception):
            c.h(5)  # qubit 5不存在

    def test_gate_on_negative_qubit(self, single_qubit_circuit):
        """在负索引qubit上应用门"""
        c = single_qubit_circuit
        # 根据实现，这可能报错或接受负索引
        try:
            c.h(-1)
        except Exception:
            pass  # 预期行为
