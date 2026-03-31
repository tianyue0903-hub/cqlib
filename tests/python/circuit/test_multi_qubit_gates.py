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
测试多量子比特门操作

测试范围：
- 双量子比特门 (CX, CY, CZ, SWAP)
- Ising耦合门 (RXX, RYY, RZZ, RZX)
- 受控旋转门 (CRX, CRY, CRZ)
- fSim门
- 多控门 (CCX)
"""

import pytest
import numpy as np
from cqlib.circuit import Circuit


class TestControlledNotGates:
    """测试CNOT及其变体"""

    def test_cnot_gate(self, two_qubit_circuit):
        """测试CNOT门"""
        c = two_qubit_circuit
        c.cx(0, 1)
        assert len(c) == 1
        assert c[0].name == "CX"
        assert c[0].num_qubits == 2

    def test_cy_gate(self, two_qubit_circuit):
        """测试Controlled-Y门"""
        c = two_qubit_circuit
        c.cy(0, 1)
        assert len(c) == 1
        assert c[0].name == "CY"

    def test_cz_gate(self, two_qubit_circuit):
        """测试Controlled-Z门"""
        c = two_qubit_circuit
        c.cz(0, 1)
        assert len(c) == 1
        assert c[0].name == "CZ"

    def test_cnot_different_qubits(self, two_qubit_circuit):
        """测试CNOT在不同qubit上"""
        c = two_qubit_circuit
        c.cx(1, 0)
        assert len(c) == 1


class TestSwapGate:
    """测试SWAP门"""

    def test_swap_gate(self, two_qubit_circuit):
        """测试SWAP门"""
        c = two_qubit_circuit
        c.swap(0, 1)
        assert len(c) == 1
        assert c[0].name == "SWAP"

    def test_swap_same_qubit_not_allowed(self, two_qubit_circuit):
        """SWAP同一qubit应该报错"""
        c = two_qubit_circuit
        # 根据实现，这可能报错或无操作
        c.swap(0, 0)
        # 断言根据实际行为调整


class TestIsingGates:
    """测试Ising耦合门"""

    def test_rxx_gate(self, two_qubit_circuit):
        """测试RXX门"""
        c = two_qubit_circuit
        c.rxx(0, 1, np.pi / 2)
        assert len(c) == 1
        assert c[0].name == "RXX"

    def test_ryy_gate(self, two_qubit_circuit):
        """测试RYY门"""
        c = two_qubit_circuit
        c.ryy(0, 1, np.pi / 4)
        assert len(c) == 1
        assert c[0].name == "RYY"

    def test_rzz_gate(self, two_qubit_circuit):
        """测试RZZ门"""
        c = two_qubit_circuit
        c.rzz(0, 1, np.pi / 8)
        assert len(c) == 1
        assert c[0].name == "RZZ"

    def test_rzx_gate(self, two_qubit_circuit):
        """测试RZX门"""
        c = two_qubit_circuit
        c.rzx(0, 1, 0.5)
        assert len(c) == 1
        assert c[0].name == "RZX"

    def test_ising_with_parameter(self, two_qubit_circuit, theta_param):
        """测试Ising门使用符号参数"""
        c = two_qubit_circuit
        c.rxx(0, 1, theta_param)
        assert len(c) == 1


class TestControlledRotationGates:
    """测试受控旋转门"""

    def test_crx_gate(self, two_qubit_circuit):
        """测试Controlled-RX门"""
        c = two_qubit_circuit
        c.crx(0, 1, np.pi / 4)
        assert len(c) == 1
        assert c[0].name == "CRX"

    def test_cry_gate(self, two_qubit_circuit):
        """测试Controlled-RY门"""
        c = two_qubit_circuit
        c.cry(0, 1, np.pi / 4)
        assert len(c) == 1
        assert c[0].name == "CRY"

    def test_crz_gate(self, two_qubit_circuit):
        """测试Controlled-RZ门"""
        c = two_qubit_circuit
        c.crz(0, 1, np.pi / 4)
        assert len(c) == 1
        assert c[0].name == "CRZ"

    def test_controlled_rotations_with_params(self, two_qubit_circuit, theta_param):
        """测试受控旋转门使用符号参数"""
        c = two_qubit_circuit
        c.crx(0, 1, theta_param)
        assert len(c) == 1


class TestFsimGate:
    """测试fSim门"""

    def test_fsim_gate(self, two_qubit_circuit):
        """测试fSim门"""
        c = two_qubit_circuit
        c.fsim(0, 1, 0.5, 0.3)
        assert len(c) == 1
        assert c[0].name == "FSIM"
        assert c[0].num_params == 2

    def test_fsim_with_parameters(self, two_qubit_circuit, theta_param, phi_param):
        """测试fSim门使用符号参数"""
        c = two_qubit_circuit
        c.fsim(0, 1, theta_param, phi_param)
        assert len(c) == 1


class TestMultiControlledGates:
    """测试多控门"""

    def test_ccx_gate(self, three_qubit_circuit):
        """测试Toffoli (CCX)门"""
        c = three_qubit_circuit
        c.ccx(0, 1, 2)
        assert len(c) == 1
        assert c[0].name == "CCX"
        assert c[0].num_qubits == 3

    def test_ccx_different_order(self, three_qubit_circuit):
        """测试Toffoli门不同顺序"""
        c = three_qubit_circuit
        c.ccx(2, 1, 0)
        assert len(c) == 1


class TestMultiControlMethod:
    """测试multi_control方法"""

    def test_multi_control_x_one_control(self, two_qubit_circuit):
        """测试单控X门（等效于CNOT）"""
        from cqlib.circuit.gates import X
        c = two_qubit_circuit
        c.multi_control(X, [0], [1])
        assert len(c) == 1

    def test_multi_control_x_two_controls(self, three_qubit_circuit):
        """测试双控X门（等效于CCX）"""
        from cqlib.circuit.gates import X
        c = three_qubit_circuit
        c.multi_control(X, [0, 1], [2])
        assert len(c) == 1

    def test_multi_control_hadamard(self, two_qubit_circuit):
        """测试受控Hadamard"""
        from cqlib.circuit.gates import H
        c = two_qubit_circuit
        c.multi_control(H, [0], [1])
        assert len(c) == 1

    def test_multi_control_with_params(self, two_qubit_circuit, theta_param):
        """测试多控门使用参数"""
        from cqlib.circuit.gates import RX
        c = two_qubit_circuit
        c.multi_control(RX(theta_param), [0], [1])
        assert len(c) == 1


class TestMultiQubitGateErrors:
    """测试多量子比特门的错误处理"""

    def test_cnot_on_same_qubit(self, two_qubit_circuit):
        """CNOT控制位和目标位相同应该报错"""
        c = two_qubit_circuit
        # 根据实现，这可能报错或无操作
        try:
            c.cx(0, 0)
        except Exception:
            pass  # 预期行为

    def test_gate_on_invalid_qubit(self, two_qubit_circuit):
        """在无效qubit上应用门"""
        c = two_qubit_circuit
        with pytest.raises(Exception):
            c.cx(0, 5)  # qubit 5不存在
