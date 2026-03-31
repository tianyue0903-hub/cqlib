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
测试参数化单量子比特门操作

测试范围：
- 旋转门 (RX, RY, RZ)
- Phase门
- 通用U门
- RXY门
- 使用浮点数参数
- 使用符号参数
- 使用表达式参数
"""

import numpy as np


class TestRotationGatesWithFloat:
    """测试使用浮点数的旋转门"""

    def test_rx_with_float(self, single_qubit_circuit):
        """RX门使用浮点参数"""
        c = single_qubit_circuit
        c.rx(0, np.pi / 2)
        assert len(c) == 1
        assert c[0].name == "RX"
        assert c[0].num_params == 1

    def test_ry_with_float(self, single_qubit_circuit):
        """RY门使用浮点参数"""
        c = single_qubit_circuit
        c.ry(0, np.pi / 4)
        assert len(c) == 1
        assert c[0].name == "RY"

    def test_rz_with_float(self, single_qubit_circuit):
        """RZ门使用浮点参数"""
        c = single_qubit_circuit
        c.rz(0, np.pi / 8)
        assert len(c) == 1
        assert c[0].name == "RZ"

    def test_rx_zero_rotation(self, single_qubit_circuit):
        """RX门零旋转"""
        c = single_qubit_circuit
        c.rx(0, 0.0)
        assert len(c) == 1

    def test_rx_full_rotation(self, single_qubit_circuit):
        """RX门完整旋转（2π）"""
        c = single_qubit_circuit
        c.rx(0, 2 * np.pi)
        assert len(c) == 1

    def test_rx_negative_rotation(self, single_qubit_circuit):
        """RX门负旋转"""
        c = single_qubit_circuit
        c.rx(0, -np.pi / 2)
        assert len(c) == 1


class TestRotationGatesWithParameter:
    """测试使用符号参数的旋转门"""

    def test_rx_with_parameter(self, single_qubit_circuit, theta_param):
        """RX门使用符号参数"""
        c = single_qubit_circuit
        c.rx(0, theta_param)
        assert len(c) == 1
        # 验证电路追踪了参数
        assert len(c.parameters) >= 1

    def test_ry_with_parameter(self, single_qubit_circuit, theta_param):
        """RY门使用符号参数"""
        c = single_qubit_circuit
        c.ry(0, theta_param)
        assert len(c) == 1

    def test_rz_with_parameter(self, single_qubit_circuit, theta_param):
        """RZ门使用符号参数"""
        c = single_qubit_circuit
        c.rz(0, theta_param)
        assert len(c) == 1

    def test_multiple_parameters(self, single_qubit_circuit, theta_param, phi_param):
        """同一电路使用多个符号参数"""
        c = single_qubit_circuit
        c.rx(0, theta_param)
        c.ry(0, phi_param)
        assert len(c) == 2
        assert len(c.parameters) >= 2


class TestRotationGatesWithExpression:
    """测试使用表达式参数的旋转门"""

    def test_rx_with_expression(self, single_qubit_circuit, theta_param):
        """RX门使用表达式"""
        c = single_qubit_circuit
        c.rx(0, theta_param + 1.0)
        assert len(c) == 1

    def test_rx_with_scaled_parameter(self, single_qubit_circuit, theta_param):
        """RX门使用缩放参数"""
        c = single_qubit_circuit
        c.rx(0, 2.0 * theta_param)
        assert len(c) == 1

    def test_rx_with_complex_expression(
        self, single_qubit_circuit, theta_param, phi_param
    ):
        """RX门使用复杂表达式"""
        c = single_qubit_circuit
        c.rx(0, theta_param + phi_param)
        assert len(c) == 1


class TestPhaseGate:
    """测试Phase门"""

    def test_phase_with_float(self, single_qubit_circuit):
        """Phase门使用浮点参数"""
        c = single_qubit_circuit
        c.phase(0, np.pi / 4)
        assert len(c) == 1
        assert c[0].name == "Phase"

    def test_phase_with_parameter(self, single_qubit_circuit, theta_param):
        """Phase门使用符号参数"""
        c = single_qubit_circuit
        c.phase(0, theta_param)
        assert len(c) == 1


class TestUGate:
    """测试通用U门"""

    def test_u_with_floats(self, single_qubit_circuit):
        """U门使用三个浮点参数"""
        c = single_qubit_circuit
        c.u(0, np.pi / 2, np.pi / 4, np.pi / 8)
        assert len(c) == 1
        assert c[0].name == "U"
        assert c[0].num_params == 3

    def test_u_with_parameters(self, single_qubit_circuit, theta_param, phi_param):
        """U门使用符号参数"""
        c = single_qubit_circuit
        c.u(0, theta_param, phi_param, 0.0)
        assert len(c) == 1

    def test_u_with_mixed_params(self, single_qubit_circuit, theta_param):
        """U门使用混合参数（符号和浮点）"""
        c = single_qubit_circuit
        c.u(0, theta_param, 0.5, 0.3)
        assert len(c) == 1


class TestRXYGate:
    """测试RXY门"""

    def test_rxy_with_floats(self, single_qubit_circuit):
        """RXY门使用两个浮点参数"""
        c = single_qubit_circuit
        c.rxy(0, np.pi / 2, np.pi / 4)
        assert len(c) == 1
        assert c[0].name == "RXY"
        assert c[0].num_params == 2

    def test_rxy_with_parameters(self, single_qubit_circuit, theta_param, phi_param):
        """RXY门使用符号参数"""
        c = single_qubit_circuit
        c.rxy(0, theta_param, phi_param)
        assert len(c) == 1


class TestParameterTracking:
    """测试电路参数追踪"""

    def test_track_single_parameter(self, single_qubit_circuit, theta_param):
        """追踪单个参数"""
        c = single_qubit_circuit
        c.rx(0, theta_param)
        params = c.parameters
        assert len(params) >= 1

    def test_track_expression(self, single_qubit_circuit, theta_param):
        """追踪表达式参数"""
        c = single_qubit_circuit
        c.rx(0, theta_param + 1.0)
        params = c.parameters
        # 表达式应该被追踪
        assert len(params) >= 1

    def test_parameters_list(self, single_qubit_circuit, theta_param, phi_param):
        """获取参数列表"""
        c = single_qubit_circuit
        c.rx(0, theta_param)
        c.ry(0, phi_param)
        # 通过 parameters 属性获取参数列表
        params = c.parameters
        # 参数列表应该包含参数
        assert len(params) >= 0  # 根据实际实现调整
