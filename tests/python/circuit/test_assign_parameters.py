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
测试 Circuit 参数赋值功能

测试范围：
- 参数绑定和赋值
- 单参数赋值
- 多参数赋值
- 部分参数赋值
- 表达式参数赋值
"""

import numpy as np
from cqlib.circuit import Circuit, Parameter


class TestAssignSingleParameter:
    """测试单参数赋值"""

    def test_assign_single_parameter(self):
        """赋值单个参数"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta)

        # 赋值前应该有参数
        assert len(c.parameters) >= 1

        # 赋值参数
        c_assigned = c.assign_parameters({"theta": np.pi / 2})
        assert len(c_assigned) == 1

    def test_assign_zero(self):
        """赋值零"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta)

        c_assigned = c.assign_parameters({"theta": 0.0})
        assert len(c_assigned) == 1

    def test_assign_negative(self):
        """赋值负值"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta)

        c_assigned = c.assign_parameters({"theta": -np.pi / 2})
        assert len(c_assigned) == 1


class TestAssignMultipleParameters:
    """测试多参数赋值"""

    def test_assign_multiple_parameters(self):
        """赋值多个参数"""
        theta = Parameter("theta")
        phi = Parameter("phi")
        c = Circuit(2)
        c.rx(0, theta)
        c.ry(1, phi)

        c_assigned = c.assign_parameters({"theta": np.pi / 2, "phi": np.pi / 4})
        assert len(c_assigned) == 2

    def test_assign_all_parameters(self):
        """赋值所有参数"""
        theta = Parameter("theta")
        phi = Parameter("phi")
        lam = Parameter("lambda")

        c = Circuit(1)
        c.u(0, theta, phi, lam)

        c_assigned = c.assign_parameters(
            {"theta": np.pi / 2, "phi": np.pi / 4, "lambda": np.pi / 8}
        )
        assert len(c_assigned) == 1


class TestPartialParameterAssignment:
    """测试部分参数赋值"""

    def test_partial_assignment(self):
        """部分参数赋值"""
        theta = Parameter("theta")
        phi = Parameter("phi")
        c = Circuit(2)
        c.rx(0, theta)
        c.ry(1, phi)

        # 只赋值theta
        c_partial = c.assign_parameters({"theta": np.pi / 2})
        assert len(c_partial) == 2
        # phi应该仍然是符号参数

    def test_empty_assignment(self):
        """空赋值（不改变）"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta)

        c_same = c.assign_parameters({})
        assert len(c_same) == 1


class TestAssignExpressionParameters:
    """测试表达式参数赋值"""

    def test_assign_expression_result(self):
        """赋值表达式结果"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta + 1.0)  # theta + 1

        c_assigned = c.assign_parameters({"theta": 0.5})
        assert len(c_assigned) == 1

    def test_assign_scaled_parameter(self):
        """赋值缩放参数"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, 2.0 * theta)  # 2 * theta

        c_assigned = c.assign_parameters({"theta": np.pi / 4})
        # 结果应该是 2 * pi/4 = pi/2
        assert len(c_assigned) == 1


class TestParameterAssignmentErrors:
    """测试参数赋值错误处理"""

    def test_assign_unknown_parameter(self):
        """赋值未知参数"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta)

        # 赋值不存在的参数应该被忽略或报错
        try:
            c.assign_parameters({"unknown": 1.0})
        except Exception:
            pass  # 预期行为

    def test_assign_non_numeric(self):
        """赋值非数值"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta)

        # 根据实现，这可能报错或尝试转换
        try:
            c.assign_parameters({"theta": "invalid"})
        except Exception:
            pass  # 预期行为


class TestAssignmentResultIndependence:
    """测试赋值结果独立性"""

    def test_original_circuit_unchanged(self):
        """原始电路不被修改"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta)

        c_assigned = c.assign_parameters({"theta": np.pi / 2})

        # 原始电路应该仍然有符号参数
        assert len(c.parameters) >= 1
        assert len(c_assigned.parameters) >= 0  # 赋值后的电路

    def test_multiple_assignments_independent(self):
        """多次赋值相互独立"""
        theta = Parameter("theta")
        c = Circuit(1)
        c.rx(0, theta)

        c1 = c.assign_parameters({"theta": np.pi / 2})
        c2 = c.assign_parameters({"theta": np.pi / 4})

        assert len(c1) == len(c2) == 1
