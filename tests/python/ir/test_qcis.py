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
测试 QCIS IR 导入导出

测试范围：
- QCIS字符串解析（loads）
- QCIS文件加载（load）
- QCIS字符串生成（dumps）
- QCIS文件保存（dump）

注意：QCIS 只支持特定的门集合，不包括 CNOT/CX
"""

import pytest
import tempfile
import os
from cqlib.circuit import Circuit
from cqlib.ir.qcis import loads, load, dumps, dump


class TestQcisLoads:
    """测试QCIS解析"""

    def test_loads_empty_circuit(self):
        """解析空电路"""
        c = loads("")
        assert len(c) == 0

    def test_loads_single_qubit_gates(self):
        """解析单量子比特门"""
        qcis = """
H Q0
X Q1
Y Q2
Z Q3
"""
        c = loads(qcis)
        assert len(c) == 4

    def test_loads_hadamard(self):
        """解析Hadamard"""
        qcis = "H Q0"
        c = loads(qcis)
        assert len(c) == 1

    def test_loads_cz_gate(self):
        """解析CZ门（QCIS支持）"""
        qcis = """
H Q0
CZ Q0 Q1
"""
        c = loads(qcis)
        assert len(c) == 2

    def test_loads_parametric(self):
        """解析参数化门"""
        qcis = """
RX Q0 0.5
RY Q1 0.3
RZ Q2 0.2
"""
        c = loads(qcis)
        assert len(c) == 3


class TestQcisDumps:
    """测试QCIS生成"""

    def test_dumps_empty_circuit(self):
        """生成空电路"""
        c = Circuit(1)
        qcis = dumps(c)
        assert isinstance(qcis, str)

    def test_dumps_single_qubit(self):
        """生成单量子比特电路"""
        c = Circuit(2)
        c.h(0)
        c.x(1)
        qcis = dumps(c)
        assert isinstance(qcis, str)
        assert len(qcis) > 0

    def test_dumps_cz_circuit(self):
        """生成含CZ的电路"""
        c = Circuit(2)
        c.h(0)
        c.cz(0, 1)
        qcis = dumps(c)
        assert isinstance(qcis, str)


class TestQcisFileIO:
    """测试QCIS文件IO"""

    def test_load_from_file(self):
        """从文件加载"""
        qcis_content = """
H Q0
CZ Q0 Q1
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qcis', delete=False) as f:
            f.write(qcis_content)
            temp_path = f.name

        try:
            c = load(temp_path)
            assert len(c) == 2
        finally:
            os.unlink(temp_path)

    def test_dump_to_file(self):
        """保存到文件"""
        c = Circuit(2)
        c.h(0)
        c.cz(0, 1)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.qcis', delete=False) as f:
            temp_path = f.name

        try:
            dump(c, temp_path)
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)
