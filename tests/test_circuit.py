import numpy as np

from cqlib.circuit import Circuit, Parameter, UnitaryGate
from cqlib.circuit.gates import H


def test_circuit_basic():
    theta = Parameter("theta")
    c = Circuit([1, 2, 3])
    print(c.qubits)
    c.h(1)
    c.rx(1, theta)

    c.measure(1)
    c.multi_control(H, controls=[1, 3], targets=[2])

    # 测试自定义幺正门
    g = UnitaryGate("gate", 1).with_matrix(np.array([[1, 0], [0, 1]]))
    g = UnitaryGate("gate", 1).with_matrix([[1, 0], [0, 1]])

    # 应用自定义门到电路
    c.unitary(g, [1])

    # 测试两比特自定义门
    g2 = UnitaryGate("two_qubit", 2).with_matrix(np.eye(4))
    c.unitary(g2, [1, 2])

    # No assertion yet, just checking it doesn't crash
