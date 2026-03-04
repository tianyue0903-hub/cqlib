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

"""QCIS format support for quantum circuits.

QCIS (Quantum Circuit Intermediate Representation) is Telecom Quantum's native
quantum circuit format. Each line represents a gate operation:

    GATE_NAME QUBIT_LIST [PARAMETER_LIST]

Supported Gates:
    - Native gates: X2P, X2M, Y2P, Y2M, XY2P, XY2M, CZ, RZ, I
    - Standard gates: X, Y, Z, H, S, SD, T, TD
    - Parameterized gates: RX, RY, RXY
    - Directives: B (Barrier), M (Measurement)

Functions:
    loads: Parse a QCIS string into a Circuit.
    load: Parse a QCIS file into a Circuit.
    dumps: Serialize a Circuit to a QCIS string.
    dump: Serialize a Circuit to a QCIS file.

Example:
    >>> from cqlib.ir import qcis
    >>> from cqlib import Circuit, Qubit
    >>>
    >>> # Parse QCIS
    >>> qcis_code = '''
    ... H Q0
    ... CZ Q0 Q1
    ... RZ Q0 3.14159
    ... M Q0 Q1
    ... '''
    >>> circuit = qcis.loads(qcis_code)
    >>>
    >>> # Serialize to QCIS
    >>> output = qcis.dumps(circuit)
"""

from .._native import ir as _ir_module

_qcis_module = _ir_module.qcis

# Export functions from the qcis submodule
dump = _qcis_module.dump
dumps = _qcis_module.dumps
load = _qcis_module.load
loads = _qcis_module.loads

__all__ = [
    "dump",
    "dumps",
    "load",
    "loads",
]
