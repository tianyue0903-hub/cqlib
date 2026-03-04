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

"""OpenQASM 2.0 format support for quantum circuits.

This module provides functions to parse and serialize OpenQASM 2.0 programs.
OpenQASM (Open Quantum Assembly Language) is a hardware-agnostic intermediate
representation for quantum circuits.

Functions:
    loads: Parse an OpenQASM 2.0 string into a Circuit.
    load: Parse an OpenQASM 2.0 file into a Circuit.
    dumps: Serialize a Circuit to an OpenQASM 2.0 string.
    dump: Serialize a Circuit to an OpenQASM 2.0 file.

Example:
    >>> from cqlib.ir import qasm2
    >>> from cqlib import Circuit, Qubit
    >>>
    >>> # Parse OpenQASM 2.0
    >>> qasm_code = '''
    ... OPENQASM 2.0;
    ... include "qelib1.inc";
    ... qreg q[2];
    ... h q[0];
    ... cx q[0], q[1];
    ... '''
    >>> circuit = qasm2.loads(qasm_code)
    >>>
    >>> # Serialize to OpenQASM 2.0
    >>> output = qasm2.dumps(circuit)
"""

from .._native import ir as _ir_module

_qasm2_module = _ir_module.qasm2

# Export functions from the qasm2 submodule
dump = _qasm2_module.dump
dumps = _qasm2_module.dumps
load = _qasm2_module.load
loads = _qasm2_module.loads

__all__ = [
    "dump",
    "dumps",
    "load",
    "loads",
]
