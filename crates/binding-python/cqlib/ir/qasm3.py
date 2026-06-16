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

"""OpenQASM 3.0 format support for quantum circuits.

This module provides functions to parse and serialize OpenQASM 3.0 programs.
OpenQASM 3.0 is a quantum assembly language for circuit descriptions with
classical declarations, control flow, and standard-library gates.

Functions:
    loads: Parse an OpenQASM 3.0 string into a Circuit.
    load: Parse an OpenQASM 3.0 file into a Circuit.
    dumps: Serialize a Circuit to an OpenQASM 3.0 string.
    dump: Serialize a Circuit to an OpenQASM 3.0 file.

Example:
    >>> from cqlib.ir import qasm3
    >>>
    >>> qasm_code = '''
    ... OPENQASM 3;
    ... include "stdgates.inc";
    ... qubit[2] q;
    ... h q[0];
    ... cx q[0], q[1];
    ... '''
    >>> circuit = qasm3.loads(qasm_code)
    >>> output = qasm3.dumps(circuit)
"""

from .._native import ir as _ir_module

_qasm3_module = _ir_module.qasm3

# Export functions from the qasm3 submodule
dump = _qasm3_module.dump
dumps = _qasm3_module.dumps
load = _qasm3_module.load
loads = _qasm3_module.loads

__all__ = [
    "dump",
    "dumps",
    "load",
    "loads",
]
