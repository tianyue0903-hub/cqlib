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

"""Intermediate Representation (IR) module for quantum circuit formats.

This module provides functions to parse and serialize quantum circuits in
various formats including OpenQASM 2.0 and QCIS.

Supported Formats:
    - OpenQASM 2.0: IBM's quantum assembly language
    - QCIS: Telecom Quantum's native circuit format

Submodules:
    qasm2: OpenQASM 2.0 format support
    qcis: QCIS format support

Example:
    >>> from cqlib.ir import qasm2, qcis
    >>> from cqlib import Circuit
    >>>
    >>> # Parse OpenQASM 2.0
    >>> circuit = qasm2.loads('OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0];')
    >>>
    >>> # Convert to QCIS
    >>> qcis_str = qcis.dumps(circuit)
    >>> print(qcis_str)
"""

from . import qasm2
from . import qcis

__all__ = ["qasm2", "qcis"]
