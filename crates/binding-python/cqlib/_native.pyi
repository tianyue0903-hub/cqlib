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
Native extension module for Cqlib.

This module provides Rust-implemented quantum circuit functionality exposed to Python via PyO3.
Type stubs are organized in submodules mirroring the Rust source structure.
"""

# Re-export all public types from circuit submodules
from .circuit.bit import Qubit
from .circuit.circuit import Circuit
from .circuit.parameter import Parameter
from .circuit.gates.standard import StandardGate
from .circuit.gates.unitary import UnitaryGate

__all__ = [
    "Qubit",
    "Circuit",
    "Parameter",
    "StandardGate",
    "UnitaryGate",
]
