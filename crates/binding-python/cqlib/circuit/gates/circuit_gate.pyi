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

from typing_extensions import final


@final
class CircuitGate:
    """A quantum gate defined by a quantum circuit.
    
    CircuitGate allows you to define custom gates by wrapping a Circuit object.
    The gate can have symbolic parameters that are mapped to the internal circuit's
    parameters when the gate is applied.
    """
    pass
