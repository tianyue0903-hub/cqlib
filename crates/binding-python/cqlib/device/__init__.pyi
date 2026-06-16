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
Quantum device characterization for noise-aware compilation.

This subpackage provides:

- **Device modeling**: :class:`Device`, :class:`Topology`, :class:`Layout`
- **Qubit properties**: :class:`QubitProp`, :class:`InstructionProp`,
  :class:`EdgeProp`
- **Qubit identifiers**: :class:`LogicalQubit`, :class:`PhysicalQubit`
- **Noise models**: :class:`NoiseModel`, :class:`SingleQubitNoise`,
  :class:`TwoQubitNoise`, :class:`ReadoutError`, :class:`OperationKey`
- **Execution results**: :class:`Outcome`, :class:`Status`,
  :class:`ExecutionResult`

Key Usage — device construction::

    >>> from cqlib.device import Device, Topology, QubitProp
    >>>
    >>> # Quick line topology
    >>> dev = Device.line("my_device", num_qubits=5)
    >>>
    >>> # Or explicit construction
    >>> topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CX")])
    >>> dev = Device("my_device", [0, 1, 2], topo)
    >>> dev.default_t1 = 100.0
    >>> dev.default_t2 = 50.0

Key Usage — layout and routing::

    >>> from cqlib.device import Layout
    >>>
    >>> layout = Layout([0, 1], [100, 101, 102])
    >>> layout.swap_physical(100, 101)  # SWAP routing step
"""

from .device_impl import InstructionProp, QubitProp, EdgeProp, Device
from .topology import Topology
from .layout import Layout
from .qubit import LogicalQubit, PhysicalQubit
from .noise import (
    SingleQubitNoise,
    TwoQubitNoise,
    ReadoutError,
    OperationKey,
    NoiseModel,
)
from .result import Outcome, Status, ExecutionResult

__all__ = [
    "Topology",
    "InstructionProp",
    "QubitProp",
    "EdgeProp",
    "Device",
    "Layout",
    "LogicalQubit",
    "PhysicalQubit",
    "SingleQubitNoise",
    "TwoQubitNoise",
    "ReadoutError",
    "OperationKey",
    "NoiseModel",
    "Outcome",
    "Status",
    "ExecutionResult",
]
