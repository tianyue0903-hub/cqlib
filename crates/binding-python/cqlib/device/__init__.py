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

- **Device modeling**: Device, Topology, Layout
- **Qubit properties**: QubitProp, InstructionProp, EdgeProp
- **Qubit identifiers**: LogicalQubit, PhysicalQubit
- **Noise models**: NoiseModel, SingleQubitNoise, TwoQubitNoise, ReadoutError, OperationKey
- **Execution results**: Outcome, Status, ExecutionResult

Key Usage — device construction::

    >>> from cqlib.device import Device, Topology, QubitProp
    >>>
    >>> # Quick line topology
    >>> dev = Device.line("my_device", num_qubits=5)
    >>>
    >>> # Explicit construction
    >>> topo = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CX")])
    >>> dev = Device("my_device", [0, 1, 2], topo)
    >>> dev.default_t1 = 100.0
    >>> dev.default_t2 = 50.0

Key Usage — layout and routing::

    >>> from cqlib.device import Layout, LogicalQubit, PhysicalQubit
    >>>
    >>> layout = Layout([0, 1], [100, 101, 102])
    >>> layout.swap_physical(100, 101)

Key Usage — noise model::

    >>> from cqlib.device import NoiseModel, SingleQubitNoise, ReadoutError
    >>> from cqlib.circuit import StandardGate
    >>>
    >>> model = NoiseModel()
    >>> model.add_readout_error(0, ReadoutError(0.02, 0.01))
    >>> model.add_single_qubit_error(StandardGate.H, 0, SingleQubitNoise.depolarizing(0.001))
"""

from .._native import device as _device_module

Topology = _device_module.Topology
InstructionProp = _device_module.InstructionProp
QubitProp = _device_module.QubitProp
EdgeProp = _device_module.EdgeProp
Device = _device_module.Device
Layout = _device_module.Layout
LogicalQubit = _device_module.LogicalQubit
PhysicalQubit = _device_module.PhysicalQubit
SingleQubitNoise = _device_module.SingleQubitNoise
TwoQubitNoise = _device_module.TwoQubitNoise
ReadoutError = _device_module.ReadoutError
OperationKey = _device_module.OperationKey
NoiseModel = _device_module.NoiseModel
Outcome = _device_module.Outcome
Status = _device_module.Status
ExecutionResult = _device_module.ExecutionResult

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
