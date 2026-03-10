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
Python bindings for the quantum device module.

This module provides Python access to quantum hardware characterization data,
including device topology, qubit properties, noise models, and execution results.

# Example

```python
from cqlib.device import Device, Topology, QubitProp
from datetime import datetime, timezone

# Create a device topology
topology = Topology([0, 1, 2], [(0, 1, "CX"), (1, 2, "CX")])

# Initialize a device with calibration data
device = Device("superconducting_qpu", [0, 1, 2], topology)
device.calibration_time = datetime.now(timezone.utc)

# Set qubit properties
prop = QubitProp(readout_error=0.01)
prop.t1 = 50.0  # microseconds
prop.t2 = 25.0
device.add_qubit_properties(0, prop)
```
"""

from .._native import device as _device_module

Topology = _device_module.Topology
InstructionProp = _device_module.InstructionProp
QubitProp = _device_module.QubitProp
EdgeProp = _device_module.EdgeProp
Device = _device_module.Device
Layout = _device_module.Layout
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
    "SingleQubitNoise",
    "TwoQubitNoise",
    "ReadoutError",
    "OperationKey",
    "NoiseModel",
    "Outcome",
    "Status",
    "ExecutionResult",
]
