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

"""Permissions and hard limits for ancillary-resource allocation.

``ResourcePolicy`` describes what a compiler is allowed to use;
``ResourceLimits`` describes what the target can physically accommodate.
Keeping these concepts separate prevents an optimization preference from
silently overriding a device capacity constraint.
"""

from ..._native import compile as _compile_module

_resource_module = _compile_module.resource

ResourcePolicy = _resource_module.ResourcePolicy
ResourceLimits = _resource_module.ResourceLimits

__all__ = ["ResourcePolicy", "ResourceLimits"]
