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

"""Device-aware circuit routing transforms."""

from ..._native import compile as _compile_module

_routing_module = _compile_module.transform.routing

RoutedCircuit = _routing_module.RoutedCircuit
SabreRouteResult = _routing_module.SabreRouteResult
route_with_layout = _routing_module.route_with_layout
route_sabre = _routing_module.route_sabre

__all__ = [
    "RoutedCircuit",
    "SabreRouteResult",
    "route_with_layout",
    "route_sabre",
]
