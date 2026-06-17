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

"""Ansatz templates are not part of the active Python binding yet.

The Rust ansatz wrappers currently depend on the Python QIS bindings, which
are restored in a separate migration step.
"""

raise ImportError(
    "cqlib.circuit.ansatz is not available until the Python QIS bindings are restored"
)
