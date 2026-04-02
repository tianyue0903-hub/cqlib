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

from ..circuit import Circuit

def draw_text(
    circuit: Circuit,
    *,
    line_width: int | None = None,
    initial_state: bool = False,
    reverse_bits: bool = False,
    show_params: bool = True,
    decompose_circuit_gates: bool = False,
) -> str:
    """Render a circuit as unicode text."""
    ...

def draw_figure(
    circuit: Circuit,
    *,
    fold: int | None = None,
    initial_state: bool = False,
    reverse_bits: bool = False,
    show_params: bool = True,
    decompose_circuit_gates: bool = False,
    output_path: str | None = None,
) -> str:
    """Render a circuit as SVG string.

    The runtime object also supports inline SVG display in notebook frontends.
    """
    ...
