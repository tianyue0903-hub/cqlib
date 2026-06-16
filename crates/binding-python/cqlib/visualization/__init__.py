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

from __future__ import annotations

from .._native import draw_figure as _draw_figure
from .._native import draw_text


class _InlineSvg(str):
    """String SVG wrapper with rich display support for notebook frontends."""

    def _repr_svg_(self) -> str:
        return str(self)


def draw_figure(
    circuit,
    *,
    fold=None,
    initial_state=False,
    reverse_bits=False,
    show_params=True,
    decompose_circuit_gates=False,
    output_path=None,
):
    """Render a circuit as SVG.

    In notebook frontends, the return value displays inline when used as the
    last expression in a cell.
    """
    svg = _draw_figure(
        circuit,
        fold=fold,
        initial_state=initial_state,
        reverse_bits=reverse_bits,
        show_params=show_params,
        decompose_circuit_gates=decompose_circuit_gates,
        output_path=output_path,
    )
    return _InlineSvg(svg)


__all__ = ["draw_text", "draw_figure"]
