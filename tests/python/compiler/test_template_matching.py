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
Template matching compiler tests.

Test coverage:
- basic template matching behavior
- strict parameter compatibility behavior
- matching does not skip intermediate gates
- heuristic argument path acceptance
- no-match behavior
"""

from cqlib.circuit import Circuit, Parameter
from cqlib.compiler import TemplateMatching


def _simple_circuit() -> Circuit:
    """Builds H-CX-H circuit."""
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.h(1)
    return circuit


def _simple_template() -> Circuit:
    """Builds H-CX template."""
    template = Circuit(2)
    template.h(0)
    template.cx(0, 1)
    return template


def _hh_template() -> Circuit:
    """Builds H-H identity template."""
    template = Circuit(1)
    template.h(0)
    template.h(0)
    return template


def _cxcx_template() -> Circuit:
    """Builds CX-CX identity template."""
    template = Circuit(2)
    template.cx(0, 1)
    template.cx(0, 1)
    return template


class TestTemplateMatching:
    """Tests template matching API behavior."""

    def test_basic_matching(self) -> None:
        """Finds at least one match on a simple pattern."""
        matcher = TemplateMatching()
        matches = matcher.run(_simple_circuit(), _simple_template())
        assert len(matches) >= 1
        assert len(matches[0][0]) == 2

    def test_parameter_exactness(self) -> None:
        """Requires exact parameter compatibility for parametric gates."""
        circuit = Circuit(1)
        circuit.rx(0, 0.1)

        template = Circuit(1)
        template.rx(0, 0.2)

        matcher = TemplateMatching()
        matches = matcher.run(circuit, template)
        assert matches == []

    def test_hh_template_does_not_match_hxh(self) -> None:
        """Does not match H-H across a non-commuting middle X gate."""
        circuit = Circuit(1)
        circuit.h(0)
        circuit.x(0)
        circuit.h(0)

        matcher = TemplateMatching()
        matches = matcher.run(circuit, _hh_template())
        assert matches == []

    def test_cxcx_template_does_not_match_cx_h_cx(self) -> None:
        """Does not match CX-CX across a middle Hadamard gate."""
        matcher = TemplateMatching()

        for hadamard_qubit in (0, 1):
            circuit = Circuit(2)
            circuit.cx(0, 1)
            circuit.h(hadamard_qubit)
            circuit.cx(0, 1)

            matches = matcher.run(circuit, _cxcx_template())
            assert matches == []

    def test_symbolic_parameter_exactness(self) -> None:
        """Matches symbolic parameters only when expressions are equal."""
        theta = Parameter("theta")
        phi = Parameter("phi")

        circuit = Circuit(1)
        circuit.rx(0, theta)

        same_template = Circuit(1)
        same_template.rx(0, theta)
        different_template = Circuit(1)
        different_template.rx(0, phi)

        matcher = TemplateMatching()
        same = matcher.run(circuit, same_template)
        different = matcher.run(circuit, different_template)
        assert len(same) >= 1
        assert different == []

    def test_heuristic_arguments(self) -> None:
        """Accepts heuristic arguments and returns stable match shape."""
        matcher = TemplateMatching()
        matches = matcher.run(
            _simple_circuit(),
            _simple_template(),
            qubit_fixing_cnt=1,
            prune_depth=3,
            prune_width=1,
        )
        assert isinstance(matches, list)
