# This code is part of Cqlib.
#
# (C) Copyright China Telecom Quantum Group 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http:#www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Template optimization compiler tests.

Test coverage:
- default template loading and execution
- custom-template cancellation behavior
- iterative optimization behavior
- file-based template loading behavior
"""

from pathlib import Path

from cqlib.circuit import Circuit
from cqlib.compiler import TemplateOptimization


def _hh_template() -> Circuit:
    """Builds H-H cancellation template."""
    template = Circuit(1)
    template.h(0)
    template.h(0)
    return template


class TestTemplateOptimization:
    """Tests template optimization API behavior."""

    def test_default_templates_execute(self) -> None:
        """Loads default templates when no explicit input is provided."""
        circuit = Circuit(2)
        circuit.h(0)
        circuit.h(0)
        circuit.cx(0, 1)

        optimizer = TemplateOptimization()
        optimized = optimizer.execute(circuit)

        assert optimizer.template_count() >= 1
        assert len(optimized.operations) <= len(circuit.operations)

    def test_custom_template_cancellation(self) -> None:
        """Applies explicit cancellation template and removes redundant gates."""
        circuit = Circuit(2)
        circuit.h(0)
        circuit.h(0)
        circuit.cx(0, 1)

        optimizer = TemplateOptimization([_hh_template()], qubit_fixing_cnt=1, prune_depth=3, prune_width=1)
        optimized = optimizer.execute(circuit)
        assert len(optimized.operations) == 1

    def test_iterative_optimization(self) -> None:
        """Runs iterative optimization until no further size decrease occurs."""
        circuit = Circuit(1)
        for _ in range(3):
            circuit.h(0)
            circuit.h(0)

        optimizer = TemplateOptimization([_hh_template()])
        optimized = optimizer.execute_iterative(circuit, max_iterations=10)
        assert len(optimized.operations) == 0

    def test_json_template_file_loading(self, tmp_path: Path) -> None:
        """Loads templates from JSON file and executes optimization."""
        json_text = """
{
  "version": 1,
  "templates": [
    {
      "name": "hh_cancel",
      "gates": [
        { "gate": "H", "qubits": [0] },
        { "gate": "H", "qubits": [0] }
      ]
    }
  ]
}
"""
        template_path = tmp_path / "templates.json"
        template_path.write_text(json_text, encoding="utf-8")

        circuit = Circuit(1)
        circuit.h(0)
        circuit.h(0)

        optimizer = TemplateOptimization(template_file=str(template_path))
        optimized = optimizer.execute(circuit)
        assert len(optimized.operations) == 0

    def test_qcis_template_file_loading(self, tmp_path: Path) -> None:
        """Loads templates from QCIS file split by `---` separators."""
        qcis_text = """
H Q0
H Q0
---
CZ Q0 Q1
CZ Q0 Q1
"""
        template_path = tmp_path / "templates.qcis"
        template_path.write_text(qcis_text, encoding="utf-8")

        circuit = Circuit(2)
        circuit.cz(0, 1)
        circuit.cz(0, 1)

        optimizer = TemplateOptimization(template_file=str(template_path))
        optimized = optimizer.execute(circuit)
        assert len(optimized.operations) == 0
