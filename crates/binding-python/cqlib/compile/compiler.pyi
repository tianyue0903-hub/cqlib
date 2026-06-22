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

from typing import Optional

from cqlib.circuit import Circuit, Instruction
from cqlib.device import Device, Layout

class CompileMode:
    """Optimization effort selected for the compiler workflow.

    ``CompileMode`` controls how much optimization and cleanup effort the
    compiler spends while preserving circuit semantics.

    - :meth:`normal` uses conservative production defaults.
    - :meth:`enhanced` increases bounded rewrite and routing cleanup effort.

    Example::

        from cqlib.compile import CompileMode, compile
        from cqlib.circuit import Circuit

        circuit = Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        result = compile(circuit, mode=CompileMode.enhanced())
        assert result.mode == CompileMode.enhanced()
    """

    @staticmethod
    def normal() -> CompileMode:
        """Return the normal production compiler mode."""
        ...
    @staticmethod
    def enhanced() -> CompileMode:
        """Return the enhanced compiler mode."""
        ...
    def __copy__(self) -> CompileMode:
        """Return this immutable mode value."""
        ...
    def __deepcopy__(self, memo: dict) -> CompileMode:
        """Return this immutable mode value."""
        ...
    def __eq__(self, other: CompileMode) -> bool:
        """Return whether two values select the same compiler mode."""
        ...
    def __hash__(self) -> int:
        """Return a hash value for dictionaries and sets."""
        ...

class WorkflowStepReport:
    """Per-step execution record produced by a compiler workflow run.

    Step reports are returned in execution order. They are intended for
    diagnostics, tests, and compile logs. Step names describe workflow
    positions such as ``"route.sabre"`` or ``"translate.target_basis"``.

    Example::

        result = compile(circuit)
        for step in result.steps:
            print(step.stage, step.name, step.changed, step.skipped)
    """

    @property
    def stage(self) -> str:
        """Coarse workflow stage, such as ``"init"``, ``"optimization"``, or ``"output"``."""
        ...
    @property
    def name(self) -> str:
        """Workflow-local step name, such as ``"canonicalize.output"``."""
        ...
    @property
    def changed(self) -> bool:
        """Whether this step changed the circuit representation."""
        ...
    @property
    def skipped(self) -> bool:
        """Whether this step was intentionally skipped."""
        ...
    @property
    def reason(self) -> Optional[str]:
        """Optional skip reason or configuration note."""
        ...
    def __copy__(self) -> WorkflowStepReport:
        """Return a shallow copy of this report."""
        ...
    def __deepcopy__(self, memo: dict) -> WorkflowStepReport:
        """Return a deep copy of this report."""
        ...

class CompileResult:
    """Result returned by :func:`compile`.

    The result owns the compiled circuit and the workflow diagnostics. The
    input circuit passed to :func:`compile` is not modified.

    Example::

        from cqlib.compile import compile
        from cqlib.circuit import Circuit

        circuit = Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        result = compile(circuit)
        compiled = result.circuit
        assert compiled.num_qubits == circuit.num_qubits
        assert result.steps
    """

    @property
    def circuit(self) -> Circuit:
        """Compiled circuit returned by the workflow."""
        ...
    @property
    def changed(self) -> bool:
        """Whether any workflow step changed the input representation."""
        ...
    @property
    def mode(self) -> CompileMode:
        """Compiler mode used for this run."""
        ...
    @property
    def steps(self) -> list[WorkflowStepReport]:
        """Workflow step reports in execution order."""
        ...
    def __copy__(self) -> CompileResult:
        """Return a shallow copy of this result."""
        ...
    def __deepcopy__(self, memo: dict) -> CompileResult:
        """Return a deep copy of this result."""
        ...

def compile(
    circuit: Circuit,
    *,
    mode: CompileMode | None = None,
    target_basis: list[Instruction] | None = None,
    device: Device | None = None,
    initial_layout: Layout | None = None,
    seed: int | None = None,
) -> CompileResult:
    """Compile a circuit with the configured compiler workflow.

    Args:
        circuit: Logical input circuit. The function does not mutate it.
        mode: Optimization effort. ``None`` selects :meth:`CompileMode.normal`.
        target_basis: Optional final target instruction basis. The current core
            workflow accepts standard-gate instructions here, created with
            :meth:`cqlib.circuit.Instruction.from_standard_gate`.
        device: Optional hardware target. When provided, compilation may route
            the circuit through the device topology and may use device native
            gates as the target basis when ``target_basis`` is ``None``.
        initial_layout: Optional logical-to-physical layout. This is valid only
            when ``device`` is provided.
        seed: Optional deterministic seed for heuristic layout/routing stages.

    Returns:
        A :class:`CompileResult` containing the compiled circuit, selected mode,
        changed flag, and workflow step reports.

    Raises:
        ValueError: If the compiler rejects the input configuration or a
            transform precondition is not satisfied.

    Examples:
        Logical-only compilation::

            from cqlib.compile import compile
            from cqlib.circuit import Circuit

            circuit = Circuit(2)
            circuit.h(0)
            circuit.cx(0, 1)

            result = compile(circuit)
            print(result.circuit.operations)

        Compile with an explicit target basis::

            from cqlib.compile import compile
            from cqlib.circuit import Circuit, Instruction, StandardGate

            circuit = Circuit(2)
            circuit.cx(0, 1)

            basis = [
                Instruction.from_standard_gate(StandardGate.H),
                Instruction.from_standard_gate(StandardGate.CZ),
            ]
            result = compile(circuit, target_basis=basis)

        Compile for a line device::

            from cqlib.compile import compile
            from cqlib.circuit import Circuit
            from cqlib.device import Device

            circuit = Circuit(3)
            circuit.cx(0, 2)

            result = compile(circuit, device=Device.line("line-3", 3), seed=7)
            assert any(step.name == "route.sabre" for step in result.steps)
    """
    ...

__all__: list[str]
