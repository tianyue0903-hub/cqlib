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

from collections.abc import Sequence
from typing import Optional

from cqlib.circuit import Circuit, Instruction
from cqlib.device import Device, Layout
from .resource import ResourcePolicy

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

class CompileConfig:
    """Immutable compiler workflow configuration snapshot.

    Mutable inputs such as the target basis, device, and initial layout are
    copied during construction. Properties also return copies, so subsequent
    caller-side mutations do not change this configuration.

    Validation that depends on multiple fields or on the input circuit is
    performed when the configuration is used by :class:`CompilerWorkflow` or
    :func:`compile`.
    """

    def __init__(
        self,
        *,
        mode: CompileMode | None = None,
        target_basis: Sequence[str | Instruction] | None = None,
        device: Device | None = None,
        initial_layout: Layout | None = None,
        resource_policy: ResourcePolicy | None = None,
        seed: int | None = None,
    ) -> None:
        """Create an immutable compiler workflow configuration snapshot.

        Args:
            mode: Optimization effort. ``None`` selects
                :meth:`CompileMode.normal`.
            target_basis: Optional explicit target instruction basis. Entries
                may be case-insensitive standard-gate names or instructions.
            device: Optional hardware target used for capacity, routing, and
                native-gate constraints.
            initial_layout: Optional initial logical-to-physical layout. A
                target device is required when this is set.
            resource_policy: Ancillary-resource permissions. ``None`` uses the
                conservative default policy.
            seed: Optional deterministic layout/routing seed.

        Raises:
            ValueError: If a target-basis gate name is unknown.
        """
        ...
    @property
    def mode(self) -> CompileMode:
        """Optimization workflow mode."""
        ...
    @property
    def target_basis(self) -> list[Instruction] | None:
        """Copied explicit target basis, or ``None`` when unspecified."""
        ...
    @property
    def device(self) -> Device | None:
        """Copied target device, or ``None`` when unspecified."""
        ...
    @property
    def initial_layout(self) -> Layout | None:
        """Copied initial layout, or ``None`` when unspecified."""
        ...
    @property
    def resource_policy(self) -> ResourcePolicy:
        """Copied ancillary-resource policy."""
        ...
    @property
    def seed(self) -> int | None:
        """Deterministic layout/routing seed, or ``None``."""
        ...
    def __copy__(self) -> CompileConfig:
        """Return an independent shallow copy of this snapshot."""
        ...
    def __deepcopy__(self, memo: dict) -> CompileConfig:
        """Return an independent deep copy of this snapshot."""
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

class CompilerWorkflow:
    """Reusable compiler optimization workflow.

    The workflow owns an immutable configuration snapshot and may be run over
    multiple circuits. Running it never mutates the input circuit.
    """

    def __init__(self, config: CompileConfig | None = None) -> None:
        """Create a workflow, using a default configuration when omitted."""
        ...
    @property
    def config(self) -> CompileConfig:
        """Return an independent copy of the workflow configuration."""
        ...
    def run(self, circuit: Circuit) -> CompileResult:
        """Compile a circuit without modifying it.

        Raises:
            ValueError: If the configuration, input circuit, or a transform
                precondition is invalid.
        """
        ...

def compile(
    circuit: Circuit,
    *,
    mode: CompileMode | None = None,
    target_basis: Sequence[str | Instruction] | None = None,
    device: Device | None = None,
    initial_layout: Layout | None = None,
    resource_policy: ResourcePolicy | None = None,
    seed: int | None = None,
) -> CompileResult:
    """Compile a circuit with the configured compiler workflow.

    Args:
        circuit: Logical input circuit. The function does not mutate it.
        mode: Optimization effort. ``None`` selects :meth:`CompileMode.normal`.
        target_basis: Optional final target instruction basis. Entries may be
            standard-gate names (case-insensitive) or standard-gate instructions
            created with :meth:`cqlib.circuit.Instruction.from_standard_gate`.
        device: Optional hardware target. When provided, compilation may route
            the circuit through the device topology and may use device native
            gates as the target basis when ``target_basis`` is ``None``.
        initial_layout: Optional logical-to-physical layout. This is valid only
            when ``device`` is provided.
        resource_policy: Permissions for compiler-created clean ancillas and
            dirty borrowing. ``None`` uses the conservative default, which
            creates no ancillas and does not borrow input qubits. Device
            capacity remains a separate hard bound.
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

            basis = ["H", Instruction.from_standard_gate(StandardGate.CZ)]
            result = compile(circuit, target_basis=basis)

        Compile for a line device::

            from cqlib.compile import compile
            from cqlib.circuit import Circuit
            from cqlib.device import Device

            circuit = Circuit(3)
            circuit.cx(0, 2)

            result = compile(circuit, device=Device.line("line-3", 3), seed=7)
            assert any(step.name == "route.sabre" for step in result.steps)

        Permit clean ancillas for multi-controlled-gate synthesis::

            from cqlib.compile import compile
            from cqlib.compile.resource import ResourcePolicy

            result = compile(
                circuit,
                resource_policy=ResourcePolicy(
                    max_pre_layout_clean_ancillas=2,
                ),
            )
    """
    ...

__all__: list[str]
