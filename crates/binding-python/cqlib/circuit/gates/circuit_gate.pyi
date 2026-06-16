"""Frozen circuits and circuit-defined composite gates.

Building reusable custom gates
-------------------------------

:class:`FrozenCircuit` captures an immutable circuit definition.  Once frozen,
it can be wrapped by a :class:`CircuitGate` and reused across multiple circuits
as a single composite gate::

    from cqlib import Circuit, Qubit
    from cqlib.circuit.gates import FrozenCircuit, CircuitGate

    # Build a 2-qubit Bell-pair sub-circuit
    bell = Circuit(2)
    bell.h(0)
    bell.cx(0, 1)

    # Freeze and wrap as a reusable gate
    frozen = FrozenCircuit(
        qubits=[Qubit(0), Qubit(1)],
        operations=bell.operations(),
    )
    bell_gate = CircuitGate("Bell", frozen)

    # Use it in a larger circuit
    big = Circuit(4)
    big.append_circuit_gate(bell_gate, [0, 1])
    big.append_circuit_gate(bell_gate, [2, 3])

When the sub-circuit carries symbolic parameters, bind them positionally
via ``params`` on :meth:`~cqlib.circuit.Circuit.append_circuit_gate`.

Typical lifecycle
-----------------

1. Build a circuit with :class:`~cqlib.circuit.Circuit`.
2. Freeze it with :class:`FrozenCircuit`, optionally via
   :meth:`~cqlib.circuit.Circuit.to_gate(name)` which returns a ready-made
   :class:`CircuitGate`.
3. Apply the gate to other circuits via
   :meth:`~cqlib.circuit.Circuit.append_circuit_gate`.

Classes
-------

- :class:`FrozenCircuit` — immutable, validated snapshot of a circuit's
  :class:`~cqlib.circuit.ValueOperation` list and qubit layout.
- :class:`CircuitGate` — named, reusable composite gate backed by a frozen
  circuit definition.
"""

from ..bit import Qubit
from ..classical import ClassicalType
from ..operation import ValueOperation

class FrozenCircuit:
    """Immutable circuit snapshot suitable for use inside a gate definition.

    Created from construction-IR operations.  Once built, the internal
    operation sequence cannot mutate, guaranteeing that every
    :class:`CircuitGate` that references this definition observes the same
    behaviour.

    Use :class:`FrozenCircuit` directly when you need to keep the raw
    circuit around for inspection or custom gate logic.  For the common
    case, :meth:`Circuit.to_gate(name) <cqlib.circuit.Circuit.to_gate>`
    returns a ready-made :class:`CircuitGate` without exposing this type.
    """

    def __init__(
        self,
        qubits: list[Qubit],
        operations: list[ValueOperation],
        classical_vars: list[ClassicalType] | None = ...,
        classical_values: list[ClassicalType] | None = ...,
    ) -> None:
        """Create a frozen circuit from construction-IR parts.

        Args:
            qubits: Qubits in storage order.
            operations: Serialised :class:`~cqlib.circuit.ValueOperation`
                entries.
            classical_vars: Types of mutable classical variables, if any.
            classical_values: Types of immutable classical values produced
                by measurement, if any.

        Raises:
            CircuitError: If the circuit IR does not pass validation.
        """
        ...

    @property
    def qubits(self) -> list[Qubit]:
        """Qubits in storage order."""
        ...

    @property
    def num_operations(self) -> int:
        """Number of stored operations."""
        ...

    @property
    def operations(self) -> list[ValueOperation]:
        """Self-contained operations with circuit parameters resolved.

        Each returned :class:`~cqlib.circuit.ValueOperation` carries
        its own :class:`~cqlib.circuit.Parameter` values, not
        circuit-local parameter-table indices.
        """
        ...

    @property
    def symbols(self) -> list[str]:
        """Symbolic parameter names in insertion order."""
        ...

    def __repr__(self) -> str: ...


class CircuitGate:
    """Composite gate defined by an immutable :class:`FrozenCircuit`.

    Supports Python subclassing for custom gate families (e.g., oracles,
    fixed-architecture ansatz blocks).

    Args:
        name: A descriptive name for the gate (appears in circuit
            visualisations and serialisation).
        circuit: The frozen circuit definition backing this gate.
    """

    def __init__(self, name: str, circuit: FrozenCircuit) -> None:
        """Create a named gate from a frozen circuit.

        Raises:
            CircuitError: If *circuit* contains non-gate operations
                (e.g. directives) that cannot be used inside a gate.
        """
        ...

    @property
    def name(self) -> str:
        """The gate name."""
        ...

    @property
    def num_qubits(self) -> int:
        """Number of qubits used by the definition."""
        ...

    @property
    def num_params(self) -> int:
        """Number of positional symbolic parameters accepted by this gate."""
        ...

    @property
    def symbols(self) -> list[str]:
        """Symbolic parameter names in positional (application) order."""
        ...

    @property
    def circuit(self) -> FrozenCircuit:
        """The immutable circuit definition."""
        ...

    def inverse(self) -> CircuitGate:
        """Return the inverse circuit gate.

        The inverse reverses every operation in the underlying circuit and
        preserves the same parameter slots.  If any operation is not
        invertible a ``CircuitError`` is raised.

        Raises:
            CircuitError: When the underlying circuit contains irreversible
                operations.
        """
        ...

    def __eq__(self, other: object) -> bool:
        """Equality compares logical identity (name and circuit definition).

        Two gates with the same name but different frozen circuits are not
        considered equal.
        """
        ...

    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...
