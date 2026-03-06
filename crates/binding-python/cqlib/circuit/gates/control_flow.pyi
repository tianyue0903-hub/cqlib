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

from typing import Optional
from typing_extensions import final
from ..bit import Qubit
from ..operation import Operation

@final
class ConditionView:
    """A classical condition based on a measurement outcome.

    Used in control flow operations to determine which branch to execute.
    Represents a condition like "if qubit 0 measures to 1".
    """

    def __init__(self, qubit: Qubit, target: int) -> None:
        """Creates a new condition view.

        Args:
            qubit: The qubit whose measurement result to check.
            target: The target value to compare against (typically 0 or 1).
        """
        ...

    @property
    def qubit(self) -> Qubit:
        """Returns the qubit associated with this condition."""
        ...

    @property
    def target(self) -> int:
        """Returns the target value for comparison."""
        ...

    def __repr__(self) -> str: ...

@final
class IfElseGate:
    """A conditional quantum operation based on a classical condition.

    Contains a true branch that executes when condition is met, and optionally a false branch.
    """

    def __init__(
        self,
        condition: ConditionView,
        true_body: list[Operation],
        false_body: Optional[list[Operation]] = None,
    ) -> None:
        """Creates a new if-else gate.

        Args:
            condition: The condition to evaluate.
            true_body: Operations to execute when condition is true.
            false_body: Optional operations to execute when condition is false.
        """
        ...

    @property
    def condition(self) -> ConditionView:
        """Returns the condition for this gate."""
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used in this gate."""
        ...

    @property
    def num_params(self) -> int:
        """Returns the number of parameters."""
        ...

    def __repr__(self) -> str: ...

@final
class WhileLoopGate:
    """A while-loop quantum operation.

    Executes quantum operations repeatedly while a classical condition remains true.
    """

    def __init__(self, condition: ConditionView, body: list[Operation]) -> None:
        """Creates a new while-loop gate.

        Args:
            condition: The condition to evaluate before each iteration.
            body: The operations to execute in each iteration.
        """
        ...

    @property
    def condition(self) -> ConditionView:
        """Returns the condition for this loop."""
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used in this gate."""
        ...

    @property
    def num_params(self) -> int:
        """Returns the number of parameters."""
        ...

    def __repr__(self) -> str: ...

@final
class ControlFlow:
    """A control flow operation (if-else or while-loop).

    Wraps different types of control flow operations for use in circuits.
    """

    @staticmethod
    def if_else(
        condition: ConditionView,
        true_body: list[Operation],
        false_body: Optional[list[Operation]] = None,
    ) -> "ControlFlow":
        """Creates a new if-else control flow.

        Args:
            condition: The condition to evaluate.
            true_body: Operations to execute when condition is true.
            false_body: Optional operations to execute when condition is false.
        """
        ...

    @staticmethod
    def while_loop(condition: ConditionView, body: list[Operation]) -> "ControlFlow":
        """Creates a new while-loop control flow.

        Args:
            condition: The condition to evaluate before each iteration.
            body: Operations to execute in each iteration.
        """
        ...

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used in this control flow."""
        ...

    @property
    def num_params(self) -> int:
        """Returns the number of parameters."""
        ...

    @property
    def is_if_else(self) -> bool:
        """Returns True if this is an if-else control flow."""
        ...

    @property
    def is_while_loop(self) -> bool:
        """Returns True if this is a while-loop control flow."""
        ...

    def as_if_else(self) -> Optional[IfElseGate]:
        """Returns the IfElseGate if this is an if-else, None otherwise."""
        ...

    def as_while_loop(self) -> Optional[WhileLoopGate]:
        """Returns the WhileLoopGate if this is a while-loop, None otherwise."""
        ...

    def __repr__(self) -> str: ...
