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

class Parameter:
    """A symbolic parameter used in parameterized quantum circuits (PQC).

    This class intelligently detects the input type:
    - If a number is passed, creates a numeric parameter (e.g., `Parameter(3.14)` creates 3.14)
    - If a string that looks like a pure number is passed, creates a numeric parameter
    - Otherwise, creates a symbolic parameter (e.g., `Parameter("theta")`)
    """

    def __init__(self, value: int | float | str) -> None:
        """Create a new parameter.

        Args:
            value: The value (number or string symbol name).

        Examples:
            >>> Parameter(3.14)       # Creates numeric parameter 3.14
            >>> Parameter("3.14")     # Also creates numeric parameter 3.14
            >>> Parameter("theta")    # Creates symbolic parameter 'theta'
            >>> Parameter("x + 1")   # Creates expression
        """
        ...

    @staticmethod
    def from_expression(expr: str) -> "Parameter":
        """Parse a mathematical expression string into a Parameter.

        Args:
            expr: The expression string to parse (e.g., "theta + 1", "pi/2", "sin(x)").

        Returns:
            A new Parameter representing the parsed expression.
        """
        ...

    @staticmethod
    def pi() -> "Parameter":
        """Returns a new parameter representing the mathematical constant Pi (π)."""
        ...

    @staticmethod
    def e() -> "Parameter":
        """Returns a new parameter representing the mathematical constant Euler's number (e)."""
        ...

    def evaluate(self, bindings: Optional[dict[str, float]] = None) -> float:
        """Evaluates the parameter expression given a set of variable bindings."""
        ...

    def simplify(self, max_iterations: Optional[int] = None) -> "Parameter":
        """Applies algebraic and trigonometric simplification rules."""
        ...

    def derivative(self, var: str) -> "Parameter":
        """Calculate the derivative of the expression with respect to the specified variable."""
        ...

    @property
    def symbols(self) -> list[str]:
        """Retrieves all unique symbols (variables) used in this parameter expression."""
        ...

    # Arithmetic operators
    def __add__(self, other: "Parameter" | float) -> "Parameter": ...
    def __sub__(self, other: "Parameter" | float) -> "Parameter": ...
    def __mul__(self, other: "Parameter" | float) -> "Parameter": ...
    def __truediv__(self, other: "Parameter" | float) -> "Parameter": ...
    def __pow__(self, other: "Parameter" | float) -> "Parameter": ...
    def __neg__(self) -> "Parameter": ...
    def __eq__(self, other: object) -> bool: ...

    # Reverse arithmetic operators
    def __radd__(self, other: float) -> "Parameter": ...
    def __rsub__(self, other: float) -> "Parameter": ...
    def __rmul__(self, other: float) -> "Parameter": ...
    def __rtruediv__(self, other: float) -> "Parameter": ...

    # Mathematical functions
    def sin(self) -> "Parameter": ...
    def cos(self) -> "Parameter": ...
    def tan(self) -> "Parameter": ...
    def asin(self) -> "Parameter": ...
    def acos(self) -> "Parameter": ...
    def atan(self) -> "Parameter": ...
    def exp(self) -> "Parameter": ...
    def ln(self) -> "Parameter": ...
    def log(self, base: Optional["Parameter"] = None) -> "Parameter": ...
    def sqrt(self) -> "Parameter": ...
    def abs(self) -> "Parameter": ...
    def pow(self, val: "Parameter" | float) -> "Parameter":
        """Returns the power of this parameter raised to the given exponent.

        Args:
            val: The exponent (can be a float or Parameter).

        Returns:
            A new parameter representing `self^val`.

        Example:
            >>> x = Parameter("x")
            >>> y = Parameter("y")
            >>> result = x.pow(y)  # x^y
            >>> result = x.pow(2)  # x^2
        """
        ...

    def replace(self, symbol: str, param: "Parameter") -> "Parameter":
        """Replaces all occurrences of a symbol with another parameter expression.

        Args:
            symbol: The name of the symbol to replace.
            param: The parameter expression to substitute.

        Returns:
            A new parameter with the substitution applied.

        Example:
            >>> x = Parameter("x")
            >>> expr = x + Parameter(2.0)
            >>> y = Parameter("y")
            >>> replacement = y * Parameter(3.0)
            >>> new_expr = expr.replace("x", replacement)  # (y * 3) + 2
        """
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
