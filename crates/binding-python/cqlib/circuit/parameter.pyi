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

from typing import Optional, Dict, List, Union

class Parameter:
    """
    A symbolic parameter for quantum circuits.
    """

    def __init__(self, name: str) -> None:
        """
        Creates a new symbolic parameter with the given name.

        Args:
            name (str): The name of the symbol.
        """
        ...

    @staticmethod
    def from_float(val: float) -> "Parameter":
        """
        Creates a constant parameter from a float value.

        Args:
            val (float): The constant value.
        """
        ...

    @staticmethod
    def pi() -> "Parameter":
        """
        Returns a parameter representing the mathematical constant Pi (π).
        """
        ...

    @staticmethod
    def e() -> "Parameter":
        """
        Returns a parameter representing the mathematical constant Euler's number (e).
        """
        ...

    def evaluate(self, bindings: Optional[Dict[str, float]] = None) -> float:
        """
        Evaluates the parameter expression given a set of variable bindings.

        Args:
            bindings (Optional[Dict[str, float]]): A dictionary mapping symbol names to their values.

        Returns:
            float: The computed value.

        Raises:
            ValueError: If a symbol is missing or a math error occurs.
        """
        ...

    def simplify(self, max_iterations: Optional[int] = None) -> "Parameter":
        """
        Simplifies the parameter expression.

        Args:
            max_iterations (Optional[int]): The maximum number of simplification passes.

        Returns:
            Parameter: A new simplified parameter.
        """
        ...

    def derivative(self, var: str) -> "Parameter":
        """
        Computes the symbolic derivative with respect to a variable.

        Args:
            var (str): The variable to differentiate by.

        Returns:
            Parameter: The derivative expression.
        """
        ...

    @property
    def symbols(self) -> List[str]:
        """
        Returns a list of unique symbols in the expression.
        """
        ...

    def abs(self) -> "Parameter": ...
    def sqrt(self) -> "Parameter": ...
    def exp(self) -> "Parameter": ...
    def sin(self) -> "Parameter": ...
    def cos(self) -> "Parameter": ...
    def tan(self) -> "Parameter": ...
    def asin(self) -> "Parameter": ...
    def acos(self) -> "Parameter": ...
    def atan(self) -> "Parameter": ...
    def ln(self) -> "Parameter": ...
    def log(self, base: Optional["Parameter"] = None) -> "Parameter":
        """
        Logarithm with an arbitrary base.

        Args:
            base (Optional[Parameter]): The base of the logarithm. If None, uses natural logarithm (base e).
        """
        ...

    def __add__(self, other: Union["Parameter", float]) -> "Parameter": ...
    def __radd__(self, other: Union["Parameter", float]) -> "Parameter": ...
    def __sub__(self, other: Union["Parameter", float]) -> "Parameter": ...
    def __rsub__(self, other: Union["Parameter", float]) -> "Parameter": ...
    def __mul__(self, other: Union["Parameter", float]) -> "Parameter": ...
    def __rmul__(self, other: Union["Parameter", float]) -> "Parameter": ...
    def __truediv__(self, other: Union["Parameter", float]) -> "Parameter": ...
    def __rtruediv__(self, other: Union["Parameter", float]) -> "Parameter": ...
    def __pow__(self, other: Union["Parameter", float]) -> "Parameter": ...
    def __neg__(self) -> "Parameter": ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
