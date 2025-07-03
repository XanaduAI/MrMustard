# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the classes to describe sets of parameters."""

import io
from collections.abc import Sequence
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from mrmustard.math.backend_manager import BackendManager

from .parameters import Constant, Variable

math = BackendManager()

__all__ = ["ParameterSet"]


class ParameterSet:
    r"""
    A set of parameters.

    ``ParameterSet`` can store both constant and variable parameters. It provides fast access to
    both classes of parameters, as well as to their names.

    .. code::

      const1 = Constant(1.2345, "const1")
      const2 = Constant(2.3456, "const2")
      var1 = Variable(3.4567, "var1")

      ps = ParameterSet()
      ps.add_parameter(const1)
      ps.add_parameter(const2)
      ps.add_parameter(var1)

      ps.names  # returns `["const1", "const2", "var1"]`
      ps.constants  # returns `{"const1": const1, "const2": const2}`
      ps.variable  # returns `{"var1": var1}`
    """

    def __init__(self):
        self._names: list[str] = []
        self._constants: dict[str, Constant] = {}
        self._variables: dict[str, Variable] = {}

    @property
    def constants(self) -> dict[str, Constant]:
        r"""
        The constant parameters in this parameter set.
        """
        return self._constants

    @property
    def variables(self) -> dict[str, Variable]:
        r"""
        The variable parameters in this parameter set.
        """
        return self._variables

    @property
    def all_parameters(self) -> dict[str, Constant | Variable]:
        r"""
        The constant and variable parameters in this parameter set.
        """
        return self.constants | self.variables

    @property
    def names(self) -> Sequence[str]:
        r"""
        The names of all the parameters in this parameter set, in the order in which they
        were added.
        """
        return self._names

    def add_parameter(self, parameter: Constant | Variable) -> None:
        r"""
        Adds a parameter to this parameter set.

        Args:
            parameter: A constant or variable parameter.

        Raises:
            ValueError: If this parameter set already contains a parameter with the same
                name as that of the given parameter.
        """
        name = parameter.name

        if name in self.names:
            msg = f"A parameter with name ``{name}`` is already part of this parameter set."
            raise ValueError(msg)
        self._names.append(name)

        # updates dictionary and dynamically creates an attribute
        if isinstance(parameter, Constant):
            self.constants[name] = parameter
            self.__dict__[name] = self.constants[name]
        elif isinstance(parameter, Variable):
            self.variables[parameter.name] = parameter
            self.__dict__[name] = self.variables[name]

    def tagged_variables(self, tag: str) -> dict[str, Variable]:
        r"""
        Returns a dictionary whose keys are tagged names of the variables in this parameter set, and whose
        values are the variables in this parameter set. Tagging is done by prepending the string ``f"{tag}"/``
        to variables' original names.
        """
        ret = {}
        for k, v in self.variables.items():
            ret[f"{tag}/{k}"] = v
        return ret

    def to_dict(self) -> dict[str, Any]:
        r"""
        Returns a dictionary representation of this parameter set such that
        it is compatible with the signature of built-in circuit components.
        """
        ret = {}
        for name, param in self.all_parameters.items():
            ret[name] = param.value
            if isinstance(param, Variable):
                ret[name + "_trainable"] = True
                ret[name + "_bounds"] = param.bounds
        return ret

    def to_string(self, decimals: int) -> str:
        r"""
        Returns a string representation of the parameter values, separated by commas and rounded
        to the specified number of decimals.

        Args:
            decimals (int): number of decimals to round to

        Returns:
            str: string representation of the parameter values
        """
        strings = []
        for name in self.names:
            param = self.constants.get(name) or self.variables.get(name)
            if len(param.value.shape) == 0:  # don't show arrays
                # conversion to numpy is necessary to ensure consistent string formatting
                # across backends
                string = str(np.round(math.asnumpy(param.value), decimals))
            else:
                string = f"{name}"
            strings.append(string)
        return ", ".join(strings)

    def __getitem__(self, items: int | Sequence[int]):
        r"""
        Returns a parameter set that contains slices of the parameters in this parameter set.

        In particular:
            * If a parameter's value is a number, the returned parameter set contains the same
              parameter.
            * If a parameter's value is an array, the returned parameter set contains the same
              parameter with sliced value.

        .. code-block::

            >>> from mrmustard.math.parameter_set import ParameterSet
            >>> from mrmustard.math.parameters import Constant, Variable
            >>> import numpy as np

            >>> const1 = Constant(1, "c1")
            >>> const2 = Constant([2, 3, 4], "c2")
            >>> var1 = Variable(5, "v1")
            >>> var2 = Variable([6, 7, 8], "v2")

            >>> ps = ParameterSet()
            >>> ps.add_parameter(const1)
            >>> ps.add_parameter(const2)
            >>> ps.add_parameter(var1)
            >>> ps.add_parameter(var2)

            >>> assert np.allclose(ps[0].constants["c1"].value, 1)
            >>> assert np.allclose(ps[0].constants["c2"].value, 2)
            >>> assert np.allclose(ps[0].variables["v1"].value, 5)
            >>> assert np.allclose(ps[0].variables["v2"].value, 6)

            >>> assert np.allclose(ps[1, 2].constants["c1"].value, 1)
            >>> assert np.allclose(ps[1, 2].constants["c2"].value, [3, 4])
            >>> assert np.allclose(ps[1, 2].variables["v1"].value, 5)
            >>> assert np.allclose(ps[1, 2].variables["v2"].value, [7, 8])
        """
        if isinstance(items, int):
            items = [items]
        items = math.astensor(items)

        ret = ParameterSet()

        for name, const in self._constants.items():
            if const.value.shape != ():
                val = math.gather(const.value, items)
                ret.add_parameter(Constant(val, name))
            else:
                ret.add_parameter(const)

        for name, var in self._variables.items():
            if var.value.shape != ():
                val = math.gather(var.value, items)
                ret.add_parameter(Variable(val, name, var.bounds, var.update_fn))
            else:
                ret.add_parameter(var)

        return ret

    def _format_value(self, param: Constant | Variable) -> tuple[str, str]:
        r"""Format parameter value and shape strings."""
        try:
            value = math.asnumpy(param.value)
        except (ValueError, TypeError, AttributeError):
            return "array", str(getattr(param.value, "shape", "scalar"))

        # Handle arrays
        if hasattr(param.value, "shape") and param.value.shape != ():
            shape_str = str(param.value.shape)

            # Check if values should be formatted as integers
            is_integer_like = np.issubdtype(value.dtype, np.integer) or np.all(
                np.equal(np.mod(value, 1), 0),
            )

            flat = value.flat
            if len(flat) <= 3:
                # Small arrays: preserve structure, format integers appropriately
                if is_integer_like:
                    # Simple approach: convert string representation
                    value_str = str(value.astype(int).tolist())
                else:
                    value_str = str(value.tolist())
            else:
                # Large arrays: show preview with ellipsis
                if is_integer_like:
                    preview = [str(int(x)) for x in flat[:3]]
                else:
                    preview = [f"{x:.3g}" for x in flat[:3]]
                value_str = f"[{', '.join(preview)}, ...]"
            return value_str, shape_str

        # Handle scalars
        if np.issubdtype(value.dtype, np.integer):
            value_str = str(int(value))
        elif np.isrealobj(value):
            value_str = f"{float(value):.6g}"
        else:
            value_str = f"{complex(value):.6g}"

        return value_str, "scalar"

    def _format_dtype(self, param: Constant | Variable) -> str:
        r"""Format parameter dtype string."""
        try:
            dtype_str = str(param.value.dtype)
            common_dtypes = {"float64", "float32", "complex128", "complex64"}
            for dtype in common_dtypes:
                if dtype in dtype_str:
                    return dtype
            return dtype_str
        except (ValueError, TypeError, AttributeError):
            return "unknown"

    def _format_bounds(self, param: Constant | Variable) -> str:
        r"""Format parameter bounds string."""
        if not isinstance(param, Variable):
            return "—"

        bounds = param.bounds
        if bounds == (None, None):
            return "(-∞, +∞)"

        low = "-∞" if bounds[0] is None else f"{bounds[0]:.3g}"
        high = "+∞" if bounds[1] is None else f"{bounds[1]:.3g}"
        return f"({low}, {high})"

    def __repr__(self) -> str:
        r"""
        Returns a rich-formatted string representation of this parameter set.
        """
        if not self:
            return "ParameterSet()"

        table = Table(title=f"ParameterSet ({len(self.names)} parameters)", show_header=True)

        table.add_column("Name", style="#FFB3B3", header_style="#FFB3B3", no_wrap=True)
        table.add_column("Type", style="#FFCC99", header_style="#FFCC99", width=9)
        table.add_column("Value", style="#FFFFBA", header_style="#FFFFBA")
        table.add_column("Dtype", style="#BAFFC9", header_style="#BAFFC9", width=10)
        table.add_column("Bounds", style="#B3E5FF", header_style="#B3E5FF")
        table.add_column("Shape", style="#E1BAFF", header_style="#E1BAFF")

        for name in self.names:
            param = self.all_parameters[name]
            param_type = "Constant" if isinstance(param, Constant) else "Variable"

            value_str, shape_str = self._format_value(param)
            dtype_str = self._format_dtype(param)
            bounds_str = self._format_bounds(param)

            table.add_row(name, param_type, value_str, dtype_str, bounds_str, shape_str)

        with io.StringIO() as string_buffer:
            console = Console(file=string_buffer, width=100, legacy_windows=False)
            console.print(table)
            return string_buffer.getvalue().strip()

    def __bool__(self) -> bool:
        r"""
        ``False`` if this parameter set is empty, ``True`` otherwise.
        """
        return bool(self._constants or self._variables)

    def __eq__(self, other: object) -> bool:
        r"""
        Returns whether ``other`` is equivalent to this parameter set.
        """
        if not isinstance(other, ParameterSet):
            return False
        return (
            self._names == other._names
            and self._constants == other._constants
            and self._variables == other._variables
        )
