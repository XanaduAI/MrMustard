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

"""This module contains the classes to describe a dictionary of parameters."""

from __future__ import annotations

import io
from collections import UserDict
from copy import copy as _copy
from copy import deepcopy as _deepcopy
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from mrmustard.math.backend_manager import BackendManager

from .parameters import Constant, Variable, format_dtype, format_value

math = BackendManager()

__all__ = ["ParameterDict"]


class ParameterDict(UserDict):
    r"""A dictionary-like class for storing parameters.

    >>> c1 = Constant(1.2345, "const1")
    >>> c2 = Constant(2.3456, "const2")
    >>> v1 = Variable(3.4567, "var1")
    >>> pd = ParameterDict(c1, c2, some_name=v1)
    >>> assert pd.names == ['const1', 'const2', 'some_name']
    >>> assert pd.constants == {"const1": c1, "const2": c2}
    >>> assert pd.variables == {"some_name": v1}

    Args:
        *args: Constant or Variable parameters.
        **kwargs: Constant or Variable parameters by name.
    """

    def __init__(self, *args: Constant | Variable, **kwargs: Constant | Variable) -> None:
        super().__init__()
        for arg in args:
            self.data[arg.name] = arg
            if not isinstance(arg, (Constant, Variable)):
                raise ValueError(f"Argument {arg} is not a Constant or Variable")
        for key, value in kwargs.items():
            self.data[key] = value

    def __deepcopy__(self, memo: dict[int, Any]) -> ParameterDict:
        new = type(self)()
        new.data = _deepcopy(self.data, memo)
        return new

    def copy(self) -> ParameterDict:
        return _copy(self)

    def __setitem__(self, key: str, value: Constant | Variable) -> None:
        self.data[key] = value

    def __getattr__(self, key: str) -> Constant | Variable:
        return self.data[key]

    def __hash__(self) -> int:
        return hash(tuple(self.data.values()))

    def __tuple__(self) -> tuple[Constant | Variable, ...]:
        return tuple(self.data.values())

    @property
    def names(self) -> list[str]:
        return list(self.data.keys())

    @property
    def constants(self) -> ParameterDict:
        r"""Returns a ParameterDict of constant parameters in this ParameterDict."""
        return ParameterDict(
            **{name: param for name, param in self.data.items() if isinstance(param, Constant)}
        )

    @property
    def variables(self) -> ParameterDict:
        r"""Returns a ParameterDict of variable parameters in this ParameterDict."""
        return ParameterDict(
            **{name: param for name, param in self.data.items() if isinstance(param, Variable)}
        )

    def to_string(self, decimals: int) -> str:
        r"""Returns a string representation of the parameter values, separated by commas and rounded
        to the specified number of decimals.

        Args:
            decimals (int): number of decimals to round to

        Returns:
            str: string representation of the parameter values
        """
        strings = []
        for name, param in self.data.items():
            if len(param.value.shape) == 0:  # don't show arrays
                string = str(np.round(math.asnumpy(param.value), decimals))
            else:
                string = f"{name}"
            strings.append(string)
        return ", ".join(strings)

    def __repr__(self) -> str:
        r"""Returns a rich-formatted string representation of this parameter set."""
        if not self:
            return "ParameterDict()"

        table = Table(title=f"ParameterDict ({len(self.names)} parameters)", show_header=True)

        table.add_column("Name", style="#FFB3B3", header_style="#FFB3B3", no_wrap=True)
        table.add_column("Type", style="#FFCC99", header_style="#FFCC99", width=9)
        table.add_column("Value", style="#FFFFBA", header_style="#FFFFBA")
        table.add_column("Dtype", style="#BAFFC9", header_style="#BAFFC9", width=10)
        table.add_column("Shape", style="#E1BAFF", header_style="#E1BAFF")

        for name in self.names:
            param = self.data[name]
            param_type = "Constant" if isinstance(param, Constant) else "Variable"

            value_str, shape_str = format_value(param)
            dtype_str = format_dtype(param)

            table.add_row(name, param_type, value_str, dtype_str, shape_str)

        with io.StringIO() as string_buffer:
            console = Console(file=string_buffer, width=100, legacy_windows=False)
            console.print(table)
            return string_buffer.getvalue().strip()
