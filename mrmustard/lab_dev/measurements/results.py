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

"""
This module contains the base classes for the available measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from mrmustard.physics.representations import Fock
from ..circuit_components import CircuitComponent

__all__ = ["Results"]


class Results:
    r"""
    A class to store results.

    Arguments:
        data: The data.
    """

    def __init__(self, data: Optional[dict[str, tuple[int, CircuitComponent]]] = None):
        self._data = data or {}

    @property
    def data(self) -> dict[str, tuple[int, CircuitComponent]]:
        r"""
        The data in this result object.
        """
        return self._data

    def add(self, tag: str, leftover: CircuitComponent) -> None:
        try:
            self._data[tag][0] += 1
        except KeyError:
            self._data[tag] = [1, leftover]

    def __repr__(self) -> str:
        msg = ""
        for t, (n, _) in self.data.items():
            msg += f"'{t}': {n}, "
            if len(msg) >= 50:
                msg += "..."
                return f"Results({msg})"
        return f"Results({msg[:-2]})"
