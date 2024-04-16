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
A set of components that do not correspond to physical elements of a circuit, but can be used to
perform useful mathematical calculations.
"""

# pylint: disable=super-init-not-called, protected-access

from __future__ import annotations

from typing import Sequence

from mrmustard.physics import triples
from .circuit_components import CircuitComponent
from ..physics.representations import Bargmann

__all__ = ["TraceOut"]


class TraceOut(CircuitComponent):
    r"""
    A circuit component to perform trace-out operations.

    Args:
        modes: The modes to trace out.
    """

    def __init__(
        self,
        modes: Sequence[int],
    ):
        super().__init__("Tr", modes_in_ket=modes, modes_in_bra=modes)

    @property
    def representation(self) -> Bargmann:
        return Bargmann(*triples.identity_Abc(len(self.modes)))