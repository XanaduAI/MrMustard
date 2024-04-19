# Copyright 2024 Xanadu Quantum Technologies Inc.

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

__all__ = ["_DsMap"]


class _DsMap(CircuitComponent):
    r"""The s-parametrized Dgate as a channel.
    This will be used as an internal Channel for representation transformation.

    Args:
        num_modes: number of modes of this channel.
        s: The `s` parameter of this channel.
    """

    def __init__(
        self,
        modes: Sequence[int],
        s: float,
    ):
        super().__init__(
            "_DsMap",
            modes_out_bra=modes,
            modes_in_bra=modes,
            modes_out_ket=modes,
            modes_in_ket=modes,
        )
        self.s = s

    @property
    def representation(self) -> Bargmann:
        return Bargmann(*triples.displacement_map_s_parametrized_Abc(self.s, len(self.modes)))
