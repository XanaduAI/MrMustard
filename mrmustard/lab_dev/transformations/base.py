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
This module contains the base classes for the available quantum states.
"""

from __future__ import annotations

from typing import Sequence

from ..circuit_components import CircuitComponent

__all__ = ["Transformation", "Unitary", "Channel"]


class Transformation(CircuitComponent):
    r"""
    Base class for all transformations.
    """

    def __rshift__(self, other: CircuitComponent):
        raise NotImplementedError


class Unitary(Transformation):
    r"""
    Base class for all unitary transformations.

    Arguments:
        name: The name of this transformation.
        modes: The modes that this transformation acts on.
    """

    def __init__(self, name: str, modes: Sequence[int]):
        super().__init__(name, modes_in_ket=modes, modes_out_ket=modes)


class Channel(Transformation):
    r"""
    Base class for all non-unitary transformations.

    Arguments:
        name: The name of this transformation.
        modes: The modes that this transformation acts on.
    """

    def __init__(self, name: str, modes: Sequence[int]):
        super().__init__(
            name, modes_in_ket=modes, modes_out_ket=modes, modes_in_bra=modes, modes_out_bra=modes
        )
