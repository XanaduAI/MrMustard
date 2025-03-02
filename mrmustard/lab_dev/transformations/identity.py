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
The classes representing an identity gate.
"""

from __future__ import annotations

from .base import Unitary
from ...physics.ansatz import PolyExpAnsatz
from ...physics import triples

__all__ = ["Identity"]


class Identity(Unitary):
    r"""
    The identity gate.

    .. code-block ::

        >>> from mrmustard.lab_dev import Identity

        >>> unitary = Identity(modes=(1, 2))
        >>> assert unitary.modes == (1, 2)

    Args:
        modes: The modes this gate is applied to.
    """

    short_name = "I"

    def __init__(
        self,
        modes: tuple[int, ...],
    ):
        super().__init__(name="Identity")
        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(fn=triples.identity_Abc, n_modes=len(modes)),
        ).representation
