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
The class representing a generic gaussian gate.
"""

from __future__ import annotations

from typing import Sequence
from mrmustard.utils.typing import RealMatrix

from .base import Unitary
from ...physics.representations import Bargmann
from ..utils import make_parameter

__all__ = ["Ggate"]


class Ggate(Unitary):
    r"""
    The generic N-mode Gaussian gate.

    .. code-block ::

        >>> from mrmustard import math
        >>> from mrmustard.lab_dev import Ggate, Vacuum, Identity

        >>> U = Ggate(modes=[0], symplectic=math.random_symplectic(1))
        >>> assert isinstance(Vacuum([0]) >> U, Ket)
        >>> assert U >> U.dual == Identity([0])

    Args:
        modes: The modes this gate is applied to.
        symplectic: The symplectic matrix of the gate in the XXPP ordering.
    """

    short_name = "G"

    def __init__(
        self,
        modes: Sequence[int],
        symplectic: RealMatrix,
        symplectic_trainable: bool = False,
    ):
        super().__init__(modes_out=modes, modes_in=modes, name="Ggate")
        S = make_parameter(symplectic_trainable, symplectic, "symplectic", (None, None))
        self.parameter_set.add_parameter(S)

        self._representation = Bargmann.from_function(
            fn=lambda s: Unitary.from_symplectic(modes, s).bargmann_triple(),
            s=self.parameter_set.symplectic,
        )
