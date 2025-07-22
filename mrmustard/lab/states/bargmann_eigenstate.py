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
The class representing a Bargmann eigenstate.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from ..utils import make_parameter
from .ket import Ket

__all__ = ["BargmannEigenstate"]


class BargmannEigenstate(Ket):
    r"""
    The `N`-mode Bargmann eigenstate.

    Args:
        modes: A list of modes.
        alpha: The displacement of the state (i.e., the eigen-value).

    Notes:
        The only difference with ``Coherent(modes, alphas)`` is in its `c` parameter (and hence, does not have unit norm).

    .. code-block::

        >>> from mrmustard.lab import BargmannEigenstate

        >>> state = BargmannEigenstate(1, 0.1 + 0.5j)
        >>> assert state.modes == (1,)

    .. details::

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = 0 , b = \alpha, c = 1.

    """

    short_name = "Be"

    def __init__(
        self,
        mode: int | tuple[int],
        alpha: complex | Sequence[complex] = 0.0j,
        alpha_trainable: bool = False,
        alpha_bounds: tuple[complex | None, complex | None] = (None, None),
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(name="BargmannEigenstate")

        self.parameters.add_parameter(
            make_parameter(
                is_trainable=alpha_trainable,
                value=alpha,
                name="alpha",
                bounds=alpha_bounds,
                dtype=math.complex128,
            ),
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=triples.bargmann_eigenstate_Abc,
            alpha=self.parameters.alpha,
        )
        self._wires = Wires(modes_out_ket=set(mode))
