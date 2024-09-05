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
The class representing a rotation gate.
"""

from __future__ import annotations

from typing import Sequence

from .base import Unitary
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter, reshape_params

__all__ = ["Rgate"]


class Rgate(Unitary):
    r"""
    The rotation gate.

    If ``theta`` is an iterable, its length must be equal to `1` or `N`. If its length is equal to `1`,
    all the modes share the same ``theta``.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Rgate

        >>> unitary = Rgate(modes=[1, 2], phi=0.1)
        >>> assert unitary.modes == [1, 2]

    Args:
        modes: The modes this gate is applied to.
        theta: The rotation angles.
        theta_bounds: The bounds for ``theta``.
        theta_trainable: Whether ``theta`` is a trainable variable.
    """

    short_name = "R"

    def __init__(
        self,
        modes: Sequence[int],
        phi: float | Sequence[float] = 0.0,
        phi_trainable: bool = False,
        phi_bounds: tuple[float | None, float | None] = (0.0, None),
    ):
        super().__init__(modes_out=modes, modes_in=modes, name="Rgate")
        (phis,) = list(reshape_params(len(modes), phi=phi))
        self._add_parameter(make_parameter(phi_trainable, phis, "phi", phi_bounds))

        self._representation = Bargmann.from_function(fn=triples.rotation_gate_Abc, theta=self.phi)
