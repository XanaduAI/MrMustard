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

from .base import Operation
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter, reshape_params

__all__ = ["FockDamping"]


class FockDamping(Operation):
    r"""
    The Fock damping operator.

    If ``damping`` is an iterable, its length must be equal to `1` or `N`. If it length is equal to `1`,
    all the modes share the same damping.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import FockDamping, Coherent

        >>> operator = FockDamping(modes=[0], damping=0.1)
        >>> input_state = Coherent(modes=[0], x=1, y=0.5)
        >>> output_state = input_state >> operator
        >>> assert operator.modes == [0]
        >>> assert np.allclose(operator.damping.value, [0.1, 0.1])
        >>> assert output_state.L2_norm < 1

    Args:
        modes: The modes this gate is applied to.
        damping: The damping parameter.
        damping_trainable: Whether the damping is a trainable variable.
        damping_bounds: The bounds for the damping.

    .. details::

        Its ``(A,b,c)`` triple is given by

        .. math::
            A &= e^{-\beta}\begin{bmatrix}
                    O_N & I_N & \\
                    I_N & O_N &

                \end{bmatrix} \\ \\
            b &= O_{2N} \\ \\
            c &= 1\:.
    """

    def __init__(
        self,
        modes: Sequence[int],
        damping: float | Sequence[float] | None = 0.0,
        damping_trainable: bool = False,
        damping_bounds: tuple[float | None, float | None] = (0.0, None),
    ):
        super().__init__(modes_out=modes, modes_in=modes, name="FockDamping")
        (betas,) = list(reshape_params(len(modes), damping=damping))
        self._add_parameter(
            make_parameter(
                damping_trainable,
                betas,
                "damping",
                damping_bounds,
                None,
            )
        )
        self._representation = Bargmann.from_function(
            fn=triples.fock_damping_Abc, beta=self.damping
        )
