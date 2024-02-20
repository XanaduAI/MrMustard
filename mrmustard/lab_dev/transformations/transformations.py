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
The classes representing transformations in quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Iterable, Tuple, Union

from mrmustard import math
from .base import Unitary, Channel
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter

__all__ = ["Attenuator", "Dgate"]


class Dgate(Unitary):
    r"""
    Phase space displacement gate.

    If ``x`` and/or ``y`` are iterables, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Dgate

        >>> gate = Dgate(modes=[1, 2], 0.1, [0.2, 0.3])
        >>> assert gate.modes == [1, 2]
        >>> assert np.allclose(gate.x.value, [0.1, 0.1])
        >>> assert np.allclose(gate.y.value, [0.2, 0.3])

    To apply mode-specific values use a list of floats, one can optionally set bounds for each
    parameter, which the optimizer will respect.

    Args:
        modes: The modes this gate is applied to.
        x: The displacements along the `x` axis.
        x_bounds: The bounds for the displacement along the `x` axis.
        x_trainable: Whether `x` is a trainable variable.
        y: The displacements along the `y` axis.
        y_bounds: The bounds for the displacement along the `y` axis.
        y_trainable: Whether `y` is a trainable variable.

    .. details::

        The displacement gate is a Gaussian gate defined as

        .. math::
            D(\alpha) = \exp(\alpha a^\dagger -\alpha^* a) = \exp\left(-i\sqrt{2}(\re(\alpha) \hat{p} -\im(\alpha) \hat{x})/\sqrt{\hbar}\right)

        where :math:`\alpha = x + iy`.
    """

    def __init__(
        self,
        modes: Iterable[int] = None,
        x: Union[float, Iterable[float]] = 0.0,
        y: Union[float, Iterable[float]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ) -> None:
        super().__init__("Dgate", modes=modes)
        self._add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, y, "y", y_bounds))

    @property
    def representation(self) -> Bargmann:
        num_modes = len(self.modes)

        xs = math.atleast_1d(self.x.value)
        if len(xs) == 1:
            xs = math.astensor([xs[0] for _ in range(num_modes)])
        ys = math.atleast_1d(self.y.value)
        if len(ys) == 1:
            ys = math.astensor([ys[0] for _ in range(num_modes)])

        return Bargmann(*triples.displacement_gate_Abc(xs, ys))


class Attenuator(Channel):
    r"""The noisy attenuator channel.

    If ``transmissivity`` is an iterable, its length must be equal to `1` or `N`. If it length is equal to `1`,
    all the modes share the same transmissivity.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Attenuator

        >>> channel = Attenuator(modes=[1, 2], 0.1)
        >>> assert channel.modes == [1, 2]
        >>> assert np.allclose(channel.transmissivity.value, [0.1, 0.1])

    Args:
        modes: The modes this gate is applied to.
        transmissivity: The transmissivity.
        transmissivity_trainable: Whether the transmissivity is a trainable variable.
        transmissivity_bounds: The bounds for the transmissivity.

    .. details::

        The attenuator is defined as

        .. math::
            ??@yuan
    """

    def __init__(
        self,
        modes: Optional[Iterable[int]] = None,
        transmissivity: Union[Optional[float], Optional[list[float]]] = 1.0,
        transmissivity_trainable: bool = False,
        transmissivity_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
    ):
        super().__init__(modes=modes, name="Att")
        self._add_parameter(
            make_parameter(
                transmissivity_trainable,
                transmissivity,
                "transmissivity",
                transmissivity_bounds,
                None,
            )
        )

    @property
    def representation(self) -> Bargmann:
        eta = self.transmissivity.value
        return Bargmann(*triples.attenuator_Abc(eta))
