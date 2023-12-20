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

from typing import Optional, Sequence, Tuple, Union
import numpy as np

from mrmustard import math
from ..physics.representations import Bargmann
from ..utils.typing import Batch, ComplexMatrix, ComplexTensor, ComplexVector, Mode
from .circuits import Circuit
from .circuit_components import CircuitComponent
from .utils import make_parameter

__all__ = ["Dgate", "Transformation", "Unitary"]


class Transformation(CircuitComponent):
    r"""
    Base class for all transformations.
    """

    def __rshift__(self, other: CircuitComponent):
        r"""
        Returns a ``Circuit`` with two components, light-copied from ``self`` and ``other``.
        """
        return Circuit([self, other])


class Unitary(Transformation):
    r"""
    Base class for all unitary transformations.

    Arguments:
        name: The name of this unitary.
        modes: The modes that this unitary acts on.
    """

    def __init__(self, name, modes):
        super().__init__(name, modes_in_ket=modes, modes_out_ket=modes)


class Dgate(Unitary):
    r"""

    If ``len(modes) > 1`` the gate is applied in parallel to all of the modes provided.

    If a parameter is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats. One can optionally set bounds for each
    parameter, which the optimizer will respect.

    Args:
        x (float or List[float]): the list of displacements along the x axis
        x_bounds (float, float): bounds for the displacement along the x axis
        x_trainable (bool): whether x is a trainable variable
        y (float or List[float]): the list of displacements along the y axis
        y_bounds (float, float): bounds for the displacement along the y axis
        y_trainable bool: whether y is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        x: Union[float, Sequence[float]] = 0.0,
        y: Union[float, Sequence[float]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[Sequence[int]] = None,
    ) -> None:
        m = max(len(math.atleast_1d(x)), len(math.atleast_1d(y)))
        super().__init__("Dgate", modes=modes or list(range(m)))
        self._add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, y, "y", y_bounds))

    @property
    def representation(self) -> Bargmann:
        num_modes = len(self.modes)

        xs = math.atleast_1d(self.x.value)
        if len(xs) == 1:
            xs = np.array([xs[0] for _ in range(num_modes)])
        ys = math.atleast_1d(self.y.value)
        if len(ys) == 1:
            ys = np.array([ys[0] for _ in range(num_modes)])

        A = np.kron(np.array([[0, 1], [1, 0]]), math.eye(num_modes))
        B = math.concat([xs + 1j * ys, -xs + 1j * ys], axis=0)
        C = np.prod([np.exp(-abs(x + 1j * y) ** 2 / 2) for x, y in zip(xs, ys)])

        return Bargmann(A, B, C)
    

class Channel(Transformation):
    r"""
    Base class for all non-unitary transformations.

    Arguments:
        name: The name of this channel.
        modes: The modes that this channel acts on.
    """

    def __init__(self, name, modes):
        super().__init__(
            name, modes_in_ket=modes, modes_out_ket=modes, modes_in_bra=modes, modes_out_bra=modes
        )
    
    
class Attenuator(Channel):
    r"""The noisy attenuator channel.

    It corresponds to mixing with a thermal environment and applying the pure loss channel. The pure
    lossy channel is recovered for nbar = 0 (i.e. mixing with vacuum).

    The CPT channel is given by

    .. math::

        X = sqrt(transmissivity) * I
        Y = (1-transmissivity) * (2*nbar + 1) * (hbar / 2) * I

    If ``len(modes) > 1`` the gate is applied in parallel to all of the modes provided.
    If ``transmissivity`` is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats.

    One can optionally set bounds for `transmissivity`, which the optimizer will respect.

    Args:
        transmissivity (float or List[float]): the list of transmissivities
        nbar (float): the average number of photons in the thermal state
        transmissivity_trainable (bool): whether transmissivity is a trainable variable
        nbar_trainable (bool): whether nbar is a trainable variable
        transmissivity_bounds (float, float): bounds for the transmissivity
        nbar_bounds (float, float): bounds for the average number of photons in the thermal state
        modes (optional, List[int]): the list of modes this gate is applied to
    """
    def __init__(
        self,
        transmissivity: Union[Optional[float], Optional[list[float]]] = 1.0,
        nbar: float = 0.0,
        transmissivity_trainable: bool = False,
        nbar_trainable: bool = False,
        transmissivity_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        nbar_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        modes: Optional[list[int]] = None,
    ):
        super().__init__(
            modes=modes or list(range(len(math.atleast_1d(transmissivity)))),
            name="Att",
        )
        self._add_parameter(
            make_parameter(
                transmissivity_trainable,
                transmissivity,
                "transmissivity",
                transmissivity_bounds,
                None,
            )
        )
        self._add_parameter(make_parameter(nbar_trainable, nbar, "nbar", nbar_bounds))
