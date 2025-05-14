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
The class representing a beam splitter gate.
"""

from __future__ import annotations

from typing import Sequence
from dataclasses import replace

from mrmustard import math
from mrmustard.utils.typing import ComplexTensor
from .base import Unitary
from ...physics.ansatz import PolyExpAnsatz, ArrayAnsatz
from ...physics import triples
from ..utils import make_parameter
from ...physics.wires import Wires, ReprEnum
from ...physics.representations import Representation
from ...physics import fock_utils

__all__ = ["BSgate"]


class BSgate(Unitary):
    r"""
    The beam splitter gate.

    Args:
        modes: The pair of modes of the beam splitter gate.
        theta: The transmissivity angle.
        phi: The phase angle.
        theta_trainable: Whether ``theta`` is a trainable variable.
        phi_trainable: Whether ``phi`` is a trainable variable.
        theta_bounds: The bounds for ``theta``.
        phi_bounds: The bounds for ``phi``.

        .. code-block ::

        >>> from mrmustard.lab import BSgate

        >>> unitary = BSgate(modes=(1, 2), theta=0.1)
        >>> assert unitary.modes == (1, 2)
        >>> assert unitary.parameters.theta.value == 0.1
        >>> assert unitary.parameters.phi.value == 0.0

    .. details::

        The beamsplitter gate is a Gaussian gate defined by

        .. math::
            S = \begin{bmatrix}
                    \text{Re}(U) & -\text{Im}(U)\\
                    \text{Im}(U) & \text{Re}(U)
                \end{bmatrix} \text{ and }
            d = O_4\:,

        with

        .. math::
            U &= \begin{bmatrix}
                    \text{cos}(\theta) & -e^{-i\phi}\text{sin}(\theta)\\
                    e^{i\phi}\text{sin}(\theta) & \text{cos}(\theta)
                \end{bmatrix} \\

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = \begin{bmatrix}
                    O_2 & U \\
                    U^{T} & O_2
                \end{bmatrix} \text{, }
            b = O_{4} \text{, and }
            c = 1
    """

    short_name = "BS"

    def __init__(
        self,
        modes: tuple[int, int],
        theta: float | Sequence[float] = 0.0,
        phi: float | Sequence[float] = 0.0,
        theta_trainable: bool = False,
        phi_trainable: bool = False,
        theta_bounds: tuple[float | None, float | None] = (None, None),
        phi_bounds: tuple[float | None, float | None] = (None, None),
    ):
        super().__init__(name="BSgate")
        self.parameters.add_parameter(make_parameter(theta_trainable, theta, "theta", theta_bounds))
        self.parameters.add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))
        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.beamsplitter_gate_Abc,
                theta=self.parameters.theta,
                phi=self.parameters.phi,
            ),
        ).representation

    def fock_array(
        self, shape: int | Sequence[int] = None, method: str = "vanilla"
    ) -> ComplexTensor:
        r"""
        Returns the unitary representation of the Beam Splitter gate in the Fock basis.

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it defaults to
                ``settings.DEFAULT_FOCK_SIZE``.
            method: The method to use to compute the Fock array. Available methods are:
                - ``"vanilla"``: The default method.
                - ``"schwinger"``: Use the Schwinger representation to compute the Fock array.
        Returns:
            array: The Fock representation of this component.
        """
        if isinstance(shape, int):
            shape = (shape,) * (2 * self.ansatz.num_vars)
        shape = tuple(shape) or tuple(self.auto_shape())
        if len(shape) != 4:
            raise ValueError(f"Expected Fock shape of length {4}, got length {len(shape)}")

        if self.ansatz.batch_shape:
            theta, phi = math.broadcast_arrays(
                self.parameters.theta.value, self.parameters.phi.value
            )
            theta = math.reshape(theta, (-1,))
            phi = math.reshape(phi, (-1,))
            ret = math.astensor(
                [
                    fock_utils.beamsplitter(t, p, shape=shape, method=method)
                    for t, p in zip(theta, phi)
                ]
            )
            ret = math.reshape(ret, self.ansatz.batch_shape + shape)
        else:
            ret = fock_utils.beamsplitter(
                self.parameters.theta.value,
                self.parameters.phi.value,
                shape=shape,
                method=method,
            )
        return ret

    def to_fock(self, shape: int | Sequence[int] | None = None) -> BSgate:
        batch_dims = self.ansatz.batch_dims - 1 if self.ansatz._lin_sup else self.ansatz.batch_dims
        fock = ArrayAnsatz(self.fock_array(shape), batch_dims=batch_dims)
        fock._original_abc_data = self.ansatz.triple
        ret = self.__class__(self.modes, **self.parameters.to_dict())
        wires = Wires.from_wires(
            quantum={replace(w, repr=ReprEnum.FOCK) for w in self.wires.quantum},
            classical={replace(w, repr=ReprEnum.FOCK) for w in self.wires.classical},
        )
        ret._representation = Representation(fock, wires)
        return ret
