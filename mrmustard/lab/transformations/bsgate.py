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

from collections.abc import Sequence

from mrmustard import math
from mrmustard.utils.typing import ComplexTensor

from ...physics import triples
from ...physics.ansatz import PolyExpAnsatz
from ...physics.wires import Wires
from .base import Unitary

__all__ = ["BSgate"]


class BSgate(Unitary):
    r"""
    The beam splitter gate.

    Args:
        modes: The pair of modes of the beam splitter gate.
        theta: The transmissivity angle.
        phi: The phase angle.

        .. code-block::

        >>> from mrmustard.lab import BSgate

        >>> unitary = BSgate(modes=(1, 2), theta=0.1)
        >>> assert unitary.modes == (1, 2)

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
    ):
        A, b, c = triples.beamsplitter_gate_Abc(theta=theta, phi=phi)
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_in_ket=set(modes), modes_out_ket=set(modes))

        def specialized_fock(shape, method="stable", **kwargs):
            """Optimized Fock computation using beamsplitter formula."""
            theta_tensor = math.astensor(theta)
            phi_tensor = math.astensor(phi)
            if ansatz.batch_shape:
                theta_local, phi_local = math.broadcast_arrays(theta_tensor, phi_tensor)
                theta_local = math.reshape(theta_local, (-1,))
                phi_local = math.reshape(phi_local, (-1,))
                ret = math.astensor(
                    [
                        math.beamsplitter(t, p, shape=shape, method=method)
                        for t, p in zip(theta_local, phi_local)
                    ],
                )
                ret = math.reshape(ret, ansatz.batch_shape + shape)
                if ansatz._lin_sup:
                    ret = math.sum(ret, axis=ansatz.batch_dims - 1)
            else:
                ret = math.beamsplitter(theta_tensor, phi_tensor, shape=shape, method=method)
            return ret

        self._specialized_fock = specialized_fock

        super().__init__(ansatz=ansatz, wires=wires, name="BSgate")

    def fock_array(
        self,
        shape: int | Sequence[int] | None = None,
        method: str = "stable",
    ) -> ComplexTensor:
        r"""
        Returns the unitary representation of the Beam Splitter gate in the Fock basis.

        Args:
            shape: The shape of the returned representation. If ``shape`` is given as an ``int``,
                it is broadcasted to all the dimensions. If not given, it defaults to
                ``settings.DEFAULT_FOCK_SIZE``.
            method: The method to use to compute the Fock array. Available methods are:
                - ``"vanilla"``: standard recurrence relation (not numerically stable, but slightly faster than the stable one).
                - ``"schwinger"``: Use the Schwinger representation to compute the Fock array.
                - ``"stable"``: Use the stable implementation of the beamsplitter. (default)

        Returns:
            array: The Fock representation of this component.

        Raises:
            ValueError: If the shape is not valid for the component.
        """
        shape = self._check_fock_shape(shape)
        return self._specialized_fock(shape, method=method)
