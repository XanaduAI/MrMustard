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

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import ComplexTensor

from .base import Unitary
from .builtins import beamsplitter_gate, beamsplitter_gate_fock

__all__ = ["BSgate"]


class BSgate(Unitary):
    r"""
    The beam splitter gate.

    >>> from mrmustard.lab import BSgate
    >>> unitary = BSgate(modes=(1, 2), theta=0.1)
    >>> assert unitary.modes == (1, 2)
    >>> assert unitary.parameters.theta.value == 0.1
    >>> assert unitary.parameters.phi.value == 0.0

    Args:
        modes: The pair of modes of the beam splitter gate.
        theta: The transmissivity angle.
        phi: The phase angle.

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
        theta: float | Sequence[float] | Parameter = 0.0,
        phi: float | Sequence[float] | Parameter = 0.0,
    ):
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (beamsplitter_gate, ("theta", "phi", "lin_sup")),
                    ReprEnum.FOCK: (
                        beamsplitter_gate_fock,
                        ("theta", "phi", "shape", "method", "lin_sup"),
                    ),
                }
            ),
            wires=Wires(modes_in_ket=set(modes), modes_out_ket=set(modes)),
            name=self.__class__.__name__,
        )
        self.parameters["theta"] = Parameter.from_cc_init(theta, "float64", f"{self.name}/theta")
        self.parameters["phi"] = Parameter.from_cc_init(phi, "float64", f"{self.name}/phi")

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
        if self.ansatz_factory is None:
            raise ValueError("CircuitComponent has no ansatz factory.")
        shape = self._check_fock_shape(shape)
        ansatz_factory = self.ansatz_factory
        if ansatz_factory.ansatz_dict.get(ReprEnum.FOCK, None) is None:
            ansatz_factory = self.to_fock(shape).ansatz_factory
        return (
            ansatz_factory(
                **self.parameters,
                representation=ReprEnum.FOCK,
                shape=shape,
                method=method,
            )
            .reduce(shape)
            .array
        )
