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

"""The class representing a displacement gate."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .base import Unitary
from .builtins import displacement_gate, displacement_gate_fock

__all__ = ["Dgate"]


class Dgate(Unitary):
    r"""The displacement gate.

    >>> from mrmustard.lab import Dgate
    >>> unitary = Dgate(mode=1, alpha=0.1 + 0.2j)
    >>> assert unitary.modes == (1,)
    >>> assert unitary.parameters.alpha.value == 0.1 + 0.2j

    Args:
        mode: The mode this gate is applied to.
        alpha: The displacement in the complex phase space.
        name: A name for the gate. If not provided, the class name will be used.

    .. details::

        For any :math:`\bar{\alpha} = \bar{x} + i\bar{y}` of length :math:`N`, the :math:`N`-mode
        displacement gate is defined by

        .. math::
            S = I_N \text{ and } r = \sqrt{2\hbar}\big[\text{Re}(\bar{\alpha}), \text{Im}(\bar{\alpha})\big].

        Its ``(A,b,c)`` triple is given by

        .. math::
            A &= \begin{bmatrix}
                    O_N & I_N\\
                    I_N & O_N
                \end{bmatrix} \\ \\
            b &= \begin{bmatrix}
                    \bar{\alpha} & -\bar{\alpha}^*
                \end{bmatrix} \\ \\
            c &= \text{exp}\big(-|\bar{\alpha}^2|/2\big).
    """

    short_name = "D"

    def __init__(
        self,
        mode: int | tuple[int],
        alpha: complex | Sequence[complex] | Parameter = 0.0 + 0.0j,
        name: str | None = None,
    ) -> None:
        mode = (mode,) if not isinstance(mode, tuple) else mode
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (displacement_gate, ("alpha", "lin_sup")),
                    ReprEnum.FOCK: (displacement_gate_fock, ("alpha", "shape", "lin_sup")),
                }
            ),
            wires=Wires(modes_in_ket=set(mode), modes_out_ket=set(mode)),
            name=name,
        )
        self.parameters["alpha"] = Parameter.from_cc_init(alpha, "complex128", f"{self.name}/alpha")
