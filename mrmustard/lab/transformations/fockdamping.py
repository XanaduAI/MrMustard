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

"""The class representing a rotation gate."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .base import Operation
from .builtins import fock_damping_operation

__all__ = ["FockDamping"]


class FockDamping(Operation):
    r"""The Fock damping operator.

    >>> from mrmustard.lab import FockDamping, Coherent
    >>> operator = FockDamping(mode=0, damping=0.1)
    >>> input_state = Coherent(mode=0, alpha=1 + 0.5j)
    >>> output_state = input_state >> operator
    >>> assert operator.modes == (0,)
    >>> assert operator.parameters.damping.value == 0.1
    >>> assert output_state.L2_norm < 1

    Args:
        mode: The mode this gate is applied to.
        damping: The damping parameter.
        name: A name for the operator. If not provided, the class name will be used.

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

    short_name = "FDamp"

    def __init__(
        self,
        mode: int | tuple[int],
        damping: float | Sequence[float] | Parameter = 0.0,
        name: str | None = None,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (fock_damping_operation, ("damping", "lin_sup"))}
            ),
            wires=Wires(modes_in_ket=set(mode), modes_out_ket=set(mode)),
            name=name,
        )
        self.parameters["damping"] = Parameter.from_cc_init(
            damping, "float64", f"{self.name}/damping"
        )
