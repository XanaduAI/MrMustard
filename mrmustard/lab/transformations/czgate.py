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

"""The class representing a controlled-phase gate."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .base import Unitary
from .builtins import cz_gate

__all__ = ["CZgate"]


class CZgate(Unitary):
    r"""Controlled Z gate.

    >>> from mrmustard.lab import CZgate
    >>> gate = CZgate((0, 1), s=0.5)
    >>> assert gate.modes == (0, 1)
    >>> assert gate.parameters.s.value == 0.5

    Args:
        modes: The pair of modes of the controlled-Z gate.
        s: The control parameter.
        name: A name for the gate. If not provided, the class name will be used.

    .. details::
        We have that the controlled-Z gate is defined as

        .. math::

            C_Z = \exp(is q_1 \otimes q_2 / \hbar).

        Reference: https://arxiv.org/pdf/2110.03247.pdf, Equation 8.
        https://arxiv.org/pdf/1110.3234.pdf, Equation 161.
    """

    short_name = "CZ"

    def __init__(
        self,
        modes: tuple[int, int],
        s: float | Sequence[float] | Parameter = 0.0,
        name: str | None = None,
    ):
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (cz_gate, ("s", "lin_sup"))}
            ),
            wires=Wires(modes_in_ket=set(modes), modes_out_ket=set(modes)),
            name=name,
        )
        self.parameters["s"] = Parameter.from_cc_init(s, "float64", f"{self.name}/s")
