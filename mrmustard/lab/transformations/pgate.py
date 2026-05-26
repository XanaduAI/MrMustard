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

"""The class representing a quadratic phase gate."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .base import Unitary
from .builtins import p_gate

__all__ = ["Pgate"]


class Pgate(Unitary):
    r"""Quadratic phase gate.

    Args:
        modes: The modes this gate is applied to.
        shearing: The shearing parameter.
        name: A name for the gate. If not provided, the class name will be used.

    .. details::
        The quadratic phase gate is defined as

        .. math::

            P = \exp(i s q^2 / 2 \hbar)

    Reference: https://strawberryfields.ai/photonics/conventions/gates.html
    """

    short_name = "P"

    def __init__(
        self,
        mode: int | tuple[int],
        shearing: float | Sequence[float] | Parameter = 0.0,
        name: str | None = None,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (p_gate, ("shearing", "lin_sup"))}
            ),
            wires=Wires(modes_in_ket=set(mode), modes_out_ket=set(mode)),
            name=name,
        )
        self.parameters["shearing"] = Parameter.from_cc_init(
            shearing, "float64", f"{self.name}/shearing"
        )
