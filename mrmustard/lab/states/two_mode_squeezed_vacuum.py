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

"""The class representing a two-mode squeezed vacuum state."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import two_mode_squeezed_vacuum_state
from .ket import Ket

__all__ = ["TwoModeSqueezedVacuum"]


class TwoModeSqueezedVacuum(Ket):
    r"""The two-mode squeezed vacuum state.

    >>> from mrmustard.lab import TwoModeSqueezedVacuum, S2gate, Vacuum
    >>> state = TwoModeSqueezedVacuum(modes=(0, 1), r=0.3, phi=0.2)
    >>> assert state == Vacuum((0,1)) >> S2gate((0, 1), r=0.3, phi=0.2)

    Args:
        modes: The modes of the two-mode squeezed vacuum state.
        r: The squeezing magnitude.
        phi: The squeezing angle.
        name: A name for the state. If not provided, the class name will be used.

    Returns:
        A ``Ket`` type object that represents the two-mode squeezed vacuum state.
    """

    short_name = "TMSq"

    def __init__(
        self,
        modes: tuple[int, int],
        r: float | Sequence[float] | Parameter = 0.0,
        phi: float | Sequence[float] | Parameter = 0.0,
        name: str | None = None,
    ):
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (two_mode_squeezed_vacuum_state, ("r", "phi", "lin_sup"))
                }
            ),
            wires=Wires(modes_out_ket=set(modes)),
            name=name,
        )
        self.parameters["r"] = Parameter.from_cc_init(r, "float64", f"{self.name}/r")
        self.parameters["phi"] = Parameter.from_cc_init(phi, "float64", f"{self.name}/phi")
