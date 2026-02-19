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
The class representing a squeezed vacuum state.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import squeezed_vacuum_state, squeezed_vacuum_state_fock
from .ket import Ket

__all__ = ["SqueezedVacuum"]


class SqueezedVacuum(Ket):
    r"""
    The squeezed vacuum state in Bargmann representation.

    >>> from mrmustard.lab import SqueezedVacuum, Vacuum, Sgate
    >>> state = SqueezedVacuum(mode=0, r=0.3, phi=0.2)
    >>> assert state == Vacuum(0) >> Sgate(0, r=0.3, phi=0.2)

    Args:
        mode: The mode of the squeezed vacuum state.
        r: The squeezing magnitude.
        phi: The squeezing angle.
    """

    short_name = "Sq"

    def __init__(
        self,
        mode: int | tuple[int],
        r: float | Sequence[float] | Parameter = 0.0,
        phi: float | Sequence[float] | Parameter = 0.0,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (squeezed_vacuum_state, ("r", "phi", "lin_sup")),
                    ReprEnum.FOCK: (squeezed_vacuum_state_fock, ("r", "phi", "shape", "lin_sup")),
                }
            ),
            wires=Wires(modes_out_ket=set(mode)),
            name=self.__class__.__name__,
        )
        self.parameters["r"] = Parameter.from_cc_init(r, "float64", f"{self.name}/r")
        self.parameters["phi"] = Parameter.from_cc_init(phi, "float64", f"{self.name}/phi")
