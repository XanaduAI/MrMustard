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
The class representing a displaced squeezed state.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import displaced_squeezed_vacuum_state
from .ket import Ket

__all__ = ["DisplacedSqueezed"]


class DisplacedSqueezed(Ket):
    r"""
    The displaced squeezed state in Bargmann representation.

    >>> from mrmustard.lab import DisplacedSqueezed, Vacuum, Sgate, Dgate
    >>> state = DisplacedSqueezed(mode=0, alpha=1, r=0.2, phi=0.3)
    >>> assert state == Vacuum(0) >> Sgate(0, r=0.2, phi=0.3) >> Dgate(0, alpha=1)

    Args:
        mode: The mode of the displaced squeezed state.
        alpha: The complex displacement.
        r: The squeezing magnitude.
        phi: The squeezing angle.

    Returns:
        A ``Ket``.
    """

    short_name = "DSq"

    def __init__(
        self,
        mode: int,
        alpha: complex | Sequence[complex] | Parameter = 0.0j,
        r: float | Sequence[float] | Parameter = 0.0,
        phi: float | Sequence[float] | Parameter = 0.0,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (
                        displaced_squeezed_vacuum_state,
                        ("alpha", "r", "phi", "lin_sup"),
                    )
                }
            ),
            wires=Wires(modes_out_ket=set(mode)),
            name=self.__class__.__name__,
        )
        self.parameters["alpha"] = Parameter.from_cc_init(alpha, "complex128", f"{self.name}/alpha")
        self.parameters["r"] = Parameter.from_cc_init(r, "float64", f"{self.name}/r")
        self.parameters["phi"] = Parameter.from_cc_init(phi, "float64", f"{self.name}/phi")
