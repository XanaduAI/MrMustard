# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The class representing a squeezed thermal state."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import squeezed_thermal_state
from .dm import DM

__all__ = ["SqueezedThermal"]


class SqueezedThermal(DM):
    r"""The squeezed thermal state in Bargmann representation.

    >>> from mrmustard.lab import Sgate, SqueezedThermal, Thermal, Vacuum
    >>> state = SqueezedThermal(mode=0, nbar=1, r=0.2, phi=0.3)
    >>> assert state == Thermal(0, nbar=1) >> Sgate(0, r=0.2, phi=0.3)

    Args:
        mode: The mode of the squeezed thermal state.
        nbar: The expected number of photons.
        r: The squeezing magnitude.
        phi: The squeezing angle.
        name: A name for the state. If not provided, the class name will be used.

    Returns:
        A ``DM`` type object that represents the squeezed thermal state.
    """

    short_name = "SqTh"

    def __init__(
        self,
        mode: int | tuple[int],
        nbar: float | Sequence[float] | Parameter = 0.0,
        r: float | Sequence[float] | Parameter = 0.0,
        phi: float | Sequence[float] | Parameter = 0.0,
        name: str | None = None,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (
                        squeezed_thermal_state,
                        ("nbar", "r", "phi", "lin_sup"),
                    )
                }
            ),
            wires=Wires(modes_out_bra=set(mode), modes_out_ket=set(mode)),
            name=name,
        )
        self.parameters["nbar"] = Parameter.from_cc_init(nbar, "float64", f"{self.name}/nbar")
        self.parameters["r"] = Parameter.from_cc_init(r, "float64", f"{self.name}/r")
        self.parameters["phi"] = Parameter.from_cc_init(phi, "float64", f"{self.name}/phi")
