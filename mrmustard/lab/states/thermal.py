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

"""The class representing a thermal state."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .builtins import thermal_state
from .dm import DM

__all__ = ["Thermal"]


class Thermal(DM):
    r"""The thermal state in Bargmann representation.

    >>> from mrmustard.lab import Thermal
    >>> state = Thermal(1, nbar=3)
    >>> assert state.modes == (1,)

    Args:
        mode: The mode of the thermal state.
        nbar: The expected number of photons.
        name: A name for the state. If not provided, the class name will be used.

    Returns:
        A ``DM`` type object that represents the thermal state.
    """

    short_name = "Th"

    def __init__(
        self,
        mode: int | tuple[int],
        nbar: float | Sequence[float] | Parameter = 0.0,
        name: str | None = None,
    ) -> None:
        mode = (mode,) if not isinstance(mode, tuple) else mode
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (thermal_state, ("nbar", "lin_sup"))}
            ),
            wires=Wires(modes_out_bra=set(mode), modes_out_ket=set(mode)),
            name=name,
        )
        self.parameters["nbar"] = Parameter.from_cc_init(nbar, "float64", f"{self.name}/nbar")
