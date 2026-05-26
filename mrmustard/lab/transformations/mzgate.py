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

"""The class representing a Mach-Zehnder gate."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires

from .base import Unitary
from .builtins import mz_gate

__all__ = ["MZgate"]


class MZgate(Unitary):
    r"""Mach-Zehnder gate.

    It supports two conventions:
        1. if ``internal=True``, both phases act inside the interferometer: ``phi_a`` on the upper arm, ``phi_b`` on the lower arm.
        2. if ``internal = False``, both phases act on the upper arm: ``phi_a`` before the first BS, ``phi_b`` after the first BS.

    >>> from mrmustard.lab import MZgate
    >>> mz = MZgate((0, 1), phi_a=0.1, phi_b=0.2)
    >>> assert mz.modes == (0, 1)
    >>> assert mz.parameters.phi_a.value == 0.1
    >>> assert mz.parameters.phi_b.value == 0.2

    Args:
        modes: The pair of modes of the MZ gate.
        phi_a: The phase in the upper arm of the MZ interferometer.
        phi_b: The phase in the lower arm or external of the MZ interferometer.
        internal: Whether phases are both in the internal arms.
        name: A name for the gate. If not provided, the class name will be used.
    """

    short_name = "MZ"

    def __init__(
        self,
        modes: tuple[int, int],
        phi_a: float | Sequence[float] | Parameter = 0.0,
        phi_b: float | Sequence[float] | Parameter = 0.0,
        internal: bool = False,
        name: str | None = None,
    ):
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (mz_gate, ("phi_a", "phi_b", "internal", "lin_sup"))
                },
                internal=internal,
            ),
            wires=Wires(modes_in_ket=set(modes), modes_out_ket=set(modes)),
            name=name,
        )
        self.parameters["phi_a"] = Parameter.from_cc_init(phi_a, "float64", f"{self.name}/phi_a")
        self.parameters["phi_b"] = Parameter.from_cc_init(phi_b, "float64", f"{self.name}/phi_b")
