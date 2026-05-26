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

from .base import Unitary
from .builtins import rotation_gate

__all__ = ["Rgate"]


class Rgate(Unitary):
    r"""The rotation gate.

    >>> from mrmustard.lab import Rgate
    >>> unitary = Rgate(mode=1, theta=0.1)
    >>> assert unitary.modes == (1,)

    Args:
        mode: The mode this gate is applied to.
        theta: The rotation angle.
        name: A name for the gate. If not provided, the class name will be used.
    """

    short_name = "R"

    def __init__(
        self,
        mode: int | tuple[int],
        theta: float | Sequence[float] | Parameter = 0.0,
        name: str | None = None,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        name = name if name is not None else self.__class__.__name__
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (rotation_gate, ("theta", "lin_sup"))}
            ),
            wires=Wires(modes_in_ket=set(mode), modes_out_ket=set(mode)),
            name=name,
        )
        self.parameters["theta"] = Parameter.from_cc_init(theta, "float64", f"{self.name}/theta")
