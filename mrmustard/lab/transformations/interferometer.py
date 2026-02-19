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

"""
The class representing an Interferometer gate.
"""

from __future__ import annotations

from mrmustard import math
from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import ComplexMatrix

from .base import Unitary
from .builtins import interferometer_gate

__all__ = ["Interferometer"]


class Interferometer(Unitary):
    r"""
    N-mode interferometer.

    It corresponds to a Ggate with zero mean and a ``2N x 2N`` unitary symplectic matrix.

    >>> from mrmustard import math
    >>> from mrmustard.lab import Interferometer
    >>> unitary = Interferometer(modes=(1, 2), unitary=math.eye(2))
    >>> assert unitary.modes == (1, 2)
    >>> assert math.allclose(unitary.symplectic, math.eye(4))

    Args:
        modes: The modes this gate is applied to.
        unitary: A unitary matrix. For N modes it must have shape `(N,N)`.

    Raises:
        ValueError: If the size of the unitary does not match the number of modes.
    """

    short_name = "I"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        unitary: ComplexMatrix | Parameter,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        num_modes = len(modes)
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (interferometer_gate, ("unitary", "lin_sup"))}
            ),
            wires=Wires(modes_in_ket=set(modes), modes_out_ket=set(modes)),
            name=self.__class__.__name__,
        )
        self.parameters["unitary"] = Parameter.from_cc_init(
            unitary, "complex128", f"{self.name}/unitary"
        )
        if (size := self.parameters.unitary.value.shape[-1]) != num_modes:
            raise ValueError(
                f"The size of the unitary must match the number of modes: {size} =/= {num_modes}",
            )

    @classmethod
    def random(cls, modes: int | tuple[int, ...], seed: int | None = None) -> Interferometer:
        r"""
        Returns a random Interferometer.

        Args:
            modes: The modes of the Interferometer.
            seed: The random seed. If ``None``, the global seed is used.

        Returns:
            The random Interferometer.

        Raises:
            ValueError: if ``modes`` is an empty tuple.
        """
        modes = (modes,) if isinstance(modes, int) else modes
        if len(modes) == 0:
            raise ValueError("Cannot create a random Interferometer with no modes.")
        unitary = math.random_unitary(len(modes), seed)
        return cls(modes, unitary)
