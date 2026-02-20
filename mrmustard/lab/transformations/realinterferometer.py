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
The class representing a RealInterferometer gate.
"""

from __future__ import annotations

from mrmustard import math
from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import RealMatrix

from .base import Unitary
from .builtins import real_interferometer_gate

__all__ = ["RealInterferometer"]


class RealInterferometer(Unitary):
    r"""
    N-mode interferometer parametrized by an NxN orthogonal matrix (or 2N x 2N block-diagonal orthogonal matrix).
    Does not mix q's and p's.

    >>> from mrmustard import math
    >>> from mrmustard.lab import RealInterferometer, Identity
    >>> ri = RealInterferometer([0, 1], orthogonal = math.eye(2))
    >>> assert ri == Identity((0,1))

    Args:
        modes: The modes this gate is applied to.
        orthogonal: A real unitary (orthogonal) matrix.  For N modes it must have shape `(N,N)`.
    """

    short_name = "RI"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        orthogonal: RealMatrix | Parameter,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={
                    ReprEnum.BARGMANN: (real_interferometer_gate, ("orthogonal", "lin_sup"))
                }
            ),
            wires=Wires(modes_in_ket=set(modes), modes_out_ket=set(modes)),
            name=self.__class__.__name__,
        )
        self.parameters["orthogonal"] = Parameter.from_cc_init(
            orthogonal, "float64", f"{self.name}/orthogonal"
        )
        if (size := self.parameters.orthogonal.value.shape[-1]) != len(modes):
            raise ValueError(
                f"The size of the orthogonal matrix must match the number of modes: {size} =/= {len(modes)}",
            )

    @classmethod
    def random(cls, modes: int | tuple[int, ...], seed: int | None = None) -> RealInterferometer:
        r"""
        Returns a random RealInterferometer.

        Args:
            modes: The modes of the RealInterferometer.
            seed: The random seed. If ``None``, the global seed is used.

        Returns:
            The random RealInterferometer.

        Raises:
            ValueError: if ``modes`` is an empty tuple.
        """
        modes = (modes,) if isinstance(modes, int) else modes
        if len(modes) == 0:
            raise ValueError("Cannot create a random RealInterferometer with no modes.")
        orthogonal = math.random_orthogonal(len(modes), seed)
        return cls(modes, orthogonal)
