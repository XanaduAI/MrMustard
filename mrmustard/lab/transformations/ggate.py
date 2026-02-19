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
The class representing a generic gaussian gate.
"""

from __future__ import annotations

from mrmustard import math
from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import RealMatrix

from .base import Unitary
from .builtins import gaussian_gate

__all__ = ["Ggate"]


class Ggate(Unitary):
    r"""
    The generic N-mode Gaussian gate.

    >>> from mrmustard import math
    >>> from mrmustard.lab import Ggate, Vacuum, Identity, Ket
    >>> U = Ggate.random(modes=0)
    >>> assert isinstance(Vacuum(0) >> U, Ket)
    >>> assert U >> U.dual == Identity(0)

    Args:
        modes: The modes this gate is applied to.
        symplectic: The symplectic matrix of the gate in the XXPP ordering.
    """

    short_name = "G"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        symplectic: RealMatrix | Parameter,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (gaussian_gate, ("symplectic", "lin_sup"))}
            ),
            wires=Wires(modes_in_ket=set(modes), modes_out_ket=set(modes)),
            name=self.__class__.__name__,
        )
        self.parameters["symplectic"] = Parameter.from_cc_init(
            symplectic, "float64", f"{self.name}/symplectic"
        )

    @property
    def symplectic(self):
        return self.parameters.symplectic.value

    @classmethod
    def random(
        cls, modes: int | tuple[int, ...], max_r: float = 1.0, seed: int | None = None
    ) -> Ggate:
        r"""
        Returns a random Ggate.

        Args:
            modes: The modes of the Ggate.
            max_r: Maximum squeezing parameter over which we make random choices.
            seed: The random seed. If ``None``, the global seed is used.

        Returns:
            The random Ggate.

        Raises:
            ValueError: if ``modes`` is an empty tuple.
        """
        modes = (modes,) if isinstance(modes, int) else modes
        if len(modes) == 0:
            raise ValueError("Cannot create a random Ggate with no modes.")
        symplectic = math.random_symplectic(len(modes), max_r=max_r, seed=seed)
        return cls(modes, symplectic)
