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
from mrmustard.math.parameters import update_orthogonal
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.utils.typing import RealMatrix

from .base import Unitary
from ..utils import make_parameter
from ...physics import symplectics

__all__ = ["RealInterferometer"]


class RealInterferometer(Unitary):
    r"""
    N-mode interferometer parametrized by an NxN orthogonal matrix (or 2N x 2N block-diagonal orthogonal matrix).
    Does not mix q's and p's.

    Args:
        modes: The modes this gate is applied to.
        orthogonal: A real unitary (orthogonal) matrix.  For N modes it must have shape `(N,N)`. If ``None``, a random orthogonal is generated.
        orthogonal_trainable: Whether ``orthogonal`` is trainable.

    .. code-block::

        >>> from mrmustard import math
        >>> from mrmustard.lab_dev import RealInterferometer, Identity
        >>> ri = RealInterferometer([0, 1], orthogonal = math.eye(2))
        >>> assert ri == Identity((0,1))
    """

    short_name = "RI"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        orthogonal: RealMatrix | None = None,
        orthogonal_trainable: bool = False,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        num_modes = len(modes)
        if orthogonal is not None and orthogonal.shape[-1] != num_modes:
            raise ValueError(
                f"The size of the orthogonal matrix must match the number of modes: {orthogonal.shape[-1]} =/= {num_modes}"
            )

        orthogonal = orthogonal if orthogonal is not None else math.random_orthogonal(num_modes)

        super().__init__(name="RealInterferometer")
        self.parameters.add_parameter(
            make_parameter(
                orthogonal_trainable, orthogonal, "orthogonal", (None, None), update_orthogonal
            )
        )
        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=lambda ortho: Unitary.from_symplectic(
                    modes, symplectics.realinterferometer_symplectic(ortho)
                ).bargmann_triple(),
                ortho=self.parameters.orthogonal,
            ),
        ).representation
