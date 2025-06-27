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
The class representing a Gaussian random noise channel.
"""

from __future__ import annotations

from mrmustard import math, settings
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import RealMatrix

from ...physics import triples
from ..utils import make_parameter
from .base import Channel

__all__ = ["GaussRandNoise"]


class GaussRandNoise(Channel):
    r"""
    The Gaussian random noise channel.


    Args:
        modes: The modes the channel is applied to. The number of modes must match half of the size of ``Y``.
        Y: The Y matrix of the Gaussian random noise channel.
        Y_trainable: Whether ``Y`` is trainable.

    .. code-block::

        >>> import numpy as np
        >>> from mrmustard.lab import GaussRandNoise

        >>> channel = GaussRandNoise(modes=(1, 2), Y = 0.2 * np.eye(4))
        >>> assert channel.modes == (1, 2)
        >>> assert math.allclose(channel.parameters.Y.value, 0.2 * np.eye(4))

    Raises:
        ValueError: If the number of modes does not match half of the size of ``Y``.

    .. details::
        The Bargmann representation of the channel is computed via the formulas provided in the paper:
        https://arxiv.org/pdf/2209.06069

        The channel maps an inout covariance matrix ``cov`` as

        .. math::

                cov \mapsto cov + Y.
    """

    short_name = "GRN"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        Y: RealMatrix,
        Y_trainable: bool = False,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        if Y.shape[-1] // 2 != len(modes):
            raise ValueError(
                f"The number of modes {len(modes)} does not match the dimension of the "
                f"Y matrix {Y.shape[-1] // 2}.",
            )

        Y_eigenvectors_real = math.real(math.eigvals(Y))
        math.error_if(
            Y_eigenvectors_real,
            Y_eigenvectors_real < -settings.ATOL,
            "The input Y matrix has negative eigen-values.",
        )

        super().__init__(name="GRN~")
        self.parameters.add_parameter(
            make_parameter(is_trainable=Y_trainable, value=Y, name="Y", bounds=(None, None)),
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=triples.gaussian_random_noise_Abc,
            Y=self.parameters.Y,
        )
        self._wires = Wires(
            modes_in_bra=set(modes),
            modes_out_bra=set(modes),
            modes_in_ket=set(modes),
            modes_out_ket=set(modes),
        )
