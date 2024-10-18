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
The class representing a noisy attenuator channel.
"""

from __future__ import annotations

from typing import Sequence
from mrmustard.utils.typing import RealMatrix
from .base import Channel
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter

__all__ = ["GRN"]


class GRN(Channel):
    r"""
    The Gaussian random noise channel.

    Y_train determines whether or not the Y matrix is trainable.
    The number of modes must match half of the size of the Y matrix.
    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import GRN

        >>> channel = GRN(modes=[1, 2], Y = .2 * np.eye(4))
        >>> assert channel.modes == [1, 2]
        >>> assert np.allclose(channel.Y.value, .2 * np.eye(4))
    
    Args:
        modes: The modes the channel is applied to
        Y: The Y matrix of the Gaussian random noise
        Y_train: whether the Y matrix is a trainable variable
    
    ..details::
        The Bargmann representation of the channel is computed via the formulas provided in the paper:
        https://arxiv.org/pdf/2209.06069
    """

    short_name = "GRN"

    def __init__(
        self,
        modes: Sequence[int],
        Y: RealMatrix,
        Y_train: bool = False,
    ):

        if Y.shape[-1] // 2 != len(modes):
            raise ValueError(
                f"The number of modes does not match the dimension of the "
                f"Y matrix: {Y.shape[-1] // 2} =/= {len(modes)}."
            )

        super().__init__(modes_out=modes, modes_in=modes, name="GRN")
        self._add_parameter(make_parameter(Y_train, value=Y, name="Y", bounds=(None, None)))

        self._representation = Bargmann.from_function(
            fn=triples.gaussian_random_noise_Abc, Y=self.Y.value
        )
