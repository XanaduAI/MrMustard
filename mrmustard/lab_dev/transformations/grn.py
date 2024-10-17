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

from mrmustard.utils.typing import RealMatrix
from typing import Sequence
from .base import Channel
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter, reshape_params

__all__ = ["Attenuator"]


class GRN(Channel):
    r"""
    The Gaussian random noise channel.
    """

    short_name = "GRN"

    def __init__(
        self,
        modes: Sequence[int],
        Y: RealMatrix,
        Y_train: bool = False,
    ):

        super().__init__(modes_out=modes, modes_in=modes, name="GRN")
        self._add_parameter(make_parameter(Y_train, value=Y, name="Y", bounds=(None, None)))

        self._representation = Bargmann.from_function(
            fn=triples.gaussian_random_noise_Abc, Y=self.Y.value
        )
